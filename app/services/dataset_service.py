"""
Dataset service — create datasets and scan source_uri for recordings.
No snippet generation yet.
"""

import hashlib
import os
from typing import List, Optional

import soundfile as sf
from sqlalchemy import exists, and_, or_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models.dataset import Dataset as DatasetModel, user_datasets
from app.models.recording import Recording as RecordingModel
from app.models.team import Team as TeamModel
from app.models.team import TeamMembership as TeamMembershipModel
from app.models.team import TeamRole
from app.models.user import User, UserRole
from app.schemas.dataset import DatasetCreate
from app.config import settings

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


class DatasetService:
    def __init__(self, db: Session):
        self.db = db

    # ---------------------------------------------------------
    # Dataset operations
    # ---------------------------------------------------------

    def create_dataset(self, dataset_in: DatasetCreate, current_user: User) -> DatasetModel:
        """
        Create dataset with uniqueness check on (team_id, source_uri).

        Raises:
            ValueError("team_id_required") if a non-admin creates dataset without team_id.
            ValueError("duplicate_dataset") if the dataset already exists.
            ValueError("team_not_found") if team_id is invalid.
            ValueError("invalid_source_uri") if source_uri path does not exist.
        """

        # Non-admin users MUST supply a team_id
        if current_user.role != UserRole.ADMIN and dataset_in.team_id is None:
            raise ValueError("team_id_required")

        # Validate team
        if dataset_in.team_id is not None:
            team = self.db.query(TeamModel).filter(TeamModel.id == dataset_in.team_id).first()
            if not team:
                raise ValueError("team_not_found")

        # Validate source_uri path before committing
        self.validate_source_uri(dataset_in.source_uri)

        # Proactive duplicate check for (team_id, source_uri)
        duplicate = (
            self.db.query(
                exists().where(
                    and_(
                        DatasetModel.team_id == dataset_in.team_id,
                        DatasetModel.source_uri == dataset_in.source_uri,
                    )
                )
            ).scalar()
        )
        if duplicate:
            raise ValueError("duplicate_dataset")

        dataset = DatasetModel(**dataset_in.dict())
        self.db.add(dataset)
        try:
            self.db.commit()
        except IntegrityError:
            self.db.rollback()
            # Fallback if uniqueness was enforced only at DB level
            raise ValueError("duplicate_dataset")

        self.db.refresh(dataset)
        return dataset

    def delete_dataset(self, dataset: DatasetModel) -> None:
        """
        Delete dataset and its recordings (cascade).
        """
        self.db.delete(dataset)
        self.db.commit()

    def list_datasets(self, current_user: User, skip: int = 0, limit: int = 100):
        # Admins see everything
        if current_user.role == UserRole.ADMIN:
            return (
                self.db.query(DatasetModel)
                .offset(skip)
                .limit(limit)
                .all()
            )

        # Non-admin users: datasets from teams where user is any member (owner or user)
        member_team_ids = (
            self.db.query(TeamModel.id)
            .join(TeamMembershipModel)
            .filter(TeamMembershipModel.user_id == current_user.id)
            .all()
        )
        member_team_ids = [t[0] for t in member_team_ids]

        # Datasets with direct access granted via invitation (user_datasets table)
        direct_access_ids = (
            self.db.query(user_datasets.c.dataset_id)
            .filter(user_datasets.c.user_id == current_user.id)
            .all()
        )
        direct_access_ids = [r[0] for r in direct_access_ids]

        if not member_team_ids and not direct_access_ids:
            return []

        filters = []
        if member_team_ids:
            filters.append(DatasetModel.team_id.in_(member_team_ids))
        if direct_access_ids:
            filters.append(DatasetModel.id.in_(direct_access_ids))

        return (
            self.db.query(DatasetModel)
            .filter(or_(*filters))
            .offset(skip)
            .limit(limit)
            .all()
        )

    def get_dataset(self, dataset_id: int) -> Optional[DatasetModel]:
        return self.db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()

    def get_dataset_for_user(self, dataset_id: int, current_user: User) -> Optional[DatasetModel]:
        """Fetch a dataset only if the user is allowed to access it.

        - Admins: unrestricted access.
        - Others: must be a member (any role) of the dataset's team, or have
          direct access via the user_datasets grant table.
        Returns None if the dataset doesn't exist or the user has no access.
        """
        dataset = self.get_dataset(dataset_id)
        if not dataset:
            return None
        if current_user.role == UserRole.ADMIN:
            return dataset

        if dataset.team_id is not None:
            membership = (
                self.db.query(TeamMembershipModel)
                .filter(
                    TeamMembershipModel.team_id == dataset.team_id,
                    TeamMembershipModel.user_id == current_user.id,
                )
                .first()
            )
            return dataset if membership else None

        # Dataset not yet assigned to a team — fall back to direct-access grant
        direct = (
            self.db.query(user_datasets)
            .filter(
                user_datasets.c.user_id == current_user.id,
                user_datasets.c.dataset_id == dataset_id,
            )
            .first()
        )
        return dataset if direct else None

    def can_delete_dataset(self, dataset: DatasetModel, current_user: User) -> bool:
        """Return True if the user may delete this dataset.

        - Admins: always allowed.
        - Others: must be OWNER of the dataset's team.
        """
        if current_user.role == UserRole.ADMIN:
            return True
        if dataset.team_id is None:
            return False
        # Role comparison done in Python to avoid enum name/value mismatch
        # with the native PostgreSQL ENUM type.
        membership = (
            self.db.query(TeamMembershipModel)
            .filter(
                TeamMembershipModel.team_id == dataset.team_id,
                TeamMembershipModel.user_id == current_user.id,
            )
            .first()
        )
        return bool(membership and membership.role == TeamRole.OWNER)

    # ---------------------------------------------------------
    # Path validation
    # ---------------------------------------------------------

    def validate_source_uri(self, source_uri: str) -> None:
        """
        Validate that the source_uri path exists and is a directory.
        
        Raises:
            ValueError("invalid_source_uri") if the path does not exist or is not a directory.
        """
        DATA_ROOT = settings.DATA_ROOT or "/data"
        dataset_path = os.path.join(DATA_ROOT, source_uri)
        
        if not os.path.isdir(dataset_path):
            raise ValueError("invalid_source_uri")

    # ---------------------------------------------------------
    # Recording discovery
    # ---------------------------------------------------------

    def scan_recordings(self, dataset: DatasetModel) -> List[RecordingModel]:
        """
        Walk dataset.source_uri (relative to DATA_ROOT),
        detect audio files, and create Recording rows.

        Returns a list of newly created recordings.
        """
        DATA_ROOT = settings.DATA_ROOT or "/data"
        dataset_path = os.path.join(DATA_ROOT, dataset.source_uri)

        if not os.path.isdir(dataset_path):
            raise ValueError(f"Invalid dataset path: {dataset_path}")

        audio_files = self._scan_audio_files(dataset_path)
        new_recordings: List[RecordingModel] = []

        for fpath in audio_files:
            rec = self._get_or_create_recording(dataset, fpath)
            if rec is not None:
                new_recordings.append(rec)

        return new_recordings

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------

    def _scan_audio_files(self, root_dir: str) -> List[str]:
        audio_files: List[str] = []
        for root, _dirs, files in os.walk(root_dir):
            for filename in files:
                _, ext = os.path.splitext(filename.lower())
                if ext in AUDIO_EXTENSIONS:
                    audio_files.append(os.path.join(root, filename))
        return sorted(audio_files)

    def _compute_checksum(self, filepath: str) -> str:
        """Compute a SHA-256 checksum for the given file."""
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()

    def _get_or_create_recording(
            self, dataset: DatasetModel, filepath: str
    ) -> Optional[RecordingModel]:
        # Store relative path (relative to DATA_ROOT) for portability
        DATA_ROOT = settings.DATA_ROOT or "/data"
        if filepath.startswith(DATA_ROOT):
            relative_path = os.path.relpath(filepath, DATA_ROOT)
        else:
            relative_path = filepath
        
        existing = (
            self.db.query(RecordingModel)
            .filter(
                RecordingModel.dataset_id == dataset.id,
                RecordingModel.file_path == relative_path,
            )
            .first()
        )
        if existing:
            return None

        # Extract duration & sample rate from audio
        try:
            info = sf.info(filepath)
            duration = float(info.frames) / float(info.samplerate)
            sample_rate = int(info.samplerate)
        except Exception as e:
            # Log error but skip unreadable audio files
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to read audio file {filepath}: {e}")
            return None

        checksum = self._compute_checksum(filepath)

        rec = RecordingModel(
            dataset_id=dataset.id,
            file_path=relative_path,  # Store relative path
            file_name=os.path.basename(filepath),
            duration=duration,
            sample_rate=sample_rate,
            extra_metadata=None,
            audio_sha256=checksum,
        )
        self.db.add(rec)
        self.db.commit()
        self.db.refresh(rec)
        return rec
