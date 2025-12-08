"""
Dataset service — create datasets and scan source_uri for recordings.
No snippet generation yet.
"""

import os
import hashlib
from typing import List, Optional

import soundfile as sf
from sqlalchemy import exists, and_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models.dataset import Dataset
from app.models.recording import Recording
from app.models.team import Team
from app.models.user import User
from app.schemas.dataset import DatasetCreate

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


class DatasetService:
    def __init__(self, db: Session):
        self.db = db

    # ---------------------------------------------------------
    # Dataset operations
    # ---------------------------------------------------------

    def create_dataset(self, dataset_in: DatasetCreate, current_user: User) -> Dataset:
        """
        Create dataset with uniqueness check on (team_id, source_uri).

        Raises:
            ValueError("duplicate_dataset") if the dataset already exists.
            ValueError("team_not_found") if team_id is invalid.
        """
        # Validate team
        if dataset_in.team_id is not None:
            team = self.db.query(Team).filter(Team.id == dataset_in.team_id).first()
            if not team:
                raise ValueError("team_not_found")
        # Admin-created datasets (team_id = None) are claimable later.

        # Proactive duplicate check for (team_id, source_uri)
        duplicate = (
            self.db.query(
                exists().where(
                    and_(
                        Dataset.team_id == dataset_in.team_id,
                        Dataset.source_uri == dataset_in.source_uri,
                    )
                )
            ).scalar()
        )
        if duplicate:
            raise ValueError("duplicate_dataset")

        dataset = Dataset(**dataset_in.dict())

        self.db.add(dataset)
        try:
            self.db.commit()
        except IntegrityError:
            self.db.rollback()
            # Fallback if uniqueness was enforced only at DB level
            raise ValueError("duplicate_dataset")

        self.db.refresh(dataset)
        return dataset

    def delete_dataset(self, dataset: Dataset) -> None:
        """
        Delete dataset and its recordings (cascade).
        """
        self.db.delete(dataset)
        self.db.commit()

    def claim_dataset(self, dataset: Dataset, user: User) -> Dataset:
        """
        Allow a user to claim ownership of an admin-created dataset (team_id NULL).
        """
        dataset.team_id = user.team_id
        self.db.commit()
        self.db.refresh(dataset)
        return dataset

    def list_datasets(self, skip: int, limit: int) -> List[Dataset]:
        return (
            self.db.query(Dataset)
            .order_by(Dataset.created_at.desc())
            .offset(skip)
            .limit(limit)
            .all()
        )

    def get_dataset(self, dataset_id: int) -> Optional[Dataset]:
        return self.db.query(Dataset).filter(Dataset.id == dataset_id).first()

    # ---------------------------------------------------------
    # Recording discovery
    # ---------------------------------------------------------

    def scan_recordings(self, dataset: Dataset) -> List[Recording]:
        """
        Walk dataset.source_uri (relative to INTERNAL_DATA_ROOT, default /data),
        detect audio files, and create Recording rows.

        Returns a list of newly created recordings.
        """
        INTERNAL_DATA_ROOT = os.getenv("INTERNAL_DATA_ROOT", "/data")
        dataset_path = os.path.join(INTERNAL_DATA_ROOT, dataset.source_uri)

        if not os.path.isdir(dataset_path):
            raise ValueError(f"Invalid dataset path: {dataset_path}")

        audio_files = self._scan_audio_files(dataset_path)
        new_recordings: List[Recording] = []

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
        self, dataset: Dataset, filepath: str
    ) -> Optional[Recording]:
        existing = (
            self.db.query(Recording)
            .filter(
                Recording.dataset_id == dataset.id,
                Recording.file_path == filepath,
            )
            .first()
        )
        if existing:
            return None  # not "new"

        # Extract duration & sample rate from audio
        try:
            info = sf.info(filepath)
            duration = float(info.frames) / float(info.samplerate)
            sample_rate = int(info.samplerate)
        except Exception:
            # Skip unreadable audio files silently for now
            return None

        checksum = self._compute_checksum(filepath)

        rec = Recording(
            dataset_id=dataset.id,
            file_path=filepath,
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
