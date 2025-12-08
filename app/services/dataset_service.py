"""
Dataset service — create datasets and scan source_uri for recordings.
No snippet generation yet.
"""

import os
from typing import List, Optional

import soundfile as sf
from sqlalchemy import exists
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
        """
        # Validate team
        if dataset_in.team_id is not None:
            team = self.db.query(Team).filter(Team.id == dataset_in.team_id).first()
            if not team:
                raise ValueError("team_not_found")
        else:
            # Admin-created datasets: team_id = None, owner will be claimable later
            pass

            # Proactive duplicate check
            duplicate = (
                self.db.query(
                    exists().where(
                        Dataset.team_id == dataset_in.team_id,
                        Dataset.source_uri == dataset_in.source_uri,
                    )
                )
                .scalar()
            )
            if duplicate:
                raise ValueError("duplicate_dataset")

        dataset = Dataset(**dataset_in.dict())

        self.db.add(dataset)
        try:
            self.db.commit()
        except IntegrityError:
            self.db.rollback()
            raise ValueError("duplicate_dataset")

        self.db.refresh(dataset)
        return dataset

    def delete_dataset(self, dataset: Dataset) -> None:
        """
        Delete dataset and recordings (cascade).
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
        Walk dataset.source_uri (relative to DATA_ROOT mounted as /data),
        detect audio files, and create Recording rows.

        Returns a list of newly created recordings.
        """
        # Resolve absolute path under DATA_ROOT
        INTERNAL_DATA_ROOT = os.getenv("INTERNAL_DATA_ROOT", "/data")
        dataset_path = os.path.join(INTERNAL_DATA_ROOT, dataset.source_uri)

        if not os.path.isdir(dataset_path):
            raise ValueError(f"Invalid dataset path: {dataset_path}")

        audio_files = self._scan_audio_files(dataset_path)
        new_recordings = []

        for fpath in audio_files:
            rec = self._get_or_create_recording(dataset, fpath)
            if rec is not None:
                new_recordings.append(rec)

        return new_recordings

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------

    def _scan_audio_files(self, root_dir: str) -> List[str]:
        audio_files = []
        for root, _dirs, files in os.walk(root_dir):
            for filename in files:
                _, ext = os.path.splitext(filename.lower())
                if ext in AUDIO_EXTENSIONS:
                    audio_files.append(os.path.join(root, filename))
        return sorted(audio_files)

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
            return None

        rec = Recording(
            dataset_id=dataset.id,
            file_path=filepath,
            file_name=os.path.basename(filepath),
            duration=duration,
            sample_rate=sample_rate,
            extra_metadata=None,
        )
        self.db.add(rec)
        self.db.commit()
        self.db.refresh(rec)
        return rec

#TODO: compute checksum and include in db entry