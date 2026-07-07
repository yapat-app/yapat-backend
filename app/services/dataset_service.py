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
from app.schemas.dataset import DatasetCreate, DatasetUpdate
from app.core.permissions import check_team_owner_membership
from app.config import settings
from app.utils.recording_filename_metadata import (
    location_source_for_filename,
    parse_location_from_filename,
    parse_datetime_from_filename,
)

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

    def user_can_access_dataset(self, user: User, dataset_id: int) -> bool:
        dataset = self.get_dataset(dataset_id)
        if dataset is None:
            return False
        if user.role == UserRole.ADMIN:
            return True

        if dataset.team_id is not None:
            membership = (
                self.db.query(TeamMembershipModel.id)
                .filter(
                    TeamMembershipModel.team_id == dataset.team_id,
                    TeamMembershipModel.user_id == user.id,
                )
                .first()
            )
            if membership is not None:
                return True

        direct_access = (
            self.db.query(user_datasets.c.dataset_id)
            .filter(
                user_datasets.c.user_id == user.id,
                user_datasets.c.dataset_id == dataset_id,
            )
            .first()
        )
        return direct_access is not None

    def user_has_wssed_access(self, user: User) -> bool:
        """True when the user can open WSSED (admins always; others need focal datasets)."""
        if user.role == UserRole.ADMIN:
            return True

        from app.models.dataset import DatasetType

        query = self.db.query(DatasetModel.id).filter(
            DatasetModel.dataset_type == DatasetType.FOCAL_RECORDINGS
        )
        member_team_ids = [
            row[0]
            for row in self.db.query(TeamMembershipModel.team_id)
            .filter(TeamMembershipModel.user_id == user.id)
            .all()
        ]
        direct_access_ids = [
            row[0]
            for row in self.db.query(user_datasets.c.dataset_id)
            .filter(user_datasets.c.user_id == user.id)
            .all()
        ]
        filters = []
        if member_team_ids:
            filters.append(DatasetModel.team_id.in_(member_team_ids))
        if direct_access_ids:
            filters.append(DatasetModel.id.in_(direct_access_ids))
        if not filters:
            return False
        query = query.filter(or_(*filters))
        return query.first() is not None

    def get_focal_dataset_for_user(
        self, user: User, dataset_id: int
    ) -> Optional[DatasetModel]:
        from app.models.dataset import DatasetType

        dataset = self.get_dataset(dataset_id)
        if dataset is None or dataset.dataset_type != DatasetType.FOCAL_RECORDINGS:
            return None
        if not self.user_can_access_dataset(user, dataset_id):
            return None
        return dataset

    def get_dataset(self, dataset_id: int) -> Optional[DatasetModel]:
        return self.db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()

    def user_can_manage_dataset(self, user: User, dataset: DatasetModel) -> bool:
        """Platform admin or team owner for the dataset's team."""
        if user.role == UserRole.ADMIN:
            return True
        if dataset.team_id is None:
            return False
        return check_team_owner_membership(user, dataset.team_id, self.db)

    @staticmethod
    def _validate_spectrogram_range(
        f_min: Optional[float],
        f_max: Optional[float],
    ) -> None:
        effective_min = 0.0 if f_min is None else float(f_min)
        if f_min is not None and f_min < 0:
            raise ValueError("invalid_spectrogram_f_min")
        if f_max is not None:
            f_max_f = float(f_max)
            if f_max_f <= 0:
                raise ValueError("invalid_spectrogram_f_max")
            if f_max_f <= effective_min:
                raise ValueError("spectrogram_f_max_lte_min")

    def update_dataset(
        self,
        dataset_id: int,
        update_in: DatasetUpdate,
        current_user: User,
    ) -> DatasetModel:
        dataset = self.get_dataset(dataset_id)
        if dataset is None:
            raise ValueError("not_found")
        if not self.user_can_manage_dataset(current_user, dataset):
            raise ValueError("forbidden")

        data = update_in.model_dump(exclude_unset=True)
        if not data:
            return dataset

        if "source_uri" in data and data["source_uri"] is not None:
            self.validate_source_uri(data["source_uri"])

        next_f_min = data.get("spectrogram_f_min_hz", dataset.spectrogram_f_min_hz)
        next_f_max = data.get("spectrogram_f_max_hz", dataset.spectrogram_f_max_hz)
        if "spectrogram_f_min_hz" in data or "spectrogram_f_max_hz" in data:
            self._validate_spectrogram_range(next_f_min, next_f_max)

        for key, value in data.items():
            setattr(dataset, key, value)

        self.db.commit()
        self.db.refresh(dataset)
        return dataset

    # ---------------------------------------------------------
    # Path validation
    # ---------------------------------------------------------

    def validate_source_uri(self, source_uri: str) -> None:
        """
        Validate that the source_uri path exists and is a directory.
        
        Raises:
            ValueError("invalid_source_uri") if the path does not exist or is not a directory.
        """
        if not source_uri or not str(source_uri).strip():
            raise ValueError("invalid_source_uri")

        DATA_ROOT = settings.DATA_ROOT or "/data"
        dataset_path = os.path.normpath(os.path.join(DATA_ROOT, source_uri))

        # Reject path traversal outside DATA_ROOT
        data_root_norm = os.path.normpath(DATA_ROOT)
        if not dataset_path.startswith(data_root_norm + os.sep) and dataset_path != data_root_norm:
            raise ValueError("invalid_source_uri")

        if not os.path.isdir(dataset_path):
            raise ValueError("invalid_source_uri")

    def _resolve_path_under_data_root(self, relative: Optional[str]) -> tuple[str, str, Optional[str]]:
        """
        Resolve a relative path under DATA_ROOT.

        Returns:
            (absolute_dir, current_path, parent_path)
            where current_path/parent_path use forward slashes relative to DATA_ROOT.
        """
        data_root_norm = os.path.normpath(settings.DATA_ROOT or "/data")
        rel = (relative or "").strip().strip("/").replace("\\", "/")
        parts = [p for p in rel.split("/") if p and p not in (".", "..")]

        if parts:
            absolute = os.path.normpath(os.path.join(data_root_norm, *parts))
            if not absolute.startswith(data_root_norm + os.sep):
                raise ValueError("invalid_path")
        else:
            absolute = data_root_norm

        current_path = "/".join(parts)
        parent_path = "/".join(parts[:-1]) if len(parts) > 1 else ("" if len(parts) == 1 else None)
        if len(parts) == 1:
            parent_path = ""

        return absolute, current_path, parent_path

    @staticmethod
    def _directory_has_child_dirs(dir_path: str) -> bool:
        try:
            for name in os.listdir(dir_path):
                if name.startswith("."):
                    continue
                if os.path.isdir(os.path.join(dir_path, name)):
                    return True
        except OSError:
            return False
        return False

    def list_available_source_paths(self, prefix: Optional[str] = None) -> dict:
        """
        List immediate child directories under DATA_ROOT or under ``prefix``.

        Args:
            prefix: Optional path relative to DATA_ROOT.

        Returns:
            dict with keys ``data_root``, ``current_path``, ``parent_path``, ``paths``.
        """
        data_root_norm = os.path.normpath(settings.DATA_ROOT or "/data")

        if not os.path.isdir(data_root_norm):
            return {
                "data_root": data_root_norm,
                "current_path": "",
                "parent_path": None,
                "paths": [],
            }

        try:
            browse_dir, current_path, parent_path = self._resolve_path_under_data_root(prefix)
        except ValueError:
            return {
                "data_root": data_root_norm,
                "current_path": "",
                "parent_path": None,
                "paths": [],
            }

        if not os.path.isdir(browse_dir):
            return {
                "data_root": data_root_norm,
                "current_path": current_path,
                "parent_path": parent_path if current_path else None,
                "paths": [],
            }

        entries: List[dict] = []
        try:
            names = sorted(os.listdir(browse_dir), key=str.lower)
        except OSError:
            names = []

        for name in names:
            if name.startswith("."):
                continue
            full_path = os.path.join(browse_dir, name)
            if not os.path.isdir(full_path):
                continue
            child_path = f"{current_path}/{name}" if current_path else name
            entries.append({
                "path": child_path,
                "name": name,
                "has_children": self._directory_has_child_dirs(full_path),
            })

        return {
            "data_root": data_root_norm,
            "current_path": current_path,
            "parent_path": parent_path if current_path else None,
            "paths": entries,
        }

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

    @staticmethod
    def _location_metadata_from_filename(file_name: str) -> Optional[dict]:
        """
        Extract every piece of extra_metadata derivable from a filename in one
        pass — location (PAM site / FNJV locality) and, independently,
        recorded_date/recorded_time (PAM convention only). Each key is present
        only when its own parser matched; a filename can yield location alone,
        date/time alone, both, or neither.
        """
        meta: dict = {}

        location = parse_location_from_filename(file_name)
        if location:
            meta["location"] = location
            source = location_source_for_filename(file_name)
            if source:
                meta["location_source"] = source

        datetime_parsed = parse_datetime_from_filename(file_name)
        if datetime_parsed:
            recorded_date, recorded_time_seconds = datetime_parsed
            meta["recorded_date"] = recorded_date
            meta["recorded_time"] = recorded_time_seconds

        return meta or None

    def backfill_recording_locations(self, dataset_id: int) -> int:
        """
        Parse location and recorded date/time from file names for recordings
        missing either in extra_metadata. Returns the number of rows updated.
        Only queries recordings actually missing a value, to avoid a full
        table scan on every filter-feed request for datasets already backfilled.
        """
        from sqlalchemy import cast, String as SAString

        bind = self.db.get_bind()
        dialect = bind.dialect.name

        # Filter to rows missing location OR recorded_date — either means
        # this row hasn't been through the current parser yet.
        base_q = self.db.query(RecordingModel).filter(
            RecordingModel.dataset_id == dataset_id
        )
        if dialect == "postgresql":
            base_q = base_q.filter(
                or_(
                    RecordingModel.extra_metadata.op("->>")("location").is_(None),
                    RecordingModel.extra_metadata.op("->>")("recorded_date").is_(None),
                )
            )
        elif dialect == "sqlite":
            from sqlalchemy import func as sa_func

            base_q = base_q.filter(
                or_(
                    sa_func.json_extract(
                        RecordingModel.extra_metadata, "$.location"
                    ).is_(None),
                    sa_func.json_extract(
                        RecordingModel.extra_metadata, "$.recorded_date"
                    ).is_(None),
                )
            )
        else:
            base_q = base_q.filter(
                or_(
                    cast(RecordingModel.extra_metadata["location"], SAString).is_(None),
                    cast(RecordingModel.extra_metadata["recorded_date"], SAString).is_(None),
                )
            )

        recs = base_q.all()
        updated = 0
        for rec in recs:
            parsed = self._location_metadata_from_filename(rec.file_name)
            if not parsed:
                continue
            meta = dict(rec.extra_metadata or {})
            meta.update(parsed)
            rec.extra_metadata = meta
            updated += 1
        if updated:
            self.db.commit()
        return updated

    def list_recording_locations(self, dataset_id: int) -> List[str]:
        """Distinct location values for a dataset (after optional backfill)."""
        self.backfill_recording_locations(dataset_id)
        rows = (
            self.db.query(RecordingModel.extra_metadata)
            .filter(RecordingModel.dataset_id == dataset_id)
            .all()
        )
        locations: set[str] = set()
        for (meta,) in rows:
            if not meta or not isinstance(meta, dict):
                continue
            loc = meta.get("location")
            if isinstance(loc, str) and loc.strip():
                locations.add(loc.strip())
        return sorted(locations)

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
        file_name = os.path.basename(filepath)
        extra_metadata = self._location_metadata_from_filename(file_name)

        rec = RecordingModel(
            dataset_id=dataset.id,
            file_path=relative_path,  # Store relative path
            file_name=file_name,
            duration=duration,
            sample_rate=sample_rate,
            extra_metadata=extra_metadata,
            audio_sha256=checksum,
        )
        self.db.add(rec)
        self.db.commit()
        self.db.refresh(rec)
        return rec

    # ---------------------------------------------------------
    # Dataset explorer - scan physical directory structure
    # ---------------------------------------------------------

    def get_dataset_structure(self, dataset: DatasetModel) -> dict:
        """
        Scan the physical directory structure of a dataset and return
        species (subfolders) with their audio files.
        
        Returns:
            dict with structure:
            {
                'species': [
                    {
                        'name': 'species_folder_name',
                        'file_count': 3,
                        'files': [
                            {'filename': 'file.wav', 'file_path': 'relative/path', 'size': 12345},
                            ...
                        ]
                    },
                    ...
                ]
            }
        """
        DATA_ROOT = settings.DATA_ROOT or "/data"
        dataset_path = os.path.join(DATA_ROOT, dataset.source_uri)
        
        if not os.path.isdir(dataset_path):
            raise ValueError(f"Invalid dataset path: {dataset_path}")
        
        species_list = []
        root_audio_files = []
        
        # Scan immediate subdirectories as species folders
        try:
            entries = sorted(os.listdir(dataset_path))
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to list directory {dataset_path}: {e}")
            return {'species': []}
        
        for entry in entries:
            entry_path = os.path.join(dataset_path, entry)
            
            # Flat focal-recording datasets often store wavs directly at the root.
            if os.path.isfile(entry_path):
                _, ext = os.path.splitext(entry.lower())
                if ext in AUDIO_EXTENSIONS:
                    try:
                        file_size = os.path.getsize(entry_path)
                    except Exception:
                        file_size = None

                    root_audio_files.append({
                        'filename': entry,
                        'file_path': os.path.relpath(entry_path, DATA_ROOT),
                        'size': file_size
                    })
                continue

            if not os.path.isdir(entry_path):
                continue
            
            # Skip hidden directories
            if entry.startswith('.'):
                continue
            
            # This is a species folder - scan for audio files
            audio_files = []
            try:
                for filename in sorted(os.listdir(entry_path)):
                    file_path = os.path.join(entry_path, filename)
                    
                    # Check if it's a file (not directory)
                    if not os.path.isfile(file_path):
                        continue
                    
                    # Check if it's an audio file
                    _, ext = os.path.splitext(filename.lower())
                    if ext not in AUDIO_EXTENSIONS:
                        continue
                    
                    # Get file size
                    try:
                        file_size = os.path.getsize(file_path)
                    except Exception:
                        file_size = None
                    
                    # Store relative path from DATA_ROOT
                    relative_path = os.path.relpath(file_path, DATA_ROOT)
                    
                    audio_files.append({
                        'filename': filename,
                        'file_path': relative_path,
                        'size': file_size
                    })
            
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to scan species folder {entry_path}: {e}")
                continue
            
            # Only include species folders that have audio files
            if audio_files:
                species_list.append({
                    'name': entry,
                    'file_count': len(audio_files),
                    'files': audio_files
                })

        if root_audio_files:
            species_list.insert(0, {
                'name': 'Recordings',
                'file_count': len(root_audio_files),
                'files': root_audio_files,
            })
        
        return {'species': species_list}
