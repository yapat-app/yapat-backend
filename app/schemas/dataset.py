"""
Dataset schemas
"""

from datetime import datetime
from typing import Optional, List, Any, Dict
from enum import Enum

from pydantic import BaseModel, field_validator


class DatasetType(str, Enum):
    PAM = "PAM"
    FOCAL_RECORDINGS = "FOCAL_RECORDINGS"


class DatasetBase(BaseModel):
    name: str
    description: Optional[str] = None
    source_uri: Optional[str] = None
    dataset_type: DatasetType = DatasetType.PAM


class DatasetCreate(DatasetBase):
    team_id: Optional[int] = None  # Optional for admins, required for regular users
    spectrogram_f_min_hz: Optional[float] = None
    spectrogram_f_max_hz: Optional[float] = None
    # Marks this dataset as reference-only training data (never surfaced for
    # annotation; other datasets/teams opt in via reference links). Admin-set.
    is_reference: bool = False


class DatasetUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    source_uri: Optional[str] = None
    dataset_type: Optional[DatasetType] = None
    spectrogram_f_min_hz: Optional[float] = None
    spectrogram_f_max_hz: Optional[float] = None
    retrain_after_threshold: Optional[int] = None
    is_reference: Optional[bool] = None

    @field_validator("spectrogram_f_min_hz", "spectrogram_f_max_hz")
    @classmethod
    def validate_positive_hz(cls, v: Optional[float], info) -> Optional[float]:
        if v is None:
            return None
        if v < 0:
            raise ValueError(f"{info.field_name} must be non-negative")
        return v

    @field_validator("retrain_after_threshold")
    @classmethod
    def validate_retrain_threshold(cls, v: Optional[int]) -> Optional[int]:
        if v is None:
            return None
        if v < 1:
            raise ValueError("retrain_after_threshold must be at least 1")
        return v


class Dataset(DatasetBase):
    id: int
    team_id: Optional[int] = None
    dataset_type: Optional[DatasetType] = DatasetType.PAM
    default_snippet_set_id: Optional[int] = None
    spectrogram_f_min_hz: Optional[float] = None
    spectrogram_f_max_hz: Optional[float] = None
    quick_labels: Optional[List[Dict[str, Any]]] = None
    retrain_after_threshold: Optional[int] = None
    is_reference: bool = False
    created_at: datetime
    updated_at: Optional[datetime] = None
    recording_count: Optional[int] = None  # Number of recordings in this dataset
    is_ready_for_feed: bool = False  # True when default snippet set exists and is READY

    class Config:
        from_attributes = True


class DatasetCreationResponse(BaseModel):
    dataset: Dataset
    process_task_id: Optional[str] = None
    snippet_config_id: Optional[int] = None
    embedding_job_id: Optional[int] = None

    class Config:
        orm_mode = True


class AudioFile(BaseModel):
    """Represents an audio file in the dataset explorer"""
    filename: str
    file_path: str  # Relative path from DATA_ROOT
    size: Optional[int] = None  # File size in bytes


class SpeciesFolder(BaseModel):
    """Represents a species folder (subfolder) in a dataset"""
    name: str
    file_count: int
    files: List[AudioFile]


class DatasetExplorerResponse(BaseModel):
    """Response for dataset explorer endpoint showing species and their files"""
    dataset_id: int
    dataset_name: str
    source_uri: str
    species: List[SpeciesFolder]


class AvailableDatasetPath(BaseModel):
    """A directory under DATA_ROOT that can be registered as a dataset source_uri."""
    path: str  # full path relative to DATA_ROOT
    name: str  # segment name in the current listing
    has_children: bool = False


class AvailableDatasetPathsResponse(BaseModel):
    """Directories available on the mounted data volume (relative to DATA_ROOT)."""
    data_root: str
    current_path: str = ""
    parent_path: Optional[str] = None
    paths: List[AvailableDatasetPath]


class RecordingLocationsResponse(BaseModel):
    """Distinct site/locality values parsed from recording file names."""
    locations: List[str]