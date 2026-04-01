"""
Dataset schemas
"""

from datetime import datetime
from typing import Optional, List
from enum import Enum

from pydantic import BaseModel


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


class DatasetUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    source_uri: Optional[str] = None
    dataset_type: Optional[DatasetType] = None


class Dataset(DatasetBase):
    id: int
    team_id: Optional[int] = None
    dataset_type: Optional[DatasetType] = DatasetType.PAM
    default_snippet_set_id: Optional[int] = None
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