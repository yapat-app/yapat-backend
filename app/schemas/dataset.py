"""
Dataset schemas
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class DatasetBase(BaseModel):
    name: str
    description: Optional[str] = None
    source_uri: str


class DatasetCreate(DatasetBase):
    team_id: Optional[int] = None  # Optional for admins, required for regular users


class DatasetUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    source_uri: Optional[str] = None


class Dataset(DatasetBase):
    id: int
    team_id: Optional[int] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class DatasetCreationResponse(BaseModel):
    dataset: Dataset
    process_task_id: Optional[str] = None
    snippet_config_id: Optional[int] = None
    embedding_job_id: Optional[int] = None

    class Config:
        orm_mode = True