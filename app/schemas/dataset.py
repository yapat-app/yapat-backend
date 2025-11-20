"""
Dataset schemas
"""

from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class DatasetBase(BaseModel):
    name: str
    description: Optional[str] = None
    source_uri: Optional[str] = None


class DatasetCreate(DatasetBase):
    team_id: Optional[int] = None  # Optional for admins, required for regular users


class DatasetUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    source_uri: Optional[str] = None


class Dataset(DatasetBase):
    id: int
    team_id: Optional[int] = None
    source_uri: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

