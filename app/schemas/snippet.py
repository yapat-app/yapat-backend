"""
Snippet schemas
"""

from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List, Any


class SnippetBase(BaseModel):
    start_time: float
    end_time: float
    duration: float


class SnippetCreate(SnippetBase):
    recording_id: int
    file_path: Optional[str] = None
    embedding: Optional[List[float]] = None


class SnippetUpdate(BaseModel):
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration: Optional[float] = None
    file_path: Optional[str] = None
    embedding: Optional[List[float]] = None
    is_annotated: Optional[bool] = None


class Snippet(SnippetBase):
    id: int
    recording_id: int
    file_path: Optional[str] = None
    embedding: Optional[List[float]] = None
    is_annotated: bool
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

