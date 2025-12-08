"""
Recording schemas
"""

from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Dict, Any


class RecordingBase(BaseModel):
    file_name: str
    duration: Optional[float] = None
    sample_rate: Optional[float] = None
    extra_metadata: Optional[Dict[str, Any]] = None


class RecordingCreate(RecordingBase):
    dataset_id: int
    file_path: str


class Recording(RecordingBase):
    id: int
    dataset_id: int
    file_path: str
    audio_sha256: str
    created_at: datetime

    class Config:
        from_attributes = True
