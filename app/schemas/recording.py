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


class RecordingMetadataItem(BaseModel):
    """Compact per-recording metadata used by the annotation-hub location and
    date/time filters — just the three fields those filters read out of
    extra_metadata, so the client no longer has to page the full recordings
    table to build them."""
    recording_id: int
    location: Optional[str] = None
    recorded_date: Optional[str] = None
    recorded_time: Optional[float] = None


class RecordingMetadataSummary(BaseModel):
    dataset_id: int
    items: list[RecordingMetadataItem]
