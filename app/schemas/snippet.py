"""
Snippet and SnippetConfig schemas
"""

from datetime import datetime
from pydantic import BaseModel
from typing import Optional


# ---------------------------
# Snippet schemas
# ---------------------------

class SnippetBase(BaseModel):
    start_time: float
    duration: float
    snippet_config_id: int


class Snippet(SnippetBase):
    id: int
    recording_id: int
    created_at: datetime

    class Config:
        from_attributes = True


# ---------------------------
# SnippetConfig schemas
# ---------------------------

class SnippetConfigBase(BaseModel):
    window_size: float
    step_size: float
    overlap: float


class SnippetConfigCreate(SnippetConfigBase):
    dataset_id: int


class SnippetConfig(SnippetConfigBase):
    id: int
    dataset_id: int
    created_at: datetime

    class Config:
        from_attributes = True
