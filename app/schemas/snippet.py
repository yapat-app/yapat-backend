"""
Snippet schemas (updated for SnippetSet architecture)
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


# ---------------------------------------------------------
# Snippet Base
# ---------------------------------------------------------

class SnippetBase(BaseModel):
    start_time: float
    duration: float
    snippet_set_id: int


# ---------------------------------------------------------
# Snippet Response
# ---------------------------------------------------------

class Snippet(SnippetBase):
    id: int
    recording_id: int
    end_time: float
    created_at: datetime

    class Config:
        from_attributes = True


class UserFeedSnapshot(BaseModel):
    """Snapshot from user_feeds; request_params holds dataset_id and other args for cross-device restore."""

    id: int
    method: str
    created_at: datetime
    response: List[Snippet]
    request_params: Optional[Dict[str, Any]] = None

