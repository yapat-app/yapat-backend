"""
Study interaction event schemas
"""

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel
from datetime import datetime
from typing import Optional, Dict, Any, List


# Server-side guardrail: reject absurd batches outright.
MAX_EVENTS_PER_BATCH = 200


class StudyEventIn(BaseModel):
    """A single interaction event as sent by the client.

    The client sends camelCase keys (sessionId, eventType, …) so we use
    an alias generator to accept both camelCase (wire format) and snake_case.
    """

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,  # also accept snake_case (e.g. in tests)
    )

    session_id: str = Field(..., max_length=64)
    phase_id: Optional[str] = Field(None, max_length=32)
    dataset_id: Optional[int] = None
    snippet_set_id: Optional[int] = None
    snippet_id: Optional[int] = None
    timestamp: datetime = Field(..., description="Participant-clock ISO 8601 timestamp")
    event_type: str = Field(..., max_length=64)
    payload: Optional[Dict[str, Any]] = None
    duration_ms: Optional[int] = None


class StudyEventBatchCreate(BaseModel):
    """A batch of events. `token` is an optional auth fallback for the
    navigator.sendBeacon unload path, which cannot set an Authorization header."""
    events: List[StudyEventIn] = Field(..., max_length=MAX_EVENTS_PER_BATCH)
    token: Optional[str] = None


class StudyEventBatchResponse(BaseModel):
    accepted: int
    session_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Admin read schemas
# ---------------------------------------------------------------------------

class StudyLogUserSummary(BaseModel):
    """Aggregate view of one user's study event data."""
    user_id: int
    username: str
    session_count: int
    event_count: int
    last_seen: datetime


class StudyLogSessionSummary(BaseModel):
    """Aggregate view of one session."""
    session_id: str
    event_count: int
    first_event_at: datetime
    last_event_at: datetime
    phase_ids: List[str]
    duration_minutes: Optional[float] = None


class StudyLogEventRow(BaseModel):
    """A single event row for the admin event table."""
    id: int
    client_ts: datetime
    event_type: str
    phase_id: Optional[str] = None
    dataset_id: Optional[int] = None
    snippet_id: Optional[int] = None
    payload: Optional[Dict[str, Any]] = None
    duration_ms: Optional[int] = None


class StudyLogEventsPage(BaseModel):
    """Paginated event list for a session."""
    session_id: str
    total: int
    events: List[StudyLogEventRow]
