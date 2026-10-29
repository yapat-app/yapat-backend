"""
Study interaction event model

Stores a per-participant, timestamped stream of UI interaction events captured
during the YAPAT user study (Active Learning annotation flow). One row per event.
Payload is free-form JSON kept small (IDs + scalars), never full prediction objects
or audio content.
"""

from sqlalchemy import (
    Column, Integer, BigInteger, String, DateTime, ForeignKey, JSON, Index
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database import Base


class StudyEvent(Base):
    __tablename__ = "study_events"

    id = Column(BigInteger, primary_key=True, index=True)

    # Per-tab session generated client-side (sessionStorage). Groups one
    # participant sitting/tab into an ordered event stream.
    session_id = Column(String(64), nullable=False, index=True)

    # Always stamped server-side from the authenticated user — the client
    # envelope userId is never trusted.
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )

    phase_id = Column(String(32), nullable=True, index=True)
    dataset_id = Column(
        Integer, ForeignKey("datasets.id", ondelete="SET NULL"), nullable=True
    )
    snippet_set_id = Column(Integer, nullable=True)
    snippet_id = Column(Integer, nullable=True)

    event_type = Column(String(64), nullable=False, index=True)
    payload = Column(JSON, nullable=True)
    duration_ms = Column(Integer, nullable=True)

    # Participant-clock timestamp from the event envelope (ISO 8601 → datetime).
    client_ts = Column(DateTime(timezone=True), nullable=False)
    # Server receipt time — kept separate so analysis can detect clock skew.
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", foreign_keys=[user_id])

    __table_args__ = (
        Index("ix_study_events_session_ts", "session_id", "client_ts"),
        Index("ix_study_events_user_ts", "user_id", "client_ts"),
    )
