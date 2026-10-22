"""
RecordingEmbeddingFailure: a recording left with snippets that have no embedding.

Written by the add-recordings pipeline (app/tasks/ingest_tasks.py) after its
embedding stage: one row per recording with missing snippets, removed once all
of the recording's snippets have a vector. The UI shows the count, lists the
reasons, and can retry or dismiss them.
"""

from sqlalchemy import Column, DateTime, ForeignKey, Index, Integer, Text, UniqueConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database import Base


class RecordingEmbeddingFailure(Base):
    __tablename__ = "recording_embedding_failures"

    id = Column(Integer, primary_key=True, index=True)

    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)
    snippet_set_id = Column(Integer, ForeignKey("snippet_sets.id", ondelete="CASCADE"), nullable=False)
    recording_id = Column(Integer, ForeignKey("recordings.id", ondelete="CASCADE"), nullable=False)

    # Snippets of this recording still without an embedding after the last attempt.
    snippets_missing = Column(Integer, nullable=False)
    # Error of the whole recording (file missing, decode error, ...) or a note
    # that BirdNET returned nothing for some of its snippets.
    reason = Column(Text, nullable=False)

    last_job_id = Column(Integer, ForeignKey("dataset_ingest_jobs.id", ondelete="SET NULL"), nullable=True)
    last_attempt_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Set when the user chose to continue without these snippets: normal
    # add-recordings runs then stop retrying them (an explicit retry still does).
    dismissed_at = Column(DateTime(timezone=True), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    recording = relationship("Recording")

    __table_args__ = (
        UniqueConstraint("recording_id", "snippet_set_id", name="uq_recording_embedding_failure"),
        Index("ix_recording_embedding_failures_dataset", "dataset_id", "dismissed_at"),
    )
