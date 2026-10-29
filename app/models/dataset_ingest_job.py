"""
DatasetIngestJob: one "add recordings to an existing dataset" run.

scan -> segment -> embed -> finalize -> score, over only what is new since the
last run. Existing recordings, snippets, vectors, annotations and predictions
are never modified.
"""

import enum

from sqlalchemy import Column, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database import Base
from app.models.embedding import EnumValue
from app.models.types import PortableJSONB


class IngestJobStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class IngestStage(str, enum.Enum):
    PENDING = "pending"
    SCANNING = "scanning"
    SEGMENTING = "segmenting"
    EMBEDDING = "embedding"
    FINALIZING = "finalizing"
    SCORING = "scoring"
    COMPLETED = "completed"
    FAILED = "failed"


class IngestMode(str, enum.Enum):
    # Scan the folder for new files, then embed + score what is new.
    SCAN = "scan"
    # No scan: embed only the snippets of recordings whose embedding failed
    # (including dismissed ones), then score them.
    RETRY_FAILED = "retry_failed"


ACTIVE_INGEST_STATUSES = (IngestJobStatus.PENDING, IngestJobStatus.RUNNING)


def empty_summary() -> dict:
    return {
        "files_found": 0,
        "recordings_new": 0,
        "recordings_with_metadata": 0,
        "recordings_missing_metadata": 0,
        "files_skipped": 0,
        "files_unreadable": 0,
        "snippets_new": 0,
        "snippets_embedded": 0,
        "snippets_failed": 0,
        # Recordings (and their snippets) left without embeddings by this run.
        "embedding_failures": {"recordings": 0, "snippets": 0},
        "snippets_scored": 0,
        "scoring_skipped_reason": None,
    }


class DatasetIngestJob(Base):
    __tablename__ = "dataset_ingest_jobs"

    id = Column(Integer, primary_key=True, index=True)

    dataset_id = Column(
        Integer,
        ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False,
    )

    # The dataset's default snippet set at job creation. New snippets go here.
    snippet_set_id = Column(
        Integer,
        ForeignKey("snippet_sets.id", ondelete="CASCADE"),
        nullable=False,
    )

    # embedding_vectors.embedding_job_id is NOT NULL, so new vectors are attached
    # to the snippet set's original embedding job instead of creating a second
    # EmbeddingJob (which would show up as a running embedding job in the UI).
    embedding_job_id = Column(
        Integer,
        ForeignKey("embedding_jobs.id", ondelete="CASCADE"),
        nullable=False,
    )

    requested_by_user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )

    # source_uri the dataset was switched to by this run (None = rescan only).
    source_uri = Column(String, nullable=True)

    mode = Column(
        EnumValue(IngestMode),
        nullable=False,
        default=IngestMode.SCAN,
        server_default=IngestMode.SCAN.value,
    )

    status = Column(
        EnumValue(IngestJobStatus),
        nullable=False,
        default=IngestJobStatus.PENDING,
    )
    stage = Column(
        EnumValue(IngestStage),
        nullable=False,
        default=IngestStage.PENDING,
    )

    # Counters filled in as stages finish; keys from empty_summary().
    summary = Column(PortableJSONB, nullable=False, default=empty_summary)

    celery_task_id = Column(String, nullable=True)
    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    dataset = relationship("Dataset")
    snippet_set = relationship("SnippetSet")
    embedding_job = relationship("EmbeddingJob")

    __table_args__ = (
        Index("ix_dataset_ingest_jobs_dataset_status", "dataset_id", "status"),
    )
