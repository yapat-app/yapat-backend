"""Schemas for POST/GET /api/datasets/{id}/add-recordings."""

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


IngestStatusLiteral = Literal["pending", "running", "completed", "failed"]
IngestStageLiteral = Literal[
    "pending", "scanning", "segmenting", "embedding", "finalizing", "scoring", "completed", "failed"
]


IngestModeLiteral = Literal["scan", "retry_failed"]


class AddRecordingsRequest(BaseModel):
    source_uri: Optional[str] = Field(
        None,
        description=(
            "Folder under DATA_ROOT that contains the dataset's current folder. "
            "Omit to rescan the current folder."
        ),
    )
    retry_failed_only: bool = Field(
        False,
        description=(
            "Skip the folder scan and only re-embed the recordings whose embedding "
            "failed (dismissed ones included). Cannot be combined with source_uri."
        ),
    )


class EmbeddingFailureCount(BaseModel):
    recordings: int = 0
    snippets: int = 0


class IngestSummary(BaseModel):
    files_found: int = 0
    recordings_new: int = 0
    # New recordings whose file name gave location + recorded_date + recorded_time.
    recordings_with_metadata: int = 0
    recordings_missing_metadata: int = 0
    files_skipped: int = 0
    files_unreadable: int = 0
    snippets_new: int = 0
    snippets_embedded: int = 0
    snippets_failed: int = 0
    # Recordings this run left with snippets that have no embedding.
    embedding_failures: EmbeddingFailureCount = Field(default_factory=EmbeddingFailureCount)
    snippets_scored: int = 0
    scoring_skipped_reason: Optional[str] = None


class IngestJobResponse(BaseModel):
    job_id: int
    dataset_id: int
    mode: IngestModeLiteral = "scan"
    status: IngestStatusLiteral
    stage: IngestStageLiteral
    done: int = 0
    total: int = 0
    percent: Optional[float] = None
    stalled: bool = False
    error: Optional[str] = None
    summary: IngestSummary
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class ActiveIngestJob(BaseModel):
    id: int
    mode: IngestModeLiteral = "scan"
    status: IngestStatusLiteral
    stage: IngestStageLiteral
    started_at: Optional[datetime] = None


class EmbeddingFailureItem(BaseModel):
    recording_id: int
    file_name: Optional[str] = None
    file_path: Optional[str] = None
    snippets_missing: int
    reason: str
    last_attempt_at: Optional[datetime] = None
    last_job_id: Optional[int] = None
    dismissed: bool = False


class EmbeddingFailureList(BaseModel):
    # Totals over the current, non-dismissed failures of the dataset.
    recordings: int
    snippets: int
    dismissed_recordings: int
    # Page of items (non-dismissed first unless include_dismissed is false).
    items: List[EmbeddingFailureItem]


class DismissFailuresResponse(BaseModel):
    dismissed: int
