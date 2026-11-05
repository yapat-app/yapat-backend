from typing import List, Optional, Dict, Any
from pydantic import BaseModel, field_validator
from datetime import datetime


# ---------------------------------------------------------
# Embedding Model
# ---------------------------------------------------------

class EmbeddingModel(BaseModel):
    """Schema for embedding model (from database)."""
    id: int
    name: str
    version: Optional[str] = None
    description: Optional[str] = None
    window_size: float
    step_size: float
    overlap: float
    requires_fixed_window: bool = True
    requires_fixed_step: bool = True
    requires_fixed_overlap: bool = True
    dim: int
    sample_rate: Optional[int] = None
    # False when this deployment has no weights for the model, so it cannot be run yet.
    weights_available: bool = False

    @field_validator("requires_fixed_window", "requires_fixed_step", "requires_fixed_overlap", mode="before")
    @classmethod
    def convert_int_to_bool(cls, v):
        """Convert Integer (0/1) to bool if needed."""
        if isinstance(v, int):
            return bool(v)
        return v

    class Config:
        from_attributes = True


# ---------------------------------------------------------
# Create Embedding Job
# ---------------------------------------------------------

class EmbeddingJobCreateRequest(BaseModel):
    embedding_model_id: int
    # Optional overrides — usually ignored for strict models
    window_size: Optional[float] = None
    step_size: Optional[float] = None
    overlap: Optional[float] = None


# ---------------------------------------------------------
# Embedding Job Response
# ---------------------------------------------------------

class EmbeddingJobResponse(BaseModel):
    embedding_job_id: int
    snippet_set_id: int
    model_id: int
    celery_task_id: Optional[str]
    status: str

    class Config:
        from_attributes = True


# ---------------------------------------------------------
# SnippetSet Schemas
# ---------------------------------------------------------

class EmbeddingJobProgress(BaseModel):
    """Live progress of an embedding job (Redis counter, see services/embedding_progress)."""
    embedding_job_id: int
    dataset_id: int
    status: str  # pending | running | completed | failed (EmbeddingJob.status)
    stage: str  # segmenting | embedding | completed | failed | pending
    done: Optional[int] = None  # snippets processed; None while segmenting/unknown
    total: Optional[int] = None
    percent: Optional[float] = None
    stalled: bool = False  # RUNNING but no progress for STALL_AFTER_SECONDS
    last_update: Optional[int] = None  # unix seconds of the last increment
    error_message: Optional[str] = None


class EmbeddingOption(BaseModel):
    """One embedding model on a dataset, as offered by the hub's model switcher."""
    embedding_model_id: int
    name: str
    version: Optional[str] = None
    dim: int
    snippet_set_id: int
    window_size: float
    step_size: float
    overlap: float
    embedding_job_id: int
    status: str  # pending | running | completed | failed (EmbeddingJob.status)
    # True for the model the hub opens first.
    is_default: bool
    # True when another model uses the same clips, so annotations are shared with it.
    shares_snippet_set: bool
    # Classifier of this model on this dataset; None until one is trained.
    active_checkpoint_id: Optional[int] = None
    # The classifier's family, or the family to use when training the first one.
    model_family_name: str


class SnippetSet(BaseModel):
    """Schema for snippet set response."""
    id: int
    dataset_id: int
    # Models that finished embedding this set, first to finish first.
    embedding_model_ids: List[int] = []
    # The first of embedding_model_ids. Kept for existing clients; a set can have several.
    embedding_model_id: Optional[int] = None
    window_size: float
    step_size: float
    overlap: float
    status: str
    created_at: datetime

    class Config:
        from_attributes = True


class SnippetSetWithStats(SnippetSet):
    """SnippetSet with annotation statistics."""
    annotation_count: int
    annotated_snippet_count: int
    total_snippet_count: int
    has_annotations: bool


class SnippetSetDeleteRequest(BaseModel):
    """Request to delete a snippet set."""
    acknowledge_annotation_loss: bool = False
    
    class Config:
        json_schema_extra = {
            "example": {
                "acknowledge_annotation_loss": True
            }
        }


class SnippetSetDeleteResponse(BaseModel):
    """Response after deleting a snippet set."""
    deleted_snippet_set_id: int
    deleted_annotation_count: int
    deleted_snippet_count: int
    message: str
