from typing import Optional
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
