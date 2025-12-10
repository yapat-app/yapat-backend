from typing import Optional
from pydantic import BaseModel


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
