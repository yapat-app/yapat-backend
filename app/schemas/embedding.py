from pydantic import BaseModel


class EmbeddingJobCreateRequest(BaseModel):
    embedding_model_id: int


class EmbeddingJobResponse(BaseModel):
    embedding_job_id: int
    snippet_config_id: int
    model_id: int
    celery_task_id: str | None
    status: str

    class Config:
        orm_mode = True
