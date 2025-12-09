"""
Embedding job endpoints
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_current_active_user
from app.models.embedding import EmbeddingJob, EmbeddingModel
from app.models.dataset import Dataset
from app.services.embedding_service import EmbeddingService
from app.tasks.embedding_tasks import run_embedding
from app.schemas.embedding import (
    EmbeddingJobCreateRequest,
    EmbeddingJobResponse,
)
from app.models.user import User

router = APIRouter()


@router.post(
    "/datasets/{dataset_id}/embeddings",
    response_model=EmbeddingJobResponse,
)
def create_embedding_job(
    dataset_id: int,
    payload: EmbeddingJobCreateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Create an embedding job for a dataset.

    Steps:
    - validate dataset + model
    - create EmbeddingJob + SnippetConfig
    - trigger Celery run_embedding(job_id)
    """

    service = EmbeddingService(db)

    dataset = db.query(Dataset).filter_by(id=dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    model = service.get_model(payload.embedding_model_id)

    # Create embedding job = job + snippet_config
    job = service.create_embedding_job(dataset, model)

    # Trigger celery
    task = run_embedding.delay(job.id)

    # Store Celery task id
    service.update_job_status(job.id, job.status, celery_task_id=task.id)

    return EmbeddingJobResponse(
        embedding_job_id=job.id,
        snippet_config_id=job.snippet_config.id,
        model_id=model.id,
        celery_task_id=task.id,
        status=job.status.value,
    )


@router.get("/datasets/{dataset_id}/embeddings")
def list_embedding_jobs(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """List all embedding jobs for a dataset."""
    service = EmbeddingService(db)
    return service.list_jobs_for_dataset(dataset_id)


@router.get("/embeddings/{job_id}", response_model=EmbeddingJobResponse)
def get_embedding_job(
    job_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Retrieve a single embedding job."""

    service = EmbeddingService(db)
    job = service.get_job(job_id)

    return EmbeddingJobResponse(
        embedding_job_id=job.id,
        snippet_config_id=job.snippet_config.id,
        model_id=job.embedding_model_id,
        celery_task_id=job.celery_task_id,
        status=job.status.value,
    )
