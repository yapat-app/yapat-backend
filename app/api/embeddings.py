"""
Embedding job endpoints (updated for SnippetSet architecture)
"""

from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_current_active_user
from app.models.dataset import Dataset
from app.services.embedding_service import EmbeddingService
from app.tasks.embedding_tasks import run_embedding
from app.schemas.embedding import (
    EmbeddingModel,
    EmbeddingJobCreateRequest,
    EmbeddingJobResponse,
)
from app.models.user import User

router = APIRouter()


# ---------------------------------------------------------
# Embedding Models
# ---------------------------------------------------------

@router.get("/embedding-models", response_model=List[EmbeddingModel])
def list_embedding_models(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """List all available embedding models."""
    service = EmbeddingService(db)
    return service.list_models()


@router.get("/embedding-models/{model_id}", response_model=EmbeddingModel)
def get_embedding_model(
    model_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Retrieve a single embedding model by ID."""
    service = EmbeddingService(db)
    try:
        return service.get_model(model_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ---------------------------------------------------------
# Create Embedding Job
# ---------------------------------------------------------

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
    - create or reuse SnippetSet
    - create EmbeddingJob(snippet_set_id)
    - trigger Celery
    """

    service = EmbeddingService(db)

    # --------------------------
    # Validate dataset
    # --------------------------
    dataset = (
        db.query(Dataset)
        .filter(Dataset.id == dataset_id)
        .first()
    )
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # --------------------------
    # Validate embedding model
    # --------------------------
    model = service.get_model(payload.embedding_model_id)

    # --------------------------
    # Create job (SnippetSet + EmbeddingJob)
    # --------------------------
    job = service.create_embedding_job(
        dataset,
        model,
        window_size=payload.window_size,
        step_size=payload.step_size,
        overlap=payload.overlap,
    )

    # --------------------------
    # Trigger Celery worker
    # --------------------------
    task = run_embedding.delay(job.id)
    service.update_job_status(job.id, job.status, celery_task_id=task.id)

    return EmbeddingJobResponse(
        embedding_job_id=job.id,
        snippet_set_id=job.snippet_set_id,
        model_id=model.id,
        celery_task_id=task.id,
        status=job.status.value,
    )


# ---------------------------------------------------------
# List Jobs
# ---------------------------------------------------------

@router.get("/datasets/{dataset_id}/embeddings")
def list_embedding_jobs(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """List all embedding jobs for a dataset."""
    service = EmbeddingService(db)
    return service.list_jobs_for_dataset(dataset_id)


# ---------------------------------------------------------
# Retrieve Single Job
# ---------------------------------------------------------

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
        snippet_set_id=job.snippet_set_id,
        model_id=job.embedding_model_id,
        celery_task_id=job.celery_task_id,
        status=job.status.value,
    )
