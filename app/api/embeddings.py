"""
Embedding job endpoints (updated for SnippetSet architecture)
"""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_current_active_user
from app.models.dataset import Dataset
from app.services.embedding_service import EmbeddingService
from app.services.snippet_set_service import SnippetSetService
from app.tasks.embedding_tasks import run_embedding
from app.models.embedding import SnippetSet as SnippetSetModel, SnippetSetStatus
from app.schemas.embedding import (
    EmbeddingModel,
    EmbeddingJobCreateRequest,
    EmbeddingJobResponse,
    SnippetSet,
    SnippetSetWithStats,
    SnippetSetDeleteRequest,
    SnippetSetDeleteResponse,
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

    # NOTE:
    # In the SnippetSet architecture, the embedding pipeline materializes snippets
    # and marks the SnippetSet READY inside `run_embedding`. For new datasets,
    # `default_snippet_set_id` may be null until the first embedding job runs.
    # Therefore we must not block job creation based on default snippet set state.

    # --------------------------
    # Validate embedding model
    # --------------------------
    model = service.get_model(payload.embedding_model_id)

    # --------------------------
    # Create job (SnippetSet + EmbeddingJob)
    # --------------------------
    try:
        job = service.create_embedding_job(
            dataset,
            model,
            window_size=payload.window_size,
            step_size=payload.step_size,
            overlap=payload.overlap,
        )
    except ValueError as e:
        # Handle duplicate job error
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e)
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


# ---------------------------------------------------------
# SnippetSet Management
# ---------------------------------------------------------

@router.get("/datasets/{dataset_id}/snippet-sets", response_model=List[SnippetSet])
def list_snippet_sets(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    List all snippet sets for a dataset.
    
    A snippet set represents a specific segmentation configuration
    (window size, step size, overlap) for a dataset.
    """
    # Validate dataset exists
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    service = SnippetSetService(db)
    return service.list_for_dataset(dataset_id)


@router.get("/snippet-sets/{snippet_set_id}", response_model=SnippetSetWithStats)
def get_snippet_set(
    snippet_set_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Get a single snippet set with annotation statistics.
    
    Returns the snippet set configuration along with stats on:
    - Total number of annotations
    - Number of annotated snippets
    - Total number of snippets
    - Whether it has any annotations (protection flag)
    """
    service = SnippetSetService(db)
    
    try:
        snippet_set = service.get(snippet_set_id)
        stats = service.get_annotation_stats(snippet_set_id)
        
        return SnippetSetWithStats(
            id=snippet_set.id,
            dataset_id=snippet_set.dataset_id,
            embedding_model_id=snippet_set.embedding_model_id,
            window_size=snippet_set.window_size,
            step_size=snippet_set.step_size,
            overlap=snippet_set.overlap,
            status=snippet_set.status.value,
            created_at=snippet_set.created_at,
            **stats
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete(
    "/snippet-sets/{snippet_set_id}",
    response_model=SnippetSetDeleteResponse,
    status_code=status.HTTP_200_OK
)
def delete_snippet_set(
    snippet_set_id: int,
    delete_request: SnippetSetDeleteRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Delete a snippet set with annotation loss protection.
    
    **IMPORTANT**: Deleting a snippet set will delete all associated snippets
    and their annotations. This operation cannot be undone.
    
    **Protection Mechanism**:
    - If the snippet set contains annotations, deletion will fail unless
      you explicitly acknowledge the data loss by setting
      `acknowledge_annotation_loss: true` in the request body.
    
    **Use Case**:
    - Safe to delete: Snippet sets with no annotations (e.g., test configurations)
    - Requires acknowledgment: Snippet sets with any annotations
    
    **Example Request Body**:
    ```json
    {
        "acknowledge_annotation_loss": true
    }
    ```
    """
    service = SnippetSetService(db)
    
    try:
        # Check stats first to provide informative error
        stats = service.get_annotation_stats(snippet_set_id)
        
        if stats["annotation_count"] > 0 and not delete_request.acknowledge_annotation_loss:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "annotation_loss_not_acknowledged",
                    "message": (
                        f"Cannot delete snippet set: it contains {stats['annotation_count']} "
                        f"annotation(s) across {stats['annotated_snippet_count']} snippet(s). "
                        f"To prevent accidental data loss, you must explicitly acknowledge this "
                        f"by setting 'acknowledge_annotation_loss: true' in your request."
                    ),
                    "annotation_count": stats["annotation_count"],
                    "annotated_snippet_count": stats["annotated_snippet_count"],
                    "total_snippet_count": stats["total_snippet_count"],
                }
            )
        
        # Perform safe deletion
        result = service.safe_delete(
            snippet_set_id,
            allow_with_annotations=delete_request.acknowledge_annotation_loss
        )
        
        return SnippetSetDeleteResponse(**result)
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        # This shouldn't happen due to above pre-check, but just in case
        raise HTTPException(status_code=400, detail=str(e))
