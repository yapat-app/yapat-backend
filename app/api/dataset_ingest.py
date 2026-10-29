"""
Add recordings to an existing dataset (yapat-backend#70).

Mounted under /api/datasets.
"""

from typing import Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.api.deps import get_current_active_user, get_db
from app.models.user import User
from app.schemas.dataset_ingest import (
    AddRecordingsRequest,
    DismissFailuresResponse,
    EmbeddingFailureList,
    IngestJobResponse,
)
from app.services.dataset_ingest_service import (
    DatasetIngestService,
    IngestRequestError,
    job_to_response,
)
from app.services.dataset_service import DatasetService
from app.tasks.ingest_tasks import run_dataset_ingest

router = APIRouter()


@router.post(
    "/{dataset_id}/add-recordings",
    response_model=IngestJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def start_add_recordings(
    dataset_id: int,
    payload: Optional[AddRecordingsRequest] = Body(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    svc = DatasetIngestService(db)
    try:
        job = svc.create_job(
            dataset_id,
            current_user,
            source_uri=payload.source_uri if payload else None,
            retry_failed_only=bool(payload and payload.retry_failed_only),
        )
    except IngestRequestError as e:
        db.rollback()
        body = {"detail": e.code}
        if e.status_code == status.HTTP_409_CONFLICT:
            body["job_id"] = e.job_id
        return JSONResponse(status_code=e.status_code, content=body)

    task = run_dataset_ingest.delay(job.id)
    job.celery_task_id = task.id
    db.commit()
    db.refresh(job)
    return job_to_response(job)


@router.get("/{dataset_id}/add-recordings/{job_id}", response_model=IngestJobResponse)
def get_add_recordings_job(
    dataset_id: int,
    job_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    datasets = DatasetService(db)
    if datasets.get_dataset(dataset_id) is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if not datasets.user_can_access_dataset(current_user, dataset_id):
        raise HTTPException(status_code=403, detail="Not enough permissions")

    job = DatasetIngestService(db).get_job(dataset_id, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Ingest job not found")
    response = job_to_response(job)
    # Release the read transaction before returning (pollers hit this every 2 s).
    db.commit()
    return response


@router.get("/{dataset_id}/embedding-failures", response_model=EmbeddingFailureList)
def list_embedding_failures(
    dataset_id: int,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    include_dismissed: bool = Query(False),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Recordings whose snippets were left without embeddings, with the reason."""
    datasets = DatasetService(db)
    if datasets.get_dataset(dataset_id) is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if not datasets.user_can_access_dataset(current_user, dataset_id):
        raise HTTPException(status_code=403, detail="Not enough permissions")
    result = DatasetIngestService(db).list_failures(
        dataset_id, skip=skip, limit=limit, include_dismissed=include_dismissed
    )
    db.commit()
    return result


@router.post("/{dataset_id}/embedding-failures/dismiss", response_model=DismissFailuresResponse)
def dismiss_embedding_failures(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Continue without the failed recordings: they stop counting as failures and
    are no longer retried by normal add-recordings runs (retry_failed_only still does)."""
    datasets = DatasetService(db)
    dataset = datasets.get_dataset(dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if not datasets.user_can_manage_dataset(current_user, dataset):
        raise HTTPException(status_code=403, detail="Only admins and team owners can dismiss failures")
    return {"dismissed": DatasetIngestService(db).dismiss_failures(dataset_id)}
