"""
Task management endpoints for Celery tasks
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from celery.result import AsyncResult

from app.api.deps import get_db, get_current_active_user
from app.models.user import User
from app.celery_app import celery_app

# Import the actual, existing tasks
from app.tasks.processing_tasks import (
    scan_dataset,
    process_dataset,
)

router = APIRouter()


# ------------------------------------------------------
# Task status
# ------------------------------------------------------
@router.get("/status/{task_id}")
def get_task_status(
    task_id: str,
    current_user: User = Depends(get_current_active_user),
):
    """
    Query Celery task status + metadata.
    """
    task = AsyncResult(task_id, app=celery_app)

    resp = {
        "task_id": task_id,
        "status": task.state,
        "ready": task.ready(),
        "successful": task.successful() if task.ready() else None,
        "failed": task.failed() if task.ready() else None,
    }

    # include results if task finished
    if task.ready():
        if task.successful():
            resp["result"] = task.result
        else:
            resp["error"] = str(task.info)
    else:
        # pending progress metadata
        if task.info:
            resp["meta"] = task.info

    return resp


# ------------------------------------------------------
# Dataset-level operations
# ------------------------------------------------------
@router.post("/datasets/{dataset_id}/scan")
def trigger_dataset_scan(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Trigger scan for audio files inside a dataset.
    """
    from app.models.dataset import Dataset

    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if ds is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    task = scan_dataset.delay(dataset_id)

    return {
        "task_id": task.id,
        "status": "started",
        "dataset_id": dataset_id,
    }


@router.post("/datasets/{dataset_id}/process")
def trigger_dataset_processing(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    High-level dataset pipeline task.

    Currently: scan recordings → return submitted scan task ID.
    Future: add snippet-generation orchestration.
    """
    from app.models.dataset import Dataset

    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if ds is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    task = process_dataset.delay(dataset_id)

    return {
        "task_id": task.id,
        "status": "submitted",
        "dataset_id": dataset_id,
    }


# ------------------------------------------------------
# Cancel task
# ------------------------------------------------------
@router.delete("/cancel/{task_id}")
def cancel_task(
    task_id: str,
    current_user: User = Depends(get_current_active_user),
):
    """
    Cancel a running Celery task.
    """
    celery_app.control.revoke(task_id, terminate=True)

    return {
        "task_id": task_id,
        "status": "cancelled",
        "message": "Task cancellation requested",
    }
