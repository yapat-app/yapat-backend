"""
Task management endpoints for Celery tasks
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from celery.result import AsyncResult

from app.api.deps import get_db, get_current_active_user
from app.models.user import User
from app.celery_app import celery_app
from app.tasks import (
    process_recording,
    generate_snippets_for_recording,
    scan_and_process_dataset,
)

router = APIRouter()


@router.get("/status/{task_id}")
def get_task_status(
    task_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """
    Get status of a Celery task
    
    Args:
        task_id: ID of the task to check
        
    Returns:
        Task status and result/metadata
    """
    task_result = AsyncResult(task_id, app=celery_app)
    
    response = {
        "task_id": task_id,
        "status": task_result.state,
        "ready": task_result.ready(),
        "successful": task_result.successful() if task_result.ready() else None,
        "failed": task_result.failed() if task_result.ready() else None,
    }
    
    # Add result if completed
    if task_result.ready():
        if task_result.successful():
            response["result"] = task_result.result
        elif task_result.failed():
            response["error"] = str(task_result.info)
    else:
        # Add progress metadata if available
        if task_result.info:
            response["meta"] = task_result.info
    
    return response


@router.post("/recordings/{recording_id}/process")
def trigger_recording_processing(
    recording_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Trigger processing for a recording (metadata extraction and snippet generation)
    
    Args:
        recording_id: ID of the recording
        
    Returns:
        Task ID and initial status
    """
    from app.models.recording import Recording
    
    # Verify recording exists
    recording = db.query(Recording).filter(Recording.id == recording_id).first()
    if not recording:
        raise HTTPException(status_code=404, detail="Recording not found")
    
    # Trigger task
    task = process_recording.delay(recording_id)
    
    return {
        "task_id": task.id,
        "status": "started",
        "recording_id": recording_id
    }


@router.post("/recordings/{recording_id}/generate-snippets")
def trigger_snippet_generation(
    recording_id: int,
    window_duration_sec: float = 3.0,
    hop_duration_sec: float = 1.5,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Trigger snippet generation for a recording
    
    Args:
        recording_id: ID of the recording
        window_duration_sec: Duration of each snippet
        hop_duration_sec: Hop size between snippets
        
    Returns:
        Task ID and initial status
    """
    from app.models.recording import Recording
    
    # Verify recording exists
    recording = db.query(Recording).filter(Recording.id == recording_id).first()
    if not recording:
        raise HTTPException(status_code=404, detail="Recording not found")
    
    # Trigger task
    task = generate_snippets_for_recording.delay(
        recording_id,
        window_duration_sec,
        hop_duration_sec
    )
    
    return {
        "task_id": task.id,
        "status": "started",
        "recording_id": recording_id,
        "window_duration_sec": window_duration_sec,
        "hop_duration_sec": hop_duration_sec
    }


@router.post("/datasets/{dataset_id}/scan")
def trigger_dataset_scan(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Trigger dataset scanning and processing
    Scans directory for audio files and processes them
    
    Args:
        dataset_id: ID of the dataset
        
    Returns:
        Task ID and initial status
    """
    from app.models.dataset import Dataset
    
    # Verify dataset exists
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Trigger task
    task = scan_and_process_dataset.delay(dataset_id)
    
    return {
        "task_id": task.id,
        "status": "started",
        "dataset_id": dataset_id
    }


@router.delete("/cancel/{task_id}")
def cancel_task(
    task_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """
    Cancel a running task
    
    Args:
        task_id: ID of the task to cancel
        
    Returns:
        Cancellation status
    """
    celery_app.control.revoke(task_id, terminate=True)
    
    return {
        "task_id": task_id,
        "status": "cancelled",
        "message": "Task cancellation requested"
    }

