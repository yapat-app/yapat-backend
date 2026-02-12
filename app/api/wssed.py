"""
WSSED API endpoints

Provides REST API for WSSED training, detection, predictions, and feedback.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks, status
from sqlalchemy.orm import Session
from typing import List, Optional

from app.api.deps import get_db, get_current_active_user
from app.models.user import User
from app.schemas.wssed import (
    TrainingJobCreate,
    TrainingJob,
    TrainingStatus,
    Prediction,
    RecordingTimeline,
    FeedbackSubmit,
    FeedbackResponse,
    FeedbackStats,
    SpeciesList,
    DetectionResponse,
    FeedbackType
)
from app.services.wssed_service import WSSEDService
from app.tasks.wssed_tasks import trigger_wssed_training, trigger_wssed_detection

router = APIRouter()


# ============ TRAINING ENDPOINTS ============

@router.post("/training-jobs", response_model=TrainingJob, status_code=status.HTTP_201_CREATED)
async def create_training_job(
    job_in: TrainingJobCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Create and start a new WSSED training job.
    
    The training will run on a remote GPU server. This endpoint returns immediately
    with the job details. Use GET /training-jobs/{job_id} to poll for status updates.
    """
    service = WSSEDService(db)
    
    try:
        # Create training job
        job = service.create_training_job(
            dataset_id=job_in.dataset_id,
            hyperparameters=job_in.hyperparameters
        )
        
        # Trigger training in background
        background_tasks.add_task(trigger_wssed_training, job.id)
        
        return job
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create training job: {str(e)}")


@router.get("/training-jobs/{job_id}", response_model=TrainingJob)
async def get_training_job(
    job_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Get training job status and metrics.
    
    If the job is still training, this will poll the GPU server for the latest status.
    """
    service = WSSEDService(db)
    job = service.get_training_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    # Poll GPU server for latest status if still training
    if job.status == TrainingStatus.TRAINING:
        try:
            job = await service.update_training_status(job_id)
        except Exception:
            # If polling fails, just return current status
            pass
    
    return job


@router.get("/training-jobs", response_model=List[TrainingJob])
def list_training_jobs(
    dataset_id: Optional[int] = Query(None, description="Filter by dataset ID"),
    status: Optional[TrainingStatus] = Query(None, description="Filter by status"),
    skip: int = Query(0, ge=0, description="Number of items to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Number of items to return"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    List all training jobs with optional filtering.
    """
    service = WSSEDService(db)
    return service.list_training_jobs(dataset_id, status, skip, limit)


# ============ DETECTION ENDPOINTS ============

@router.post("/training-jobs/{job_id}/detect", response_model=DetectionResponse)
async def trigger_detection(
    job_id: int,
    threshold: float = Query(0.5, ge=0, le=1, description="Detection confidence threshold"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Run detection using a trained model.
    
    This will apply the model to all recordings in the dataset and store predictions.
    The operation runs asynchronously on the GPU server.
    """
    service = WSSEDService(db)
    
    job = service.get_training_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    if job.status != TrainingStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Training job must be completed. Current status: {job.status.value}"
        )
    
    try:
        # Trigger detection in background
        background_tasks.add_task(trigger_wssed_detection, job_id, threshold)
        
        return DetectionResponse(
            message="Detection started",
            task_id=f"detection_{job_id}",
            training_job_id=job_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to trigger detection: {str(e)}")


# ============ PREDICTION ENDPOINTS ============

@router.get("/predictions", response_model=List[Prediction])
def list_predictions(
    recording_id: Optional[int] = Query(None, description="Filter by recording ID"),
    training_job_id: Optional[int] = Query(None, description="Filter by training job ID"),
    species_name: Optional[str] = Query(None, description="Filter by species name"),
    threshold: float = Query(0.0, ge=0, le=1, description="Minimum confidence threshold (ignored if uncertain_range is provided)"),
    uncertain_range: Optional[float] = Query(None, ge=0, le=0.5, description="Filter for uncertain predictions close to 0.5. For example, 0.1 will show predictions where 0.4 <= confidence <= 0.6"),
    feedback_filter: Optional[FeedbackType] = Query(None, description="Filter by feedback status"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    List predictions with filtering options.
    
    Use this to browse all predictions or filter by specific criteria.
    
    To show uncertain predictions (close to 0.5), use the uncertain_range parameter.
    For example, uncertain_range=0.1 will show predictions where confidence is between 0.4 and 0.6.
    """
    service = WSSEDService(db)
    return service.list_predictions(
        recording_id=recording_id,
        training_job_id=training_job_id,
        species_name=species_name,
        threshold=threshold,
        uncertain_range=uncertain_range,
        feedback_filter=feedback_filter,
        skip=skip,
        limit=limit
    )


@router.get("/recordings/{recording_id}/predictions-timeline", response_model=RecordingTimeline)
def get_recording_timeline(
    recording_id: int,
    training_job_id: Optional[int] = Query(None, description="Filter by specific training job"),
    threshold: float = Query(0.5, ge=0, le=1, description="Minimum confidence threshold (ignored if uncertain_range is provided)"),
    uncertain_range: Optional[float] = Query(None, ge=0, le=0.5, description="Filter for uncertain predictions close to 0.5. For example, 0.1 will show predictions where 0.4 <= confidence <= 0.6"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Get all predictions for a recording formatted for timeline visualization.
    
    This endpoint is optimized for displaying predictions on an audio timeline player.
    It returns predictions sorted by time with their confidence scores and feedback status.
    
    To show uncertain predictions (close to 0.5), use the uncertain_range parameter.
    For example, uncertain_range=0.1 will show predictions where confidence is between 0.4 and 0.6.
    """
    service = WSSEDService(db)
    
    try:
        return service.get_recording_timeline(recording_id, training_job_id, threshold, uncertain_range)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ============ FEEDBACK ENDPOINTS ============

@router.post("/predictions/{prediction_id}/feedback", response_model=FeedbackResponse)
def submit_feedback(
    prediction_id: int,
    feedback_in: FeedbackSubmit,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Submit feedback on a prediction (accept or reject).
    
    - **ACCEPTED**: Creates a strong_positive label (species present) for retraining
    - **REJECTED**: Creates a strong_negative label (species absent) for retraining
    
    If 5 or more feedbacks have been submitted since the last training,
    this will automatically trigger a retraining job.
    """
    service = WSSEDService(db)
    
    try:
        result = service.submit_feedback(prediction_id, feedback_in.feedback)
        
        message = f"Feedback submitted successfully"
        if result['retraining_triggered']:
            message += ". Retraining automatically triggered."
        
        return FeedbackResponse(
            success=True,
            retraining_triggered=result['retraining_triggered'],
            feedback_count=result['feedback_count'],
            message=message
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {str(e)}")


@router.get("/training-jobs/{job_id}/feedback-stats", response_model=FeedbackStats)
def get_feedback_stats(
    job_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Get feedback statistics for a training job.
    
    Returns counts of accepted, rejected, and pending predictions,
    plus the number of new feedbacks since the job was created.
    """
    service = WSSEDService(db)
    
    job = service.get_training_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    return service.get_feedback_stats(job_id)


# ============ RETRAINING ENDPOINTS ============

@router.post("/training-jobs/{job_id}/retrain", response_model=TrainingJob, status_code=status.HTTP_201_CREATED)
def manual_retrain(
    job_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Manually trigger retraining with accumulated feedback.
    
    This creates a new training job that includes:
    - Original weak labels from the dataset
    - Strong labels from accepted predictions
    - Negative examples from rejected predictions
    """
    service = WSSEDService(db)
    
    try:
        new_job = service.trigger_retraining(job_id)
        
        # Trigger training in background
        background_tasks.add_task(trigger_wssed_training, new_job.id)
        
        return new_job
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to trigger retraining: {str(e)}")


# ============ SPECIES ENDPOINTS ============

@router.get("/datasets/{dataset_id}/species", response_model=SpeciesList)
def get_dataset_species(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Get list of species detected in a dataset.
    
    Species are extracted from the FNJV filename format.
    """
    service = WSSEDService(db)
    
    try:
        return service.get_dataset_species(dataset_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get species list: {str(e)}")
