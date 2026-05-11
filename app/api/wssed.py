"""
WSSED API endpoints

Exposes training job management, active-learning suggestions, label
submission, retrain, and prediction histogram.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_current_active_user
from app.models.user import User
from app.schemas.wssed import (
    ActiveLearningLabel,
    ActiveLearningResponse,
    PredictionHistogram,
    RetrainBody,
    WSSEDTrainingJobCreate,
    WSSEDTrainingJobResponse,
    WSSEDTrainingStatusResponse,
)
from app.services.wssed_service import WSSEDService

router = APIRouter()


# ============ Training jobs ============

@router.post(
    "/training-jobs",
    response_model=WSSEDTrainingJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_training_job(
    body: WSSEDTrainingJobCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Create and dispatch a full WSSED training job to the GPU server.

    Returns the job_id immediately; poll GET /training-jobs/{job_id}/status
    to track progress.
    """
    svc = WSSEDService(db)
    try:
        job = svc.create_training_job(
            dataset_id=body.dataset_id,
            model_name=body.model_name,
            hyperparameters=body.hyperparameters,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create training job: {e}")

    from app.tasks.wssed_tasks import trigger_wssed_training
    trigger_wssed_training.delay(job.id)

    return WSSEDTrainingJobResponse(
        job_id=job.id,
        status=job.status.value,
        message=f"Training job {job.id} dispatched. Poll GET /training-jobs/{job.id}/status.",
    )


@router.get(
    "/training-jobs/{job_id}/status",
    response_model=WSSEDTrainingStatusResponse,
)
def get_training_job_status(
    job_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Return the current status of a WSSED training job.

    The status is read from the local DB (updated by the Celery
    poll_training_status task).  A quick GPU-server probe is also attempted
    when the job is still in TRAINING state so the response stays fresh.
    """
    svc = WSSEDService(db)
    data = svc.get_training_job_status(job_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Training job {job_id} not found")

    # If still running, try a live GPU probe (best-effort, non-blocking)
    if data["status"] == "TRAINING":
        try:
            import asyncio
            job = asyncio.run(svc.update_training_status(job_id))
            data = svc.get_training_job_status(job_id)
        except Exception:
            pass  # keep DB value on probe failure

    return WSSEDTrainingStatusResponse(**data)


# ============ Active learning – suggestions ============

@router.get("/suggestions", response_model=ActiveLearningResponse)
def get_suggestions(
    dataset_id: int = Query(...),
    snippet_set_id: int = Query(...),
    species_name: str = Query(...),
    threshold: float = Query(0.0, ge=0.0, le=1.0),
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Return unlabeled snippet suggestions for active learning."""
    svc = WSSEDService(db)
    try:
        result = svc.get_suggestions(
            dataset_id=dataset_id,
            snippet_set_id=snippet_set_id,
            species_name=species_name,
            threshold=threshold,
            limit=limit,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch suggestions: {e}")
    return result


# ============ Active learning – label ============

@router.post("/label", status_code=status.HTTP_204_NO_CONTENT)
def submit_label(
    body: ActiveLearningLabel,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Accept or reject a snippet for the given species model."""
    svc = WSSEDService(db)
    try:
        svc.submit_label(
            snippet_set_id=body.snippet_set_id,
            dataset_id=body.dataset_id,
            species_name=body.species_name,
            snippet_id=body.snippet_id,
            label=body.label,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit label: {e}")


# ============ Active learning – retrain ============

@router.post("/retrain", status_code=status.HTTP_202_ACCEPTED)
def retrain(
    body: RetrainBody,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Trigger a species-level retrain after labeling."""
    svc = WSSEDService(db)
    try:
        job = svc.retrain(
            snippet_set_id=body.snippet_set_id,
            dataset_id=body.dataset_id,
            species_name=body.species_name,
            device=body.device or "cpu",
            epochs=body.epochs or 10,
            lr=body.lr or 0.001,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to trigger retrain: {e}")

    return {"job_id": job.id, "status": job.status.value}


# ============ Histogram ============

@router.get("/histogram", response_model=PredictionHistogram)
def get_histogram(
    model_id: int = Query(...),
    snippet_set_id: int = Query(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Return a confidence-score histogram for a species model."""
    svc = WSSEDService(db)
    try:
        result = svc.get_histogram(model_id=model_id, snippet_set_id=snippet_set_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compute histogram: {e}")
    return result
