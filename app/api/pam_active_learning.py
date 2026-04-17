"""
PAM Active Learning API endpoints

REST API for the PAM-specific active learning flow:
  - Model checkpoint management
  - Inference + scoring  (synchronous — typically fast)
  - Feedback (accept / reject / modify)
  - Retrain (manual + auto) — dispatched as background Celery tasks
  - Job polling
"""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from typing import List, Optional

from app.api.deps import get_db, get_current_active_user
from app.models.user import User
from app.schemas.pam_active_learning import (
    ALCheckpointCreate,
    ALCheckpointResponse,
    ALRunInferenceRequest,
    ALFeedbackSubmit,
    ALFeedbackResponse,
    ALRetrainRequest,
    ALStats,
    ALTrainFromScratchRequest,
    ALPredictionListResponse,
    ALJobDispatch,
    ALRetrainJobStatusResponse,
)
from app.services.pam_active_learning_service import PAMActiveLearningService

router = APIRouter()


# ============ MODEL CHECKPOINT ENDPOINTS ============

@router.post(
    "/checkpoints",
    response_model=ALCheckpointResponse,
    status_code=status.HTTP_201_CREATED,
)
def register_checkpoint(
    body: ALCheckpointCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Register (or update) a model checkpoint for a PAM dataset."""
    svc = PAMActiveLearningService(db)
    try:
        ckpt = svc.register_checkpoint(
            dataset_id=body.dataset_id,
            model_family_name=body.model_family_name,
            version=body.version,
            checkpoint_path=body.checkpoint_path,
            model_type=body.model_type,
            hyperparameters=body.hyperparameters,
            is_base=body.is_base,
            parent_checkpoint_id=body.parent_checkpoint_id,
        )
        return ckpt
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/checkpoints", response_model=List[ALCheckpointResponse])
def list_checkpoints(
    dataset_id: Optional[int] = Query(None, description="Filter by dataset"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """List all registered PAM model checkpoints."""
    svc = PAMActiveLearningService(db)
    return svc.list_active_family_checkpoints(dataset_id=dataset_id)


@router.get("/checkpoints/{checkpoint_id}", response_model=ALCheckpointResponse)
def get_checkpoint(
    checkpoint_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Get a single checkpoint by ID."""
    svc = PAMActiveLearningService(db)
    ckpt = svc._get_checkpoint(checkpoint_id)
    if ckpt is None:
        raise HTTPException(status_code=404, detail=f"Checkpoint {checkpoint_id} not found.")
    return ckpt


# ============ INFERENCE + SCORING ============

@router.post(
    "/inference/get-or-create",
    response_model=ALPredictionListResponse,
    status_code=status.HTTP_200_OK,
)
def get_or_create_predictions(
    body: ALRunInferenceRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Return cached predictions for the active checkpoint, or run inference if
    none exist yet (or force_refresh=true).
    """
    service = PAMActiveLearningService(db)
    try:
        return service.get_or_create_predictions(body)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve predictions: {str(e)}",
        )


# ============ FEEDBACK ============

@router.post(
    "/feedback",
    response_model=ALFeedbackResponse,
    status_code=status.HTTP_201_CREATED,
)
def submit_feedback(
    body: ALFeedbackSubmit,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Submit snippet-level feedback (ACCEPT / REJECT / MODIFY).

    If the feedback threshold is reached, an auto-retrain job is created and
    dispatched to the background worker automatically.  The response includes
    retrain_triggered=true and auto_retrain_job_id so the caller can poll job
    status via GET /retrain/jobs/{auto_retrain_job_id}.
    """
    service = PAMActiveLearningService(db)
    try:
        result = service.submit_feedback(body)

        if result.get("auto_retrain_job_id"):
            from app.tasks.pam_al_tasks import pam_al_auto_retrain
            pam_al_auto_retrain.delay(
                checkpoint_id=result["auto_retrain_checkpoint_id"],
                job_id=result["auto_retrain_job_id"],
            )

        return result

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit feedback: {str(e)}",
        )


# ============ TRAIN FROM SCRATCH / COLD START ============

@router.post(
    "/train-from-scratch",
    response_model=ALJobDispatch,
    status_code=status.HTTP_202_ACCEPTED,
)
def train_from_scratch(
    body: ALTrainFromScratchRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Start a cold-start training run asynchronously.

    Creates a checkpoint (LOADING) and a retrain job (PENDING) immediately,
    dispatches the work to the pam_al Celery queue, and returns the job_id
    for polling.  Poll GET /retrain/jobs/{job_id} to track progress.
    """
    svc = PAMActiveLearningService(db)
    try:
        ckpt, job = svc.setup_train_from_scratch(body)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to set up training: {str(e)}")

    from app.tasks.pam_al_tasks import pam_al_train_from_scratch
    pam_al_train_from_scratch.delay(checkpoint_id=ckpt.id, job_id=job.id)

    return ALJobDispatch(
        job_id=job.id,
        checkpoint_id=ckpt.id,
        status="PENDING",
        message=(
            f"Cold-start training job {job.id} dispatched. "
            f"Poll GET /retrain/jobs/{job.id} for status."
        ),
    )


# ============ MANUAL RETRAIN ============

@router.post(
    "/retrain/manual",
    response_model=ALJobDispatch,
    status_code=status.HTTP_202_ACCEPTED,
)
def manual_retrain(
    body: ALRetrainRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Trigger a manual retrain asynchronously.

    Creates a new checkpoint (LOADING) and retrain job (PENDING) immediately,
    dispatches the work to the pam_al Celery queue, and returns the job_id
    for polling.  Poll GET /retrain/jobs/{job_id} to track progress.
    """
    svc = PAMActiveLearningService(db)
    try:
        ckpt, job = svc.setup_manual_retrain(body)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to set up manual retrain: {str(e)}")

    from app.tasks.pam_al_tasks import pam_al_manual_retrain
    pam_al_manual_retrain.delay(checkpoint_id=ckpt.id, job_id=job.id)

    return ALJobDispatch(
        job_id=job.id,
        checkpoint_id=ckpt.id,
        status="PENDING",
        message=(
            f"Manual retrain job {job.id} dispatched. "
            f"Poll GET /retrain/jobs/{job.id} for status."
        ),
    )


# ============ JOB POLLING ============

@router.get(
    "/retrain/jobs",
    response_model=List[ALRetrainJobStatusResponse],
)
def list_retrain_jobs(
    dataset_id: Optional[int] = Query(None, description="Filter by dataset"),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """List recent retrain jobs, optionally filtered by dataset."""
    svc = PAMActiveLearningService(db)
    return svc.list_retrain_jobs(dataset_id=dataset_id, limit=limit)


@router.get(
    "/retrain/jobs/{job_id}",
    response_model=ALRetrainJobStatusResponse,
)
def get_retrain_job(
    job_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Poll the status of a retrain job.

    Returns the current status (PENDING → RUNNING → COMPLETED | FAILED)
    along with result_metrics on success or error_message on failure.
    """
    svc = PAMActiveLearningService(db)
    job = svc.get_retrain_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Retrain job {job_id} not found.")
    return job
