"""
REST API for WSSED (weakly supervised sound event detection).

Routes cover GPU-backed training job lifecycle, active-learning workflows
(suggestions, labels, species-level retrain), and prediction histograms.
Training is executed asynchronously: this layer persists job state and
delegates remote work to Celery and the WSSED GPU service.
"""

import logging

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
    WSSEDDatasetArtifactsResponse,
    WSSEDRegisterALResponse,
)
from app.services.wssed_service import WSSEDService

logger = logging.getLogger(__name__)

router = APIRouter()


# --- Training jobs -----------------------------------------------------------

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
    Create a training job record and enqueue remote training on the GPU server.

    Returns HTTP 202 with ``job_id`` while work runs asynchronously. Poll
    ``GET /training-jobs/{job_id}/status`` for state transitions. If the Celery
    broker rejects the task, the job is marked FAILED and HTTP 503 is returned.
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

    try:
        celery_task_id = svc.enqueue_wssed_training_dispatch(job.id)
    except Exception as e:
        logger.exception("WSSED training Celery enqueue failed job_id=%s", job.id)
        svc.fail_training_job(
            job.id,
            f"Failed to queue training task (is Redis/broker up and worker listening on "
            f"queue 'default'?): {e}",
        )
        raise HTTPException(
            status_code=503,
            detail=(
                f"Training job {job.id} was created but could not be queued to Celery: {e}"
            ),
        ) from e

    logger.info(
        "WSSED training dispatched job_id=%s celery_task_id=%s", job.id, celery_task_id
    )

    return WSSEDTrainingJobResponse(
        job_id=job.id,
        status=job.status.value,
        message=f"Training job {job.id} dispatched. Poll GET /training-jobs/{job.id}/status.",
    )


@router.post(
    "/training-jobs/{job_id}/dispatch",
    response_model=WSSEDTrainingJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def dispatch_training_job(
    job_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Re-enqueue a training job that is still ``PENDING`` in the database.

    Use when ``POST /training-jobs`` succeeded but ``trigger_wssed_training``
    was never consumed (for example, transient broker or worker issues).
    Returns HTTP 409 if the job is not ``PENDING``.
    """
    svc = WSSEDService(db)
    data = svc.get_training_job_status(job_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Training job {job_id} not found")
    if data["status"] != "PENDING":
        raise HTTPException(
            status_code=409,
            detail=f"Job {job_id} is {data['status']}; only PENDING jobs can be re-dispatched.",
        )

    try:
        celery_task_id = svc.enqueue_wssed_training_dispatch(job_id)
    except Exception as e:
        logger.exception("WSSED training Celery re-dispatch failed job_id=%s", job_id)
        svc.fail_training_job(job_id, f"Failed to re-queue training task: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Could not queue training job {job_id} to Celery: {e}",
        ) from e

    logger.info(
        "WSSED training re-dispatched job_id=%s celery_task_id=%s",
        job_id,
        celery_task_id,
    )

    return WSSEDTrainingJobResponse(
        job_id=job_id,
        status=data["status"],
        message=f"Training job {job_id} queued to Celery (task id {celery_task_id}).",
    )


@router.get(
    "/datasets/{dataset_id}/artifacts",
    response_model=WSSEDDatasetArtifactsResponse,
)
async def get_dataset_artifacts(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Report existing BirdNET embeddings and checkpoints on the GPU server."""
    svc = WSSEDService(db)
    try:
        data = await svc.get_dataset_artifacts(dataset_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        logger.exception("Failed to fetch WSSED artifacts for dataset %s", dataset_id)
        raise HTTPException(
            status_code=502,
            detail=f"Could not reach WSSED GPU server: {e}",
        ) from e
    return WSSEDDatasetArtifactsResponse(**data)


@router.get(
    "/training-jobs/latest",
    response_model=WSSEDTrainingStatusResponse,
)
async def get_latest_training_job_status(
    dataset_id: int = Query(..., description="YAPAT dataset id"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Return the most recent training job for a dataset.

    Used by the WSSED UI to restore state after a page refresh.
    """
    from app.config import settings

    svc = WSSEDService(db)
    job = svc.get_latest_training_job(dataset_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail=f"No training jobs found for dataset {dataset_id}",
        )

    if job.status.value == "TRAINING" and svc.is_status_stale(
        job.id, settings.WSSED_POLL_INTERVAL
    ):
        try:
            await svc.update_training_status(job.id)
        except Exception:
            pass

    data = svc.get_training_job_status(job.id)
    if data is None:
        raise HTTPException(status_code=404, detail="Training job not found")
    data.pop("_updated_at", None)
    return WSSEDTrainingStatusResponse(**data)


@router.get(
    "/training-jobs/{job_id}/status",
    response_model=WSSEDTrainingStatusResponse,
)
async def get_training_job_status(
    job_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Return training job status from the application database.

    For ``TRAINING`` jobs, a conditional refresh queries the GPU server only when
    the last persisted probe is older than ``settings.WSSED_POLL_INTERVAL``
    seconds, limiting load on the remote service during frequent client polling.
    """
    from app.config import settings

    svc = WSSEDService(db)
    data = svc.get_training_job_status(job_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Training job {job_id} not found")

    if data["status"] == "TRAINING" and svc.is_status_stale(
        job_id, settings.WSSED_POLL_INTERVAL
    ):
        try:
            await svc.update_training_status(job_id)
            data = svc.get_training_job_status(job_id)
        except Exception:
            # Best-effort refresh; return last known DB state if the GPU is unreachable.
            pass

    data.pop("_updated_at", None)
    return WSSEDTrainingStatusResponse(**data)


@router.post(
    "/training-jobs/{job_id}/register-al",
    response_model=WSSEDRegisterALResponse,
)
async def register_training_job_for_al(
    job_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Sync a completed WSSED job from the GPU server, copy weights into
    PAM_CHECKPOINTS_DIR, and register an Active Learning model family.
    """
    svc = WSSEDService(db)
    try:
        result = await svc.register_training_job_for_al(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to register WSSED job %s for AL", job_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return WSSEDRegisterALResponse(**result)


# --- Active learning: suggestions --------------------------------------------

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
    """List unlabeled snippets above a confidence threshold for a species model."""
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


# --- Active learning: labels -------------------------------------------------

@router.post("/label", status_code=status.HTTP_204_NO_CONTENT)
def submit_label(
    body: ActiveLearningLabel,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Record user feedback (accept or reject) for one snippet on a species model."""
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


# --- Active learning: retrain ------------------------------------------------

@router.post("/retrain", status_code=status.HTTP_202_ACCEPTED)
def retrain(
    body: RetrainBody,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Start a species-scoped retrain after labeling; enqueues the same Celery path
    as full-dataset training with hyperparameters derived from the request body.
    """
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


# --- Prediction histogram ----------------------------------------------------

@router.get("/histogram", response_model=PredictionHistogram)
def get_histogram(
    model_id: int = Query(...),
    snippet_set_id: int = Query(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Return a histogram of model confidence scores over snippets in a snippet set."""
    svc = WSSEDService(db)
    try:
        result = svc.get_histogram(model_id=model_id, snippet_set_id=snippet_set_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compute histogram: {e}")
    return result
