"""
PAM Active Learning API endpoints

REST API for the PAM-specific active learning flow:
  - Model checkpoint management
  - Inference + scoring  (synchronous — typically fast)
  - Feedback (accept / reject / modify)
  - Retrain (manual + auto) — dispatched as background Celery tasks
  - Job polling
"""

import logging
import os

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
    ALFeedbackCountResponse,
    ALRetrainRequest,
    ALStats,
    ALTrainFromScratchRequest,
    ALTrainingPathDefaultsResponse,
    ALPredictionListResponse,
    ALJobDispatch,
    ALRetrainJobStatusResponse,
    ALLabeledSnippetsResponse,
    ALSnippetLabelsResponse,
    ALSnippetLabel,
)
from app.services.pam_al.service import PAMActiveLearningService

logger = logging.getLogger(__name__)

router = APIRouter()


# ============ MODEL CHECKPOINT ENDPOINTS ============

@router.get(
    "/datasets/{dataset_id}/training-path-defaults",
    response_model=ALTrainingPathDefaultsResponse,
)
def get_training_path_defaults(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Return default metadata and label-config paths for cold-start training."""
    from app.config import settings
    from app.utils.pam_training_paths import resolve_pam_training_paths

    svc = PAMActiveLearningService(db)
    try:
        ds = svc.get_dataset_for_training_paths(dataset_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    data_root = settings.DATA_ROOT or "/data"
    try:
        metadata_path, label_config_path = resolve_pam_training_paths(
            data_root,
            ds.source_uri,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return ALTrainingPathDefaultsResponse(
        metadata_path=metadata_path,
        label_config_path=label_config_path,
        source_uri=ds.source_uri,
    )


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
    """List active PAM model checkpoints (one per model family).

    This returns only checkpoints referenced by ALModelFamilyState.active_model_checkpoint_id.
    """
    svc = PAMActiveLearningService(db)
    return svc.list_active_family_checkpoints(dataset_id=dataset_id)


@router.get("/checkpoints/all", response_model=List[ALCheckpointResponse])
def list_all_checkpoints(
    dataset_id: Optional[int] = Query(None, description="Filter by dataset"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """List all registered PAM model checkpoints (including inactive)."""
    svc = PAMActiveLearningService(db)
    return svc.list_checkpoints(dataset_id=dataset_id)


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


@router.get("/checkpoints/{checkpoint_id}/species", response_model=List[str])
def get_checkpoint_species(
    checkpoint_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Return the species list stored in the checkpoint's label config file."""
    from app.services.pam_al._checkpoint_helpers import load_species_from_label_config

    svc = PAMActiveLearningService(db)
    ckpt = svc._get_checkpoint(checkpoint_id)
    if ckpt is None:
        raise HTTPException(status_code=404, detail=f"Checkpoint {checkpoint_id} not found.")
    if not ckpt.label_config_path:
        raise HTTPException(status_code=404, detail="Checkpoint has no label config path.")
    try:
        return load_species_from_label_config(ckpt.label_config_path)
    except ValueError as e:
        # The checkpoint exists and the file might exist, but its contents are invalid.
        # Treat this as a client/configuration error, not "not found".
        raise HTTPException(status_code=400, detail=str(e))


@router.post(
    "/checkpoints/{checkpoint_id}/activate",
    status_code=status.HTTP_200_OK,
)
def activate_checkpoint(
    checkpoint_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Mark a registered checkpoint as the active checkpoint for its (dataset_id, model_family_name).

    This affects which checkpoint is used by inference/feedback/retrain flows that resolve the
    active model via ALModelFamilyState.
    """
    from app.models.pam_active_learning import ALModelCheckpoint, ALModelStatus
    from app.services.pam_al import _checkpoint_helpers as ckpt_h

    ckpt = (
        db.query(ALModelCheckpoint)
        .filter(ALModelCheckpoint.id == checkpoint_id)
        .one_or_none()
    )
    if ckpt is None:
        raise HTTPException(status_code=404, detail=f"Checkpoint {checkpoint_id} not found.")

    if ckpt.status != ALModelStatus.AVAILABLE:
        raise HTTPException(
            status_code=400,
            detail=f"Checkpoint {checkpoint_id} is not AVAILABLE (status={ckpt.status}).",
        )

    ckpt_h.set_active_family_checkpoint(db, ckpt.dataset_id, ckpt.model_family_name, ckpt.id)
    db.commit()

    return {
        "dataset_id": ckpt.dataset_id,
        "model_family_name": ckpt.model_family_name,
        "active_checkpoint_id": ckpt.id,
    }

@router.get("/species-default", response_model=List[str])
def get_default_species(
    current_user: User = Depends(get_current_active_user),
):
    """Return the species list from the user-study labels file.

    If the labels file is not present in this deployment, return an empty list
    (HTTP 200) instead of 404 so the frontend degrades gracefully without
    spamming this endpoint.
    """
    from app.config import settings
    from app.services.pam_al._checkpoint_helpers import load_species_from_label_config

    default_path = os.path.join(settings.DATA_ROOT, "labels.json")
    if not os.path.isfile(default_path):
        return []
    try:
        return load_species_from_label_config(default_path)
    except ValueError:
        return []


# ============ INFERENCE + SCORING ============

@router.post(
    "/inference/get-or-create",
    response_model=ALPredictionListResponse | ALJobDispatch,
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
        # Fast path: if predictions exist and no refresh requested, return them immediately.
        if not body.force_refresh:
            try:
                return service.get_or_create_predictions(body)
            except ValueError:
                db.rollback()
                raise
            except Exception as fast_err:
                # Sync path failed (OOM, timeout, DB error, etc.) — roll back so the
                # async fallback below can use the same session.
                db.rollback()
                logger.warning(
                    "Sync inference fast path failed for dataset_id=%s model_family=%s; "
                    "falling back to async job: %s",
                    body.dataset_id,
                    body.model_family_name,
                    fast_err,
                    exc_info=True,
                )

        # Async path: create a job record and let the pam_al worker handle inference.
        from datetime import datetime, timezone

        from app.models.pam_active_learning import ALRetrainJob, ALRetrainStatus
        from app.services.pam_al import _checkpoint_helpers as ckpt_h

        model_ckpt = ckpt_h.get_active_checkpoint_for_model_family(db, body.dataset_id, body.model_family_name)
        if model_ckpt is None:
            # No checkpoint: keep previous behavior (random suggestions).
            return service.get_or_create_predictions(body)

        job = ALRetrainJob(
            model_checkpoint_id=model_ckpt.id,
            dataset_id=body.dataset_id,
            trigger="inference",
            feedback_count=0,
            status=ALRetrainStatus.PENDING,
            created_at=datetime.now(timezone.utc),
        )
        db.add(job)
        db.commit()
        db.refresh(job)

        from app.tasks.pam_al_tasks import pam_al_create_predictions

        pam_al_create_predictions.delay(job_id=job.id, inference_body=body.model_dump())

        return ALJobDispatch(
            job_id=job.id,
            checkpoint_id=model_ckpt.id,
            status=ALRetrainStatus.PENDING,
            message=(
                f"Inference job {job.id} dispatched. "
                f"Poll GET /retrain/jobs/{job.id} for status."
            ),
        )

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve predictions: {str(e)}",
        )


# ============ FEEDBACK ============

@router.get(
    "/feedback-count",
    response_model=ALFeedbackCountResponse,
    status_code=status.HTTP_200_OK,
)
def get_feedback_count(
    dataset_id: int = Query(..., description="Dataset ID"),
    model_family_name: str = Query(..., description="Model family name"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Current feedback counter used for auto-retrain gating."""
    service = PAMActiveLearningService(db)
    try:
        return service.get_feedback_count_since_retrain(dataset_id, model_family_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load feedback count: {e}")

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
        # Ensure feedback is attributable; also used when persisting confirmed labels.
        if body.user_id is None:
            body.user_id = current_user.id
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


# ============ USER STUDY-MODE HELPERS ============

@router.get(
    "/labeled-snippets",
    response_model=ALLabeledSnippetsResponse,
)
def list_labeled_snippets(
    dataset_id: int = Query(..., description="Dataset ID"),
    snippet_set_id: Optional[int] = Query(
        None,
        description="Optional snippet-set scope (recommended for large datasets).",
    ),
    scope: str = Query(
        "any",
        description="any=all sources, user=only current user's USER labels",
    ),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Snippet IDs that already have at least one annotation. Used by the
    visualisation to mark the labeled pool (border / highlight).
    """
    svc = PAMActiveLearningService(db)
    try:
        snippet_ids = svc.list_labeled_snippets(
            dataset_id,
            snippet_set_id,
            scope=scope,
            user_id=current_user.id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load labeled snippets: {e}")
    return ALLabeledSnippetsResponse(
        dataset_id=dataset_id,
        snippet_set_id=snippet_set_id,
        snippet_ids=snippet_ids,
    )


@router.get(
    "/snippet-labels",
    response_model=ALSnippetLabelsResponse,
)
def list_snippet_labels(
    dataset_id: int = Query(..., description="Dataset ID"),
    snippet_set_id: Optional[int] = Query(
        None,
        description="Optional snippet-set scope (recommended for large datasets).",
    ),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Per-snippet ground-truth / user labels — feeds the `actual_label` color
    filter on the projection view.
    """
    svc = PAMActiveLearningService(db)
    try:
        items = svc.list_snippet_labels(dataset_id, snippet_set_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load snippet labels: {e}")
    return ALSnippetLabelsResponse(
        dataset_id=dataset_id,
        snippet_set_id=snippet_set_id,
        items=[ALSnippetLabel(**item) for item in items],
    )
