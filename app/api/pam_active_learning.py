"""
PAM Active Learning API endpoints

REST API for the PAM-specific active learning flow:
  - Model checkpoint management
  - Inference + scoring
  - Feedback (accept / reject / modify)
  - Retrain (auto after N=5 + manual)
  - Statistics
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
    ALPredictionResponse,
    ALFeedbackSubmit,
    ALFeedbackResponse,
    ALRetrainRequest,
    ALRetrainJobResponse,
    ALStats,
    ALTrainFromScratchRequest
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
    """
    Register (or update) a model checkpoint for a PAM dataset.

    This is the "model checkout" step — it records which model version
    will be used for inference in the active learning loop.
    """
    svc = PAMActiveLearningService(db)
    try:
        ckpt = svc.register_checkpoint(
            dataset_id=body.dataset_id,
            name=body.name,
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
    return svc.list_checkpoints(dataset_id=dataset_id)


@router.get("/checkpoints/{checkpoint_id}", response_model=ALCheckpointResponse)
def get_checkpoint(
    checkpoint_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Get a single checkpoint by ID."""
    svc = PAMActiveLearningService(db)
    try:
        return svc.get_checkpoint(checkpoint_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ============ INFERENCE + SCORING ============


@router.post(
    "/inference/get-or-create",
    response_model=list[ALPredictionResponse],
    status_code=status.HTTP_200_OK,
)
def get_or_create_predictions(
    body: ALRunInferenceRequest,
    db: Session = Depends(get_db),
):
    service = PAMActiveLearningService(db)

    try:
        return service.get_or_create_predictions(body)

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve predictions: {str(e)}",
        )

@router.post(
    "/retrain/manual",
    response_model=ALCheckpointResponse,
    status_code=status.HTTP_201_CREATED,
)
def manual_retrain(
    body: ALRetrainRequest,
    db: Session = Depends(get_db),
):
    service = PAMActiveLearningService(db)

    try:
        return service.manual_retrain(body)

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Manual retraining failed: {str(e)}",
        )


# ============ TRAIN FROM SCRATCH / COLD START TRAINING ============

@router.post(
    "/train-from-scratch",
    response_model=ALCheckpointResponse,
    status_code=status.HTTP_201_CREATED,
)
def train_from_scratch(
    body: ALTrainFromScratchRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    svc = PAMActiveLearningService(db)
    try:
        ckpt = svc.train_from_scratch(body)
        return ckpt
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Cold-start training failed: {e}",
        )

@router.post(
    "/feedback",
    response_model=ALFeedbackResponse,
    status_code=status.HTTP_201_CREATED,
)
def submit_feedback(
    body: ALFeedbackSubmit,
    db: Session = Depends(get_db),
):
    service = PAMActiveLearningService(db)

    try:
        result = service.submit_feedback(body)
        return result

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit feedback: {str(e)}",
        )

