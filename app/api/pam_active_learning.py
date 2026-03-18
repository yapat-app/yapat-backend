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
    ALInferenceResult,
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
    response_model=PAMCheckpointResponse,
    status_code=status.HTTP_201_CREATED,
)
def register_checkpoint(
    body: PAMCheckpointCreate,
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


@router.get("/checkpoints", response_model=List[PAMCheckpointResponse])
def list_checkpoints(
    dataset_id: Optional[int] = Query(None, description="Filter by dataset"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """List all registered PAM model checkpoints."""
    svc = PAMActiveLearningService(db)
    return svc.list_checkpoints(dataset_id=dataset_id)


@router.get("/checkpoints/{checkpoint_id}", response_model=PAMCheckpointResponse)
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

@router.post("/inference", response_model=PAMInferenceResult)
def run_inference(
    body: PAMRunInferenceRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Run the classifier on a snippet set, score & rank predictions,
    and return the top-k most informative snippets for labeling.

    Flow: model checkout → classifier inference → combined scoring →
          top-k selection → persist & return predictions.
    """
    svc = PAMActiveLearningService(db)
    try:
        result = svc.run_inference(
            model_checkpoint_id=body.model_checkpoint_id,
            snippet_set_id=body.snippet_set_id,
            k=body.k,
            device=body.device,
        )
        return PAMInferenceResult(
            predictions=[
                PAMPredictionResponse.model_validate(p, from_attributes=True)
                for p in result["predictions"]
            ],
            total_scored=result["total_scored"],
            model_info=result["model_info"],
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Inference failed: {e}"
        )


# # ============ FEEDBACK ============
#
# @router.post("/feedback", response_model=PAMFeedbackResponse)
# def submit_feedback(
#     body: PAMFeedbackSubmit,
#     db: Session = Depends(get_db),
#     current_user: User = Depends(get_current_active_user),
# ):
#     """
#     Submit feedback on a prediction: ACCEPT / REJECT / MODIFY.
#
#     When action=MODIFY, provide the corrected label in ``modified_label``.
#
#     If 5 or more feedbacks have been submitted since the last retrain,
#     retraining is triggered automatically.
#     """
#     svc = PAMActiveLearningService(db)
#     try:
#         result = svc.submit_feedback(
#             prediction_id=body.prediction_id,
#             action=body.action.value,
#             user_id=current_user.id,
#             modified_label=body.modified_label,
#             notes=body.notes,
#         )
#         return PAMFeedbackResponse(
#             id=result["feedback_id"],
#             prediction_id=result["prediction_id"],
#             action=result["action"],
#             modified_label=result["modified_label"],
#             created_at=result["created_at"],
#             feedback_count_since_retrain=result["feedback_count_since_retrain"],
#             retrain_triggered=result["retrain_triggered"],
#         )
#     except ValueError as e:
#         raise HTTPException(status_code=400, detail=str(e))
#     except Exception as e:
#         raise HTTPException(
#             status_code=500, detail=f"Feedback submission failed: {e}"
#         )
#
#
# # ============ RETRAIN ============
#
# @router.post(
#     "/retrain",
#     response_model=ALRetrainJobResponse,
#     status_code=status.HTTP_201_CREATED,
# )
# def retrain(
#     body: ALRetrainRequest,
#     db: Session = Depends(get_db),
#     current_user: User = Depends(get_current_active_user),
# ):
#     """
#     Retraining for a PAM model checkpoint.
#     """
#     svc = PAMActiveLearningService(db)
#     try:
#         job = svc.manual_retrain(
#             model_checkpoint_id=body.model_checkpoint_id,
#             epochs=body.epochs,
#             learning_rate=body.learning_rate,
#             device=body.device,
#         )
#         # Enrich response with new checkpoint info from result_metrics
#         metrics = job.result_metrics or {}
#         return ALRetrainJobResponse(
#             id=job.id,
#             model_checkpoint_id=job.model_checkpoint_id,
#             trigger=job.trigger,
#             feedback_count=job.feedback_count,
#             status=job.status.value,
#             result_metrics=job.result_metrics,
#             error_message=job.error_message,
#             started_at=job.started_at,
#             completed_at=job.completed_at,
#             created_at=job.created_at,
#             new_checkpoint_id=metrics.get("new_checkpoint_id"),
#             new_checkpoint_path=metrics.get("new_checkpoint_path"),
#         )
#     except ValueError as e:
#         raise HTTPException(status_code=400, detail=str(e))
#     except Exception as e:
#         raise HTTPException(
#             status_code=500, detail=f"Retrain trigger failed: {e}"
#         )


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

# @router.post("/inference", response_model=ALInferenceResult)
# def run_inference(
#     body: ALRunInferenceRequest,
#     db: Session = Depends(get_db),
#     current_user: User = Depends(get_current_active_user),
# ):
#     svc = PAMActiveLearningService(db)
#     try:
#         result = svc.run_inference(body)
#         return ALInferenceResult(
#             predictions=[
#                 ALPredictionResponse.model_validate(p, from_attributes=True)
#                 for p in result["predictions"]
#             ],
#             total_scored=result["total_scored"],
#             model_info=result["model_info"],
#         )
#     except ValueError as e:
#         raise HTTPException(status_code=400, detail=str(e))
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Inference failed: {e}")


