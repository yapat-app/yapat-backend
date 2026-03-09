"""
PAM Active Learning Pydantic schemas

Request / response models for the PAM active learning API.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


# ── Enums (mirror ORM enums for API layer) ─────────────────────────────

class ALModelStatusSchema(str, Enum):
    AVAILABLE = "AVAILABLE"
    LOADING = "LOADING"
    ERROR = "ERROR"


class ALFeedbackActionSchema(str, Enum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    MODIFY = "MODIFY"


class ALRetrainStatusSchema(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


# ── Model Checkpoint ───────────────────────────────────────────────────

class ALCheckpointCreate(BaseModel):
    """Register / checkout a model checkpoint for a dataset."""
    dataset_id: int = Field(..., description="ID of the PAM dataset")
    name: str = Field(..., description="Human-readable model name")
    version: str = Field(default="v0", description="Version tag")
    checkpoint_path: Optional[str] = Field(None, description="Filesystem path to weights (optional)")
    model_type: str = Field(default="pam_classifier", description="Classifier type identifier")
    hyperparameters: Optional[Dict[str, Any]] = None
    is_base: bool = Field(default=False, description="Mark as base model entry (uses shared base weights)")
    parent_checkpoint_id: Optional[int] = Field(None, description="Parent checkpoint ID for version lineage")


class ALCheckpointResponse(BaseModel):
    id: int
    dataset_id: int
    name: str
    version: str
    checkpoint_path: Optional[str] = None
    model_type: str
    hyperparameters: Optional[Dict[str, Any]] = None
    is_base: int = 0
    parent_checkpoint_id: Optional[int] = None
    status: PAMModelStatusSchema
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# ── Inference / Predictions ────────────────────────────────────────────

class ALRunInferenceRequest(BaseModel):
    """Trigger inference on a snippet set using a checked-out model."""
    model_checkpoint_id: int = Field(..., description="Checked-out model checkpoint ID")
    snippet_set_id: int = Field(..., description="Snippet set to run inference on")
    k: int = Field(default=20, ge=1, le=500, description="Number of top-ranked predictions to return")
    device: str = Field(default="cpu", description="cpu or cuda")


class ALPredictionResponse(BaseModel):
    id: int
    model_checkpoint_id: int
    snippet_id: int
    predicted_label: str
    uncertainty: Optional[float] = None
    diversity: Optional[float] = None
    density: Optional[float] = None
    composite: Optional[float] = None
    created_at: datetime

    class Config:
        from_attributes = True


class PAMInferenceResult(BaseModel):
    """Result returned after running inference + scoring."""
    predictions: List[ALPredictionResponse]
    total_scored: int
    model_info: Dict[str, Any]


# ── Feedback ───────────────────────────────────────────────────────────

class ALFeedbackSubmit(BaseModel):
    """Submit feedback on a single prediction."""
    prediction_id: int = Field(..., description="Prediction to give feedback on")
    action: ALFeedbackActionSchema = Field(..., description="ACCEPT, REJECT, or MODIFY")
    modified_label: Optional[str] = Field(
        None, description="Corrected label (required when action=MODIFY)"
    )
    notes: Optional[str] = None


class ALFeedbackResponse(BaseModel):
    id: int
    prediction_id: int
    action: ALFeedbackActionSchema
    modified_label: Optional[str] = None
    notes: Optional[str] = None
    created_at: datetime
    # auto-retrain status
    feedback_count_since_retrain: int
    retrain_triggered: bool

    class Config:
        from_attributes = True


# ── Retrain ────────────────────────────────────────────────────────────

class ALRetrainRequest(BaseModel):
    """Manually trigger retraining."""
    model_checkpoint_id: int = Field(..., description="Checkpoint to retrain")
    epochs: int = Field(default=5, ge=1, le=500)
    learning_rate: float = Field(default=1e-3, gt=0)
    device: str = Field(default="cpu")


class ALRetrainJobResponse(BaseModel):
    id: int
    model_checkpoint_id: int
    trigger: str
    feedback_count: int
    status: ALRetrainStatusSchema
    result_metrics: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime
    # Populated after successful retrain — the new versioned checkpoint
    new_checkpoint_id: Optional[int] = None
    new_checkpoint_path: Optional[str] = None

    class Config:
        from_attributes = True


# ── Stats ──────────────────────────────────────────────────────────────

class ALActiveLearningStats(BaseModel):
    model_checkpoint_id: int
    total_predictions: int
    total_feedback: int
    accepted: int
    rejected: int
    modified: int
    feedback_since_last_retrain: int
    retrain_jobs: int
