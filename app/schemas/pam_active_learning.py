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
    model_type: str = Field(default="pam_multi_label_classifier", description="Classifier type identifier")
    hyperparameters: Optional[Dict[str, Any]] = None
    is_base: bool = Field(default=False, description="Mark as base model entry (uses shared base weights)")
    parent_checkpoint_id: Optional[int] = Field(None, description="Parent checkpoint ID for version lineage")

class ALCheckpointHyperparameters(BaseModel):
    training_mode: Optional[str] = None

    embedding_model_id: Optional[int] = None
    metadata_path: Optional[str] = None
    label_config_path: Optional[str] = None

    min_samples_per_class: Optional[int] = None
    max_samples_per_class: Optional[int] = None

    epochs: Optional[int] = None
    learning_rate: Optional[float] = None
    batch_size: Optional[int] = None
    hidden_dim: Optional[int] = None
    dropout: Optional[float] = None
    device: Optional[str] = None

    resolved_snippet_set_id: Optional[int] = None
    n_dim: Optional[int] = None
    num_classes: Optional[int] = None
    train_samples: Optional[int] = None

    used_species: Optional[List[str]] = None
    excluded_species: Optional[List[str]] = None
    class_counts: Optional[Dict[str, int]] = None

class ALCheckpointResponse(BaseModel):
    id: int
    dataset_id: int
    name: str
    version: str
    checkpoint_path: Optional[str] = None
    model_type: str
    hyperparameters: Optional[ALCheckpointHyperparameters] = None
    is_base: int = 0
    parent_checkpoint_id: Optional[int] = None
    status: ALModelStatusSchema
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# ── Inference / Predictions ────────────────────────────────────────────

class ALRunInferenceRequest(BaseModel):
    """Trigger inference on a snippet set using a checked-out model."""
    model_checkpoint_id: int = Field(..., description="Checked-out model checkpoint ID")
    snippet_set_id: int = Field(..., description="Snippet set to run inference on")
    k: int = Field(default=20, ge=1, le=500, description="Number of top-ranked informative samples to return")
    device: str = Field(default="cpu", description="cpu or cuda")


class ALPredictionResponse(BaseModel):
    id: int
    model_checkpoint_id: int
    snippet_id: int
    predicted_label: str
    uncertainty: Optional[float] = None
    diversity: Optional[float] = None
    density: Optional[float] = None
    composite_score: Optional[float] = None
    created_at: datetime

    class Config:
        from_attributes = True


class ALInferenceResult(BaseModel):
    """Result returned after running inference + scoring."""
    predictions: List[ALPredictionResponse]
    total_labeled: int
    total_unlabeled: int
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

# ── Train from scratch ────────────────────────────────────────────────────────────

class ALTrainFromScratchRequest(BaseModel):
    """Train a fresh classifier from ground-truth labels to mitigate cold start."""
    dataset_id: int = Field(..., description="Dataset to train on")
    snippet_set_id: Optional[int] = Field(
        None,
        description="Snippet set to use; defaults to dataset.default_snippet_set_id",
    )
    embedding_model_id: int = Field(
        ...,
        description="Embedding model whose vectors should be used for training",
    )
    metadata_path: str = Field(
        ...,
        description="Path to metadata file containing ground-truth labels",
    )
    label_config_path: str = Field(..., description="Path to label config file consisting of class names to train on")
    min_samples_per_class: int = Field(
        default=5,
        ge=1,
        description="Minimum number of labeled samples required for a class to be included",
    )
    max_samples_per_class: Optional[int] = Field(
        default=None,
        ge=1,
        description="Optional cap on samples per class for balancing",
    )
    checkpoint_name: str = Field(
        default="cold_start_base",
        description="Human-readable name for the new base checkpoint",
    )
    version: str = Field(default="v0", description="Version tag for the new checkpoint")
    model_type: str = Field(default="pam_multilabel_classifier", description="Classifier type identifier")
    epochs: int = Field(default=20, ge=1, le=500)
    learning_rate: float = Field(default=1e-3, gt=0)
    batch_size: int = Field(default=32, ge=1, le=4096)
    hidden_dim: int = Field(default=128, ge=1)
    dropout: float = Field(default=0.5, ge=0.0, le=0.9)
    device: str = Field(default="cpu", description="cpu or cuda")


# ── Stats ──────────────────────────────────────────────────────────────

class ALStats(BaseModel):
    model_checkpoint_id: int
    total_predictions: int
    total_feedback: int
    accepted: int
    rejected: int
    modified: int
    feedback_since_last_retrain: int
    retrain_jobs: int

