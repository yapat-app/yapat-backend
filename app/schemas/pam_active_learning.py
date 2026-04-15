"""
PAM Active Learning Pydantic schemas

Request / response models for the PAM active learning API.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
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

class SamplingMode(str, Enum):
    RANDOM = "random"
    UNCERTAINTY = "uncertainty"
    DIVERSITY = "diversity"
    DENSITY = "density"
    COMPOSITE = "composite"

# ── Model Checkpoint ───────────────────────────────────────────────────

class ALCheckpointCreate(BaseModel):
    """Register / checkout a model checkpoint for a dataset."""
    dataset_id: int = Field(..., description="ID of the PAM dataset")
    model_family_name: str = Field(..., description="Model family name shared across checkpoint versions")
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
    model_family_name: str
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
    model_family_name: Optional[str] = Field(default=None, description="Model family name shared across checkpoint versions")
    dataset_id : int
    snippet_set_id: int = Field(..., description="Snippet set to retrieve predictions for")
    device: Optional[str] = Field(default="cpu", description="cpu or cuda")

    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    density_k: Optional[int] = Field(default=None, ge=1)
    composite_wu: Optional[float] = Field(default=None)
    composite_wd: Optional[float] = Field(default=None)
    composite_wr: Optional[float] = Field(default=None)

    force_refresh: bool = Field(
        default=False,
        description="If true, rerun inference even if predictions already exist",
    )

    sample_suggestion: bool = Field(
        default=False,
        description="If true, return ranked annotation suggestions instead of the full prediction set.",
    )
    suggestion_strategy: Optional[SamplingMode] = Field(
        default=SamplingMode.COMPOSITE,
        description="Ranking strategy used when sample_suggestion=true.",
    )
    k: Optional[int] = Field(
        default=5,
        description="Number of suggestions to return when sample_suggestion=true.",
    )


class ALPredictionResponse(BaseModel):
    id: Optional[int] = None
    model_checkpoint_id: Optional[int] = None
    snippet_id: int
    predicted_labels: Optional[List[str]] = None
    predicted_probabilities: Optional[Dict[str, float]] = None
    uncertainty: Optional[float] = None
    diversity: Optional[float] = None
    density: Optional[float] = None
    composite_score: Optional[float] = None
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class ALPredictionListResponse(BaseModel):
    mode: Literal["predictions", "suggestions"]
    model_family_name: str
    used_checkpoint_id: Optional[int] = None
    total_predictions: int
    returned_count: int
    suggestion_strategy: SamplingMode = Field(
        default=SamplingMode.COMPOSITE,
        description="uncertainty, density, diversity or composite",
    )
    k: Optional[int] = None
    rows: List[ALPredictionResponse]

class ALInferenceRow(BaseModel):
    snippet_id: int
    embedding: list[float] | None = None
    predicted_labels: list[str]
    predicted_probabilities: dict[str, float]
    uncertainty: float
    diversity: float | None
    density: float | None
    composite_score: float | None


# ── Feedback ───────────────────────────────────────────────────────────
class ALFeedbackSubmit(BaseModel):
    dataset_id: int = Field(..., description="Dataset ID")
    model_family_name: str = Field(..., description="Model family name shared across checkpoint versions")
    snippet_id: int = Field(..., description="Snippet being reviewed")
    embedding_model_id: Optional[int] = Field(
        default=1, # for birdnet
        description="Required in bootstrap mode before the first checkpoint exists",
    )
    action: ALFeedbackActionSchema = Field(
        ...,
        description="ACCEPT, REJECT, or MODIFY",
    )

    labels: Optional[List[str]] = Field(
        default=None,
        description=(
            "Final labels provided by the user. "
            "For ACCEPT this may be omitted to use predicted labels. "
            "For MODIFY this should contain replacement labels. "
            "For REJECT this is usually omitted."
        ),
    )

    notes: Optional[str] = None
    user_id: Optional[int] = None


class ALFeedbackResponse(BaseModel):
    id: int
    model_family_name: Optional[str] = None
    model_checkpoint_id: Optional[int] = None
    active_checkpoint_id: Optional[int] = None
    snippet_id: int
    action: ALFeedbackActionSchema
    final_labels: Optional[List[str]] = None
    notes: Optional[str] = None
    created_at: datetime
    feedback_count_since_retrain: int
    retrain_triggered: bool

    class Config:
        from_attributes = True


# ── Retrain ────────────────────────────────────────────────────────────

class ALRetrainRequest(BaseModel):
    dataset_id: int = Field(..., description="Dataset ID")
    model_family_name: str = Field(..., description="Model family name shared across checkpoint versions")

    run_inference: bool = Field(default=True)
    # Optional hyperparameters for manual retrain. These are read by the service layer
    # (and defaulted to values from the active checkpoint when omitted).
    epochs: Optional[int] = Field(default=None, ge=1, le=500)
    learning_rate: Optional[float] = Field(default=None)
    batch_size: Optional[int] = Field(default=None, ge=1)
    hidden_dim: Optional[int] = Field(default=None, ge=1)
    dropout: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    device: Optional[str] = Field(default=None, description="cpu or cuda")

    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    density_k: Optional[int] = Field(default=None, ge=1)
    composite_wu: Optional[float] = Field(default=None)
    composite_wd: Optional[float] = Field(default=None)
    composite_wr: Optional[float] = Field(default=None)


class ALRetrainJobResponse(BaseModel):
    id: int
    model_family_name: str
    used_checkpoint_id: int
    active_checkpoint_id: int
    epochs: int
    learning_rate: float
    batch_size: int
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
    metadata_path: Optional[str] = Field(
        default=None,
        description="Path to metadata file containing ground-truth labels",
    )
    label_config_path: Optional[str] = Field(default=None, description="Path to label config file consisting of class names to train on")
    min_samples_per_class: int = Field(
        default=1,
        ge=1,
        description="Minimum number of labeled samples required for a class to be included",
    )
    max_samples_per_class: Optional[int] = Field(
        default=None,
        ge=1,
        nullable=True,
        description="Optional cap on samples per class for balancing",
        example=None
    )
    model_family_name: str = Field(
        default="cold_start_base",
        description="Model family name shared across checkpoint versions",
    )
    version: str = Field(default="v0", description="Version tag for the new checkpoint")
    model_type: Optional[str] = Field(default="pam_multilabel_classifier", description="Classifier type identifier")
    epochs: Optional[int] = Field(default=20, ge=1, le=500)
    learning_rate: Optional[float] = Field(default=1e-3)
    batch_size: Optional[int] = Field(default=16)
    hidden_dim: Optional[int] = Field(default=128)
    dropout: Optional[float] = Field(default=0.5)
    device: Optional[str] = Field(default="cpu", description="cpu or cuda")

    run_inference: bool = Field(default=False)

    threshold: Optional[float] = Field(default=0.6)
    density_k: Optional[int] = Field(default=15)
    composite_wu: Optional[float] = Field(default=0.5)
    composite_wd: Optional[float] = Field(default=0.25)
    composite_wr: Optional[float] = Field(default=0.25)

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

class ALSingleSampleScore(BaseModel):
    uncertainty: float
    diversity: Optional[float]
    density: Optional[float]
    composite: Optional[float]


# ── Async job dispatch ──────────────────────────────────────────────────

class ALJobDispatch(BaseModel):
    """
    Returned immediately when a training or retrain request is accepted and
    dispatched to the background worker.  The client should poll
    GET /api/pam-al/retrain/jobs/{job_id} to track progress.
    """
    job_id: int
    checkpoint_id: int
    status: ALRetrainStatusSchema
    message: str


class ALRetrainJobStatusResponse(BaseModel):
    """Full status of a retrain job — use for polling."""
    id: int
    dataset_id: int
    model_checkpoint_id: int
    trigger: str
    feedback_count: int
    status: ALRetrainStatusSchema
    result_metrics: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime

    class Config:
        from_attributes = True
