"""
WSSED Pydantic schemas
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class TrainingStatus(str, Enum):
    PENDING = "PENDING"
    TRAINING = "TRAINING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class FeedbackType(str, Enum):
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"


# ============ TRAINING JOB SCHEMAS ============

class WSSEDHyperparameters(BaseModel):
    """Hyperparameters for WSSED training"""
    model_name: str = Field(default="CDur", description="Model architecture: CDur, TALNet, Baseline")
    pooling: str = Field(default="mean", description="Pooling method: max, mean, linear, exp, att, auto, power, hi, hi_plus, hi_fixed")
    epochs: int = Field(default=100, ge=1, le=500, description="Number of training epochs")
    learning_rate: float = Field(default=0.001, gt=0, le=1, description="Learning rate")
    threshold: float = Field(default=0.5, ge=0, le=1, description="Detection threshold")
    sample_rate: int = Field(default=22000, description="Audio sample rate in Hz")
    n_mels: int = Field(default=64, description="Number of mel bands")
    n_fft: int = Field(default=1100, description="FFT window size")
    hop_length: int = Field(default=550, description="Hop length for STFT")
    bag_seconds: str = Field(default="full", description="Bag duration: 'full' or integer seconds")
    instance_duration: float = Field(default=3.0, ge=0.1, le=60.0, description="Duration of instances in seconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "CDur",
                "pooling": "mean",
                "epochs": 100,
                "learning_rate": 0.001,
                "threshold": 0.5,
                "sample_rate": 22000,
                "n_mels": 64,
                "n_fft": 1100,
                "hop_length": 550,
                "bag_seconds": "full",
                "instance_duration": 3.0
            }
        }


class TrainingJobCreate(BaseModel):
    """Create a new training job"""
    dataset_id: int = Field(..., description="ID of the dataset to train on")
    hyperparameters: WSSEDHyperparameters = Field(default_factory=WSSEDHyperparameters)


class TrainingJob(BaseModel):
    """Training job response"""
    id: int
    dataset_id: int
    model_name: str
    hyperparameters: Dict[str, Any]
    status: TrainingStatus
    model_path: Optional[str] = None
    training_metrics: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


# ============ PREDICTION SCHEMAS ============

class Prediction(BaseModel):
    """Single prediction from WSSED model"""
    id: int
    training_job_id: int
    recording_id: int
    species_name: str
    start_time: float
    end_time: float
    confidence: float
    user_feedback: Optional[FeedbackType] = None
    feedback_at: Optional[datetime] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


class PredictionWithRecording(Prediction):
    """Prediction with recording details"""
    recording_file_name: str
    recording_duration: float


class TimelinePrediction(BaseModel):
    """Simplified prediction for timeline visualization"""
    prediction_id: int
    species: str
    start: float
    end: float
    confidence: float
    feedback: Optional[str] = None


class RecordingTimeline(BaseModel):
    """All predictions for a recording, formatted for timeline player"""
    recording_id: int
    file_name: str
    duration: float
    predictions: List[TimelinePrediction]


# ============ FEEDBACK SCHEMAS ============

class FeedbackSubmit(BaseModel):
    """Submit feedback on a prediction"""
    feedback: FeedbackType


class FeedbackResponse(BaseModel):
    """Response after submitting feedback"""
    success: bool
    retraining_triggered: bool
    feedback_count: int
    message: Optional[str] = None


class FeedbackStats(BaseModel):
    """Statistics about feedback for a training job"""
    training_job_id: int
    total_predictions: int
    accepted_count: int
    rejected_count: int
    pending_count: int
    feedback_since_last_training: int


# ============ DETECTION SCHEMAS ============

class DetectionRequest(BaseModel):
    """Request to run detection on a dataset"""
    training_job_id: int
    threshold: Optional[float] = Field(default=0.5, ge=0, le=1)


class DetectionResponse(BaseModel):
    """Response after triggering detection"""
    message: str
    task_id: str
    training_job_id: int


# ============ STRONG LABEL SCHEMAS ============

class StrongLabel(BaseModel):
    """Strong label created from prediction feedback (present or absent)"""
    id: int
    prediction_id: int
    recording_id: int
    species_name: str
    start_time: float
    end_time: float
    confidence: float
    label_type: str  # "strong_positive" for present, "strong_negative" for absent
    created_at: datetime
    
    class Config:
        from_attributes = True


# ============ SPECIES SCHEMAS ============

class SpeciesList(BaseModel):
    """List of species detected in dataset"""
    dataset_id: int
    species: List[str]
    count: int


# ============ RETRAINING SCHEMAS ============

class RetrainingRequest(BaseModel):
    """Request to retrain with feedback"""
    original_job_id: int
    include_feedback: bool = True


class RetrainingResponse(BaseModel):
    """Response after triggering retraining"""
    message: str
    new_job_id: int
    feedback_labels_count: int


# ============ ACTIVE LEARNING SCHEMAS ============

class SpeciesModelCreate(BaseModel):
    """Create or register a species-specific model"""
    species_name: str = Field(..., description="Name of the species")
    dataset_id: int = Field(..., description="ID of the dataset")
    model_directory: str = Field(..., description="Base directory for species models (species-specific subdirectory will be created)")
    metric_type: str = Field(default="macro", description="Metric type: 'macro' or 'micro'")
    prediction_level: str = Field(default="segment", description="Prediction level: 'segment' or 'clip'")
    model_version: Optional[str] = Field(None, description="Optional version identifier")
    hyperparameters: Optional[Dict[str, Any]] = Field(None, description="Optional model hyperparameters")


class SpeciesModel(BaseModel):
    """Species-specific model response"""
    id: int
    species_name: str
    dataset_id: int
    model_directory: str
    metric_type: str
    prediction_level: str
    model_version: Optional[str] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    status: TrainingStatus
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class ActiveLearningSuggestionsRequest(BaseModel):
    """Request for active learning suggestions"""
    snippet_set_id: int = Field(..., description="ID of the snippet set")
    species_name: str = Field(..., description="Name of the species")
    dataset_id: int = Field(..., description="ID of the dataset")
    strategy: str = Field(default="uncertainty", description="Query strategy: 'uncertainty', 'diversity', 'density', 'random'")
    k: int = Field(default=20, ge=1, le=100, description="Number of suggestions to return")
    device: str = Field(default="cpu", description="Device for computation: 'cpu' or 'cuda'")
    seed: int = Field(default=0, description="Random seed for reproducibility")


class ActiveLearningSuggestion(BaseModel):
    """Single active learning suggestion"""
    snippet_id: int
    predicted_probability: float


class ActiveLearningSuggestionsResponse(BaseModel):
    """Response with active learning suggestions"""
    snippet_ids: List[int]
    probs: List[float]
    n_labeled: int
    model_info: Dict[str, Any]
    suggestions: List[ActiveLearningSuggestion]


class ActiveLearningLabelSubmit(BaseModel):
    """Submit a single label for active learning"""
    snippet_set_id: int = Field(..., description="Snippet set ID")
    species_name: str = Field(..., description="Species name")
    dataset_id: int = Field(..., description="Dataset ID")
    snippet_id: int = Field(..., description="Snippet ID to label")
    label: int = Field(..., ge=0, le=1, description="Label: 0=reject (species not present), 1=accept (species present)")


class ActiveLearningLabelResponse(BaseModel):
    """Response after submitting active learning labels"""
    added: int
    labeled_count: int
    retrained: bool
    species_model_id: int
    train_stats: Optional[Dict[str, Any]] = None
    checkpoint: Optional[str] = None


class ActiveLearningStats(BaseModel):
    """Statistics for active learning"""
    species_model_id: int
    snippet_set_id: Optional[int] = None
    total_predictions: int
    labeled: int
    unlabeled: int
    accepted: int
    rejected: int


class SnippetLabelResponse(BaseModel):
    """Response for a snippet label"""
    id: int
    species_model_id: int
    snippet_id: int
    predicted_label: float
    confidence_score: Optional[float] = None
    user_label: Optional[FeedbackType] = None
    labeled_at: Optional[datetime] = None
    created_at: datetime

    class Config:
        from_attributes = True


class PredictionHistogramResponse(BaseModel):
    """Histogram of model predictions (0-1) for a species' snippets.
    X axis: prediction bins, Y axis: count of snippets in each bin."""
    species_model_id: int
    species_name: str
    snippet_set_id: Optional[int] = None
    bin_edges: List[float] = Field(
        ...,
        description="Bin boundaries in [0, 1], e.g. [0, 0.1, 0.2, ..., 1.0]. Length is num_bins + 1."
    )
    counts: List[int] = Field(
        ...,
        description="Number of snippets in each bin. Length equals num_bins."
    )
    total_snippets: int = Field(..., description="Total number of snippets with predictions")
