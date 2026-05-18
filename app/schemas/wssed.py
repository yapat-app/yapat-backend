"""
Pydantic schemas for WSSED API
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel


# ============ Suggestions ============

class ActiveLearningModelInfo(BaseModel):
    species_model_id: int


class ActiveLearningSuggestion(BaseModel):
    snippet_id: int
    confidence: float


class ActiveLearningResponse(BaseModel):
    model_info: ActiveLearningModelInfo
    suggestions: List[ActiveLearningSuggestion]


# ============ Label submission ============

class ActiveLearningLabel(BaseModel):
    snippet_set_id: int
    dataset_id: int
    species_name: str
    snippet_id: int
    label: int  # 0 = rejected, 1 = accepted


# ============ Retrain (species-level, post-labeling) ============

class RetrainBody(BaseModel):
    snippet_set_id: int
    dataset_id: int
    species_name: str
    device: Optional[str] = "cpu"
    epochs: Optional[int] = 10
    lr: Optional[float] = 0.001


# ============ Histogram ============

class PredictionHistogram(BaseModel):
    bin_edges: List[float]
    counts: List[int]


# ============ Full training job ============

class WSSEDTrainingJobCreate(BaseModel):
    dataset_id: int
    model_name: str = "CDur"
    hyperparameters: Dict[str, Any] = {}


class WSSEDTrainingJobResponse(BaseModel):
    job_id: int
    status: str
    message: str


class WSSEDTrainingStatusResponse(BaseModel):
    job_id: int
    status: str  # PENDING | TRAINING | COMPLETED | FAILED
    model_path: Optional[str] = None
    model_paths: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    progress: Optional[Dict[str, Any]] = None


class WSSEDDatasetArtifactsResponse(BaseModel):
    dataset_path: str
    embeddings_path: str
    embeddings_complete: bool
    embeddings_status: str
    checkpoint_exists: bool
    checkpoint_path: Optional[str] = None
    output_dir: str
    audio_count: int = 0
    npz_count: int = 0
