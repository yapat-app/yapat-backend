from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from enum import Enum

class FPVRequest(BaseModel):
    dataset_id: int
    model_family_name: str
    run_3d: bool = False


class FPVDatasetRequest(BaseModel):
    dataset_id: int
    embedding_model_id: int
    run_3d: bool = False

class FPVPointMetadata(BaseModel):
    snippet_id: int
    predicted_labels: List[str]
    uncertainty: Optional[float] = None
    diversity: Optional[float] = None
    density: Optional[float] = None
    composite_score: Optional[float] = None


class FPVProjection2D(BaseModel):
    x: List[float]
    y: List[float]


class FPVProjection3D(BaseModel):
    x: List[Optional[float]]
    y: List[Optional[float]]
    z: List[Optional[float]]


class FPVResponse(BaseModel):
    dataset_id: int
    # For checkpoint-based projections this is provided; for dataset-level projections it is omitted.
    model_family_name: Optional[str] = None
    model_checkpoint_id: Optional[int] = None
    embedding_model_id: Optional[int] = None
    points: List[FPVPointMetadata]
    projections_2d: Dict[str, FPVProjection2D]
    projections_3d: Optional[Dict[str, FPVProjection3D]] = None