from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from enum import Enum

class FPVRequest(BaseModel):
    dataset_id: int
    model_family_name: str
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
    model_family_name: str
    model_checkpoint_id: int
    points: List[FPVPointMetadata]
    projections_2d: Dict[str, FPVProjection2D]
    projections_3d: Optional[Dict[str, FPVProjection3D]] = None