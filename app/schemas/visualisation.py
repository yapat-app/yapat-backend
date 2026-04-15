from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from enum import Enum

class FPVVisibilityField(str, Enum):
    NONE = "none"
    UNCERTAINTY = "uncertainty"
    DIVERSITY = "diversity"
    DENSITY = "density"
    COMPOSITE = "composite"
    YEAR_CYCLE = "year_cycle"
    DAY_CYCLE = "day_cycle"


class FPVColorField(str, Enum):
    NONE = "none"
    PREDICTED_LABEL = "predicted_label"
    UNCERTAINTY = "uncertainty"
    DIVERSITY = "diversity"
    DENSITY = "density"
    COMPOSITE = "composite"
    YEAR_CYCLE = "year_cycle"
    DAY_CYCLE = "day_cycle"
    SOUND_TYPE = "sound_type"
    BIRDNET_LABEL = "birdnet_label"
    YAMNET_LABEL = "yamnet_label"


class FPVRequest(BaseModel):
    dataset_id: int
    model_family_name: str
    run_3d: bool = False

    color_filter_value: FPVColorField = FPVColorField.PREDICTED_LABEL
    visibility_filter_value: FPVVisibilityField = FPVVisibilityField.COMPOSITE

    visibility_range_min: Optional[float] = Field(default=None)
    visibility_range_max: Optional[float] = Field(default=None)

class FPVVisibilityRangeResponse(BaseModel):
    field: FPVVisibilityField
    min_value: float
    max_value: float
    step: float
    label: str

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

class FPVColorMetadata(BaseModel):
    field: FPVColorField
    values: List[Optional[str | float]]
    mode: str  # "continuous", "categorical", "none"

class FPVResponse(BaseModel):
    dataset_id: int
    model_family_name: str
    model_checkpoint_id: int
    color_filter_value: FPVColorField
    visibility_filter_value: FPVVisibilityField
    color: FPVColorMetadata
    points: List[FPVPointMetadata]
    projections_2d: Dict[str, FPVProjection2D]
    projections_3d: Optional[Dict[str, FPVProjection3D]] = None
