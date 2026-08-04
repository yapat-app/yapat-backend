"""
Quick label entry schemas
"""

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# Guardrail for the bulk create path. A GBIF pick sends one item; an Ontology
# Engineering label space could send many, but not thousands.
MAX_QUICK_LABELS_PER_REQUEST = 200


class QuickLabelEntryCreate(BaseModel):
    """One label to add to the caller's personal quick-label palette."""

    taxon_id: str = Field(..., min_length=1, max_length=160)
    display_name: str = Field(..., min_length=1, max_length=255)
    rank: Optional[str] = Field(None, max_length=32)
    # 'oe' is intentionally not accepted here: OE rows are dataset-wide and are
    # written service-side, not through this participant-facing endpoint.
    source: Literal["gbif", "manual"] = "gbif"


class QuickLabelEntryBatchCreate(BaseModel):
    labels: List[QuickLabelEntryCreate] = Field(
        ..., min_length=1, max_length=MAX_QUICK_LABELS_PER_REQUEST
    )


class QuickLabelEntryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    taxon_id: str
    display_name: str
    rank: Optional[str] = None
    source: str
    created_at: Optional[datetime] = None
    # True when the row belongs to the calling user, i.e. they may delete it.
    # Dataset-wide rows (Ontology Engineering) come back with owned=False.
    owned: bool
