"""
Reference data pool link schemas.

A DatasetReferenceLink attaches a reference dataset (Dataset.is_reference=True)
to either one target dataset or an entire team. See
docs/reference-data-pool-design.md.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, model_validator


class DatasetReferenceLinkCreate(BaseModel):
    reference_dataset_id: int
    dataset_id: Optional[int] = None
    team_id: Optional[int] = None

    @model_validator(mode="after")
    def _exactly_one_scope(self) -> "DatasetReferenceLinkCreate":
        if (self.dataset_id is None) == (self.team_id is None):
            raise ValueError("Exactly one of dataset_id or team_id must be set.")
        return self


class DatasetReferenceLink(BaseModel):
    id: int
    reference_dataset_id: int
    dataset_id: Optional[int] = None
    team_id: Optional[int] = None
    created_at: datetime

    class Config:
        from_attributes = True
