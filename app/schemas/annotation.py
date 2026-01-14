"""
Annotation schemas
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import datetime
from typing import Optional, Dict, Any
import re


TAXON_ID_PATTERN = re.compile(r'^[a-z]+:\d+$')


class AnnotationBase(BaseModel):
    taxon_id: Optional[str] = Field(
        None,
        description="Namespaced taxon identifier (e.g., 'gbif:2420576'). Either taxon_id or species_name must be provided.",
        pattern=r'^[a-z]+:\d+$'
    )
    species_name: Optional[str] = Field(
        None,
        description="Scientific or common species name (e.g., 'Turdus merula' or 'Common Blackbird'). Either taxon_id or species_name must be provided."
    )
    extra_metadata: Optional[Dict[str, Any]] = None
    
    @field_validator('taxon_id')
    @classmethod
    def validate_taxon_id_format(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not TAXON_ID_PATTERN.match(v):
            raise ValueError(
                "taxon_id must be in format 'namespace:key' (e.g., 'gbif:2420576')"
            )
        return v
    
    @model_validator(mode='after')
    def validate_taxon_id_or_species_name(self):
        """Ensure either taxon_id or species_name is provided"""
        if not self.taxon_id and not self.species_name:
            raise ValueError("Either 'taxon_id' or 'species_name' must be provided")
        if self.taxon_id and self.species_name:
            raise ValueError("Provide either 'taxon_id' or 'species_name', not both")
        return self


class AnnotationCreate(AnnotationBase):
    snippet_id: int


class AnnotationBatchCreate(BaseModel):
    """Create multiple annotations for a single snippet"""
    snippet_id: int
    annotations: list[AnnotationBase]


class Annotation(AnnotationBase):
    id: int
    snippet_id: int
    user_id: int
    taxon_id: str = Field(
        ...,
        description="Namespaced taxon identifier (always present in response)"
    )
    resolved_name_snapshot: str = Field(
        ...,
        description="Snapshot of resolved scientific name at time of annotation"
    )
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class AnnotationExport(BaseModel):
    """Annotation with recording and snippet metadata for export"""
    # Annotation fields
    annotation_id: int
    dataset_id: int
    snippet_id: int
    taxon_id: str
    resolved_name_snapshot: str
    confidence: Optional[float]
    created_at: datetime
    created_by: int
    
    # Recording metadata
    recording_file_name: str
    recording_file_path: str
    
    # Snippet metadata
    snippet_start_time: float
    snippet_end_time: float
    snippet_duration: float
    
    class Config:
        from_attributes = True


class DatasetAnnotationStats(BaseModel):
    """Statistics about annotation status for a dataset"""
    dataset_id: int
    total_snippets: int
    annotated_snippets: int = Field(description="Number of snippets with at least one annotation")
    not_annotated_snippets: int = Field(description="Number of snippets with no annotations")
    annotation_percentage: float = Field(description="Percentage of snippets that are annotated (0-100)")
    total_annotations: int = Field(description="Total number of annotations across all snippets")

