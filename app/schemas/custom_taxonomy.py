"""
Custom Taxonomy schemas
"""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Dict, Any, List


class TaxonomyNode(BaseModel):
    """Hierarchical taxonomy node structure"""
    id: str = Field(..., description="Unique identifier within the taxonomy")
    name: str = Field(..., description="Display name of the taxonomy node")
    rank: Optional[str] = Field(None, description="Taxonomic rank (e.g., species, genus, family)")
    parent_id: Optional[str] = Field(None, description="Parent node ID for hierarchy")
    children: Optional[List['TaxonomyNode']] = Field(default_factory=list, description="Child nodes")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class TaxonomyGenerationRequest(BaseModel):
    """Request for generating a custom taxonomy"""
    prompt: str = Field(..., min_length=10, max_length=2000, description="User prompt describing desired taxonomy")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context for generation")


class TaxonomyGenerationResponse(BaseModel):
    """Response from OE_YAPAT service with generated taxonomy"""
    taxonomy_data: Dict[str, Any] = Field(..., description="Generated taxonomy structure")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Generation metadata (model, timestamp, etc)")


class CustomTaxonomyBase(BaseModel):
    """Base schema for custom taxonomy"""
    name: str = Field(..., min_length=1, max_length=255, description="Name of the custom taxonomy")
    description: Optional[str] = Field(None, description="Description of the taxonomy purpose")


class CustomTaxonomyCreate(CustomTaxonomyBase):
    """Schema for creating a custom taxonomy"""
    team_id: int = Field(..., description="Team ID this taxonomy belongs to")
    taxonomy_data: Dict[str, Any] = Field(..., description="Hierarchical taxonomy data structure")
    is_global: Optional[bool] = Field(False, description="Whether this taxonomy is globally available (admin only)")


class CustomTaxonomyUpdate(BaseModel):
    """Schema for updating a custom taxonomy"""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    status: Optional[str] = Field(None, description="Status: draft, active, archived")


class CustomTaxonomyResponse(CustomTaxonomyBase):
    """Response schema for custom taxonomy"""
    id: int
    taxonomy_id: str = Field(..., description="Unique taxonomy identifier (e.g., 'custom:uuid')")
    team_id: int
    created_by_user_id: int
    taxonomy_data: Dict[str, Any]
    status: str
    is_global: bool
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class CustomTaxonomyListResponse(BaseModel):
    """List response for custom taxonomies"""
    taxonomies: List[CustomTaxonomyResponse]
    total: int


class TaxonResolution(BaseModel):
    """Resolved taxon details from custom taxonomy"""
    taxon_id: str
    name: str
    rank: Optional[str] = None
    parent_id: Optional[str] = None
    taxonomy_id: str = Field(..., description="ID of the taxonomy this taxon belongs to")
    taxonomy_name: str = Field(..., description="Name of the taxonomy")
    metadata: Optional[Dict[str, Any]] = None


class TaxonomyLabel(BaseModel):
    """A label/taxon within a taxonomy"""
    id: str = Field(..., description="Label identifier")
    name: str = Field(..., description="Label name")
    scientific_name: Optional[str] = Field(None, description="Scientific name if available")
    rank: Optional[str] = Field(None, description="Taxonomic rank")
    taxon_id: Optional[str] = Field(None, description="Taxon ID (e.g., 'gbif:123' for custom taxonomies)")


class AvailableTaxonomy(BaseModel):
    """A taxonomy available to the user"""
    taxonomy_id: str = Field(..., description="Taxonomy identifier (e.g., 'gbif' or 'custom:uuid')")
    name: str = Field(..., description="Taxonomy name")
    type: str = Field(..., description="Taxonomy type: 'gbif' or 'custom'")
    description: Optional[str] = Field(None, description="Taxonomy description")
    team_id: Optional[int] = Field(None, description="Team ID (for custom taxonomies)")
    is_global: bool = Field(False, description="Whether taxonomy is globally available")
    status: Optional[str] = Field(None, description="Status (for custom taxonomies: active, draft, archived)")
    labels_count: int = Field(0, description="Total number of labels in this taxonomy")
    labels_preview: List[TaxonomyLabel] = Field(default_factory=list, description="Preview of labels (if include_labels=true)")
    has_more_labels: bool = Field(False, description="Whether there are more labels beyond the preview")


class TaxonomyLabelsResponse(BaseModel):
    """Paginated response for taxonomy labels"""
    taxonomy_id: str = Field(..., description="Taxonomy identifier")
    taxonomy_name: str = Field(..., description="Taxonomy name")
    total: int = Field(..., description="Total number of labels")
    limit: int = Field(..., description="Number of labels per page")
    offset: int = Field(..., description="Current offset")
    labels: List[TaxonomyLabel] = Field(..., description="Labels for this page")


class AvailableTaxonomiesResponse(BaseModel):
    """Response with all available taxonomies for a user"""
    taxonomies: List[AvailableTaxonomy] = Field(..., description="List of available taxonomies")
    total: int = Field(..., description="Total number of taxonomies")
