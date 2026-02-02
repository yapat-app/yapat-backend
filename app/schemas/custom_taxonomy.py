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
