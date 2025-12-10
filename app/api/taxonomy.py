"""
Taxonomy endpoints for species/taxon search and resolution
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional
from pydantic import BaseModel

from app.api.deps import get_current_active_user
from app.models.user import User
from app.core import taxonomy


router = APIRouter()


# Response models
class TaxonSuggestion(BaseModel):
    taxon_id: str
    canonical_name: Optional[str]
    scientific_name: Optional[str]
    rank: Optional[str]
    kingdom: Optional[str]
    status: Optional[str]


class CommonName(BaseModel):
    name: str
    language: str


class TaxonSearchResult(BaseModel):
    taxon_id: str
    canonical_name: Optional[str]
    scientific_name: Optional[str]
    rank: Optional[str]
    kingdom: Optional[str]
    status: Optional[str]
    common_names: List[CommonName] = []
    habitats: List[str] = []
    taxonomic_status: Optional[str]


class TaxonDetails(BaseModel):
    taxon_id: str
    canonical_name: Optional[str] = None
    scientific_name: Optional[str] = None
    rank: Optional[str] = None
    status: Optional[str] = None
    kingdom: Optional[str] = None
    phylum: Optional[str] = None
    class_: Optional[str] = None  # "class" is reserved keyword
    order: Optional[str] = None
    family: Optional[str] = None
    genus: Optional[str] = None
    common_names: List[CommonName] = []
    habitats: List[str] = []
    taxonomic_status: Optional[str] = None
    match_type: Optional[str] = None  # For match endpoint
    confidence: Optional[float] = None  # For match endpoint

    class Config:
        # Allow "class" field to be mapped from "class_"
        fields = {'class_': 'class'}


class BatchResolveRequest(BaseModel):
    taxon_ids: List[str]


class BatchResolveResponse(BaseModel):
    results: dict  # taxon_id -> TaxonDetails or None


@router.get("/suggest", response_model=List[TaxonSuggestion])
def suggest_taxa(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(10, ge=1, le=50, description="Maximum number of suggestions"),
    current_user: User = Depends(get_current_active_user)
):
    """
    Fast autocomplete suggestions for taxon names.
    
    Optimized for real-time search as users type.
    Returns a simplified list of matching taxa.
    """
    suggestions = taxonomy.suggest_species(q, limit=limit)
    return suggestions


@router.get("/search", response_model=List[TaxonSearchResult])
def search_taxa(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of results"),
    rank: Optional[str] = Query(None, description="Filter by rank (SPECIES, GENUS, FAMILY, etc.)"),
    status: str = Query("ACCEPTED", description="Filter by taxonomic status"),
    current_user: User = Depends(get_current_active_user)
):
    """
    Full text search over taxon names with filtering.
    
    Supports filtering by rank and taxonomic status.
    Returns detailed information including common names and habitats.
    """
    results = taxonomy.search_species(q, limit=limit, rank=rank, status=status)
    return results


@router.get("/resolve", response_model=TaxonDetails)
def resolve_taxon(
    id: str = Query(..., description="Namespaced taxon ID (e.g., 'gbif:2420576')"),
    current_user: User = Depends(get_current_active_user)
):
    """
    Resolve a single taxon ID to detailed information.
    
    Returns full taxonomic hierarchy, common names, and other metadata.
    Used for validating taxon IDs and displaying detailed information.
    """
    # Validate format
    if not taxonomy.TAXON_ID_PATTERN.match(id):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid taxon ID format. Expected format: 'namespace:key' (e.g., 'gbif:2420576')"
        )
    
    result = taxonomy.resolve_taxon_id(id)
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"Taxon ID '{id}' not found or could not be resolved"
        )
    
    return result


@router.post("/resolve", response_model=BatchResolveResponse)
def batch_resolve_taxa(
    request: BatchResolveRequest,
    current_user: User = Depends(get_current_active_user)
):
    """
    Resolve multiple taxon IDs at once.
    
    Returns a dictionary mapping each taxon ID to its resolved data.
    If a taxon ID cannot be resolved, its value will be null.
    """
    # Validate all IDs first
    invalid_ids = []
    for taxon_id in request.taxon_ids:
        if not taxonomy.TAXON_ID_PATTERN.match(taxon_id):
            invalid_ids.append(taxon_id)
    
    if invalid_ids:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid taxon ID format for: {', '.join(invalid_ids)}"
        )
    
    results = taxonomy.batch_resolve_taxon_ids(request.taxon_ids)
    return {"results": results}


@router.get("/match", response_model=TaxonDetails)
def match_taxon_name(
    name: str = Query(..., min_length=1, description="Species name to match"),
    current_user: User = Depends(get_current_active_user)
):
    """
    Fuzzy match a species name to find the best matching taxon.
    
    Uses GBIF's name matching service with confidence scoring.
    Useful for validating free-text species names and resolving synonyms.
    
    Note: Works best with full species names (e.g., "Hyla versicolor").
    Genus names alone (e.g., "Hyla") may not match. Use /suggest or /search for genus-level queries.
    """
    result = taxonomy.match_species_name(name)
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"No match found for species name '{name}'. "
                   f"Try using /taxonomy/suggest or /taxonomy/search instead for broader queries, "
                   f"or ensure you're using a full species name (e.g., 'Hyla versicolor' instead of just 'Hyla')."
        )
    
    return result


@router.get("/validate", response_model=dict)
def validate_taxon(
    id: str = Query(..., description="Namespaced taxon ID to validate"),
    current_user: User = Depends(get_current_active_user)
):
    """
    Validate if a taxon ID exists and is resolvable.
    
    Returns a simple boolean response indicating validity.
    """
    is_valid = taxonomy.validate_taxon_id(id)
    return {"taxon_id": id, "valid": is_valid}

