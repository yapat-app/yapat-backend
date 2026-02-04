"""
Taxonomy endpoints for species/taxon search and resolution
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional
from sqlalchemy.orm import Session

from app.api.deps import get_current_active_user, get_db
from app.models.user import User
from app.models.team import TeamMembership
from app.core import taxonomy
from app.services import custom_taxonomy_service
from app.services.taxonomy_search_service import (
    search_custom_taxonomies,
    get_user_custom_taxonomy_ids,
    sort_results_by_relevance
)
from app.schemas.custom_taxonomy import AvailableTaxonomy, AvailableTaxonomiesResponse
from app.schemas.taxonomy import (
    TaxonSuggestion,
    TaxonSearchResult,
    TaxonDetails,
    BatchResolveRequest,
    BatchResolveResponse
)
from app.models.custom_taxonomy import TaxonomyStatus


router = APIRouter()


@router.get("/available", response_model=AvailableTaxonomiesResponse)
def get_available_taxonomies(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get all taxonomies available to the current user.
    
    Returns both GBIF (standard) and custom taxonomies that the user has access to.
    Custom taxonomies are filtered by:
    - Team membership
    - Global taxonomies (available to all)
    - Active status only
    """
    taxonomies_list = []
    
    # Add GBIF taxonomy (always available)
    taxonomies_list.append(AvailableTaxonomy(
        taxonomy_id="gbif",
        name="GBIF (Global Biodiversity Information Facility)",
        type="gbif",
        description="Standard global taxonomy from GBIF",
        is_global=True
    ))
    
    # Get user's team memberships
    memberships = db.query(TeamMembership).filter(
        TeamMembership.user_id == current_user.id
    ).all()
    
    # Get custom taxonomies from user's teams and global taxonomies
    seen_taxonomy_ids = {"gbif"}  # Track to avoid duplicates
    for membership in memberships:
        team_taxonomies = custom_taxonomy_service.get_available_taxonomies(
            team_id=membership.team_id,
            user_id=current_user.id,
            db=db,
            status=TaxonomyStatus.ACTIVE.value  # Only active taxonomies
        )
        for t in team_taxonomies:
            if t.taxonomy_id not in seen_taxonomy_ids:
                taxonomies_list.append(AvailableTaxonomy(
                    taxonomy_id=t.taxonomy_id,
                    name=t.name,
                    type="custom",
                    description=t.description,
                    team_id=t.team_id,
                    is_global=t.is_global,
                    status=t.status
                ))
                seen_taxonomy_ids.add(t.taxonomy_id)
    
    return AvailableTaxonomiesResponse(
        taxonomies=taxonomies_list,
        total=len(taxonomies_list)
    )


@router.get("/suggest", response_model=List[TaxonSuggestion])
def suggest_taxa(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(10, ge=1, le=50, description="Maximum number of suggestions"),
    taxonomy_ids: Optional[str] = Query(None, description="Comma-separated list of taxonomy IDs to filter by (e.g., 'gbif,custom:uuid1,custom:uuid2'). If not provided, searches all available taxonomies."),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Fast autocomplete suggestions for taxon names.
    
    Optimized for real-time search as users type.
    Can filter by specific taxonomies. If taxonomy_ids is provided, only searches
    within those taxonomies. Otherwise searches all available taxonomies.
    """
    suggestions = []
    
    # Parse taxonomy filter
    selected_taxonomies = None
    if taxonomy_ids:
        selected_taxonomies = [tid.strip() for tid in taxonomy_ids.split(",") if tid.strip()]
    
    # If no filter or GBIF is included, search GBIF
    if not selected_taxonomies or "gbif" in selected_taxonomies:
        gbif_suggestions = taxonomy.suggest_species(q, limit=limit)
        suggestions.extend(gbif_suggestions)
    
    # Search custom taxonomies if they're selected
    if selected_taxonomies:
        custom_taxonomy_ids = [tid for tid in selected_taxonomies if tid.startswith("custom:")]
        if custom_taxonomy_ids:
            custom_suggestions = search_custom_taxonomies(
                query=q,
                taxonomy_ids=custom_taxonomy_ids,
                limit=limit,
                current_user=current_user,
                db=db
            )
            suggestions.extend(custom_suggestions)
    else:
        # Search all available custom taxonomies
        all_custom_taxonomy_ids = get_user_custom_taxonomy_ids(current_user, db)
        if all_custom_taxonomy_ids:
            custom_suggestions = search_custom_taxonomies(
                query=q,
                taxonomy_ids=all_custom_taxonomy_ids,
                limit=limit,
                current_user=current_user,
                db=db
            )
            suggestions.extend(custom_suggestions)
    
    # Sort by relevance and limit results
    suggestions = sort_results_by_relevance(suggestions, q)
    return suggestions[:limit]


@router.get("/search", response_model=List[TaxonSearchResult])
def search_taxa(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of results"),
    rank: Optional[str] = Query(None, description="Filter by rank (SPECIES, GENUS, FAMILY, etc.)"),
    status: str = Query("ACCEPTED", description="Filter by taxonomic status"),
    taxonomy_ids: Optional[str] = Query(None, description="Comma-separated list of taxonomy IDs to filter by (e.g., 'gbif,custom:uuid1,custom:uuid2'). If not provided, searches all available taxonomies."),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Full text search over taxon names with filtering.
    
    Supports filtering by rank, taxonomic status, and specific taxonomies.
    Can filter by specific taxonomies. If taxonomy_ids is provided, only searches
    within those taxonomies. Otherwise searches all available taxonomies.
    Returns detailed information including common names and habitats.
    """
    results = []
    
    # Parse taxonomy filter
    selected_taxonomies = None
    if taxonomy_ids:
        selected_taxonomies = [tid.strip() for tid in taxonomy_ids.split(",") if tid.strip()]
    
    # If no filter or GBIF is included, search GBIF
    if not selected_taxonomies or "gbif" in selected_taxonomies:
        gbif_results = taxonomy.search_species(q, limit=limit, rank=rank, status=status)
        results.extend(gbif_results)
    
    # Search custom taxonomies if they're selected
    if selected_taxonomies:
        custom_taxonomy_ids = [tid for tid in selected_taxonomies if tid.startswith("custom:")]
    else:
        # Search all available custom taxonomies
        custom_taxonomy_ids = get_user_custom_taxonomy_ids(current_user, db)
    
    if custom_taxonomy_ids:
        custom_results = search_custom_taxonomies(
            query=q,
            taxonomy_ids=custom_taxonomy_ids,
            limit=limit,
            current_user=current_user,
            db=db
        )
        # Convert to TaxonSearchResult format
        for r in custom_results:
            results.append({
                "taxon_id": r["taxon_id"],
                "canonical_name": r.get("canonical_name"),
                "scientific_name": r.get("scientific_name"),
                "rank": r.get("rank"),
                "kingdom": r.get("kingdom"),
                "status": r.get("status"),
                "common_names": [],
                "habitats": [],
                "taxonomic_status": r.get("status")
            })
    
    # Sort by relevance and limit results
    results = sort_results_by_relevance(results, q)
    return results[:limit]


@router.get("/resolve", response_model=TaxonDetails)
def resolve_taxon(
    id: str = Query(..., description="Namespaced taxon ID (e.g., 'gbif:2420576' or 'custom:uuid')"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Resolve a single taxon ID to detailed information.
    
    Supports both GBIF taxonomies and custom taxonomies.
    Returns full taxonomic hierarchy, common names, and other metadata.
    Used for validating taxon IDs and displaying detailed information.
    """
    # Validate format (both GBIF and custom patterns)
    if not taxonomy.TAXON_ID_PATTERN.match(id) and not taxonomy.CUSTOM_TAXON_ID_PATTERN.match(id):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid taxon ID format. Expected format: 'namespace:key' (e.g., 'gbif:2420576' or 'custom:uuid')"
        )
    
    result = taxonomy.resolve_taxon_id(id, db_session=db)
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
    id: str = Query(..., description="Namespaced taxon ID to validate (e.g., 'gbif:2420576' or 'custom:uuid')"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Validate if a taxon ID exists and is resolvable.
    
    Supports both GBIF taxonomies and custom taxonomies.
    Returns a simple boolean response indicating validity.
    """
    is_valid = taxonomy.validate_taxon_id(id, db_session=db)
    return {"taxon_id": id, "valid": is_valid}

