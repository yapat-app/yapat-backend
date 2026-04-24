"""
Taxonomy endpoints for species/taxon search and resolution
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session

from app.api.deps import get_current_active_user, get_db
from app.models.user import User
from app.models.team import TeamMembership
from app.core.permissions import check_admin
from app.core import taxonomy
from app.services import custom_taxonomy_service
from app.services.taxonomy_search_service import (
    search_custom_taxonomies,
    get_user_custom_taxonomy_ids,
    sort_results_by_relevance
)
from app.schemas.custom_taxonomy import (
    AvailableTaxonomy,
    AvailableTaxonomiesResponse,
    TaxonomyLabel,
    TaxonomyLabelsResponse
)
from app.schemas.taxonomy import (
    TaxonSuggestion,
    TaxonSearchResult,
    TaxonDetails,
    BatchResolveRequest,
    BatchResolveResponse
)
from app.models.custom_taxonomy import TaxonomyStatus


router = APIRouter()


def extract_labels_from_taxonomy_data(taxonomy_data: Dict[str, Any]) -> List[TaxonomyLabel]:
    """
    Recursively extract all labels/nodes from a taxonomy_data structure.
    
    Args:
        taxonomy_data: The taxonomy_data JSONB structure from CustomTaxonomy
        
    Returns:
        List of TaxonomyLabel objects
    """
    labels = []
    
    def extract_from_nodes(nodes: List[Dict[str, Any]]):
        """Recursively extract labels from nodes and their children"""
        for node in nodes:
            if not isinstance(node, dict):
                continue
            
            # Extract label information
            label_id = node.get("id") or node.get("taxon_id") or ""
            name = node.get("name") or node.get("canonical_name") or ""
            scientific_name = node.get("scientific_name")
            rank = node.get("rank") or (node.get("metadata", {}) or {}).get("rank")
            taxon_id = node.get("taxon_id")
            
            # Only add if we have at least an ID or name
            if label_id or name:
                labels.append(TaxonomyLabel(
                    id=label_id,
                    name=name,
                    scientific_name=scientific_name,
                    rank=rank,
                    taxon_id=taxon_id
                ))
            
            # Recursively process children if they exist
            if "children" in node and isinstance(node["children"], list):
                extract_from_nodes(node["children"])
    
    # Extract from top-level nodes
    if isinstance(taxonomy_data, dict):
        nodes = taxonomy_data.get("nodes", [])
        if isinstance(nodes, list):
            extract_from_nodes(nodes)
    
    return labels


@router.get("/available", response_model=AvailableTaxonomiesResponse)
def get_available_taxonomies(
    include_labels: bool = Query(False, description="Include label preview in response"),
    labels_preview: int = Query(50, ge=1, le=200, description="Number of labels to preview per taxonomy"),
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
    
    By default, returns only taxonomy metadata with label counts (fast).
    Use include_labels=true to get a preview of labels.
    For full label pagination, use the /taxonomy/{taxonomy_id}/labels endpoint.
    """
    taxonomies_list = []
    
    # Add GBIF taxonomy (always available)
    # For GBIF, we can fetch a sample of species if labels are requested
    gbif_labels_preview = []
    if include_labels:
        # Fetch a sample of species from GBIF using suggest endpoint
        # We'll use multiple common queries to get a diverse sample
        try:
            sample_queries = ["bird", "fish", "plant", "insect", "mammal"]
            per_query_limit = min(labels_preview // len(sample_queries) + 1, 20)
            
            for query in sample_queries:
                if len(gbif_labels_preview) >= labels_preview:
                    break
                
                suggestions = taxonomy.suggest_species(query, limit=per_query_limit)
                for species in suggestions:
                    if len(gbif_labels_preview) >= labels_preview:
                        break
                    gbif_labels_preview.append(TaxonomyLabel(
                        id=species.get("taxon_id", ""),
                        name=species.get("canonical_name") or species.get("scientific_name", ""),
                        scientific_name=species.get("scientific_name"),
                        rank=species.get("rank"),
                        taxon_id=species.get("taxon_id")
                    ))
        except Exception:
            # If GBIF fetch fails, just return empty preview
            gbif_labels_preview = []
    
    taxonomies_list.append(AvailableTaxonomy(
        taxonomy_id="gbif",
        name="GBIF (Global Biodiversity Information Facility)",
        type="gbif",
        description="Standard global taxonomy from GBIF",
        is_global=True,
        labels_count=0,  # GBIF has millions, exact count is not practical
        labels_preview=gbif_labels_preview,
        has_more_labels=True  # GBIF always has more
    ))
    
    seen_taxonomy_ids = {"gbif"}  # Track to avoid duplicates

    # Admins can operate across teams; expose all ACTIVE custom taxonomies to admins.
    if check_admin(current_user):
        from app.models.custom_taxonomy import CustomTaxonomy

        taxonomies = (
            db.query(CustomTaxonomy)
            .filter(CustomTaxonomy.status == TaxonomyStatus.ACTIVE.value)
            .order_by(CustomTaxonomy.created_at.desc())
            .all()
        )
        for t in taxonomies:
            if t.taxonomy_id in seen_taxonomy_ids:
                continue

            all_labels = extract_labels_from_taxonomy_data(t.taxonomy_data)
            labels_count = len(all_labels)

            if include_labels:
                labels_preview_list = all_labels[:labels_preview]
                has_more = labels_count > labels_preview
            else:
                labels_preview_list = []
                has_more = labels_count > 0

            taxonomies_list.append(
                AvailableTaxonomy(
                    taxonomy_id=t.taxonomy_id,
                    name=t.name,
                    type="custom",
                    description=t.description,
                    team_id=t.team_id,
                    is_global=t.is_global,
                    status=t.status,
                    labels_count=labels_count,
                    labels_preview=labels_preview_list,
                    has_more_labels=has_more,
                )
            )
            seen_taxonomy_ids.add(t.taxonomy_id)

    else:
        # Get user's team memberships
        memberships = (
            db.query(TeamMembership)
            .filter(TeamMembership.user_id == current_user.id)
            .all()
        )

        # Get custom taxonomies from user's teams and global taxonomies
        for membership in memberships:
            team_taxonomies = custom_taxonomy_service.get_available_taxonomies(
                team_id=membership.team_id,
                user_id=current_user.id,
                db=db,
                status=TaxonomyStatus.ACTIVE.value,  # Only active taxonomies
            )
            for t in team_taxonomies:
                if t.taxonomy_id in seen_taxonomy_ids:
                    continue

                all_labels = extract_labels_from_taxonomy_data(t.taxonomy_data)
                labels_count = len(all_labels)

                if include_labels:
                    labels_preview_list = all_labels[:labels_preview]
                    has_more = labels_count > labels_preview
                else:
                    labels_preview_list = []
                    has_more = labels_count > 0

                taxonomies_list.append(
                    AvailableTaxonomy(
                        taxonomy_id=t.taxonomy_id,
                        name=t.name,
                        type="custom",
                        description=t.description,
                        team_id=t.team_id,
                        is_global=t.is_global,
                        status=t.status,
                        labels_count=labels_count,
                        labels_preview=labels_preview_list,
                        has_more_labels=has_more,
                    )
                )
                seen_taxonomy_ids.add(t.taxonomy_id)
    
    return AvailableTaxonomiesResponse(
        taxonomies=taxonomies_list,
        total=len(taxonomies_list)
    )


@router.get("/{taxonomy_id}/labels", response_model=TaxonomyLabelsResponse)
def get_taxonomy_labels(
    taxonomy_id: str,
    limit: int = Query(100, ge=1, le=500, description="Number of labels per page"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    q: Optional[str] = Query(None, description="Search query to filter labels by name"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get paginated labels for a specific taxonomy.
    
    Supports:
    - Pagination via limit and offset
    - Optional search/filter via q parameter
    - Works for custom taxonomies only (GBIF uses /suggest or /search endpoints)
    
    Returns:
    - Taxonomy metadata
    - Total label count
    - Paginated labels
    """
    # Handle GBIF separately
    if taxonomy_id == "gbif":
        raise HTTPException(
            status_code=400,
            detail="GBIF taxonomy contains millions of species. Use /taxonomy/suggest or /taxonomy/search endpoints instead."
        )
    
    # Get custom taxonomy
    taxonomy_obj = custom_taxonomy_service.get_taxonomy_by_id(taxonomy_id, db)
    
    if not taxonomy_obj:
        raise HTTPException(
            status_code=404,
            detail=f"Taxonomy '{taxonomy_id}' not found"
        )
    
    # Check access permissions
    if not taxonomy_obj.is_global and not check_admin(current_user):
        membership = (
            db.query(TeamMembership)
            .filter(
                TeamMembership.user_id == current_user.id,
                TeamMembership.team_id == taxonomy_obj.team_id,
            )
            .first()
        )

        if not membership:
            raise HTTPException(status_code=403, detail="You don't have access to this taxonomy")
    
    # Check if taxonomy is active
    if taxonomy_obj.status != TaxonomyStatus.ACTIVE:
        raise HTTPException(
            status_code=400,
            detail=f"Taxonomy is not active (status: {taxonomy_obj.status})"
        )
    
    # Extract all labels
    all_labels = extract_labels_from_taxonomy_data(taxonomy_obj.taxonomy_data)
    
    # Apply search filter if provided
    if q:
        q_lower = q.lower()
        filtered_labels = [
            label for label in all_labels
            if q_lower in label.name.lower() or 
               (label.scientific_name and q_lower in label.scientific_name.lower())
        ]
    else:
        filtered_labels = all_labels
    
    total = len(filtered_labels)
    
    # Apply pagination
    paginated_labels = filtered_labels[offset:offset + limit]
    
    return TaxonomyLabelsResponse(
        taxonomy_id=taxonomy_obj.taxonomy_id,
        taxonomy_name=taxonomy_obj.name,
        total=total,
        limit=limit,
        offset=offset,
        labels=paginated_labels
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
    # Validate format (digits key, alphanumeric key e.g. local:slug, or custom:uuid)
    if not (
        taxonomy.TAXON_ID_PATTERN.match(id)
        or taxonomy.TAXON_ID_ALNUM_PATTERN.match(id)
        or taxonomy.CUSTOM_TAXON_ID_PATTERN.match(id)
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid taxon ID format. Expected format: 'namespace:key' (e.g., 'gbif:2420576', 'local:species_slug', or 'custom:uuid')"
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
    # Validate all IDs first (digits key, alphanumeric key, or custom:uuid)
    invalid_ids = []
    for taxon_id in request.taxon_ids:
        if not (
            taxonomy.TAXON_ID_PATTERN.match(taxon_id)
            or taxonomy.TAXON_ID_ALNUM_PATTERN.match(taxon_id)
            or taxonomy.CUSTOM_TAXON_ID_PATTERN.match(taxon_id)
        ):
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

