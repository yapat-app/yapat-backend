"""
Taxonomy Search Service

Helper functions for searching across GBIF and custom taxonomies.
"""

from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session

from app.models.user import User
from app.models.team import TeamMembership
from app.models.custom_taxonomy import TaxonomyStatus
from app.core import taxonomy
from app.services import custom_taxonomy_service
from app.core.permissions import check_team_member, check_admin


def search_custom_taxonomies(
    query: str,
    taxonomy_ids: List[str],
    limit: int,
    current_user: User,
    db: Session
) -> List[Dict[str, Any]]:
    """
    Search within custom taxonomies for matching taxa.
    
    Args:
        query: Search query
        taxonomy_ids: List of custom taxonomy IDs to search
        limit: Maximum number of results
        current_user: Current user (for access control)
        db: Database session
        
    Returns:
        List of matching taxa from custom taxonomies
    """
    from app.services.custom_taxonomy_service import get_taxonomy_by_id
    
    results = []
    query_lower = query.lower()
    
    for taxonomy_id in taxonomy_ids:
        taxonomy_obj = get_taxonomy_by_id(taxonomy_id, db)
        if not taxonomy_obj:
            continue
        
        # Check access
        if not taxonomy_obj.is_global:
            if not check_team_member(current_user, taxonomy_obj.team_id, db) and not check_admin(current_user):
                continue
        
        # Check if active
        if taxonomy_obj.status != TaxonomyStatus.ACTIVE:
            continue
        
        # Search within taxonomy_data nodes
        taxonomy_data = taxonomy_obj.taxonomy_data
        nodes = taxonomy_data.get("nodes", [])
        
        for node in nodes:
            # Check if node matches query
            name = node.get("name", "").lower()
            scientific_name = (node.get("scientific_name") or "").lower()
            
            if query_lower in name or query_lower in scientific_name:
                # Get taxon_id - use the taxon_id from node if available
                # Custom taxonomy items typically have GBIF taxon_ids (e.g., "gbif:123")
                # stored in the node, which is what should be used for annotation
                taxon_id = node.get("taxon_id")
                
                # If no taxon_id in node, this is unusual but we'll use the taxonomy_id
                # as a fallback (though this shouldn't happen for properly created taxonomies)
                if not taxon_id:
                    taxon_id = taxonomy_id
                
                results.append({
                    "taxon_id": taxon_id,
                    "canonical_name": node.get("name"),
                    "scientific_name": node.get("scientific_name"),
                    "rank": node.get("rank") or node.get("metadata", {}).get("rank"),
                    "kingdom": node.get("metadata", {}).get("kingdom"),
                    "status": "ACCEPTED",  # Custom taxonomy items are always accepted
                    "taxonomy_id": taxonomy_id,
                    "taxonomy_name": taxonomy_obj.name
                })
                
                if len(results) >= limit:
                    break
        
        if len(results) >= limit:
            break
    
    return results


def get_user_custom_taxonomy_ids(
    current_user: User,
    db: Session
) -> List[str]:
    """
    Get all custom taxonomy IDs available to the user.
    
    Args:
        current_user: Current user
        db: Database session
        
    Returns:
        List of custom taxonomy IDs (e.g., ['custom:abc123', 'custom:def456'])
    """
    memberships = db.query(TeamMembership).filter(
        TeamMembership.user_id == current_user.id
    ).all()
    
    all_custom_taxonomy_ids = []
    for membership in memberships:
        team_taxonomies = custom_taxonomy_service.get_available_taxonomies(
            team_id=membership.team_id,
            user_id=current_user.id,
            db=db,
            status=TaxonomyStatus.ACTIVE.value
        )
        all_custom_taxonomy_ids.extend([t.taxonomy_id for t in team_taxonomies])
    
    # Remove duplicates
    return list(set(all_custom_taxonomy_ids))


def sort_results_by_relevance(results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """
    Sort search results by relevance to the query.
    
    Args:
        results: List of result dictionaries
        query: Search query
        
    Returns:
        Sorted list of results
    """
    query_lower = query.lower()
    
    def sort_key(r):
        name = (r.get("canonical_name") or r.get("scientific_name") or "").lower()
        if name.startswith(query_lower):
            return (0, name)
        elif query_lower in name:
            return (1, name)
        else:
            return (2, name)
    
    results.sort(key=sort_key)
    return results
