"""
GBIF integration for taxonomy

GBIF API Documentation: https://techdocs.gbif.org/en/openapi/v1/species
API Base URL: https://api.gbif.org/v1
"""

import requests
from typing import Optional, Dict, Any, List
import re


GBIF_API_BASE = "https://api.gbif.org/v1"
GBIF_API_V2_BASE = "https://api.gbif.org/v2"
TAXON_ID_PATTERN = re.compile(r'^([a-z]+):(\d+)$')
CUSTOM_TAXON_ID_PATTERN = re.compile(r'^(custom:[a-f0-9-]+)$')


def parse_taxon_id(taxon_id: str) -> Optional[Dict[str, Any]]:
    """Parse a namespaced taxon ID (e.g., 'gbif:2420576' or 'custom:uuid') into namespace and key"""
    # Try GBIF pattern first
    match = TAXON_ID_PATTERN.match(taxon_id)
    if match:
        return {
            "namespace": match.group(1),
            "key": match.group(2)
        }
    
    # Try custom taxonomy pattern
    match = CUSTOM_TAXON_ID_PATTERN.match(taxon_id)
    if match:
        return {
            "namespace": "custom",
            "key": taxon_id  # Full custom:uuid
        }
    
    return None


def suggest_species(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Fast autocomplete suggestions using GBIF suggest endpoint
    
    Optimized for real-time autocomplete as users type.
    Returns a simplified list of suggestions.
    """
    try:
        response = requests.get(
            f"{GBIF_API_BASE}/species/suggest",
            params={"q": query, "limit": limit},
            timeout=5  # 5 second timeout to prevent hanging
        )
        response.raise_for_status()
        suggestions = response.json()
        
        # Format suggestions with namespaced IDs
        formatted = []
        for item in suggestions:
            formatted.append({
                "taxon_id": f"gbif:{item['key']}",
                "canonical_name": item.get("canonicalName"),
                "scientific_name": item.get("scientificName"),
                "rank": item.get("rank"),
                "kingdom": item.get("kingdom"),
                "status": item.get("status")
            })
        return formatted
    except requests.RequestException:
        return []


def search_species(
    query: str, 
    limit: int = 10,
    rank: Optional[str] = None,
    status: str = "ACCEPTED"
) -> List[Dict[str, Any]]:
    """Full text search over name usages with filtering
    
    Args:
        query: Search query (species name)
        limit: Maximum number of results
        rank: Filter by taxonomic rank (SPECIES, GENUS, FAMILY, etc.)
        status: Filter by taxonomic status (ACCEPTED, SYNONYM, DOUBTFUL)
    """
    params = {"q": query, "limit": limit, "status": status}
    if rank:
        params["rank"] = rank
    
    try:
        response = requests.get(
            f"{GBIF_API_BASE}/species/search",
            params=params,
            timeout=10  # 10 second timeout for search queries
        )
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])
        
        # Format results with namespaced IDs and additional info
        formatted = []
        for item in results:
            formatted.append({
                "taxon_id": f"gbif:{item['key']}",
                "canonical_name": item.get("canonicalName"),
                "scientific_name": item.get("scientificName"),
                "rank": item.get("rank"),
                "kingdom": item.get("kingdom"),
                "status": item.get("status"),
                "common_names": [
                    {"name": v.get("vernacularName"), "language": v.get("language")}
                    for v in item.get("vernacularNames", [])[:3]
                ],
                "habitats": item.get("habitats", []),
                "taxonomic_status": item.get("taxonomicStatus")
            })
        return formatted
    except requests.RequestException:
        return []


def resolve_taxon_id(taxon_id: str, db_session=None) -> Optional[Dict[str, Any]]:
    """Resolve a namespaced taxon ID to detailed information
    
    Supports both GBIF and custom taxonomies.
    
    Args:
        taxon_id: Namespaced ID (e.g., 'gbif:2420576' or 'custom:uuid')
        db_session: Optional database session (required for custom taxonomies)
    
    Returns:
        Dict with resolved name and details, or None if invalid
    """
    parsed = parse_taxon_id(taxon_id)
    if not parsed:
        return None
    
    namespace = parsed["namespace"]
    key = parsed["key"]
    
    # Handle custom taxonomies
    if namespace == "custom":
        if not db_session:
            return None
        
        # Import here to avoid circular dependency
        from app.services.custom_taxonomy_service import get_taxonomy_by_id
        
        taxonomy = get_taxonomy_by_id(taxon_id, db_session)
        if not taxonomy:
            return None
        
        # For custom taxonomy root, return basic info
        return {
            "taxon_id": taxon_id,
            "canonical_name": taxonomy.name,
            "scientific_name": taxonomy.name,
            "rank": "custom_taxonomy",
            "status": taxonomy.status,
            "taxonomy_type": "custom",
            "taxonomy_id": taxonomy.taxonomy_id,
            "description": taxonomy.description
        }
    
    # Handle GBIF taxonomies
    if namespace == "gbif":
        try:
            response = requests.get(
                f"{GBIF_API_BASE}/species/{key}",
                timeout=10  # 10 second timeout for resolution queries
            )
            response.raise_for_status()
            data = response.json()
            
            return {
                "taxon_id": taxon_id,
                "canonical_name": data.get("canonicalName"),
                "scientific_name": data.get("scientificName"),
                "rank": data.get("rank"),
                "status": data.get("status"),
                "kingdom": data.get("kingdom"),
                "phylum": data.get("phylum"),
                "class": data.get("class"),
                "order": data.get("order"),
                "family": data.get("family"),
                "genus": data.get("genus"),
                "common_names": [
                    {"name": v.get("vernacularName"), "language": v.get("language")}
                    for v in data.get("vernacularNames", [])
                ],
                "habitats": data.get("habitats", []),
                "taxonomic_status": data.get("taxonomicStatus"),
                "taxonomy_type": "gbif"
            }
        except requests.RequestException:
            return None
    
    return None


def batch_resolve_taxon_ids(taxon_ids: List[str]) -> Dict[str, Optional[Dict[str, Any]]]:
    """Resolve multiple taxon IDs at once
    
    Args:
        taxon_ids: List of namespaced IDs (e.g., ['gbif:2420576', 'gbif:2498293'])
    
    Returns:
        Dict mapping taxon_id to resolved data (or None if failed)
    """
    results = {}
    for taxon_id in taxon_ids:
        results[taxon_id] = resolve_taxon_id(taxon_id)
    return results


def match_species_name(name: str) -> Optional[Dict[str, Any]]:
    """Fuzzy match a species name with confidence scoring
    
    Uses GBIF's v2 match service for best-match scoring.
    Useful for validation and synonym resolution.
    """
    try:
        response = requests.get(
            f"{GBIF_API_V2_BASE}/species/match",
            params={"name": name, "verbose": "true"},
            timeout=10  # 10 second timeout for match queries
        )
        response.raise_for_status()
        data = response.json()
        
        diagnostics = data.get("diagnostics", {})
        match_type = diagnostics.get("matchType")
        usage = data.get("usage")
        
        # Check if we have a valid match
        if match_type in ["EXACT", "FUZZY", "HIGHERRANK"] and usage:
            # Get full details via resolve to ensure all fields are present
            taxon_id = f"gbif:{usage.get('key')}"
            resolved = resolve_taxon_id(taxon_id)
            if resolved:
                return {
                    "taxon_id": taxon_id,
                    "canonical_name": resolved.get("canonical_name"),
                    "scientific_name": resolved.get("scientific_name"),
                    "rank": resolved.get("rank"),
                    "status": resolved.get("status"),
                    "match_type": match_type,
                    "confidence": diagnostics.get("confidence"),
                    "taxonomic_status": resolved.get("taxonomic_status"),
                    "kingdom": resolved.get("kingdom"),
                    "phylum": resolved.get("phylum"),
                    "class": resolved.get("class"),
                    "order": resolved.get("order"),
                    "family": resolved.get("family"),
                    "genus": resolved.get("genus"),
                    "common_names": resolved.get("common_names", []),
                    "habitats": resolved.get("habitats", [])
                }
        return None
    except requests.RequestException:
        return None


def validate_taxon_id(taxon_id: str, db_session=None) -> bool:
    """Validate if a taxon ID exists and is resolvable
    
    Args:
        taxon_id: Namespaced ID (e.g., 'gbif:2420576' or 'custom:uuid')
        db_session: Optional database session (required for custom taxonomies)
    
    Returns:
        True if valid and resolvable, False otherwise
    """
    result = resolve_taxon_id(taxon_id, db_session)
    return result is not None



