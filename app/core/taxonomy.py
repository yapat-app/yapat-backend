"""
GBIF integration for taxonomy
"""

import requests
from typing import Optional, Dict, Any, List


GBIF_API_BASE = "https://api.gbif.org/v1"


def search_species(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Search for species using GBIF API"""
    try:
        response = requests.get(
            f"{GBIF_API_BASE}/species/search",
            params={"q": query, "limit": limit}
        )
        response.raise_for_status()
        data = response.json()
        return data.get("results", [])
    except requests.RequestException:
        return []


def get_species_details(species_key: int) -> Optional[Dict[str, Any]]:
    """Get detailed information about a species from GBIF"""
    try:
        response = requests.get(f"{GBIF_API_BASE}/species/{species_key}")
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def validate_species_name(species_name: str) -> bool:
    """Validate if a species name exists in GBIF"""
    results = search_species(species_name, limit=1)
    if results:
        # Check if any result matches the species name
        for result in results:
            canonical_name = result.get("canonicalName", "").lower()
            if canonical_name == species_name.lower():
                return True
    return False

