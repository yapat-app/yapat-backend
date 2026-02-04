"""
Taxonomy API response schemas
"""

from pydantic import BaseModel
from typing import List, Optional


class CommonName(BaseModel):
    name: str
    language: str


class TaxonSuggestion(BaseModel):
    taxon_id: str
    canonical_name: Optional[str]
    scientific_name: Optional[str]
    rank: Optional[str]
    kingdom: Optional[str]
    status: Optional[str]


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
