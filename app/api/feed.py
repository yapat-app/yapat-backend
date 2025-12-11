"""
Feed endpoints for snippet retrieval

Supports feed generation methods:
- default: Prioritizes unannotated snippets
- random: Random sampling
- similarity: Similarity search
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_current_active_user
from app.models.user import User
from app.schemas.snippet import Snippet
from app.services.snippet_service import SnippetService

router = APIRouter()


@router.get("/", response_model=List[Snippet])
def get_feed(
        method: Optional[str] = Query(
            default=None,
            description="Feed generation method: 'random' or 'similarity'. Default prioritizes unannotated snippets."
        ),
        dataset_id: Optional[int] = Query(default=None, description="Dataset ID to filter snippets"),
        recording_id: Optional[int] = Query(default=None, description="Recording ID to filter snippets"),
        skip: int = Query(default=0, ge=0, description="Number of snippets to skip (pagination)"),
        limit: int = Query(default=100, ge=1, le=1000, description="Maximum number of snippets to return"),
        # Method-specific parameters
        status: Optional[str] = Query(default=None, description="Filter by snippet status (for 'random' method)"),
        embedding_model_id: Optional[int] = Query(default=None, description="Embedding model ID (for 'similarity' method)"),
        query_snippet_id: Optional[int] = Query(default=None, description="Snippet ID to use as query (for 'similarity' method)"),
        crop_start_sec: Optional[float] = Query(default=None, description="Crop start time in seconds (for 'similarity' method)"),
        crop_end_sec: Optional[float] = Query(default=None, description="Crop end time in seconds (for 'similarity' method)"),
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_active_user)
):
    """
    Get feed of snippets for annotation using various sampling methods.
    
    If no method is specified, defaults to prioritizing unannotated snippets.
    Supported methods: 'random', 'similarity'
    """
    snippet_service = SnippetService(db)
    
    # Route to appropriate method based on 'method' parameter
    if method is None or method == "":
        # Default: prioritize unannotated snippets
        snippets = snippet_service.get_feed(
            dataset_id=dataset_id,
            recording_id=recording_id,
            skip=skip,
            limit=limit
        )
    elif method == "random":
        snippets = snippet_service.get_feed_random(
            dataset_id=dataset_id,
            recording_id=recording_id,
            status=status,
            skip=skip,
            limit=limit
        )
    elif method == "similarity":
        if dataset_id is None:
            raise HTTPException(status_code=400, detail="dataset_id is required for 'similarity' method")
        if query_snippet_id is None:
            raise HTTPException(status_code=400, detail="query_snippet_id is required for 'similarity' method")
        snippets = snippet_service.get_feed_similarity(
            dataset_id=dataset_id,
            query_snippet_id=query_snippet_id,
            embedding_model_id=embedding_model_id,
            crop_start_sec=crop_start_sec,
            crop_end_sec=crop_end_sec,
            skip=skip,
            limit=limit
        )
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown feed method: '{method}'. Supported methods: 'random', 'similarity'"
        )
    
    return snippets


@router.get("/similar/{snippet_id}", response_model=List[Snippet])
def get_similar_snippets(
        snippet_id: int,
        embedding_model_id: int = Query(default=1, description="Embedding model ID"),
        limit: int = Query(default=10, ge=1, le=100, description="Maximum number of similar snippets to return"),
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_active_user)
):
    """
    Get similar snippets using embedding similarity.
    
    DEPRECATED: Use GET /api/feed/?method=similarity&query_snippet_id={snippet_id} instead.
    """
    snippet_service = SnippetService(db)
    snippets = snippet_service.get_similar_snippets(
        snippet_id=snippet_id,
        embedding_model_id=embedding_model_id,
        limit=limit
    )
    return snippets
