"""
Feed endpoints for snippet retrieval
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional

from app.api.deps import get_db, get_current_active_user
from app.schemas.snippet import Snippet
from app.models.snippet import Snippet as SnippetModel
from app.models.user import User
from app.services.snippet_service import SnippetService

router = APIRouter()


@router.get("/", response_model=List[Snippet])
def get_feed(
    dataset_id: Optional[int] = None,
    recording_id: Optional[int] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get feed of snippets for annotation"""
    snippet_service = SnippetService(db)
    snippets = snippet_service.get_feed(
        dataset_id=dataset_id,
        recording_id=recording_id,
        skip=skip,
        limit=limit
    )
    return snippets


@router.get("/similar/{snippet_id}", response_model=List[Snippet])
def get_similar_snippets(
    snippet_id: int,
    limit: int = 10,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get similar snippets using embedding similarity"""
    snippet_service = SnippetService(db)
    snippets = snippet_service.get_similar_snippets(snippet_id, limit=limit)
    return snippets

