"""
Snippet endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from app.api.deps import get_db, get_current_active_user
from app.schemas.snippet import Snippet, SnippetCreate, SnippetUpdate
from app.models.snippet import Snippet as SnippetModel
from app.models.user import User

router = APIRouter()


@router.post("/", response_model=Snippet, status_code=status.HTTP_201_CREATED)
def create_snippet(
    snippet_in: SnippetCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Create a new snippet"""
    snippet = SnippetModel(**snippet_in.dict())
    db.add(snippet)
    db.commit()
    db.refresh(snippet)
    return snippet


@router.get("/", response_model=List[Snippet])
def read_snippets(
    recording_id: int = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get list of snippets"""
    query = db.query(SnippetModel)
    if recording_id:
        query = query.filter(SnippetModel.recording_id == recording_id)
    snippets = query.offset(skip).limit(limit).all()
    return snippets


@router.get("/{snippet_id}", response_model=Snippet)
def read_snippet(
    snippet_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get a specific snippet"""
    snippet = db.query(SnippetModel).filter(SnippetModel.id == snippet_id).first()
    if not snippet:
        raise HTTPException(status_code=404, detail="Snippet not found")
    return snippet

