"""
Snippet endpoints
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_current_active_user
from app.models.snippet import Snippet as SnippetModel
from app.models.user import User
from app.schemas.snippet import Snippet

router = APIRouter()


@router.get("/", response_model=List[Snippet])
def read_snippets(
    dataset_id: int,
    embedding_job_id: int,
    recording_id: Optional[int] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    List snippets belonging to a specific dataset + embedding_job.
    Optionally filter further by recording.
    """

    # Snippets now belong to embedding jobs
    query = (
        db.query(SnippetModel)
        .join(SnippetModel.recording)
        .filter(SnippetModel.embedding_job_id == embedding_job_id)
        .filter(SnippetModel.recording.has(dataset_id=dataset_id))
    )

    if recording_id is not None:
        query = query.filter(SnippetModel.recording_id == recording_id)

    return query.offset(skip).limit(limit).all()


@router.get("/{snippet_id}", response_model=Snippet)
def read_snippet(
    snippet_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Retrieve a single snippet by ID."""
    snippet = (
        db.query(SnippetModel)
        .filter(SnippetModel.id == snippet_id)
        .first()
    )
    if not snippet:
        raise HTTPException(status_code=404, detail="Snippet not found")
    return snippet
