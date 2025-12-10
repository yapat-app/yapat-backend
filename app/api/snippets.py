"""
Snippet endpoints (updated for SnippetSet-based architecture)
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_current_active_user
from app.models.snippet import Snippet as SnippetModel
from app.models.embedding import SnippetSet
from app.models.user import User
from app.schemas.snippet import Snippet

router = APIRouter()


@router.get("/", response_model=List[Snippet])
def read_snippets(
    dataset_id: int,
    snippet_set_id: int,
    recording_id: Optional[int] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    List snippets belonging to a dataset and a snippet_set.
    Optionally filter further by recording.
    """

    # Validate snippet_set belongs to dataset
    ss = (
        db.query(SnippetSet)
        .filter(
            SnippetSet.id == snippet_set_id,
            SnippetSet.dataset_id == dataset_id,
        )
        .first()
    )
    if not ss:
        raise HTTPException(404, detail="SnippetSet not found for this dataset")

    query = (
        db.query(SnippetModel)
        .join(SnippetModel.recording)
        .filter(SnippetModel.snippet_set_id == snippet_set_id)
    )

    if recording_id is not None:
        query = query.filter(SnippetModel.recording_id == recording_id)

    return (
        query.order_by(SnippetModel.start_time)
        .offset(skip)
        .limit(limit)
        .all()
    )


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
