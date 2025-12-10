"""
Recording endpoints
"""

from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_current_active_user
from app.schemas.recording import Recording, RecordingCreate
from app.models.recording import Recording as RecordingModel
from app.models.user import User

router = APIRouter()


@router.post("/", response_model=Recording, status_code=status.HTTP_201_CREATED)
def create_recording(
    recording_in: RecordingCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Create a new recording.

    NOTE: audio_sha256 is currently not computed here.
    In production ingestion, recording checksums should be computed
    in a service (e.g., DatasetService.scan_recordings) before committing.
    """
    recording = RecordingModel(**recording_in.dict())
    db.add(recording)
    db.commit()
    db.refresh(recording)

    # TODO: compute and persist recording.audio_sha256

    return recording


@router.get("/", response_model=List[Recording])
def read_recordings(
    dataset_id: int,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Get list of recordings belonging to a dataset."""
    query = (
        db.query(RecordingModel)
        .filter(RecordingModel.dataset_id == dataset_id)
        .offset(skip)
        .limit(limit)
    )
    return query.all()


@router.get("/{recording_id}", response_model=Recording)
def read_recording(
    recording_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Get a specific recording."""
    recording = (
        db.query(RecordingModel)
        .filter(RecordingModel.id == recording_id)
        .first()
    )
    if not recording:
        raise HTTPException(status_code=404, detail="Recording not found")
    return recording
