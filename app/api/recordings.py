"""
Recording endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from app.api.deps import get_db, get_current_active_user
from app.schemas.recording import Recording, RecordingCreate, RecordingUpdate
from app.models.recording import Recording as RecordingModel
from app.models.user import User

router = APIRouter()


@router.post("/", response_model=Recording, status_code=status.HTTP_201_CREATED)
def create_recording(
    recording_in: RecordingCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Create a new recording"""
    recording = RecordingModel(**recording_in.dict())
    db.add(recording)
    db.commit()
    db.refresh(recording)
    return recording


@router.get("/", response_model=List[Recording])
def read_recordings(
    dataset_id: int = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get list of recordings"""
    query = db.query(RecordingModel)
    if dataset_id:
        query = query.filter(RecordingModel.dataset_id == dataset_id)
    recordings = query.offset(skip).limit(limit).all()
    return recordings


@router.get("/{recording_id}", response_model=Recording)
def read_recording(
    recording_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get a specific recording"""
    recording = db.query(RecordingModel).filter(RecordingModel.id == recording_id).first()
    if not recording:
        raise HTTPException(status_code=404, detail="Recording not found")
    return recording

