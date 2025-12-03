"""
Dataset endpoints
"""

from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_current_active_user
from app.models.user import User, UserRole
from app.schemas.dataset import Dataset, DatasetCreate
from app.services.dataset_service import DatasetService

router = APIRouter()


@router.post("/", response_model=Dataset, status_code=status.HTTP_201_CREATED)
def create_dataset(
        dataset_in: DatasetCreate,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_active_user),
):
    svc = DatasetService(db)

    is_admin = current_user.role == UserRole.ADMIN
    if not is_admin and dataset_in.team_id is None:
        raise HTTPException(
            status_code=400,
            detail="team_id is required for non-admin users",
        )

    try:
        dataset = svc.create_dataset(dataset_in, current_user)
    except ValueError as e:
        if str(e) == "duplicate_dataset":
            raise HTTPException(status_code=409, detail="Dataset already exists")
        if str(e) == "team_not_found":
            raise HTTPException(status_code=404, detail="Team not found")
        raise

    svc.scan_recordings(dataset)
    # TODO use Celery task
    return dataset


@router.get("/", response_model=List[Dataset])
def read_datasets(
        skip: int = 0,
        limit: int = 100,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_active_user),
):
    """List datasets visible to the current user."""
    svc = DatasetService(db)
    return svc.list_datasets(skip=skip, limit=limit)


@router.get("/{dataset_id}", response_model=Dataset)
def read_dataset(
        dataset_id: int,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_active_user),
):
    """Retrieve a single dataset by ID."""
    svc = DatasetService(db)
    dataset = svc.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset

