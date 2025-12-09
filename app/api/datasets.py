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

    """Create a new dataset. Admins can create datasets without team_id."""
    # Check if user is admin
    is_admin = current_user.role == UserRole.ADMIN

    # Non-admins must provide team_id
    if not is_admin and dataset_in.team_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="team_id is required for non-admin users"
        )

    try:
        dataset = svc.create_dataset(dataset_in, current_user)
    except ValueError as e:
        if str(e) == "duplicate_dataset":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Dataset already exists",
            )
        if str(e) == "team_not_found":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Team not found",
            )
        raise

    # Scan audio recordings in source_uri and create Recording objects
    svc.scan_recordings(dataset)

    # TODO: Replace with Celery workflow:
    #   1. Ensure default SnippetConfig exists (e.g. BirdNET-like).
    #   2. generate_snippets_for_dataset.delay(dataset.id, snippet_config_id)
    #
    # For now, segmentation may remain synchronous or incomplete.

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
    return svc.list_datasets(current_user=current_user, skip=skip, limit=limit)


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


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_dataset(
        dataset_id: int,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_active_user),
):
    svc = DatasetService(db)

    dataset = svc.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    is_admin = current_user.role == UserRole.ADMIN
    is_owner = True
    # TODO: Restrict delete privileges appropriately.

    # NOTE (future): Dataset deletion will delete:
    #   - SnippetConfigs
    #   - Snippets
    #   - Annotations (cascade)
    # The SnippetConfig safety logic from Issue #<id> must be respected.

    if not (is_admin or is_owner):
        raise HTTPException(status_code=403, detail="Not authorized to delete dataset")

    svc.delete_dataset(dataset)
    return None
