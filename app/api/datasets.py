"""
Dataset endpoints
"""

from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_current_active_user
from app.models.user import User, UserRole
from app.schemas.dataset import Dataset, DatasetCreate, DatasetCreationResponse
from app.services.dataset_service import DatasetService
from app.tasks.processing_tasks import process_dataset

router = APIRouter()


@router.post("/", response_model=DatasetCreationResponse, status_code=status.HTTP_201_CREATED)
def create_dataset(
        dataset_in: DatasetCreate,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_active_user),
):
    svc = DatasetService(db)

    if current_user.role != UserRole.ADMIN and dataset_in.team_id is None:
        raise HTTPException(status_code=400, detail="team_id is required for non-admin users")

    try:
        dataset = svc.create_dataset(dataset_in, current_user)
    except ValueError as e:
        if str(e) == "duplicate_dataset":
            raise HTTPException(status_code=409, detail="Dataset already exists")
        if str(e) == "team_not_found":
            raise HTTPException(status_code=404, detail="Team not found")
        if str(e) == "invalid_source_uri":
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid dataset path: {dataset_in.source_uri} does not exist or is not a directory"
            )
        raise

    # Dispatch background task for dataset processing (scanning + snippet generation)
    # Returns task ID for client tracking; None if task dispatch fails (backward compatible)
    try:
        task = process_dataset.delay(dataset.id)
        task_id = task.id
    except Exception:
        task_id = None

    return DatasetCreationResponse(
        dataset=dataset,
        process_task_id=task_id,
        snippet_config_id=None,
        embedding_job_id=None,
    )


@router.get("/", response_model=List[Dataset])
def read_datasets(
        skip: int = 0,
        limit: int = 100,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_active_user),
):
    svc = DatasetService(db)
    return svc.list_datasets(current_user=current_user, skip=skip, limit=limit)


@router.get("/{dataset_id}", response_model=Dataset)
def read_dataset(
        dataset_id: int,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_active_user),
):
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

    if not (is_admin or is_owner):
        raise HTTPException(status_code=403, detail="Not authorized to delete dataset")

    svc.delete_dataset(dataset)
    return None
