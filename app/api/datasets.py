"""
Dataset endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from app.api.deps import get_db, get_current_active_user, get_current_admin_user
from app.schemas.dataset import Dataset, DatasetCreate, DatasetUpdate
from app.models.dataset import Dataset as DatasetModel
from app.models.user import User, UserRole
from app.models.team import Team as TeamModel, TeamMembership as TeamMembershipModel, TeamRole
from app.tasks import scan_and_process_dataset

router = APIRouter()


@router.post("/", response_model=Dataset, status_code=status.HTTP_201_CREATED)
def create_dataset(
    dataset_in: DatasetCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Create a new dataset. Admins can create datasets without team_id."""
    # Check if user is admin
    is_admin = current_user.role == UserRole.ADMIN
    
    # Non-admins must provide team_id
    if not is_admin and dataset_in.team_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="team_id is required for non-admin users"
        )
    
    # If admin and no team_id provided, allow null team_id
    # If team_id is provided, validate it exists
    if dataset_in.team_id is not None:
        from app.models.team import Team
        team = db.query(Team).filter(Team.id == dataset_in.team_id).first()
        if not team:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Team not found"
            )
    
    dataset = DatasetModel(**dataset_in.dict())
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    
    # Automatically trigger dataset scanning after creation
    if dataset.source_uri:
        scan_and_process_dataset.delay(dataset.id)
    
    return dataset


@router.get("/", response_model=List[Dataset])
def read_datasets(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get list of datasets. 
    For non-admin users: returns datasets from teams where the user is an owner.
    For admin users: returns all datasets."""
    # Check if user is admin
    is_admin = current_user.role == UserRole.ADMIN
    
    if is_admin:
        # Admins can see all datasets
        datasets = db.query(DatasetModel).offset(skip).limit(limit).all()
    else:
        # Get all teams where current user is an owner
        owned_teams = db.query(TeamModel).join(
            TeamMembershipModel
        ).filter(
            TeamMembershipModel.user_id == current_user.id,
            TeamMembershipModel.role == TeamRole.OWNER
        ).all()
        
        owned_team_ids = [t.id for t in owned_teams]
        
        if not owned_team_ids:
            return []
        
        # Get all datasets from owned teams
        datasets = db.query(DatasetModel).filter(
            DatasetModel.team_id.in_(owned_team_ids)
        ).offset(skip).limit(limit).all()
    
    return datasets


@router.get("/{dataset_id}", response_model=Dataset)
def read_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get a specific dataset"""
    dataset = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset

