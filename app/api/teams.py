"""
Team endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from app.api.deps import get_db, get_current_active_user
from app.schemas.team import (
    Team, TeamCreate, TeamUpdate, TeamMembership, TeamMembershipCreate,
    TeamInvitation, TeamInvitationCreate, TeamMember
)
from app.schemas.dataset import Dataset as DatasetSchema
from app.models.team import (
    Team as TeamModel, TeamMembership as TeamMembershipModel, TeamRole,
    TeamInvitation as TeamInvitationModel
)
from app.models.user import User, User as UserModel
from app.models.dataset import Dataset as DatasetModel
from app.core.permissions import require_team_owner
from datetime import datetime, timezone, timedelta

router = APIRouter()


@router.post("/", response_model=Team, status_code=status.HTTP_201_CREATED)
def create_team(
    team_in: TeamCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Create a new team"""
    # If dataset_id is provided, validate it belongs to a team where user is owner
    if team_in.dataset_id is not None:
        # Get all teams where current user is an owner
        owned_teams = db.query(TeamModel).join(
            TeamMembershipModel
        ).filter(
            TeamMembershipModel.user_id == current_user.id,
            TeamMembershipModel.role == TeamRole.OWNER
        ).all()
        
        owned_team_ids = [t.id for t in owned_teams]
        
        # Get the dataset and verify it belongs to one of the owned teams
        dataset = db.query(DatasetModel).filter(
            DatasetModel.id == team_in.dataset_id
        ).first()
        
        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Dataset not found"
            )
        
        if dataset.team_id not in owned_team_ids:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Dataset does not belong to any team you own"
            )
    
    # Create the team
    team_data = team_in.dict(exclude={'dataset_id'})
    team = TeamModel(**team_data)
    db.add(team)
    db.commit()
    db.refresh(team)
    
    # Add creator as team owner
    membership = TeamMembershipModel(
        team_id=team.id,
        user_id=current_user.id,
        role=TeamRole.OWNER
    )
    db.add(membership)
    
    # If dataset_id was provided, assign it to the new team
    if team_in.dataset_id is not None:
        dataset.team_id = team.id
    
    db.commit()
    db.refresh(team)
    
    return team


@router.get("/available-datasets", response_model=List[DatasetSchema])
def get_available_datasets(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get list of datasets from teams where the current user is an owner.
    These datasets can be assigned when creating a new team."""
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
    ).all()
    
    return datasets


@router.get("/", response_model=List[Team])
def read_teams(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get list of teams"""
    teams = db.query(TeamModel).offset(skip).limit(limit).all()
    return teams


@router.get("/{team_id}/members", response_model=List[TeamMember])
def get_team_members(
    team_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get all members of a team with their names
    
    Returns list of team members from team_memberships table joined with users table
    to show member names and details.
    """
    # Verify team exists
    team = db.query(TeamModel).filter(TeamModel.id == team_id).first()
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")
    
    # Get all memberships for this team with user information
    memberships = db.query(TeamMembershipModel).join(
        UserModel
    ).filter(
        TeamMembershipModel.team_id == team_id
    ).all()
    
    # Build response with user information
    members = []
    for membership in memberships:
        members.append(TeamMember(
            membership_id=membership.id,
            user_id=membership.user_id,
            username=membership.user.username,
            full_name=membership.user.full_name,
            role=membership.role,
            joined_at=membership.joined_at
        ))
    
    return members


@router.post("/{team_id}/invitations", response_model=TeamInvitation, status_code=status.HTTP_201_CREATED)
def create_team_invitation(
    team_id: int,
    invitation_in: TeamInvitationCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Create a team invitation (Team owner only)"""
    # Verify team exists
    team = db.query(TeamModel).filter(TeamModel.id == team_id).first()
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")
    
    # Check if user is team owner
    require_team_owner(current_user, team_id, db)
    
    # Set default expiration (2 days from now)
    expires_at = datetime.now(timezone.utc) + timedelta(days=2)
    # Set default max_uses (unlimited by default)
    max_uses = None  # None means unlimited uses
    
    # Create invitation
    invitation = TeamInvitationModel(
        team_id=team_id,
        invited_by=current_user.id,
        target_role=invitation_in.target_role,
        expires_at=expires_at,
        max_uses=max_uses
    )
    db.add(invitation)
    db.commit()
    db.refresh(invitation)
    
    return invitation


@router.delete("/{team_id}/invitations/{invitation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_team_invitation(
    team_id: int,
    invitation_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Delete a team invitation (Team owner only)"""
    # Verify team exists
    team = db.query(TeamModel).filter(TeamModel.id == team_id).first()
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")
    
    # Check if user is team owner
    require_team_owner(current_user, team_id, db)
    
    # Get invitation
    invitation = db.query(TeamInvitationModel).filter(
        TeamInvitationModel.id == invitation_id,
        TeamInvitationModel.team_id == team_id
    ).first()
    
    if not invitation:
        raise HTTPException(status_code=404, detail="Invitation not found")
    
    db.delete(invitation)
    db.commit()
    return None



