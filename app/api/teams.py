"""
Team endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import func
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
from app.models.user import User, User as UserModel, UserRole
from app.models.dataset import Dataset as DatasetModel, user_datasets
from app.models.recording import Recording
from app.models.embedding import SnippetSet, SnippetSetStatus
from app.core.permissions import require_team_owner, require_team_member
from datetime import datetime, timezone, timedelta

router = APIRouter()


@router.post("/", response_model=Team, status_code=status.HTTP_201_CREATED)
def create_team(
    team_in: TeamCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Create a new team and optionally assign datasets.
    
    Users can assign datasets they have access to:
    - Datasets from teams where they are OWNER
    - Datasets they have direct access to (granted via invitation)
    - Admins can assign any dataset
    """
    datasets = []
    
    # If dataset_ids are provided, validate user has access to them
    if team_in.dataset_ids is not None and len(team_in.dataset_ids) > 0:
        # Check if user is admin
        is_admin = current_user.role == UserRole.ADMIN
        
        # Get all requested datasets
        datasets = db.query(DatasetModel).filter(
            DatasetModel.id.in_(team_in.dataset_ids)
        ).all()
        
        if len(datasets) != len(team_in.dataset_ids):
            found_ids = {d.id for d in datasets}
            missing_ids = set(team_in.dataset_ids) - found_ids
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Datasets not found: {list(missing_ids)}"
            )
        
        if not is_admin:
            # For non-admin users, verify they have access to all requested datasets
            # Access sources: 1) Teams where user is OWNER, 2) Direct access via invitation
            
            # Get teams where user is owner (role comparison done in Python to
            # avoid enum name/value mismatch with the native PostgreSQL ENUM type)
            owner_memberships = db.query(TeamMembershipModel).filter(
                TeamMembershipModel.user_id == current_user.id,
            ).all()
            owned_team_ids = [m.team_id for m in owner_memberships if m.role == TeamRole.OWNER]
            
            # Get datasets with direct access
            direct_access_dataset_ids = db.query(user_datasets.c.dataset_id).filter(
                user_datasets.c.user_id == current_user.id
            ).all()
            direct_access_dataset_ids = [row[0] for row in direct_access_dataset_ids]
            
            # Verify all datasets are accessible
            invalid_datasets = []
            for dataset in datasets:
                has_access = (
                    (dataset.team_id in owned_team_ids) or  # From owned team
                    (dataset.id in direct_access_dataset_ids)  # Direct access
                )
                if not has_access:
                    invalid_datasets.append(dataset.id)
            
            if invalid_datasets:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"You do not have access to these datasets: {invalid_datasets}"
                )
    
    # Create the team (is_ready stays False until a real owner joins)
    team_data = team_in.dict(exclude={'dataset_ids'})
    team = TeamModel(**team_data)
    db.add(team)
    db.commit()
    db.refresh(team)

    # Admins create teams on behalf of others — they are not added as members,
    # and the team stays is_ready=False until an owner registers via invitation.
    # Non-admin creators become the team owner immediately.
    if current_user.role != UserRole.ADMIN:
        membership = TeamMembershipModel(
            team_id=team.id,
            user_id=current_user.id,
            role=TeamRole.OWNER
        )
        db.add(membership)
        team.is_ready = True
    
    # If dataset_ids were provided, assign them to the new team
    if datasets:
        for dataset in datasets:
            dataset.team_id = team.id
            
        # Remove direct access for these datasets since they're now part of a team
        # Team membership will control access
        for dataset in datasets:
            db.execute(
                user_datasets.delete().where(
                    user_datasets.c.dataset_id == dataset.id
                )
            )
    
    db.commit()
    db.refresh(team)
    
    return team


@router.get("/available-datasets", response_model=List[DatasetSchema])
def get_available_datasets(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get list of datasets available to the user for team assignment.
    
    Returns:
    - For admins: all datasets (both with and without teams)
    - For other users: datasets from teams where they are a member (any role) +
                      datasets they have direct access to (granted via invitation)
    """
    # Check if user is admin
    is_admin = current_user.role == UserRole.ADMIN
    
    if is_admin:
        # Admins can see all datasets
        datasets = db.query(DatasetModel).all()
    else:
        # Collect datasets from two sources:
        # 1. Datasets from teams where user is any member (owner or user)
        # 2. Datasets with direct access (via invitation)
        
        # Get all teams where current user is a member (any role)
        member_teams = db.query(TeamModel).join(
            TeamMembershipModel,
            TeamModel.id == TeamMembershipModel.team_id
        ).filter(
            TeamMembershipModel.user_id == current_user.id
        ).all()
        
        member_team_ids = [t.id for t in member_teams]
        
        # Get datasets from member teams
        team_datasets = []
        if member_team_ids:
            team_datasets = db.query(DatasetModel).filter(
                DatasetModel.team_id.in_(member_team_ids)
            ).all()
        
        # Get datasets with direct access (via user_datasets table)
        direct_access_datasets = db.query(DatasetModel).join(
            user_datasets,
            DatasetModel.id == user_datasets.c.dataset_id
        ).filter(
            user_datasets.c.user_id == current_user.id
        ).all()
        
        # Combine and deduplicate
        dataset_dict = {}
        for ds in team_datasets + direct_access_datasets:
            dataset_dict[ds.id] = ds

        datasets = list(dataset_dict.values())

    # Attach recording_count to every dataset (same logic as GET /api/datasets)
    if datasets:
        dataset_ids = [ds.id for ds in datasets]
        recording_counts = (
            db.query(Recording.dataset_id, func.count(Recording.id).label("count"))
            .filter(Recording.dataset_id.in_(dataset_ids))
            .group_by(Recording.dataset_id)
            .all()
        )
        count_map = {ds_id: count for ds_id, count in recording_counts}
    else:
        count_map = {}

    # Compute feed readiness (same logic as GET /api/datasets)
    snippet_set_ids = [ds.default_snippet_set_id for ds in datasets if ds.default_snippet_set_id]
    if snippet_set_ids:
        ready_snippet_sets = (
            db.query(SnippetSet.id)
            .filter(
                SnippetSet.id.in_(snippet_set_ids),
                SnippetSet.status == SnippetSetStatus.READY,
            )
            .all()
        )
        ready_set_ids = {ss_id for (ss_id,) in ready_snippet_sets}
    else:
        ready_set_ids = set()

    result = []
    for ds in datasets:
        is_ready = (
            ds.default_snippet_set_id is not None
            and ds.default_snippet_set_id in ready_set_ids
        )
        ds_dict = {
            "id": ds.id,
            "name": ds.name,
            "description": ds.description,
            "source_uri": ds.source_uri,
            "team_id": ds.team_id,
            "default_snippet_set_id": ds.default_snippet_set_id,
            "created_at": ds.created_at,
            "updated_at": ds.updated_at,
            "recording_count": count_map.get(ds.id, 0),
            "is_ready_for_feed": is_ready,
        }
        result.append(DatasetSchema(**ds_dict))

    return result


@router.get("/", response_model=List[Team])
def read_teams(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get list of teams.

    - Admins see all teams.
    - All other users only see teams they are a member of.
    """
    if current_user.role == UserRole.ADMIN:
        teams = db.query(TeamModel).offset(skip).limit(limit).all()
    else:
        teams = (
            db.query(TeamModel)
            .join(TeamMembershipModel, TeamModel.id == TeamMembershipModel.team_id)
            .filter(TeamMembershipModel.user_id == current_user.id)
            .offset(skip)
            .limit(limit)
            .all()
        )
    return teams


@router.get("/{team_id}", response_model=Team)
def get_team(
    team_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get a single team by ID.

    Accessible by admins and team members.
    """
    team = db.query(TeamModel).filter(TeamModel.id == team_id).first()
    if not team:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Team not found")
    require_team_member(current_user, team_id, db)
    return team


@router.patch("/{team_id}", response_model=Team)
def update_team(
    team_id: int,
    team_in: TeamUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Update team name/description (team owner or admin only)."""
    team = db.query(TeamModel).filter(TeamModel.id == team_id).first()
    if not team:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Team not found")
    require_team_owner(current_user, team_id, db)
    update_data = team_in.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(team, field, value)
    db.commit()
    db.refresh(team)
    return team


@router.delete("/{team_id}/members/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_team_member(
    team_id: int,
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Remove a member from a team (team owner or admin only)."""
    team = db.query(TeamModel).filter(TeamModel.id == team_id).first()
    if not team:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Team not found")
    require_team_owner(current_user, team_id, db)
    membership = db.query(TeamMembershipModel).filter(
        TeamMembershipModel.team_id == team_id,
        TeamMembershipModel.user_id == user_id
    ).first()
    if not membership:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Member not found")
    # If removing the owner, mark team as not ready
    if membership.role == TeamRole.OWNER:
        team.is_ready = False
    db.delete(membership)
    db.commit()
    return None


@router.get("/{team_id}/datasets", response_model=List[DatasetSchema])
def get_team_datasets(
    team_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get all datasets belonging to a specific team.
    
    Only team members (or admins) can access this endpoint.
    """
    # Verify team exists
    team = db.query(TeamModel).filter(TeamModel.id == team_id).first()
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")
    
    # Check if user is a team member (or admin)
    require_team_member(current_user, team_id, db)
    
    # Get all datasets for this team
    datasets = db.query(DatasetModel).filter(
        DatasetModel.team_id == team_id
    ).all()
    
    return datasets


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


@router.delete("/{team_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_team(
    team_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Delete a team (admin or team owner only).

    Datasets belonging to the team are unassigned (team_id set to NULL) rather
    than deleted, so recordings and annotations are preserved.
    Memberships and invitations are removed automatically via cascade.
    """
    team = db.query(TeamModel).filter(TeamModel.id == team_id).first()
    if not team:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Team not found")

    require_team_owner(current_user, team_id, db)

    # Unassign datasets via ORM so the relationship is properly tracked.
    # The datasets relationship has no delete cascade, so this simply nulls the FK.
    for dataset in list(team.datasets):
        dataset.team_id = None

    db.flush()
    db.delete(team)
    db.commit()
    return None


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



