"""
Authentication endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import datetime, timezone

from app.api.deps import get_db, get_current_active_user
from app.schemas.user import User, UserCreate, LoginRequest, LoginResponse
from app.models.user import User as UserModel, UserRole
from app.models.invitation import InvitationLink as InvitationLinkModel
from app.models.team import (
    Team as TeamModel, TeamMembership as TeamMembershipModel, TeamRole,
    TeamInvitation as TeamInvitationModel
)
from app.models.dataset import Dataset, user_datasets
from app.core.security import get_password_hash, verify_password, create_access_token

router = APIRouter()


@router.post("/register", response_model=User, status_code=status.HTTP_201_CREATED)
def register(user_in: UserCreate, db: Session = Depends(get_db)):
    """Register a new user.

    Requires either an `invitation_token` (grants TEAM_OWNER role) or a
    `team_invitation_token` (grants USER role within that team).
    Open registration without a token is not permitted.
    """
    # Require at least one invitation token
    if not user_in.invitation_token and not user_in.team_invitation_token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Registration requires an invitation token (invitation_token or team_invitation_token)"
        )

    # Check if user already exists
    db_user = db.query(UserModel).filter(UserModel.username == user_in.username).first()
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    # Handle invitation tokens if provided
    invitation = None
    team_invitation = None
    user_role = UserRole.USER  # default; overridden by token type
    
    if user_in.invitation_token:
        # Validate admin invitation token (for dataset access)
        invitation = db.query(InvitationLinkModel).filter(
            InvitationLinkModel.token == user_in.invitation_token
        ).first()
        
        if not invitation:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid invitation token"
            )
        
        # Check if invitation is valid
        if not invitation.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invitation link is inactive"
            )
        
        if invitation.expires_at and invitation.expires_at < datetime.now(timezone.utc):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invitation link has expired"
            )
        
        if invitation.max_uses and invitation.used_count >= invitation.max_uses:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invitation link has reached maximum uses"
            )
        
        # Set user role to team_owner for invitation-based registration
        user_role = UserRole.TEAM_OWNER
    
    if user_in.team_invitation_token:
        # Validate team invitation token
        team_invitation = db.query(TeamInvitationModel).filter(
            TeamInvitationModel.token == user_in.team_invitation_token
        ).first()
        
        if not team_invitation:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid team invitation token"
            )
        
        # Check if team invitation is valid
        if not team_invitation.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Team invitation is inactive"
            )
        
        if team_invitation.expires_at and team_invitation.expires_at < datetime.now(timezone.utc):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Team invitation has expired"
            )
        
        if team_invitation.max_uses and team_invitation.used_count >= team_invitation.max_uses:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Team invitation has reached maximum uses ({team_invitation.max_uses})"
            )
    
    # Create new user
    hashed_password = get_password_hash(user_in.password)
    db_user = UserModel(
        username=user_in.username,
        hashed_password=hashed_password,
        full_name=user_in.full_name,  # full_name is saved here
        role=user_role
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    # If invitation token was used, grant access to datasets
    if invitation:
        # Separate datasets into two categories:
        # 1. Datasets with teams - add user to those teams as OWNER
        # 2. Datasets without teams - grant direct access
        team_ids = set()
        unassigned_datasets = []
        
        for dataset in invitation.datasets:
            if dataset.team_id is not None:
                team_ids.add(dataset.team_id)
            else:
                unassigned_datasets.append(dataset)
        
        # Add user as OWNER member to each team that owns the invited datasets
        for team_id in team_ids:
            if team_id is None:  # Safety check
                continue
                
            # Check if user is not already a member
            existing_membership = db.query(TeamMembershipModel).filter(
                TeamMembershipModel.team_id == team_id,
                TeamMembershipModel.user_id == db_user.id
            ).first()
            
            if not existing_membership:
                membership = TeamMembershipModel(
                    team_id=team_id,
                    user_id=db_user.id,
                    role=TeamRole.OWNER
                )
                db.add(membership)
            
            # Mark team as ready now that it has an owner
            team_obj = db.query(TeamModel).filter(TeamModel.id == team_id).first()
            if team_obj and not team_obj.is_ready:
                team_obj.is_ready = True
        
        # Grant direct access to datasets without teams
        for dataset in unassigned_datasets:
            # Add to user_datasets association table
            stmt = user_datasets.insert().values(
                user_id=db_user.id,
                dataset_id=dataset.id,
                granted_by_invitation_id=invitation.id
            )
            db.execute(stmt)
        
        # Increment invitation uses count
        invitation.used_count += 1
        
        db.commit()
    
    # Handle team invitation if provided
    if team_invitation:
        # Check if user is already a member of the team
        existing_membership = db.query(TeamMembershipModel).filter(
            TeamMembershipModel.team_id == team_invitation.team_id,
            TeamMembershipModel.user_id == db_user.id
        ).first()
        
        if not existing_membership:
            # Bootstrap the first member of an ownerless team as OWNER.
            # Role comparison is done in Python to avoid enum name/value
            # mismatch with the native PostgreSQL ENUM type.
            existing_team_memberships = db.query(TeamMembershipModel).filter(
                TeamMembershipModel.team_id == team_invitation.team_id,
            ).all()
            has_owner = any(m.role == TeamRole.OWNER for m in existing_team_memberships)
            assigned_role = TeamRole.OWNER if not has_owner else team_invitation.target_role

            membership = TeamMembershipModel(
                team_id=team_invitation.team_id,
                user_id=db_user.id,
                role=assigned_role
            )
            db.add(membership)

            # If an owner just joined, mark the team as ready and elevate the
            # user's global role so they can access owner-only UI features.
            if assigned_role == TeamRole.OWNER:
                db_user.role = UserRole.TEAM_OWNER
                team_obj = db.query(TeamModel).filter(
                    TeamModel.id == team_invitation.team_id
                ).first()
                if team_obj and not team_obj.is_ready:
                    team_obj.is_ready = True

            team_invitation.used_count += 1
            db.commit()
    
    return db_user


@router.post("/login", response_model=LoginResponse)
def login(
    login_data: LoginRequest,
    db: Session = Depends(get_db)
):
    """Login and get access token"""
    user = db.query(UserModel).filter(UserModel.username == login_data.username).first()
    if not user or not verify_password(login_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    
    access_token = create_access_token(data={"sub": str(user.id)})
    return LoginResponse(access_token=access_token, token_type="bearer")


@router.post("/logout", status_code=status.HTTP_200_OK)
def logout(current_user: UserModel = Depends(get_current_active_user)):
    """Logout the current user"""
    # Using a stateless JWT system, So logout has to be handled on the client-side by discarding the token
    # This endpoint provides a way to confirm logout and can be used for logging/analytics
    return {"message": "Successfully logged out"}


@router.get("/me", response_model=User)
def read_users_me(current_user: UserModel = Depends(get_current_active_user)):
    """Get current user information"""
    return current_user

