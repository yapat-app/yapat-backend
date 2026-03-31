"""
Role-based access control
"""

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.models.user import User, UserRole
from app.models.team import Team, TeamMembership, TeamRole


def check_admin(user: User) -> bool:
    """Check if user is a platform admin"""
    return user.role == UserRole.ADMIN


def check_team_owner(user: User) -> bool:
    """Check if user is a team owner (global role)"""
    return user.role == UserRole.TEAM_OWNER


def check_team_member(user: User, team_id: int, db: Session) -> bool:
    """Check if user is a member of a team"""
    membership = db.query(TeamMembership).filter(
        TeamMembership.team_id == team_id,
        TeamMembership.user_id == user.id
    ).first()
    return membership is not None


def check_team_owner_membership(user: User, team_id: int, db: Session) -> bool:
    """Check if user is an owner of a specific team.

    NOTE: intentionally avoids filtering on `role` in SQL because the
    PostgreSQL ENUM type stores the enum *name* ('OWNER') while the Python
    enum comparison in a WHERE clause would use the *value* ('owner'),
    causing a mismatch. Instead we fetch the row and compare in Python.
    """
    membership = db.query(TeamMembership).filter(
        TeamMembership.team_id == team_id,
        TeamMembership.user_id == user.id,
    ).first()
    return membership is not None and membership.role == TeamRole.OWNER


def require_admin(user: User):
    """Require user to be a platform admin"""
    if not check_admin(user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )


def require_team_member(user: User, team_id: int, db: Session):
    """Require user to be a member of the team"""
    if not (check_admin(user) or check_team_member(user, team_id, db)):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not a member of this team"
        )


def require_team_owner(user: User, team_id: int, db: Session):
    """Require user to be an owner of the team"""
    if not (check_admin(user) or check_team_owner_membership(user, team_id, db)):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not an owner of this team"
        )


def require_conversation_access(user: User, conversation, db: Session):
    """
    Require user to have access to a conversation.

    - If the conversation has a team: user must be a team member (or admin).
    - If the conversation has no team (personal): user must be the creator (or admin).
    """
    if conversation.team_id is not None:
        require_team_member(user, conversation.team_id, db)
    elif not (check_admin(user) or conversation.user_id == user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this conversation"
        )

