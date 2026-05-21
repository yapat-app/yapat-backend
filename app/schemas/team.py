"""
Team schemas
"""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List
from app.models.team import TeamRole


class TeamBase(BaseModel):
    name: str
    description: Optional[str] = None


class TeamDataset(BaseModel):
    id: int
    name: str


class TeamCreate(TeamBase):
    dataset_ids: Optional[List[int]] = Field(
        None,
        description=(
            "Optional list of dataset IDs to assign to the team. "
            "Only datasets from teams where you are an owner can be selected. "
            "Use GET /api/teams/available-datasets to get the list of available datasets."
        ),
        examples=[[1, 2, 3]],
        json_schema_extra={
            "x-link": "/api/teams/available-datasets",
            "x-link-description": "View available datasets"
        }
    )


class TeamUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


class Team(TeamBase):
    id: int
    is_ready: bool = False
    dataset_ids: List[int] = Field(default_factory=list)
    datasets: List[TeamDatasetRef] = Field(default_factory=list)
    created_at: datetime
    updated_at: Optional[datetime] = None
    datasets: Optional[List[TeamDataset]] = None

    class Config:
        from_attributes = True


class TeamMembershipBase(BaseModel):
    role: TeamRole = TeamRole.USER


class TeamMembershipCreate(TeamMembershipBase):
    user_id: int


class TeamMembership(TeamMembershipBase):
    id: int
    team_id: int
    user_id: int
    joined_at: datetime

    class Config:
        from_attributes = True


class TeamMember(BaseModel):
    """Team member with user information"""
    membership_id: int
    user_id: int
    username: str
    full_name: Optional[str] = None
    role: TeamRole
    joined_at: datetime


class TeamInvitationCreate(BaseModel):
    target_role: TeamRole = TeamRole.USER


class TeamInvitation(BaseModel):
    id: int
    team_id: int
    invited_by: Optional[int] = None
    token: str
    target_role: TeamRole
    expires_at: Optional[datetime] = None
    is_active: bool
    max_uses: Optional[int] = None  # None means unlimited
    used_count: int = 0
    created_at: datetime

    class Config:
        from_attributes = True


class TeamInvitationPublic(BaseModel):
    """Public schema for team invitation (for registration/login)"""
    token: str
    is_valid: bool
    team_name: Optional[str] = None
    team_id: Optional[int] = None
    target_role: Optional[TeamRole] = None
    expires_at: Optional[datetime] = None
    message: Optional[str] = None  # Error or info message

