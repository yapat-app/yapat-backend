"""
User schemas
"""

from pydantic import BaseModel, field_validator
from datetime import datetime
from typing import Optional, List
from app.models.user import UserRole


class UserBase(BaseModel):
    username: str
    full_name: Optional[str] = None

    @field_validator("role", mode="before")
    @classmethod
    def normalize_role(cls, v):
        """Accept DB enum names (TEAM_OWNER) and API values (team_owner)."""
        if v is None:
            return v
        if isinstance(v, UserRole):
            return v
        key = str(v).lower()
        aliases = {
            "admin": UserRole.ADMIN,
            "team_owner": UserRole.TEAM_OWNER,
            "user": UserRole.USER,
        }
        if key in aliases:
            return aliases[key]
        return v


class UserCreate(UserBase):
    password: str
    invitation_token: Optional[str] = None  # For admin-created dataset invitations → becomes TEAM_OWNER
    team_invitation_token: Optional[str] = None  # For team invitations → becomes USER member of that team
    
    @field_validator('password')
    @classmethod
    def validate_password_length(cls, v: str) -> str:
        """Validate password doesn't exceed bcrypt's 72 byte limit"""
        if len(v.encode('utf-8')) > 72:
            raise ValueError('Password cannot be longer than 72 bytes')
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        return v


class UserUpdate(BaseModel):
    username: Optional[str] = None
    full_name: Optional[str] = None
    password: Optional[str] = None
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None
    
    @field_validator('password')
    @classmethod
    def validate_password_length(cls, v: Optional[str]) -> Optional[str]:
        """Validate password doesn't exceed bcrypt's 72 byte limit"""
        if v is None:
            return v
        if len(v.encode('utf-8')) > 72:
            raise ValueError('Password cannot be longer than 72 bytes')
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        return v


class UserInDB(UserBase):
    id: int
    role: UserRole
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class User(UserInDB):
    pass


class UserMe(UserInDB):
    team_ids: List[int] = []


class LoginRequest(BaseModel):
    """Simple login request with just username and password"""
    username: str
    password: str


class LoginResponse(BaseModel):
    """Login response with access token"""
    access_token: str
    token_type: str = "bearer"

