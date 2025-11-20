"""
User schemas
"""

from pydantic import BaseModel, field_validator
from datetime import datetime
from typing import Optional
from app.models.user import UserRole


class UserBase(BaseModel):
    username: str
    full_name: Optional[str] = None
    role: UserRole = UserRole.USER


class UserCreate(UserBase):
    password: str
    invitation_token: Optional[str] = None  # For admin-created dataset invitations
    team_invitation_token: Optional[str] = None  # For team invitations
    
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
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class User(UserInDB):
    pass


class LoginRequest(BaseModel):
    """Simple login request with just username and password"""
    username: str
    password: str


class LoginResponse(BaseModel):
    """Login response with access token"""
    access_token: str
    token_type: str = "bearer"

