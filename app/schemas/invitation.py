"""
Invitation link schemas
"""

from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List


class InvitationLinkCreate(BaseModel):
    """Schema for creating an invitation link"""
    dataset_ids: List[int]
    # expires_at and max_uses are set automatically (2 days, 3 uses)


class InvitationLink(BaseModel):
    """Schema for invitation link response"""
    id: int
    token: str
    created_by: Optional[int]
    created_at: datetime
    expires_at: Optional[datetime]
    is_active: bool
    max_uses: Optional[int]
    used_count: int
    dataset_ids: List[int] = []

    class Config:
        from_attributes = True


class InvitationLinkCreateResponse(BaseModel):
    """Simplified response schema for creating an invitation link"""
    id: int
    token: str
    created_at: datetime
    expires_at: Optional[datetime]
    is_active: bool
    max_uses: Optional[int]
    used_count: int
    dataset_ids: List[int] = []

    class Config:
        from_attributes = True

