"""
Invitation link endpoints (Admin only)
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime, timedelta, timezone

from app.api.deps import get_db, get_current_admin_user
from app.schemas.invitation import (
    InvitationLink, 
    InvitationLinkCreate, 
    InvitationLinkCreateResponse
)
from app.models.invitation import InvitationLink as InvitationLinkModel
from app.models.dataset import Dataset
from app.models.user import User

router = APIRouter()


@router.post("/", response_model=InvitationLinkCreateResponse, status_code=status.HTTP_201_CREATED)
def create_invitation_link(
    invitation_in: InvitationLinkCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin_user)
):
    """Create a new invitation link (Admin only)"""
    # Verify all datasets exist
    datasets = db.query(Dataset).filter(Dataset.id.in_(invitation_in.dataset_ids)).all()
    if len(datasets) != len(invitation_in.dataset_ids):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="One or more dataset IDs are invalid"
        )
    
    # Create invitation link with defaults
    # Default: expires in 2 days, max 3 uses
    default_expires_at = datetime.now(timezone.utc) + timedelta(days=2)
    default_max_uses = 3
    
    invitation = InvitationLinkModel(
        created_by=current_user.id,
        expires_at=default_expires_at,
        max_uses=default_max_uses
    )
    invitation.datasets.extend(datasets)
    
    db.add(invitation)
    db.commit()
    db.refresh(invitation)
    
    # Return simplified response without created_by
    return InvitationLinkCreateResponse(
        id=invitation.id,
        token=invitation.token,
        created_at=invitation.created_at,
        expires_at=invitation.expires_at,
        is_active=invitation.is_active,
        max_uses=invitation.max_uses,
        used_count=invitation.used_count,
        dataset_ids=[d.id for d in invitation.datasets]
    )


@router.get("/", response_model=List[InvitationLink])
def list_invitation_links(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin_user)
):
    """List all invitation links (Admin only)"""
    invitations = db.query(InvitationLinkModel).offset(skip).limit(limit).all()
    
    # Prepare response with dataset IDs
    result = []
    for invitation in invitations:
        invitation_dict = {
            "id": invitation.id,
            "token": invitation.token,
            "created_by": invitation.created_by,
            "created_at": invitation.created_at,
            "expires_at": invitation.expires_at,
            "is_active": invitation.is_active,
            "max_uses": invitation.max_uses,
            "used_count": invitation.used_count,
            "dataset_ids": [d.id for d in invitation.datasets]
        }
        result.append(invitation_dict)
    
    return result


@router.delete("/{invitation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_invitation_link(
    invitation_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin_user)
):
    """Delete an invitation link (Admin only)"""
    invitation = db.query(InvitationLinkModel).filter(InvitationLinkModel.id == invitation_id).first()
    if not invitation:
        raise HTTPException(status_code=404, detail="Invitation link not found")
    
    db.delete(invitation)
    db.commit()
    return None

