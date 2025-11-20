"""
Invitation link model
"""

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean, Table
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import secrets

from app.database import Base


# Association table for invitation links and datasets
invitation_datasets = Table(
    'invitation_datasets',
    Base.metadata,
    Column('invitation_id', Integer, ForeignKey('invitation_links.id', ondelete="CASCADE"), primary_key=True),
    Column('dataset_id', Integer, ForeignKey('datasets.id', ondelete="CASCADE"), primary_key=True)
)


class InvitationLink(Base):
    __tablename__ = "invitation_links"

    id = Column(Integer, primary_key=True, index=True)
    token = Column(String, unique=True, index=True, nullable=False, default=lambda: secrets.token_urlsafe(32))
    created_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    max_uses = Column(Integer, nullable=True)  # None means unlimited
    used_count = Column(Integer, default=0, nullable=False)
    
    # Relationships
    creator = relationship("User", foreign_keys=[created_by])
    datasets = relationship("Dataset", secondary=invitation_datasets, backref="invitation_links")

