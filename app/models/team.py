"""
Team and TeamMembership models
"""

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Enum, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum
import secrets

from app.database import Base


class TeamRole(str, enum.Enum):
    """Team-specific roles"""
    OWNER = "owner"
    USER = "user"


class Team(Base):
    __tablename__ = "teams"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, index=True)
    description = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    memberships = relationship("TeamMembership", back_populates="team", cascade="all, delete-orphan")
    datasets = relationship("Dataset", back_populates="team", cascade="all, delete-orphan")
    classifiers = relationship("Classifier", back_populates="team", cascade="all, delete-orphan")
    invitations = relationship("TeamInvitation", back_populates="team", cascade="all, delete-orphan")


class TeamMembership(Base):
    __tablename__ = "team_memberships"

    id = Column(Integer, primary_key=True, index=True)
    team_id = Column(Integer, ForeignKey("teams.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    role = Column(Enum(TeamRole), default=TeamRole.USER, nullable=False)
    joined_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    team = relationship("Team", back_populates="memberships")
    user = relationship("User", back_populates="team_memberships")


class TeamInvitation(Base):
    __tablename__ = "team_invitations"

    id = Column(Integer, primary_key=True, index=True)
    team_id = Column(Integer, ForeignKey("teams.id", ondelete="CASCADE"), nullable=False)
    invited_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    token = Column(String, unique=True, index=True, nullable=False, default=lambda: secrets.token_urlsafe(32))
    email = Column(String, nullable=True)  # Optional: pre-invite specific email
    target_role = Column(Enum(TeamRole), default=TeamRole.USER, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    max_uses = Column(Integer, nullable=True)  # None means unlimited uses
    used_count = Column(Integer, default=0, nullable=False)  # Track how many times used
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    team = relationship("Team", back_populates="invitations")
    inviter = relationship("User", foreign_keys=[invited_by])

