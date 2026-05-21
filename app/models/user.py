"""
User model
"""

from sqlalchemy import Column, Integer, String, Boolean, DateTime, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum

from app.database import Base


class UserRole(str, enum.Enum):
    """Global user roles"""
    ADMIN = "admin"
    TEAM_OWNER = "team_owner"
    USER = "user"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String, nullable=True)
    role = Column(Enum(UserRole), default=UserRole.USER, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    team_memberships = relationship("TeamMembership", back_populates="user", cascade="all, delete-orphan")
    annotations = relationship(
        "Annotation",
        back_populates="user",
        cascade="all, delete-orphan",
    )

    al_annotations = relationship(
        "ALSnippetAnnotation",
        back_populates="user",
    )

    @property
    def team_ids(self):
        return [m.team_id for m in (self.team_memberships or [])]

