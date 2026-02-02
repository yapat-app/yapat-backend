"""
Custom Taxonomy models
"""

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Boolean, UniqueConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import JSONB
import enum

from app.database import Base


class TaxonomyStatus(str, enum.Enum):
    """Status of custom taxonomy"""
    DRAFT = "draft"
    ACTIVE = "active"
    ARCHIVED = "archived"


class CustomTaxonomy(Base):
    __tablename__ = "custom_taxonomies"

    __table_args__ = (
        UniqueConstraint("team_id", "name", name="uq_custom_taxonomy_team_name"),
    )

    id = Column(Integer, primary_key=True, index=True)
    taxonomy_id = Column(String(255), nullable=False, unique=True, index=True)
    team_id = Column(Integer, ForeignKey("teams.id", ondelete="CASCADE"), nullable=False, index=True)
    created_by_user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    taxonomy_data = Column(JSONB, nullable=False)
    status = Column(String(50), nullable=False, default=TaxonomyStatus.ACTIVE)
    is_global = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    team = relationship("Team", backref="custom_taxonomies")
    created_by = relationship("User", backref="created_taxonomies")
    conversations = relationship("TaxonomyConversation", back_populates="custom_taxonomy")
