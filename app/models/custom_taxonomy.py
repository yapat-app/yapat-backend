"""
Custom Taxonomy models
"""

from sqlalchemy import (
    Column, Integer, String, DateTime, ForeignKey, Text, Boolean,
    UniqueConstraint, Index, text,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum

from app.database import Base
from app.models.types import PortableJSONB


class TaxonomyStatus(str, enum.Enum):
    """Status of custom taxonomy"""
    DRAFT = "draft"
    SUBMITTED = "submitted"  # Finalized label-space version awaiting team-owner promotion
    ACTIVE = "active"
    ARCHIVED = "archived"


class CustomTaxonomy(Base):
    __tablename__ = "custom_taxonomies"

    __table_args__ = (
        # Naming is scoped per (team, dataset): a team-level label space and a
        # dataset-scoped (e.g. admin-created) one may independently use "Version N".
        UniqueConstraint("team_id", "dataset_id", "name", name="uq_custom_taxonomy_team_dataset_name"),
        # The constraint above cannot police dataset-scoped rows: SQL treats
        # NULLs as distinct, so (NULL, 4, "SCIALT") never collides with itself.
        # AL reuses these rows per label code, and two concurrent annotations of
        # a new code would otherwise each insert one -- reintroducing the
        # per-annotation taxon ids this scoping exists to prevent.
        Index(
            "uq_custom_taxonomy_dataset_name",
            "dataset_id", "name",
            unique=True,
            postgresql_where=text("team_id IS NULL"),
            sqlite_where=text("team_id IS NULL"),
        ),
    )

    id = Column(Integer, primary_key=True, index=True)
    taxonomy_id = Column(String(255), nullable=False, unique=True, index=True)
    # Nullable: admin-created label spaces are dataset-scoped and have no team.
    team_id = Column(Integer, ForeignKey("teams.id", ondelete="CASCADE"), nullable=True, index=True)
    # Dataset the label space belongs to (admin-created label spaces are dataset-only).
    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="SET NULL"), nullable=True, index=True)
    created_by_user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    taxonomy_data = Column(PortableJSONB, nullable=False)
    status = Column(String(50), nullable=False, default=TaxonomyStatus.ACTIVE)
    is_global = Column(Boolean, nullable=False, default=False)
    # True only for genuine team label-space versions created via the submit/freeze
    # flow. Distinguishes them from internal per-label taxonomies auto-created by AL
    # (which are also status="active"). Only these appear as promotable versions.
    is_label_space_version = Column(Boolean, nullable=False, default=False, server_default="false")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    # foreign_keys is explicit because Team also has an FK back to custom_taxonomies
    # (teams.active_custom_taxonomy_id), creating two FK paths between the tables.
    team = relationship("Team", backref="custom_taxonomies", foreign_keys=[team_id])
    dataset = relationship("Dataset", foreign_keys=[dataset_id])
    created_by = relationship("User", backref="created_taxonomies")
    conversations = relationship("TaxonomyConversation", back_populates="custom_taxonomy")
