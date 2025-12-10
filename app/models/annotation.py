"""
Annotation model
"""

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, JSON, Float, CheckConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database import Base


class Annotation(Base):
    __tablename__ = "annotations"

    id = Column(Integer, primary_key=True, index=True)
    snippet_id = Column(Integer, ForeignKey("snippets.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    # Namespaced taxon identifier (e.g., 'gbif:2420576')
    taxon_id = Column(String(255), nullable=False, index=True)
    # Snapshot of resolved scientific name at time of annotation
    resolved_name_snapshot = Column(String(255), nullable=False)

    confidence = Column(Float, nullable=True, default=0.8)  # Confidence score (0.0 to 1.0)
    notes = Column(Text, nullable=True)
    extra_metadata = Column(JSON, nullable=True)  # Additional annotation metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    created_by = relationship("User", back_populates="annotations")

    # Relationships
    snippet = relationship("Snippet", back_populates="annotations")
    user = relationship("User", back_populates="annotations", foreign_keys=[user_id])

    # Constraints
    __table_args__ = (
        CheckConstraint("taxon_id ~ '^[a-z]+:[0-9]+$'", name='valid_taxon_id_format'),
    )

