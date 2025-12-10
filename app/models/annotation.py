"""
Annotation model
"""

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, JSON, Float
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database import Base

# TODO Rewrite model

class Annotation(Base):
    __tablename__ = "annotations"

    id = Column(Integer, primary_key=True, index=True)
    snippet_id = Column(Integer, ForeignKey("snippets.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    taxon_id = Column(String, nullable=False, index=True)
    notes = Column(Text, nullable=True)
    extra_metadata = Column(JSON, nullable=True)  # Additional annotation metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    created_by = relationship("User", back_populates="annotations")

    # Relationships
    snippet = relationship("Snippet", back_populates="annotations")
    user = relationship("User", back_populates="annotations", foreign_keys=[user_id])

