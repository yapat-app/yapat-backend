"""
Annotation model
"""

from sqlalchemy import (
    Column, Integer, String, DateTime, ForeignKey,
    Text, JSON, Float, CheckConstraint, UniqueConstraint, event
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database import Base


# --- CHECK CONSTRAINT (PostgreSQL only) -----------------------------------

# pattern: namespace:key (key = digits or alphanumeric/underscore/hyphen) or custom:uuid
valid_taxon_constraint = CheckConstraint(
    "taxon_id ~ '^([a-z]+:[a-zA-Z0-9_-]+|custom:[a-f0-9-]+)$'",
    name="valid_taxon_id_format",
)


# Remove the constraint entirely when creating tables on SQLite
@event.listens_for(Base.metadata, "before_create")
def remove_taxon_constraint_if_sqlite(target, connection, **kw):
    if connection.dialect.name == "sqlite":
        table = Annotation.__table__
        if valid_taxon_constraint in table.constraints:
            table.constraints.remove(valid_taxon_constraint)


# --- MODEL ----------------------------------------------------------------

class Annotation(Base):
    __tablename__ = "annotations"

    id = Column(Integer, primary_key=True, index=True)
    snippet_id = Column(Integer, ForeignKey("snippets.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    taxon_id = Column(String(255), nullable=False, index=True)
    resolved_name_snapshot = Column(String(255), nullable=False)

    confidence = Column(Float, nullable=True, default=0.8)
    notes = Column(Text, nullable=True)
    extra_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    created_by = relationship("User", back_populates="annotations")
    snippet = relationship("Snippet", back_populates="annotations")
    user = relationship("User", back_populates="annotations", foreign_keys=[user_id])

    __table_args__ = (
        valid_taxon_constraint,
        UniqueConstraint("snippet_id", "taxon_id", name="uq_annotations_snippet_taxon"),
    )
