"""
Dataset model
"""

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, UniqueConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database import Base


class Dataset(Base):
    __tablename__ = "datasets"

    # Unique constraint: same team cannot register the same source_uri twice.
    __table_args__ = (
        UniqueConstraint("team_id", "source_uri", name="uq_dataset_team_source"),
    )

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, index=True)
    description = Column(Text, nullable=True)
    source_uri = Column(String, nullable=False, index=True)  # Path to audio files directory
    team_id = Column(Integer, ForeignKey("teams.id", ondelete="CASCADE"),
                     nullable=True)  # Nullable for admin-created datasets
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    team = relationship("Team", back_populates="datasets")
    recordings = relationship("Recording", back_populates="dataset", cascade="all, delete-orphan")
    snippet_configs = relationship(
        "SnippetConfig",
        back_populates="dataset",
        cascade="all, delete-orphan"
    )
