"""
Snippet and SnippetConfig models
"""

from sqlalchemy import Column, Integer, DateTime, ForeignKey, Float
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database import Base


class Snippet(Base):
    __tablename__ = "snippets"

    id = Column(Integer, primary_key=True, index=True)
    recording_id = Column(Integer, ForeignKey("recordings.id", ondelete="CASCADE"), nullable=False)
    snippet_config_id = Column(Integer, ForeignKey("snippet_configs.id", ondelete="CASCADE"), nullable=False)

    start_time = Column(Float, nullable=False)  # seconds
    duration = Column(Float, nullable=False)  # == window_size

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    recording = relationship("Recording", back_populates="snippets")
    annotations = relationship(
        "Annotation",
        back_populates="snippet",
        cascade="all, delete-orphan"
    )
    config = relationship(
        "SnippetConfig",
        back_populates="snippets"
    )


class SnippetConfig(Base):
    __tablename__ = "snippet_configs"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)

    overlap = Column(Float, default=0.0, nullable=False)
    window_size = Column(Float, nullable=False)
    step_size = Column(Float, nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    snippets = relationship(
        "Snippet",
        back_populates="config",
        cascade="all, delete-orphan"  # Correct place for cascade
    )
    dataset = relationship("Dataset", back_populates="snippet_configs")
