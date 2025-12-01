"""
Snippet and SnippetConfig models
"""

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Float, Text, Boolean, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database import Base


class Snippet(Base):
    __tablename__ = "snippets"

    id = Column(Integer, primary_key=True, index=True)
    recording_id = Column(Integer, ForeignKey("recordings.id", ondelete="CASCADE"), nullable=False)
    start_time = Column(Float, nullable=False)  # Start time in seconds
    end_time = Column(Float, nullable=False)  # End time in seconds
    duration = Column(Float, nullable=False)  # Duration in seconds
    file_path = Column(String, nullable=True)  # Path to extracted snippet file
    embedding = Column(JSON, nullable=True)  # Vector embedding for similarity search
    is_annotated = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    recording = relationship("Recording", back_populates="snippets")
    annotations = relationship("Annotation", back_populates="snippet", cascade="all, delete-orphan")
    config = relationship("SnippetConfig", back_populates="snippet", uselist=False, cascade="all, delete-orphan")


class SnippetConfig(Base):
    __tablename__ = "snippet_configs"

    id = Column(Integer, primary_key=True, index=True)
    snippet_id = Column(Integer, ForeignKey("snippets.id", ondelete="CASCADE"), nullable=False, unique=True)
    overlap = Column(Float, default=0.0, nullable=False)  # Overlap with previous snippet
    window_size = Column(Float, nullable=True)  # Window size used
    step_size = Column(Float, nullable=True)  # Step size used
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    snippet = relationship("Snippet", back_populates="config")

