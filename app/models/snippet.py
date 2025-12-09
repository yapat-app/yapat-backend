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
    recording_id = Column(
        Integer,
        ForeignKey("recordings.id", ondelete="CASCADE"),
        nullable=False,
    )
    # New architecture: snippets belong to an embedding job
    embedding_job_id = Column(
        Integer,
        ForeignKey("embedding_jobs.id", ondelete="CASCADE"),
        nullable=False,
    )

    start_time = Column(Float, nullable=False)  # seconds
    duration = Column(Float, nullable=False)  # == window_size

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    recording = relationship("Recording", back_populates="snippets")
    embedding_job = relationship("EmbeddingJob", back_populates="snippets")

    annotations = relationship(
        "Annotation",
        back_populates="snippet",
        cascade="all, delete-orphan",
    )


class SnippetConfig(Base):
    __tablename__ = "snippet_configs"

    id = Column(Integer, primary_key=True, index=True)

    embedding_job_id = Column(
        Integer,
        ForeignKey("embedding_jobs.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,  # enforce 1:1
    )

    window_size = Column(Float, nullable=False)
    step_size = Column(Float, nullable=False)
    overlap = Column(Float, nullable=False)

    # 1:1 relationship
    embedding_job = relationship("EmbeddingJob", back_populates="snippet_config")
