"""
Embedding-related models:
- EmbeddingModel: describes an embedding architecture (e.g., BirdNET)
- SnippetSet: defines segmentation parameters actually used for a dataset × model
- EmbeddingJob: represents an embedding computation pass over a SnippetSet
"""

from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    ForeignKey,
    Float,
    Enum,
    Text,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum

from app.database import Base


# ---------------------------
# Embedding Model
# ---------------------------
class EmbeddingModel(Base):
    """
    A model capable of producing embeddings from audio snippets.
    Most practical models require fixed input window/step parameters.
    """

    __tablename__ = "embedding_models"

    id = Column(Integer, primary_key=True, index=True)

    name = Column(String, nullable=False)
    version = Column(String, nullable=True)
    description = Column(String, nullable=True)

    # Optional metadata: HF repo, local weights, etc.
    source_uri = Column(String, nullable=True)

    # Canonical segmentation parameters (authoritative)
    window_size = Column(Float, nullable=False)
    step_size = Column(Float, nullable=False)
    overlap = Column(Float, nullable=False)

    # Strictness flags—default: required
    requires_fixed_window = Column(Integer, default=True)
    requires_fixed_step = Column(Integer, default=True)
    requires_fixed_overlap = Column(Integer, default=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    snippet_sets = relationship("SnippetSet", back_populates="embedding_model")
    embedding_jobs = relationship("EmbeddingJob", back_populates="embedding_model")


# ---------------------------
# SnippetSet
# ---------------------------
class SnippetSetStatus(str, enum.Enum):
    PENDING = "pending"
    READY = "ready"
    FAILED = "failed"


class SnippetSet(Base):
    """
    Defines segmentation for a dataset × embedding_model with fixed parameters.
    Owns the snippets produced by segmentation.
    """

    __tablename__ = "snippet_sets"

    id = Column(Integer, primary_key=True, index=True)

    dataset_id = Column(
        Integer,
        ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False,
    )

    embedding_model_id = Column(
        Integer,
        ForeignKey("embedding_models.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Final parameters used (canonical truth)
    window_size = Column(Float, nullable=False)
    step_size = Column(Float, nullable=False)
    overlap = Column(Float, nullable=False)

    status = Column(
        Enum(SnippetSetStatus),
        nullable=False,
        default=SnippetSetStatus.PENDING,
    )

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    dataset = relationship("Dataset", back_populates="snippet_sets")
    embedding_model = relationship("EmbeddingModel", back_populates="snippet_sets")
    snippets = relationship("Snippet", back_populates="snippet_set", cascade="all, delete-orphan")
    embedding_jobs = relationship("EmbeddingJob", back_populates="snippet_set")


# ---------------------------
# Embedding Job
# ---------------------------
class EmbeddingJobStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class EmbeddingJob(Base):
    """
    Represents a single embedding computation pass over a SnippetSet.
    Does NOT own snippets or segmentation parameters.
    """

    __tablename__ = "embedding_jobs"

    id = Column(Integer, primary_key=True, index=True)

    dataset_id = Column(
        Integer,
        ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False,
    )

    embedding_model_id = Column(
        Integer,
        ForeignKey("embedding_models.id", ondelete="CASCADE"),
        nullable=False,
    )

    snippet_set_id = Column(
        Integer,
        ForeignKey("snippet_sets.id", ondelete="CASCADE"),
        nullable=False,
    )

    status = Column(
        Enum(EmbeddingJobStatus),
        nullable=False,
        default=EmbeddingJobStatus.PENDING,
    )

    celery_task_id = Column(String, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    error_message = Column(Text, nullable=True)

    # Relationships
    dataset = relationship("Dataset", back_populates="embedding_jobs")
    embedding_model = relationship("EmbeddingModel", back_populates="embedding_jobs")
    snippet_set = relationship("SnippetSet", back_populates="embedding_jobs")
