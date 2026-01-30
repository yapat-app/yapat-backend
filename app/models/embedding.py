"""
Embedding-related models:
- EmbeddingModel: describes an embedding architecture (e.g., BirdNET)
- SnippetSet: defines segmentation parameters actually used for a dataset × model
- EmbeddingJob: represents an embedding computation pass over a SnippetSet
"""

import enum

from sqlalchemy import (
    ARRAY,
    JSON,
    Column,
    Integer,
    String,
    DateTime,
    ForeignKey,
    Float,
    Text,
)
from sqlalchemy.dialects.postgresql import ARRAY as PG_ARRAY
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.types import TypeDecorator

# Try to import pgvector, fallback to None if not available
try:
    from pgvector.sqlalchemy import Vector
    PGVECTOR_AVAILABLE = True
except ImportError:
    Vector = None
    PGVECTOR_AVAILABLE = False

from app.database import Base


class EnumValue(TypeDecorator):
    """Ensure enum values (not names) are stored in database."""
    impl = String
    cache_ok = True

    def __init__(self, enum_class, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enum_class = enum_class

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if isinstance(value, self.enum_class):
            # Return the enum value (lowercase string) instead of name
            return value.value
        return value

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        # Convert database value back to enum
        return self.enum_class(value)


class VectorType(TypeDecorator):
    """
    Stores list[float] as pgvector's vector type on Postgres, JSON for SQLite.
    """
    impl = JSON  # fallback for SQLite
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql" and PGVECTOR_AVAILABLE:
            # Use pgvector's native Vector type (dimension 1024 for BirdNET)
            return dialect.type_descriptor(Vector(1024))
        elif dialect.name == "postgresql":
            # Fallback to ARRAY if pgvector not installed
            return dialect.type_descriptor(PG_ARRAY(Float))
        else:
            # JSON for SQLite
            return dialect.type_descriptor(JSON)

    def process_bind_param(self, value, dialect):
        return value

    def process_result_value(self, value, dialect):
        return value


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
        EnumValue(SnippetSetStatus),
        nullable=False,
        default=SnippetSetStatus.PENDING,
    )

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    dataset = relationship(
        "Dataset",
        back_populates="snippet_sets",
        foreign_keys=[dataset_id]
    )
    embedding_model = relationship("EmbeddingModel", back_populates="snippet_sets")
    snippets = relationship("Snippet", back_populates="snippet_set", cascade="all, delete-orphan")
    embedding_jobs = relationship("EmbeddingJob", back_populates="snippet_set", cascade="all, delete-orphan")


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
        EnumValue(EmbeddingJobStatus),
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


# ---------------------------
# Embedding Vector
# ---------------------------

class EmbeddingVector(Base):
    """
    Stores a single embedding vector for a snippet × model × job.
    This is intentionally minimal for first integration step.
    """

    __tablename__ = "embedding_vectors"

    id = Column(Integer, primary_key=True, index=True)

    snippet_id = Column(
        Integer,
        ForeignKey("snippets.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
        index=True,
    )

    embedding_job_id = Column(
        Integer,
        ForeignKey("embedding_jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    embedding_model_id = Column(
        Integer,
        ForeignKey("embedding_models.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    dim = Column(Integer, nullable=False)
    vector = Column(VectorType, nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships (lightly defined; no back_populates yet)
    snippet = relationship("Snippet")
    job = relationship("EmbeddingJob")
    embedding_model = relationship("EmbeddingModel")
