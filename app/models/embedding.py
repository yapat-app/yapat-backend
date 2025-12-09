"""
Embedding-related models:
- EmbeddingModel: describes an embedding architecture (e.g., BirdNET)
- EmbeddingJob: represents an embedding task for a dataset
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
    Example: BirdNET, YAMNet, BCResNet
    """

    __tablename__ = "embedding_models"

    id = Column(Integer, primary_key=True, index=True)

    name = Column(String, nullable=False)              # "birdnet"
    version = Column(String, nullable=True)            # "2.4"
    description = Column(String, nullable=True)

    # Optional: HF model repo or local weights
    source_uri = Column(String, nullable=True)

    # Segmentation defaults for this model
    default_window_size = Column(Float, nullable=False)
    default_step_size = Column(Float, nullable=False)
    default_overlap = Column(Float, nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Backref: one model -> many jobs
    embedding_jobs = relationship("EmbeddingJob", back_populates="embedding_model")


# ---------------------------
# Job Status Enum
# ---------------------------
class EmbeddingJobStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# ---------------------------
# Embedding Job
# ---------------------------
class EmbeddingJob(Base):
    """
    Tracks a single embedding pipeline run for a dataset.
    """

    __tablename__ = "embedding_jobs"

    id = Column(Integer, primary_key=True, index=True)

    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)
    embedding_model_id = Column(Integer, ForeignKey("embedding_models.id", ondelete="CASCADE"), nullable=False)

    status = Column(Enum(EmbeddingJobStatus), nullable=False, default=EmbeddingJobStatus.PENDING)
    celery_task_id = Column(String, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    error_message = Column(Text, nullable=True)

    # Relationships
    dataset = relationship("Dataset", back_populates="embedding_jobs")
    embedding_model = relationship("EmbeddingModel", back_populates="embedding_jobs")

    # New: EmbeddingJob owns snippets (1:M)
    snippets = relationship(
        "Snippet",
        back_populates="embedding_job",
        cascade="all, delete-orphan",
    )

    # New: 1:1 link to SnippetConfig
    snippet_config = relationship(
        "SnippetConfig",
        back_populates="embedding_job",
        uselist=False,
        cascade="all, delete-orphan",
    )
