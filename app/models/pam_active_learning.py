"""
PAM Active Learning models

Data models for the PAM-specific active learning flow:
- PAMModelCheckpoint: model version/checkpoint metadata
- PAMPrediction: classifier predictions on snippets
- PAMFeedbackEvent: human-in-the-loop accept/reject/modify events
- PAMRetrainJob: retraining run metadata
"""

from sqlalchemy import (
    Column, Integer, String, DateTime, ForeignKey, Float, Text, JSON,
    Enum as SQLEnum, UniqueConstraint,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum

from app.database import Base


# ── Enums ──────────────────────────────────────────────────────────────

class PAMModelStatus(str, enum.Enum):
    AVAILABLE = "AVAILABLE"
    LOADING = "LOADING"
    ERROR = "ERROR"


class PAMFeedbackAction(str, enum.Enum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    MODIFY = "MODIFY"


class PAMRetrainStatus(str, enum.Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


# ── Model Checkpoint ───────────────────────────────────────────────────

class PAMModelCheckpoint(Base):
    """
    Tracks model versions / checkpoints used for PAM active learning.

    A checkpoint belongs to a dataset and holds a path to the weights on disk.
    The first checkpoint (``is_base=True``) points to the shared base model;
    subsequent retrained versions reference their parent via ``parent_checkpoint_id``.
    """
    __tablename__ = "pam_model_checkpoints"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(
        Integer,
        ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name = Column(String, nullable=False)
    version = Column(String, nullable=False, default="v0")
    checkpoint_path = Column(String, nullable=True)  # filesystem path to weights
    model_type = Column(String, nullable=False, default="pam_classifier")
    hyperparameters = Column(JSON, nullable=True)
    is_base = Column(Integer, nullable=False, default=0)  # 1 = base model entry
    parent_checkpoint_id = Column(
        Integer,
        ForeignKey("pam_model_checkpoints.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    status = Column(
        SQLEnum(PAMModelStatus, name="pam_model_status_enum", create_type=True),
        nullable=False,
        default=PAMModelStatus.AVAILABLE,
    )
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), nullable=True)

    # Relationships
    dataset = relationship("Dataset", back_populates="pam_model_checkpoints")
    predictions = relationship(
        "PAMPrediction", back_populates="model_checkpoint", cascade="all, delete-orphan"
    )
    retrain_jobs = relationship(
        "PAMRetrainJob", back_populates="model_checkpoint", cascade="all, delete-orphan"
    )
    parent_checkpoint = relationship(
        "PAMModelCheckpoint",
        remote_side="PAMModelCheckpoint.id",
        foreign_keys=[parent_checkpoint_id],
        uselist=False,
    )

    __table_args__ = (
        UniqueConstraint("dataset_id", "name", "version", name="uq_pam_checkpoint"),
    )


# ── Prediction ─────────────────────────────────────────────────────────

class PAMPrediction(Base):
    """
    A single classifier prediction on a snippet.

    Stores the predicted label string, numeric score, and a composite
    ranking score produced by the combined scoring module.
    """
    __tablename__ = "pam_predictions"

    id = Column(Integer, primary_key=True, index=True)
    model_checkpoint_id = Column(
        Integer,
        ForeignKey("pam_model_checkpoints.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    snippet_id = Column(
        Integer,
        ForeignKey("snippets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    predicted_label = Column(String, nullable=False)   # e.g. "bird", "frog"
    confidence = Column(Float, nullable=False)          # model confidence [0,1]
    ranking_score = Column(Float, nullable=True)        # combined scoring output
    extra_scores = Column(JSON, nullable=True)          # per-factor breakdown
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    model_checkpoint = relationship("PAMModelCheckpoint", back_populates="predictions")
    snippet = relationship("Snippet", back_populates="pam_predictions")
    feedback_events = relationship(
        "PAMFeedbackEvent", back_populates="prediction", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("model_checkpoint_id", "snippet_id", name="uq_pam_prediction"),
    )


# ── Feedback Event ─────────────────────────────────────────────────────

class PAMFeedbackEvent(Base):
    """
    A single human-in-the-loop feedback event.

    action = ACCEPT | REJECT | MODIFY
    When action == MODIFY, *modified_label* holds the corrected label.
    """
    __tablename__ = "pam_feedback_events"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(
        Integer,
        ForeignKey("pam_predictions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    action = Column(
        SQLEnum(PAMFeedbackAction, name="pam_feedback_action_enum", create_type=True),
        nullable=False,
    )
    modified_label = Column(String, nullable=True)  # only when action == MODIFY
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    prediction = relationship("PAMPrediction", back_populates="feedback_events")
    user = relationship("User")


# ── Retrain Job ────────────────────────────────────────────────────────

class PAMRetrainJob(Base):
    """
    Metadata for a retraining run triggered either automatically or manually.
    """
    __tablename__ = "pam_retrain_jobs"

    id = Column(Integer, primary_key=True, index=True)
    model_checkpoint_id = Column(
        Integer,
        ForeignKey("pam_model_checkpoints.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    trigger = Column(String, nullable=False, default="auto")  # "auto" | "manual"
    feedback_count = Column(Integer, nullable=False, default=0)
    status = Column(
        SQLEnum(PAMRetrainStatus, name="pam_retrain_status_enum", create_type=True),
        nullable=False,
        default=PAMRetrainStatus.PENDING,
    )
    result_metrics = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    model_checkpoint = relationship("PAMModelCheckpoint", back_populates="retrain_jobs")
