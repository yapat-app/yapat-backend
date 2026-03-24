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
from pgvector.sqlalchemy import Vector
from sqlalchemy.sql import func
import enum

from app.database import Base


# ── Enums ──────────────────────────────────────────────────────────────

class ALModelStatus(str, enum.Enum):
    AVAILABLE = "AVAILABLE"
    LOADING = "LOADING"
    ERROR = "ERROR"


class ALFeedbackAction(str, enum.Enum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    MODIFY = "MODIFY"


class ALRetrainStatus(str, enum.Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class ALAnnotationSource(str, enum.Enum):
    GROUND_TRUTH = "ground_truth"
    USER = "user"

# ── Model Checkpoint ───────────────────────────────────────────────────

class ALModelCheckpoint(Base):
    """
    Tracks model versions / checkpoints used for PAM active learning.

    A checkpoint belongs to a dataset and holds a path to the weights on disk.
    The first checkpoint (``is_base=True``) points to the shared base model;
    subsequent retrained versions reference their parent via ``parent_checkpoint_id``.
    """
    __tablename__ = "al_model_checkpoints"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(
        Integer,
        ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name = Column(String, nullable=False)
    version = Column(String, nullable=False, default="v0")
    checkpoint_path = Column(String, nullable=False)  # filesystem path to weights
    label_config_path = Column(String, nullable=False)
    model_type = Column(String, nullable=False, default="al_classifier")
    hyperparameters = Column(JSON, nullable=True)
    is_base = Column(Integer, nullable=False, default=0)  # 1 = base model entry
    parent_checkpoint_id = Column(
        Integer,
        ForeignKey("al_model_checkpoints.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    status = Column(
        SQLEnum(ALModelStatus, name="al_model_status_enum", create_type=True),
        nullable=False,
        default=ALModelStatus.AVAILABLE,
    )
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), nullable=True)

    # Relationships
    dataset = relationship("Dataset", back_populates="al_model_checkpoints")
    predictions = relationship(
        "ALPrediction", back_populates="model_checkpoint", cascade="all, delete-orphan"
    )
    retrain_jobs = relationship(
        "ALRetrainJob", back_populates="model_checkpoint", cascade="all, delete-orphan"
    )
    annotations = relationship(
        "ALSnippetAnnotation",
        back_populates="model_checkpoint",
    )
    parent_checkpoint = relationship(
        "ALModelCheckpoint",
        remote_side="ALModelCheckpoint.id",
        foreign_keys=[parent_checkpoint_id],
        uselist=False,
    )

    __table_args__ = (
        UniqueConstraint("dataset_id", "name", "version", name="uq_al_checkpoint"),
    )


# ── Prediction ─────────────────────────────────────────────────────────

class ALPrediction(Base):
    """
    A single classifier prediction on a snippet.

    Stores the predicted label string, numeric score, and a composite
    ranking score produced by the combined scoring module.
    """
    __tablename__ = "al_predictions"

    id = Column(Integer, primary_key=True, index=True)
    model_checkpoint_id = Column(
        Integer,
        ForeignKey("al_model_checkpoints.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    snippet_id = Column(
        Integer,
        ForeignKey("snippets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    predicted_labels = Column(JSON, nullable=False)
    predicted_probabilities = Column(JSON, nullable=True)
    # Active learning scores
    uncertainty = Column(Float, nullable=True)
    diversity = Column(Float, nullable=True)
    density = Column(Float, nullable=True)
    composite_score = Column(Float, nullable=True)

    # Intermediate representation
    embedding = Column(Vector(512), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    model_checkpoint = relationship("ALModelCheckpoint", back_populates="predictions")
    snippet = relationship("Snippet", back_populates="al_predictions")
    feedback_events = relationship(
        "ALFeedbackEvent", back_populates="prediction", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("model_checkpoint_id", "snippet_id", name="uq_al_prediction"),
    )

class ALSnippetAnnotation(Base):
    __tablename__ = "al_snippet_annotation"

    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True)
    snippet_id = Column(Integer, ForeignKey("snippets.id", ondelete="CASCADE"), nullable=False, index=True)
    label = Column(String, nullable=False, index=True)
    source = Column(
        SQLEnum(ALAnnotationSource, name="al_annotation_source_enum", create_type=True),
        nullable=False,
    )
    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    model_checkpoint_id = Column(Integer, ForeignKey("al_model_checkpoints.id", ondelete="SET NULL"), nullable=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    snippet = relationship("Snippet", back_populates="al_annotations")
    model_checkpoint = relationship("ALModelCheckpoint", back_populates="annotations")

    __table_args__ = (
        UniqueConstraint(
            "snippet_id",
            "label",
            "source",
            "user_id",
            "model_checkpoint_id",
            name="uq_al_snippet_label_source_user_ckpt",
        ),
    )

# ── Feedback Event ─────────────────────────────────────────────────────

class ALFeedbackEvent(Base):
    """
    A single human-in-the-loop feedback event.

    action = ACCEPT | REJECT | MODIFY
    When action == MODIFY, *modified_label* holds the corrected label.
    """
    __tablename__ = "al_feedback_events"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(
        Integer,
        ForeignKey("al_predictions.id", ondelete="CASCADE"),
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
        SQLEnum(ALFeedbackAction, name="al_feedback_action_enum", create_type=True),
        nullable=False,
    )
    modified_labels = Column(JSON, nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    prediction = relationship("ALPrediction", back_populates="feedback_events")
    user = relationship("User")

# ── Fields required for VIS ─────────────────────────────────────────────────────


class ALVis(Base):
    __tablename__ = "al_vis"

    id = Column(Integer, primary_key=True, index=True)

    dataset_id = Column(
        Integer,
        ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    model_checkpoint_id = Column(
        Integer,
        ForeignKey("al_model_checkpoints.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    snippet_id = Column(
        Integer,
        ForeignKey("snippets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # AL scores
    uncertainty = Column(Float, nullable=True)
    diversity = Column(Float, nullable=True)
    density = Column(Float, nullable=True)
    composite_score = Column(Float, nullable=True)

    # Model output
    model_predicted_labels = Column(JSON, nullable=True)
    model_predicted_probabilities = Column(JSON, nullable=True)

    # Trusted / user-facing labels
    trusted_labels = Column(JSON, nullable=True)

    # Classifier embedding (optional, useful if you want to avoid recomputation)
    latent_embedding = Column(JSON, nullable=True)

    # PCA
    pca_2d_x = Column(Float, nullable=True)
    pca_2d_y = Column(Float, nullable=True)
    pca_3d_x = Column(Float, nullable=True)
    pca_3d_y = Column(Float, nullable=True)
    pca_3d_z = Column(Float, nullable=True)

    # UMAP
    umap_2d_x = Column(Float, nullable=True)
    umap_2d_y = Column(Float, nullable=True)
    umap_3d_x = Column(Float, nullable=True)
    umap_3d_y = Column(Float, nullable=True)
    umap_3d_z = Column(Float, nullable=True)

    # t-SNE
    tsne_2d_x = Column(Float, nullable=True)
    tsne_2d_y = Column(Float, nullable=True)
    tsne_3d_x = Column(Float, nullable=True)
    tsne_3d_y = Column(Float, nullable=True)
    tsne_3d_z = Column(Float, nullable=True)

    # Isomap
    isomap_2d_x = Column(Float, nullable=True)
    isomap_2d_y = Column(Float, nullable=True)
    isomap_3d_x = Column(Float, nullable=True)
    isomap_3d_y = Column(Float, nullable=True)
    isomap_3d_z = Column(Float, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), nullable=True)

    model_checkpoint = relationship("ALModelCheckpoint")
    snippet = relationship("Snippet")

    __table_args__ = (
        UniqueConstraint(
            "model_checkpoint_id",
            "snippet_id",
            name="uq_al_vis_checkpoint_snippet",
        ),
    )

# ── Retrain Job ────────────────────────────────────────────────────────

class ALRetrainJob(Base):
    """
    Metadata for a retraining run triggered either automatically or manually.
    """
    __tablename__ = "al_retrain_jobs"

    id = Column(Integer, primary_key=True, index=True)
    model_checkpoint_id = Column(
        Integer,
        ForeignKey("al_model_checkpoints.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    dataset_id = Column(
        Integer,
        ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    trigger = Column(String, nullable=False, default="auto")  # "auto" | "manual"
    feedback_count = Column(Integer, nullable=False, default=0)
    status = Column(
        SQLEnum(ALRetrainStatus, name="al_retrain_status_enum", create_type=True),
        nullable=False,
        default=ALRetrainStatus.PENDING,
    )
    result_metrics = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    model_checkpoint = relationship("ALModelCheckpoint", back_populates="retrain_jobs")
