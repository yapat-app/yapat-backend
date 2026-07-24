"""
Dataset model
"""

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Table, UniqueConstraint, Enum as SQLEnum, Float, JSON, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum

from app.database import Base


class DatasetType(str, enum.Enum):
    PAM = "PAM"
    FOCAL_RECORDINGS = "FOCAL_RECORDINGS"


# Association table for user-dataset direct access
user_datasets = Table(
    'user_datasets',
    Base.metadata,
    Column('user_id', Integer, ForeignKey('users.id', ondelete="CASCADE"), primary_key=True),
    Column('dataset_id', Integer, ForeignKey('datasets.id', ondelete="CASCADE"), primary_key=True),
    Column('granted_at', DateTime(timezone=True), server_default=func.now()),
    Column('granted_by_invitation_id', Integer, ForeignKey('invitation_links.id', ondelete="SET NULL"), nullable=True)
)


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
    dataset_type = Column(SQLEnum(DatasetType), nullable=False, default=DatasetType.PAM, index=True)
    team_id = Column(Integer, ForeignKey("teams.id", ondelete="SET NULL"),
                     nullable=True)  # Nullable for admin-created datasets
    default_snippet_set_id = Column(
        Integer,
        ForeignKey("snippet_sets.id", ondelete="SET NULL"),
        nullable=True,
        index=True
    )  # Default SnippetSet for this dataset
    # Mel-spectrogram display band for annotation (Hz). Null = use 0 and Nyquist per snippet.
    spectrogram_f_min_hz = Column(Float, nullable=True)
    spectrogram_f_max_hz = Column(Float, nullable=True)
    quick_labels = Column(JSON, nullable=True)
    # PAM active-learning auto-retrain threshold override. Null = use the global
    # RETRAIN_AFTER default from active_learning/config.yaml.
    retrain_after_threshold = Column(Integer, nullable=True)
    # Marks this dataset as reference-only training data: ingested through the
    # normal scan/snippet/embed pipeline like any other dataset, but never
    # surfaced in annotation/labelling views. Other datasets/teams opt into
    # using it via DatasetReferenceLink. See docs/reference-data-pool-design.md.
    is_reference = Column(Boolean, nullable=False, default=False, server_default="false", index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    team = relationship("Team", back_populates="datasets")
    recordings = relationship("Recording", back_populates="dataset", cascade="all, delete-orphan")
    embedding_jobs = relationship(
        "EmbeddingJob",
        back_populates="dataset",
        cascade="all, delete-orphan"
    )
    snippet_sets = relationship(
        "SnippetSet",
        back_populates="dataset",
        cascade="all, delete-orphan",
        primaryjoin="Dataset.id == SnippetSet.dataset_id"
    )
    default_snippet_set = relationship(
        "SnippetSet",
        foreign_keys=[default_snippet_set_id],
        post_update=True
    )
    # Users with direct access to this dataset (via invitation, before team assignment)
    authorized_users = relationship("User", secondary=user_datasets, backref="accessible_datasets")
    # PAM Active Learning model checkpoints
    al_model_checkpoints = relationship(
        "ALModelCheckpoint",
        back_populates="dataset",
        cascade="all, delete-orphan",
    )
    # WSSED training jobs
    wssed_training_jobs = relationship(
        "WSSEDTrainingJob",
        back_populates="dataset",
        cascade="all, delete-orphan",
    )
    # WSSED species models
    wssed_species_models = relationship(
        "WSSEDSpeciesModel",
        back_populates="dataset",
        cascade="all, delete-orphan",
    )
    # Reference pool links where THIS dataset is the target (i.e. datasets it
    # draws reference training data from).
    reference_links = relationship(
        "DatasetReferenceLink",
        back_populates="dataset",
        cascade="all, delete-orphan",
        foreign_keys="DatasetReferenceLink.dataset_id",
    )
    # Reference pool links where THIS dataset is the source (i.e. who is using
    # this dataset as reference data). Only meaningful when is_reference=True.
    referenced_by_links = relationship(
        "DatasetReferenceLink",
        back_populates="reference_dataset",
        cascade="all, delete-orphan",
        foreign_keys="DatasetReferenceLink.reference_dataset_id",
    )

