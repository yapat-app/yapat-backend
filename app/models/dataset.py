"""
Dataset model
"""

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Table, UniqueConstraint, Enum as SQLEnum
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

