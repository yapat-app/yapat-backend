"""
Dataset model
"""

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Table
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database import Base


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

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, index=True)
    description = Column(Text, nullable=True)
    source_uri = Column(String, nullable=True)  # Path to audio files directory
    team_id = Column(Integer, ForeignKey("teams.id", ondelete="CASCADE"), nullable=True)  # Nullable for admin-created datasets
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    team = relationship("Team", back_populates="datasets")
    recordings = relationship("Recording", back_populates="dataset", cascade="all, delete-orphan")
    # Users with direct access to this dataset (via invitation, before team assignment)
    authorized_users = relationship("User", secondary=user_datasets, backref="accessible_datasets")

