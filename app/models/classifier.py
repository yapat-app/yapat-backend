"""
Classifier and TrainingExample models
"""

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Boolean, JSON, Float
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database import Base


class Classifier(Base):
    __tablename__ = "classifiers"

    id = Column(Integer, primary_key=True, index=True)
    team_id = Column(Integer, ForeignKey("teams.id", ondelete="CASCADE"), nullable=False)
    name = Column(String, nullable=False, index=True)
    description = Column(Text, nullable=True)
    model_path = Column(String, nullable=True)  # Path to saved model file
    model_type = Column(String, nullable=True)  # e.g., "svm", "random_forest", "neural_network"
    accuracy = Column(Float, nullable=True)  # Model accuracy score
    status = Column(String, default="training", nullable=False)  # training, ready, deployed, archived
    extra_metadata = Column(JSON, nullable=True)  # Model configuration and hyperparameters
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    team = relationship("Team", back_populates="classifiers")
    training_examples = relationship("TrainingExample", back_populates="classifier", cascade="all, delete-orphan")


class TrainingExample(Base):
    __tablename__ = "training_examples"

    id = Column(Integer, primary_key=True, index=True)
    classifier_id = Column(Integer, ForeignKey("classifiers.id", ondelete="CASCADE"), nullable=False)
    snippet_id = Column(Integer, ForeignKey("snippets.id", ondelete="CASCADE"), nullable=False)
    label = Column(String, nullable=False)  # Species name or class label
    features = Column(JSON, nullable=True)  # Feature vector
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    classifier = relationship("Classifier", back_populates="training_examples")
    snippet = relationship("Snippet")

