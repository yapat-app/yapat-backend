"""
WSSED models for training jobs, predictions, and strong labels
"""

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Float, Text, JSON, Enum as SQLEnum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum

from app.database import Base


class TrainingStatus(str, enum.Enum):
    PENDING = "PENDING"
    TRAINING = "TRAINING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class FeedbackType(str, enum.Enum):
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"


class LabelType(str, enum.Enum):
    STRONG_POSITIVE = "strong_positive"
    STRONG_NEGATIVE = "strong_negative"


class WSSEDTrainingJob(Base):
    __tablename__ = "wssed_training_jobs"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True)
    model_name = Column(String, nullable=False)
    hyperparameters = Column(JSON, nullable=False)
    status = Column(SQLEnum(TrainingStatus), nullable=False, default=TrainingStatus.PENDING, index=True)
    model_path = Column(String, nullable=True)
    training_metrics = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    dataset = relationship("Dataset", back_populates="wssed_training_jobs")
    predictions = relationship("WSSEDPrediction", back_populates="training_job", cascade="all, delete-orphan")


class WSSEDPrediction(Base):
    __tablename__ = "wssed_predictions"

    id = Column(Integer, primary_key=True, index=True)
    training_job_id = Column(Integer, ForeignKey("wssed_training_jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    recording_id = Column(Integer, ForeignKey("recordings.id", ondelete="CASCADE"), nullable=False, index=True)
    species_name = Column(String, nullable=False, index=True)
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    frame_probabilities = Column(JSON, nullable=True)
    user_feedback = Column(SQLEnum(FeedbackType), nullable=True, index=True)
    feedback_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    training_job = relationship("WSSEDTrainingJob", back_populates="predictions")
    recording = relationship("Recording", back_populates="wssed_predictions")
    strong_label = relationship("WSSEDStrongLabel", back_populates="prediction", uselist=False, cascade="all, delete-orphan")


class WSSEDStrongLabel(Base):
    __tablename__ = "wssed_strong_labels"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("wssed_predictions.id", ondelete="CASCADE"), nullable=False, unique=True)
    recording_id = Column(Integer, ForeignKey("recordings.id", ondelete="CASCADE"), nullable=False, index=True)
    species_name = Column(String, nullable=False, index=True)
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    label_type = Column(SQLEnum(LabelType), nullable=False, default=LabelType.STRONG_POSITIVE, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    prediction = relationship("WSSEDPrediction", back_populates="strong_label")
    recording = relationship("Recording", back_populates="wssed_strong_labels")
