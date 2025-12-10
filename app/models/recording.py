"""
Recording model
"""

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Float, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database import Base


class Recording(Base):
    __tablename__ = "recordings"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)

    file_path = Column(String, nullable=False)
    file_name = Column(String, nullable=False)

    duration = Column(Float, nullable=True)  # Duration in seconds
    sample_rate = Column(Float, nullable=True)

    extra_metadata = Column(JSON, nullable=True)  # Additional metadata (location, date, etc.)

    # Explicit checksum for audio integrity verification
    audio_sha256 = Column(String, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    dataset = relationship("Dataset", back_populates="recordings")
    snippets = relationship("Snippet", back_populates="recording", cascade="all, delete-orphan")
