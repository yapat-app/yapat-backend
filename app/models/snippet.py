"""
Snippet model
"""

from sqlalchemy import Column, Integer, DateTime, ForeignKey, Float
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database import Base


class Snippet(Base):
    __tablename__ = "snippets"

    id = Column(Integer, primary_key=True, index=True)

    recording_id = Column(
        Integer,
        ForeignKey("recordings.id", ondelete="CASCADE"),
        nullable=False,
    )

    snippet_set_id = Column(
        Integer,
        ForeignKey("snippet_sets.id", ondelete="CASCADE"),
        nullable=False,
    )

    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)
    duration = Column(Float, nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    recording = relationship("Recording", back_populates="snippets")
    snippet_set = relationship("SnippetSet", back_populates="snippets")

    annotations = relationship(
        "Annotation",
        back_populates="snippet",
        cascade="all, delete-orphan",
    )
    
    # WSSED active learning snippet labels
    wssed_snippet_labels = relationship(
        "WSSEDSnippetLabel",
        back_populates="snippet",
        cascade="all, delete-orphan",
    )
    # PAM active learning predictions
    pam_predictions = relationship(
        "PAMPrediction",
        back_populates="snippet",
        cascade="all, delete-orphan",
    )