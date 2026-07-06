"""
Taxonomy Conversation models
"""

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import sqlalchemy as sa
import enum

from app.database import Base
from app.models.types import PortableJSONB


class ConversationStatus(str, enum.Enum):
    """Status of taxonomy conversation"""
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class MessageRole(str, enum.Enum):
    """Role of message sender"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class TaxonomyConversation(Base):
    __tablename__ = "taxonomy_conversations"

    id = Column(Integer, primary_key=True, index=True)
    team_id = Column(Integer, ForeignKey("teams.id", ondelete="SET NULL"), nullable=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    custom_taxonomy_id = Column(Integer, ForeignKey("custom_taxonomies.id", ondelete="SET NULL"), nullable=True)
    status = Column(String(50), nullable=False, default=ConversationStatus.IN_PROGRESS)
    label_space = Column(PortableJSONB, nullable=True, default=list)  # Accumulated list of species/taxa
    is_frozen = Column(sa.Boolean, nullable=False, default=False)  # Whether the label space is frozen
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    team = relationship("Team", backref="taxonomy_conversations")
    user = relationship("User", backref="taxonomy_conversations")
    custom_taxonomy = relationship("CustomTaxonomy", back_populates="conversations")
    messages = relationship("TaxonomyMessage", back_populates="conversation", cascade="all, delete-orphan", order_by="TaxonomyMessage.created_at")


class TaxonomyMessage(Base):
    __tablename__ = "taxonomy_messages"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("taxonomy_conversations.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String(50), nullable=False)
    content = Column(Text, nullable=False)
    message_metadata = Column(PortableJSONB, nullable=True)  # "metadata" is reserved by SQLAlchemy Declarative API
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    conversation = relationship("TaxonomyConversation", back_populates="messages")
