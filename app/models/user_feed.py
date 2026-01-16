"""
User feed snapshots
"""

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON, Index
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from app.database import Base


class UserFeed(Base):
    __tablename__ = "user_feeds"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    method = Column(String, nullable=False)
    request_params = Column(JSON, nullable=True)
    response = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    user = relationship("User", backref="feeds")


# Composite index to efficiently fetch latest feeds per user/method
Index("ix_user_feeds_user_method_created_at", UserFeed.user_id, UserFeed.method, UserFeed.created_at.desc())

