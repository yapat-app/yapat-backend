"""
Review queue for labels shared between snippet sets.

When a label cannot be copied to an overlapping clip of another snippet set
automatically (the clips only partly overlap, the label may not hold for the
whole target clip, or the target already has a contradicting label), a
candidate is stored here for a person to listen to and accept, reject or
relabel. See app/services/label_sharing.py for the rules.
"""

from sqlalchemy import (
    Column, DateTime, Float, ForeignKey, Index, Integer, String, UniqueConstraint,
)
from sqlalchemy.sql import func

from app.database import Base


class LabelShareStatus:
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    RELABELLED = "relabelled"

    ALL = (PENDING, ACCEPTED, REJECTED, RELABELLED)


class LabelShareReason:
    # The clips overlap, but neither contains the other.
    PARTIAL_OVERLAP = "partial_overlap"
    # A species label on a longer clip: the call may be outside the shorter one.
    TARGET_SHORTER = "target_shorter"
    # A no-event label on a shorter clip: the rest of the longer one may have a call.
    TARGET_LONGER = "target_longer"
    # The target already has a label of the opposite kind (species vs no event).
    CONFLICT = "conflict"
    # A recheck, not a suggestion: the label this clip's reviewed label came from
    # was deleted. The label stays unless someone removes it here.
    SOURCE_REMOVED = "source_removed"


class LabelShareCandidate(Base):
    __tablename__ = "label_share_candidates"

    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)
    # None for a recheck: its source label no longer exists.
    source_annotation_id = Column(
        Integer, ForeignKey("annotations.id", ondelete="CASCADE"), nullable=True, index=True
    )
    source_snippet_id = Column(Integer, ForeignKey("snippets.id", ondelete="CASCADE"), nullable=False)
    target_snippet_id = Column(
        Integer, ForeignKey("snippets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    target_snippet_set_id = Column(
        Integer, ForeignKey("snippet_sets.id", ondelete="CASCADE"), nullable=False
    )

    taxon_id = Column(String(255), nullable=False)
    resolved_name_snapshot = Column(String(255), nullable=False)
    # The label string used in al_snippet_annotation (training and the hub).
    al_label = Column(String, nullable=False)
    reason = Column(String(32), nullable=False)
    overlap_seconds = Column(Float, nullable=False)

    status = Column(String(16), nullable=False, default=LabelShareStatus.PENDING,
                    server_default=LabelShareStatus.PENDING)
    reviewed_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    reviewed_at = Column(DateTime(timezone=True), nullable=True)
    # The label the review created on the target clip (accept or relabel).
    result_annotation_id = Column(
        Integer, ForeignKey("annotations.id", ondelete="SET NULL"), nullable=True
    )
    # A recheck: the reviewed label on the target clip to keep or remove.
    target_annotation_id = Column(
        Integer, ForeignKey("annotations.id", ondelete="SET NULL"), nullable=True, index=True
    )
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint("source_annotation_id", "target_snippet_id", name="uq_label_share_source_target"),
        Index("ix_label_share_queue", "dataset_id", "status", "target_snippet_set_id"),
    )
