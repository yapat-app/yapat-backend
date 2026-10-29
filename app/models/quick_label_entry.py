"""
Quick label entry model

Labels that entered a dataset's quick-label palette at runtime, rather than
from a model checkpoint's labels.json or the curated ``datasets.quick_labels``
column. Today the only producer is a participant picking a species from the
GBIF search box while annotating; the row is then pinned to the front of their
own quick-label chips so the next snippet needs no second search.

``user_id`` doubles as the scope discriminator:

* ``user_id`` set  -> a personal row, visible only to that participant.
* ``user_id`` NULL -> a dataset-wide row, visible to everyone on the dataset.

The NULL lane is what the Ontology Engineering service will write into once it
becomes dataset-scoped: an OE label space belongs to the dataset, so storing it
per participant would duplicate the same rows for every member of the study.

``taxon_id`` follows the namespaced convention that app/core/taxonomy.py already
parses (``gbif:2480932``, ``envo:01001867``, ``custom:<uuid>``, ``local:<slug>``),
so no new identifier scheme is introduced here.
"""

from sqlalchemy import (
    Column, Integer, BigInteger, String, DateTime, ForeignKey, Index, text
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database import Base


# Provenance of the row — which UI produced it. Deliberately separate from the
# ``taxon_id`` namespace, because the namespace cannot answer it: a "local:"
# label may come from a participant typing free text or from OE, and OE emits
# envo:/ols:/wiki:/local: interchangeably.
SOURCE_GBIF = "gbif"
SOURCE_OE = "oe"
SOURCE_MANUAL = "manual"
VALID_SOURCES = (SOURCE_GBIF, SOURCE_OE, SOURCE_MANUAL)

# Ordering weight applied on read. Hand-picked labels stay ahead of a bulk OE
# import, which would otherwise sit permanently in front of the handful of
# species a participant actually searched for.
SOURCE_PRIORITY = {SOURCE_GBIF: 0, SOURCE_MANUAL: 0, SOURCE_OE: 1}


class QuickLabelEntry(Base):
    __tablename__ = "quick_label_entries"

    # SQLite only auto-assigns rowids for INTEGER PRIMARY KEY, not BIGINT, so
    # inserts in the test suite would fail the NOT NULL check without this.
    id = Column(
        BigInteger().with_variant(Integer, "sqlite"), primary_key=True, index=True
    )

    dataset_id = Column(
        Integer, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False
    )

    # NULL => dataset-wide. Always stamped server-side from the authenticated
    # user for personal rows; the client never supplies it.
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=True
    )

    taxon_id = Column(String(160), nullable=False)
    display_name = Column(String(255), nullable=False)
    rank = Column(String(32), nullable=True)
    source = Column(String(16), nullable=False, default=SOURCE_GBIF)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", foreign_keys=[user_id])

    __table_args__ = (
        # Two *partial* unique indexes rather than one constraint: SQL treats
        # NULLs as distinct, so a plain UNIQUE(user_id, dataset_id, taxon_id)
        # would let duplicate dataset-wide rows through.
        Index(
            "uq_quick_label_entries_dataset_taxon",
            "dataset_id",
            "taxon_id",
            unique=True,
            sqlite_where=text("user_id IS NULL"),
            postgresql_where=text("user_id IS NULL"),
        ),
        Index(
            "uq_quick_label_entries_user_dataset_taxon",
            "user_id",
            "dataset_id",
            "taxon_id",
            unique=True,
            sqlite_where=text("user_id IS NOT NULL"),
            postgresql_where=text("user_id IS NOT NULL"),
        ),
        Index(
            "ix_quick_label_entries_scope",
            "dataset_id",
            "user_id",
            "created_at",
        ),
    )
