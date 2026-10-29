"""
Dataset reference link model.

Links a "reference" dataset (a normal Dataset with is_reference=True --
ingested through the usual scan/snippet/embed pipeline, but never surfaced
in annotation/labelling views) to the target dataset or team that should
draw on it as supplementary training data for PAM active learning.

At training time, a dataset's effective reference pool is the union of:
  - links where dataset_id == this dataset's id (dataset-scoped)
  - links where team_id == this dataset's team_id (team-scoped, shared by
    every dataset under that team)

See docs/reference-data-pool-design.md for the full design.
"""

from sqlalchemy import (
    Column, Integer, DateTime, ForeignKey, UniqueConstraint, CheckConstraint, event
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database import Base


# Exactly one of dataset_id / team_id must be set. Postgres-only CHECK,
# mirrors the SQLite-skip pattern used for annotation.py's valid_taxon_constraint.
one_scope_constraint = CheckConstraint(
    "(dataset_id IS NOT NULL AND team_id IS NULL) OR "
    "(dataset_id IS NULL AND team_id IS NOT NULL)",
    name="ck_dataset_reference_links_one_scope",
)


@event.listens_for(Base.metadata, "before_create")
def _remove_scope_constraint_if_sqlite(target, connection, **kw):
    if connection.dialect.name == "sqlite":
        table = DatasetReferenceLink.__table__
        if one_scope_constraint in table.constraints:
            table.constraints.remove(one_scope_constraint)


class DatasetReferenceLink(Base):
    __tablename__ = "dataset_reference_links"

    id = Column(Integer, primary_key=True, index=True)

    reference_dataset_id = Column(
        Integer,
        ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Exactly one of these two is set -- see one_scope_constraint above.
    dataset_id = Column(
        Integer,
        ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    team_id = Column(
        Integer,
        ForeignKey("teams.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    reference_dataset = relationship(
        "Dataset", foreign_keys=[reference_dataset_id], back_populates="referenced_by_links"
    )
    dataset = relationship(
        "Dataset", foreign_keys=[dataset_id], back_populates="reference_links"
    )
    team = relationship("Team", foreign_keys=[team_id], back_populates="reference_links")

    __table_args__ = (
        one_scope_constraint,
        # NULLs are distinct under Postgres uniqueness, so each constraint
        # below only "bites" for the rows where its nullable column is set --
        # exactly the scope it's meant to guard.
        UniqueConstraint("reference_dataset_id", "dataset_id", name="uq_ref_link_dataset"),
        UniqueConstraint("reference_dataset_id", "team_id", name="uq_ref_link_team"),
    )
