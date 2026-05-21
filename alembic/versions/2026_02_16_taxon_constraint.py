"""Taxon constraint (placeholder for missing revision)

Revision ID: 2026_02_16_taxon_constraint
Revises: 2026_02_13_active_learning
Create Date: 2026-02-16

This revision exists so Alembic can resolve the chain when the DB was stamped
with 2026_02_16_taxon_constraint. No schema changes; taxon constraint is
handled in the annotation model.
"""
from alembic import op

revision = '2026_02_16_taxon_constraint'
down_revision = '2026_02_13_active_learning'
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
