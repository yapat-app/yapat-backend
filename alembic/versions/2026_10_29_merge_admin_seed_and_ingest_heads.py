"""merge default admin seeding and recording ingest heads

The default admin seed and the add-recordings ingest work were developed on
separate branches and both branched from 2026_09_22_al_pred_composite_idx.
This revision joins them so that `alembic upgrade head` has a single target.
It changes no schema.

Revision ID: 2026_10_29_merge_heads
Revises: 2026_10_07_seed_default_admin, 2026_10_14_embedding_failures
Create Date: 2026-10-29
"""

revision = "2026_10_29_merge_heads"
down_revision = ("2026_10_07_seed_default_admin", "2026_10_14_embedding_failures")
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
