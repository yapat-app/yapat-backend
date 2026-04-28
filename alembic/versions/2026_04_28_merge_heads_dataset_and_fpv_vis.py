"""merge heads: dataset team FK + fpv vis projection

Revision ID: 2026_04_28_merge_heads_dataset_and_fpv_vis
Revises: 2026_03_30_dataset_team_fk_set_null, 6f1c2a9bd3e4
Create Date: 2026-04-28

This is a merge-only revision created after merging branches that introduced
independent schema changes.
"""

from alembic import op


# revision identifiers, used by Alembic.
revision = "2026_04_28_merge_heads_dataset_and_fpv_vis"
down_revision = ("2026_03_30_dataset_team_fk_set_null", "6f1c2a9bd3e4")
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass

