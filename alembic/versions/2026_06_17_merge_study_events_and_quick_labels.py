"""merge study_events and quick_labels heads

Revision ID: 2026_06_17_merge_heads
Revises: 2026_06_10_study_events, a1b2c3d4e5f6
Create Date: 2026-06-17
"""

from alembic import op
import sqlalchemy as sa


revision = "2026_06_17_merge_heads"
down_revision = ("2026_06_10_study_events", "a1b2c3d4e5f6")
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
