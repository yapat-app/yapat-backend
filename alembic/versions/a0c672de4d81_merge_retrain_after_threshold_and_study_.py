"""merge retrain_after_threshold and study_events/quick_labels heads

Revision ID: a0c672de4d81
Revises: 2026_06_17_merge_heads, b2c3d4e5f6a7
Create Date: 2026-07-14 16:21:31.670222

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'a0c672de4d81'
down_revision = ('2026_06_17_merge_heads', 'b2c3d4e5f6a7')
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass

