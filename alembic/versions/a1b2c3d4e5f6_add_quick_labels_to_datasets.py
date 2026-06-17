"""add quick_labels to datasets

Revision ID: a1b2c3d4e5f6
Revises: 2026_05_29_al_predictions_ckpt_score_idx
Create Date: 2026-06-17 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = 'a1b2c3d4e5f6'
down_revision = '2026_05_29_al_predictions_ckpt_score_idx'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('datasets', sa.Column('quick_labels', postgresql.JSONB(astext_type=sa.Text()), nullable=True))


def downgrade() -> None:
    op.drop_column('datasets', 'quick_labels')
