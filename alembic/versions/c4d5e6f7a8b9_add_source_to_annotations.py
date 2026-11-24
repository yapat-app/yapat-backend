"""add source to annotations

Revision ID: c4d5e6f7a8b9
Revises: 2026_10_29_merge_heads
Create Date: 2026-11-17 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

revision = 'c4d5e6f7a8b9'
down_revision = '2026_10_29_merge_heads'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('annotations', sa.Column('source', sa.String(length=32), nullable=True))


def downgrade() -> None:
    op.drop_column('annotations', 'source')
