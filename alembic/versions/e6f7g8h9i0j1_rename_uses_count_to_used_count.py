"""Rename uses_count to used_count

Revision ID: e6f7g8h9i0j1
Revises: d5e6f7g8h9i0
Create Date: 2025-11-20 17:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'e6f7g8h9i0j1'
down_revision = 'd5e6f7g8h9i0'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Rename uses_count column to used_count
    op.alter_column('invitation_links', 'uses_count',
                    new_column_name='used_count',
                    existing_type=sa.Integer(),
                    existing_nullable=False,
                    existing_server_default=sa.text('0'))


def downgrade() -> None:
    # Revert used_count back to uses_count
    op.alter_column('invitation_links', 'used_count',
                    new_column_name='uses_count',
                    existing_type=sa.Integer(),
                    existing_nullable=False,
                    existing_server_default=sa.text('0'))

