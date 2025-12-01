"""Remove annotation status tracking

Revision ID: h9i0j1k2l3m4
Revises: g8h9i0j1k2l3
Create Date: 2025-01-20 20:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'h9i0j1k2l3m4'
down_revision = 'g8h9i0j1k2l3'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Drop foreign key constraint on reviewed_by
    op.drop_constraint('annotations_reviewed_by_fkey', 'annotations', type_='foreignkey')
    
    # Drop status tracking columns
    op.drop_column('annotations', 'reviewed_at')
    op.drop_column('annotations', 'reviewed_by')
    op.drop_column('annotations', 'is_reviewed')


def downgrade() -> None:
    # Add back status tracking columns
    op.add_column('annotations', sa.Column('is_reviewed', sa.Boolean(), nullable=False, server_default='false'))
    op.add_column('annotations', sa.Column('reviewed_by', sa.Integer(), nullable=True))
    op.add_column('annotations', sa.Column('reviewed_at', sa.DateTime(timezone=True), nullable=True))
    
    # Recreate foreign key constraint
    op.create_foreign_key('annotations_reviewed_by_fkey', 'annotations', 'users', ['reviewed_by'], ['id'], ondelete='SET NULL')

