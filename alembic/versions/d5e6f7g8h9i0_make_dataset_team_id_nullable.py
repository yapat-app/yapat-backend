"""Make dataset team_id nullable and add source_uri

Revision ID: d5e6f7g8h9i0
Revises: c4d5e6f7g8h9
Create Date: 2025-11-20 16:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'd5e6f7g8h9i0'
down_revision = 'c4d5e6f7g8h9'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Make team_id nullable in datasets table
    op.alter_column('datasets', 'team_id',
                    existing_type=sa.Integer(),
                    nullable=True)
    
    # Add source_uri column to datasets table
    op.add_column('datasets', sa.Column('source_uri', sa.String(), nullable=True))


def downgrade() -> None:
    # Remove source_uri column
    op.drop_column('datasets', 'source_uri')
    
    # Revert team_id to not nullable
    # Note: This will fail if there are datasets with null team_id
    op.alter_column('datasets', 'team_id',
                    existing_type=sa.Integer(),
                    nullable=False)

