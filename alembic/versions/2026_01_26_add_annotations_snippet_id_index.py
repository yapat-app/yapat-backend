"""add index on annotations.snippet_id

Revision ID: 2026_01_26_annotations_index
Revises: 2026_01_15_user_feeds
Create Date: 2026-01-26

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '2026_01_26_annotations_index'
down_revision = '2026_01_15_user_feeds'
branch_labels = None
depends_on = None


def upgrade():
    # Add index on snippet_id for faster lookups
    # PostgreSQL creates an index on FK automatically, but explicit index ensures it exists
    # and helps with query performance when filtering by snippet_id
    op.create_index(
        'ix_annotations_snippet_id',
        'annotations',
        ['snippet_id'],
        unique=False
    )


def downgrade():
    op.drop_index('ix_annotations_snippet_id', table_name='annotations')
