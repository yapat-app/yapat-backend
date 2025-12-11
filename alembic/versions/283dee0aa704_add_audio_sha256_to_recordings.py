"""add_audio_sha256_to_recordings

Revision ID: 283dee0aa704
Revises: 13e73601d7de
Create Date: 2025-12-11

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '283dee0aa704'
down_revision = '13e73601d7de'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add audio_sha256 column to recordings table
    op.add_column('recordings', sa.Column('audio_sha256', sa.String(), nullable=True))


def downgrade() -> None:
    # Remove audio_sha256 column
    op.drop_column('recordings', 'audio_sha256')

