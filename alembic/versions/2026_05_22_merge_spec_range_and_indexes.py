"""merge heads: spectrogram range + snippet/recording indexes

Revision ID: 2026_05_22_merge_heads
Revises: b5dd26a8d59c, 2026_05_21_spec_range
Create Date: 2026-05-22

Merge-only revision. Required when the DB branch includes
indexes migration (b5dd26a8d59c) and main adds dataset spectrogram columns.
"""

from alembic import op

revision = "2026_05_22_merge_heads"
down_revision = ("b5dd26a8d59c", "2026_05_21_spec_range")
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
