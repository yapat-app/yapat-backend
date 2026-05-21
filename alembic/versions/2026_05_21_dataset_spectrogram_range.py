"""Add dataset-level spectrogram frequency display range

Revision ID: 2026_05_21_spec_range
Revises: 2026_05_13_wssed_progress
Create Date: 2026-05-21

"""
from alembic import op
import sqlalchemy as sa

revision = "2026_05_21_spec_range"
down_revision = "2026_05_13_wssed_model_paths"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "datasets",
        sa.Column("spectrogram_f_min_hz", sa.Float(), nullable=True),
    )
    op.add_column(
        "datasets",
        sa.Column("spectrogram_f_max_hz", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("datasets", "spectrogram_f_max_hz")
    op.drop_column("datasets", "spectrogram_f_min_hz")
