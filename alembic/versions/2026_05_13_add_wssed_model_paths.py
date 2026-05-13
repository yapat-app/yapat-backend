"""Add model_paths JSON to wssed_training_jobs

Revision ID: 2026_05_13_wssed_model_paths
Revises: 2026_05_13_wssed_progress
Create Date: 2026-05-13 14:40:00.000000

"""
from alembic import op
import sqlalchemy as sa

revision = "2026_05_13_wssed_model_paths"
down_revision = "2026_05_13_wssed_progress"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    exists = conn.execute(sa.text(
        "SELECT EXISTS (SELECT 1 FROM information_schema.columns "
        "WHERE table_name = 'wssed_training_jobs' AND column_name = 'model_paths')"
    )).scalar()
    if not exists:
        op.add_column(
            "wssed_training_jobs",
            sa.Column("model_paths", sa.JSON(), nullable=True),
        )


def downgrade() -> None:
    conn = op.get_bind()
    exists = conn.execute(sa.text(
        "SELECT EXISTS (SELECT 1 FROM information_schema.columns "
        "WHERE table_name = 'wssed_training_jobs' AND column_name = 'model_paths')"
    )).scalar()
    if exists:
        op.drop_column("wssed_training_jobs", "model_paths")
