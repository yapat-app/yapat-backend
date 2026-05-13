"""Add progress JSON to wssed_training_jobs

Revision ID: 2026_05_13_wssed_progress
Revises: 2026_05_11_wssed_upd
Create Date: 2026-05-13 10:55:00.000000

"""
from alembic import op
import sqlalchemy as sa

revision = "2026_05_13_wssed_progress"
down_revision = "2026_05_11_wssed_upd"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    exists = conn.execute(sa.text(
        "SELECT EXISTS (SELECT 1 FROM information_schema.columns "
        "WHERE table_name = 'wssed_training_jobs' AND column_name = 'progress')"
    )).scalar()
    if not exists:
        op.add_column(
            "wssed_training_jobs",
            sa.Column("progress", sa.JSON(), nullable=True),
        )


def downgrade() -> None:
    conn = op.get_bind()
    exists = conn.execute(sa.text(
        "SELECT EXISTS (SELECT 1 FROM information_schema.columns "
        "WHERE table_name = 'wssed_training_jobs' AND column_name = 'progress')"
    )).scalar()
    if exists:
        op.drop_column("wssed_training_jobs", "progress")
