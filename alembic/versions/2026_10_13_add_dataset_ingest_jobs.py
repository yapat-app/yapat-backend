"""Add dataset_ingest_jobs table for incremental "add recordings" runs

One row per scan -> segment -> embed -> finalize -> score run over the new
recordings of an existing dataset (yapat-backend#70).

Revision ID: 2026_10_13_dataset_ingest_jobs
Revises: 2026_09_22_al_pred_composite_idx
Create Date: 2026-10-13
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "2026_10_13_dataset_ingest_jobs"
down_revision = "2026_09_22_al_pred_composite_idx"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    if "dataset_ingest_jobs" in inspector.get_table_names():
        return

    json_type = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")

    op.create_table(
        "dataset_ingest_jobs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("dataset_id", sa.Integer(), nullable=False),
        sa.Column("snippet_set_id", sa.Integer(), nullable=False),
        sa.Column("embedding_job_id", sa.Integer(), nullable=False),
        sa.Column("requested_by_user_id", sa.Integer(), nullable=True),
        sa.Column("source_uri", sa.String(), nullable=True),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("stage", sa.String(), nullable=False),
        sa.Column("summary", json_type, nullable=False),
        sa.Column("celery_task_id", sa.String(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["snippet_set_id"], ["snippet_sets.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["embedding_job_id"], ["embedding_jobs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["requested_by_user_id"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_dataset_ingest_jobs_id", "dataset_ingest_jobs", ["id"])
    op.create_index(
        "ix_dataset_ingest_jobs_dataset_status",
        "dataset_ingest_jobs",
        ["dataset_id", "status"],
    )


def downgrade() -> None:
    op.drop_index("ix_dataset_ingest_jobs_dataset_status", table_name="dataset_ingest_jobs")
    op.drop_index("ix_dataset_ingest_jobs_id", table_name="dataset_ingest_jobs")
    op.drop_table("dataset_ingest_jobs")
