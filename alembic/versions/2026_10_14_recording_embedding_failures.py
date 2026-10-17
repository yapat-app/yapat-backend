"""Track recordings whose embedding failed; add dataset_ingest_jobs.mode

recording_embedding_failures holds one row per recording left with snippets
that have no embedding after an "add recordings" run (reason, count, dismissed).
dataset_ingest_jobs.mode tells a normal run ("scan") from one that only
retries those recordings ("retry_failed").

Revision ID: 2026_10_14_embedding_failures
Revises: 2026_10_14_recordings_dataset_idx
Create Date: 2026-10-14
"""

from alembic import op
import sqlalchemy as sa

revision = "2026_10_14_embedding_failures"
down_revision = "2026_10_14_recordings_dataset_idx"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)

    job_columns = {c["name"] for c in inspector.get_columns("dataset_ingest_jobs")}
    if "mode" not in job_columns:
        op.add_column(
            "dataset_ingest_jobs",
            sa.Column("mode", sa.String(), nullable=False, server_default="scan"),
        )

    if "recording_embedding_failures" in inspector.get_table_names():
        return
    op.create_table(
        "recording_embedding_failures",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("dataset_id", sa.Integer(), nullable=False),
        sa.Column("snippet_set_id", sa.Integer(), nullable=False),
        sa.Column("recording_id", sa.Integer(), nullable=False),
        sa.Column("snippets_missing", sa.Integer(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("last_job_id", sa.Integer(), nullable=True),
        sa.Column(
            "last_attempt_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("dismissed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["snippet_set_id"], ["snippet_sets.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["recording_id"], ["recordings.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["last_job_id"], ["dataset_ingest_jobs.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("recording_id", "snippet_set_id", name="uq_recording_embedding_failure"),
    )
    op.create_index("ix_recording_embedding_failures_id", "recording_embedding_failures", ["id"])
    op.create_index(
        "ix_recording_embedding_failures_dataset",
        "recording_embedding_failures",
        ["dataset_id", "dismissed_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_recording_embedding_failures_dataset", table_name="recording_embedding_failures")
    op.drop_index("ix_recording_embedding_failures_id", table_name="recording_embedding_failures")
    op.drop_table("recording_embedding_failures")
    op.drop_column("dataset_ingest_jobs", "mode")
