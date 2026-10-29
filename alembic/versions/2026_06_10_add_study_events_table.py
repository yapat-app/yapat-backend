"""Add study_events table for user-study interaction logging

Revision ID: 2026_06_10_study_events
Revises: 2026_05_29_al_predictions_ckpt_score_idx
Create Date: 2026-06-10
"""

from alembic import op
import sqlalchemy as sa

revision = "2026_06_10_study_events"
down_revision = "2026_05_29_al_predictions_ckpt_score_idx"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    if "study_events" in inspector.get_table_names():
        return

    op.create_table(
        "study_events",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("phase_id", sa.String(length=32), nullable=True),
        sa.Column("dataset_id", sa.Integer(), nullable=True),
        sa.Column("snippet_set_id", sa.Integer(), nullable=True),
        sa.Column("snippet_id", sa.Integer(), nullable=True),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("client_ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_study_events_id", "study_events", ["id"])
    op.create_index("ix_study_events_session_id", "study_events", ["session_id"])
    op.create_index("ix_study_events_user_id", "study_events", ["user_id"])
    op.create_index("ix_study_events_phase_id", "study_events", ["phase_id"])
    op.create_index("ix_study_events_event_type", "study_events", ["event_type"])
    op.create_index(
        "ix_study_events_session_ts", "study_events", ["session_id", "client_ts"]
    )
    op.create_index(
        "ix_study_events_user_ts", "study_events", ["user_id", "client_ts"]
    )


def downgrade() -> None:
    op.drop_index("ix_study_events_user_ts", table_name="study_events")
    op.drop_index("ix_study_events_session_ts", table_name="study_events")
    op.drop_index("ix_study_events_event_type", table_name="study_events")
    op.drop_index("ix_study_events_phase_id", table_name="study_events")
    op.drop_index("ix_study_events_user_id", table_name="study_events")
    op.drop_index("ix_study_events_session_id", table_name="study_events")
    op.drop_index("ix_study_events_id", table_name="study_events")
    op.drop_table("study_events")
