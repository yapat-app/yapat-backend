"""add user_feeds table

Revision ID: 2026_01_15_user_feeds
Revises: 2026_01_12_pgvector
Create Date: 2026-01-15

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "2026_01_15_user_feeds"
down_revision = "2026_01_12_pgvector"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "user_feeds",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("method", sa.String(), nullable=False),
        sa.Column("request_params", sa.JSON(), nullable=True),
        sa.Column("response", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_index(
        "ix_user_feeds_user_method_created_at",
        "user_feeds",
        ["user_id", "method", "created_at"],
    )


def downgrade():
    op.drop_index("ix_user_feeds_user_method_created_at", table_name="user_feeds")
    op.drop_table("user_feeds")

