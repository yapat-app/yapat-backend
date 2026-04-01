"""add is_ready flag to teams table

Revision ID: 2026_03_31_add_team_is_ready
Revises: 2026_03_30_dataset_team_fk_null
Create Date: 2026-03-31

A team is marked ready (is_ready=True) once it has at least one OWNER member.
Teams that were created before this migration are assumed to be ready if they
already have an owner membership, so we backfill accordingly.
"""
from alembic import op
import sqlalchemy as sa


revision = "2026_03_31_add_team_is_ready"
down_revision = "2026_03_30_dataset_team_fk_null"
branch_labels = None
depends_on = None


def upgrade():
    # Add the column with a server default of false
    op.add_column(
        "teams",
        sa.Column("is_ready", sa.Boolean(), nullable=False, server_default="false"),
    )

    # Backfill: mark existing teams as ready if they already have an OWNER member
    op.execute(
        """
        UPDATE teams
        SET is_ready = true
        WHERE id IN (
            SELECT DISTINCT team_id
            FROM team_memberships
            WHERE role = 'OWNER'
        )
        """
    )


def downgrade():
    op.drop_column("teams", "is_ready")
