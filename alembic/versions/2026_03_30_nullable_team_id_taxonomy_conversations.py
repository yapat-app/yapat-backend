"""make taxonomy_conversations.team_id nullable for teamless users

Revision ID: 2026_03_30_nullable_team_id
Revises: 2026_02_16_taxon_constraint
Create Date: 2026-03-30

Allow users without team membership to start conversations and create labels.
team_id becomes optional; personal conversations (no team) are owned by the creator.
"""
from alembic import op
import sqlalchemy as sa


revision = "2026_03_30_nullable_team_id"
down_revision = "2026_02_16_taxon_constraint"
branch_labels = None
depends_on = None


def upgrade():
    # Drop the existing NOT NULL constraint and FK (CASCADE delete), re-add as nullable with SET NULL
    op.drop_constraint("taxonomy_conversations_team_id_fkey", "taxonomy_conversations", type_="foreignkey")
    op.alter_column("taxonomy_conversations", "team_id", nullable=True)
    op.create_foreign_key(
        "taxonomy_conversations_team_id_fkey",
        "taxonomy_conversations",
        "teams",
        ["team_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade():
    # Restore NOT NULL + CASCADE - will fail if there are NULL rows
    op.drop_constraint("taxonomy_conversations_team_id_fkey", "taxonomy_conversations", type_="foreignkey")
    op.alter_column("taxonomy_conversations", "team_id", nullable=False)
    op.create_foreign_key(
        "taxonomy_conversations_team_id_fkey",
        "taxonomy_conversations",
        "teams",
        ["team_id"],
        ["id"],
        ondelete="CASCADE",
    )
