"""change datasets.team_id FK from CASCADE to SET NULL to preserve datasets on team deletion

Revision ID: 2026_03_30_dataset_team_fk_set_null
Revises: 2026_03_30_nullable_team_id
Create Date: 2026-03-30

Datasets should be preserved (unassigned) when their team is deleted.
Previously the FK used ON DELETE CASCADE which silently deleted datasets.
"""
from alembic import op
import sqlalchemy as sa


revision = "2026_03_30_dataset_team_fk_set_null"
down_revision = "2026_03_30_nullable_team_id"
branch_labels = None
depends_on = None


def upgrade():
    op.drop_constraint("datasets_team_id_fkey", "datasets", type_="foreignkey")
    op.create_foreign_key(
        "datasets_team_id_fkey",
        "datasets",
        "teams",
        ["team_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade():
    op.drop_constraint("datasets_team_id_fkey", "datasets", type_="foreignkey")
    op.create_foreign_key(
        "datasets_team_id_fkey",
        "datasets",
        "teams",
        ["team_id"],
        ["id"],
        ondelete="CASCADE",
    )
