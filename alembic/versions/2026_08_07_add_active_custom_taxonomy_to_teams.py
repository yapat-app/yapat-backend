"""Add active_custom_taxonomy_id to teams

Introduces a single team-wide active label-space pointer. The team owner promotes
one submitted CustomTaxonomy version to be the team's active label space; all team
members read that version.

Revision ID: 2026_08_07_team_active_labelspace
Revises: 2026_07_28_dataset_id_taxconv
Create Date: 2026-08-07
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "2026_08_07_team_active_labelspace"
down_revision = "2026_07_28_dataset_id_taxconv"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "teams",
        sa.Column("active_custom_taxonomy_id", sa.Integer(), nullable=True),
    )
    op.create_foreign_key(
        "fk_teams_active_custom_taxonomy",
        source_table="teams",
        referent_table="custom_taxonomies",
        local_cols=["active_custom_taxonomy_id"],
        remote_cols=["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint("fk_teams_active_custom_taxonomy", "teams", type_="foreignkey")
    op.drop_column("teams", "active_custom_taxonomy_id")
