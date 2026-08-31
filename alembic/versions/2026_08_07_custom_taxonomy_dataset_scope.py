"""Make custom_taxonomies dataset-scoped

Supports admin-created, dataset-only label spaces:
  - team_id becomes nullable (admin label spaces have no team).
  - adds dataset_id (FK datasets, ON DELETE SET NULL, indexed).
  - unique naming scope changes from (team_id, name) to (team_id, dataset_id, name).

Revision ID: 2026_08_07_custom_tax_dataset_scope
Revises: 2026_08_07_is_labelspace_version
Create Date: 2026-08-07
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "2026_08_07_custom_tax_dataset_scope"
down_revision = "2026_08_07_is_labelspace_version"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # team_id -> nullable
    op.alter_column(
        "custom_taxonomies",
        "team_id",
        existing_type=sa.Integer(),
        nullable=True,
    )

    # dataset_id column + index + FK
    op.add_column(
        "custom_taxonomies",
        sa.Column("dataset_id", sa.Integer(), nullable=True),
    )
    op.create_index(
        "ix_custom_taxonomies_dataset_id",
        "custom_taxonomies",
        ["dataset_id"],
    )
    op.create_foreign_key(
        "fk_custom_taxonomies_dataset",
        source_table="custom_taxonomies",
        referent_table="datasets",
        local_cols=["dataset_id"],
        remote_cols=["id"],
        ondelete="SET NULL",
    )

    # Unique naming scope: (team_id, name) -> (team_id, dataset_id, name)
    op.drop_constraint(
        "uq_custom_taxonomy_team_name", "custom_taxonomies", type_="unique"
    )
    op.create_unique_constraint(
        "uq_custom_taxonomy_team_dataset_name",
        "custom_taxonomies",
        ["team_id", "dataset_id", "name"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_custom_taxonomy_team_dataset_name", "custom_taxonomies", type_="unique"
    )
    op.create_unique_constraint(
        "uq_custom_taxonomy_team_name",
        "custom_taxonomies",
        ["team_id", "name"],
    )

    op.drop_constraint(
        "fk_custom_taxonomies_dataset", "custom_taxonomies", type_="foreignkey"
    )
    op.drop_index("ix_custom_taxonomies_dataset_id", table_name="custom_taxonomies")
    op.drop_column("custom_taxonomies", "dataset_id")

    op.alter_column(
        "custom_taxonomies",
        "team_id",
        existing_type=sa.Integer(),
        nullable=False,
    )
