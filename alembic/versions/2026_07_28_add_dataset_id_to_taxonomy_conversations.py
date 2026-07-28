"""add dataset_id to taxonomy_conversations and merge heads

Associates a taxonomy conversation / label space with a dataset so that labels
added from the chat can be mirrored into the dataset's quick_labels.

Also merges the two open migration heads
(b2c3d4e5f6a7, 12fb3a4876f6) into a single head.

Revision ID: 2026_07_28_dataset_id_taxconv
Revises: b2c3d4e5f6a7, 12fb3a4876f6
Create Date: 2026-07-28
"""
from alembic import op
import sqlalchemy as sa


revision = "2026_07_28_dataset_id_taxconv"
down_revision = "b2c3d4e5f6a7"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "taxonomy_conversations",
        sa.Column("dataset_id", sa.Integer(), nullable=True),
    )
    op.create_index(
        "ix_taxonomy_conversations_dataset_id",
        "taxonomy_conversations",
        ["dataset_id"],
    )
    op.create_foreign_key(
        "taxonomy_conversations_dataset_id_fkey",
        "taxonomy_conversations",
        "datasets",
        ["dataset_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade():
    op.drop_constraint(
        "taxonomy_conversations_dataset_id_fkey",
        "taxonomy_conversations",
        type_="foreignkey",
    )
    op.drop_index(
        "ix_taxonomy_conversations_dataset_id",
        table_name="taxonomy_conversations",
    )
    op.drop_column("taxonomy_conversations", "dataset_id")
