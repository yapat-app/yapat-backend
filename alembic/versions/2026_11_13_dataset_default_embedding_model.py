"""datasets.default_embedding_model_id: the hub's embedding model per dataset

Set by an admin or team owner on the dataset card. Null keeps the old rule:
the model that embedded the dataset's default snippet set first.

Revision ID: 2026_11_13_dataset_default_model
Revises: 2026_11_04_vector_indexes
Create Date: 2026-11-13
"""

import sqlalchemy as sa
from alembic import op

revision = "2026_11_13_dataset_default_model"
down_revision = "2026_11_04_vector_indexes"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("datasets", sa.Column("default_embedding_model_id", sa.Integer(), nullable=True))
    op.create_foreign_key(
        "fk_datasets_default_embedding_model",
        "datasets",
        "embedding_models",
        ["default_embedding_model_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint("fk_datasets_default_embedding_model", "datasets", type_="foreignkey")
    op.drop_column("datasets", "default_embedding_model_id")
