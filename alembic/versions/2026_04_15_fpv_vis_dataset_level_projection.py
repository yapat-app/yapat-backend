"""Support dataset-level FPV projections in fpv_vis

Revision ID: 6f1c2a9bd3e4
Revises: 3293bc64fb83
Create Date: 2026-04-15
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "6f1c2a9bd3e4"
down_revision = "3293bc64fb83"
branch_labels = None
depends_on = None


def upgrade():
    # 1) Allow model_checkpoint_id to be NULL (dataset-level projections)
    op.alter_column("fpv_vis", "model_checkpoint_id", existing_type=sa.Integer(), nullable=True)

    # 2) Add embedding_model_id to identify embedding space for dataset-level projections
    op.add_column("fpv_vis", sa.Column("embedding_model_id", sa.Integer(), nullable=True))
    op.create_foreign_key(
        "fk_fpv_vis_embedding_model_id_embedding_models",
        "fpv_vis",
        "embedding_models",
        ["embedding_model_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_index("ix_fpv_vis_embedding_model_id", "fpv_vis", ["embedding_model_id"])

    # 3) Replace the old uniqueness constraint with partial unique indexes.
    #    Postgres treats NULLs as distinct for UNIQUE constraints, so we must enforce
    #    uniqueness separately for checkpoint rows and dataset-level (NULL checkpoint) rows.
    op.drop_constraint("uq_fpv_vis_checkpoint_snippet", "fpv_vis", type_="unique")

    op.create_index(
        "uq_fpv_vis_checkpoint_snippet_nonnull",
        "fpv_vis",
        ["model_checkpoint_id", "snippet_id"],
        unique=True,
        postgresql_where=sa.text("model_checkpoint_id IS NOT NULL"),
    )
    op.create_index(
        "uq_fpv_vis_embedding_model_snippet_null_ckpt",
        "fpv_vis",
        ["embedding_model_id", "snippet_id"],
        unique=True,
        postgresql_where=sa.text("model_checkpoint_id IS NULL"),
    )


def downgrade():
    op.drop_index("uq_fpv_vis_embedding_model_snippet_null_ckpt", table_name="fpv_vis")
    op.drop_index("uq_fpv_vis_checkpoint_snippet_nonnull", table_name="fpv_vis")

    op.create_unique_constraint(
        "uq_fpv_vis_checkpoint_snippet",
        "fpv_vis",
        ["model_checkpoint_id", "snippet_id"],
    )

    op.drop_index("ix_fpv_vis_embedding_model_id", table_name="fpv_vis")
    op.drop_constraint("fk_fpv_vis_embedding_model_id_embedding_models", "fpv_vis", type_="foreignkey")
    op.drop_column("fpv_vis", "embedding_model_id")

    op.alter_column("fpv_vis", "model_checkpoint_id", existing_type=sa.Integer(), nullable=False)

