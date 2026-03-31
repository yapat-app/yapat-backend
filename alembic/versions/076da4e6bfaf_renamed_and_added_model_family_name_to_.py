from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "076da4e6bfaf"
down_revision = "41e48fb512d0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "al_model_family_state",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("dataset_id", sa.Integer(), nullable=False),
        sa.Column("model_family_name", sa.String(), nullable=False),
        sa.Column("active_model_checkpoint_id", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["active_model_checkpoint_id"], ["al_model_checkpoints.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("dataset_id", "model_family_name", name="uq_al_model_family_state"),
    )
    op.create_index(
        op.f("ix_al_model_family_state_active_model_checkpoint_id"),
        "al_model_family_state",
        ["active_model_checkpoint_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_al_model_family_state_dataset_id"),
        "al_model_family_state",
        ["dataset_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_al_model_family_state_id"),
        "al_model_family_state",
        ["id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_al_model_family_state_model_family_name"),
        "al_model_family_state",
        ["model_family_name"],
        unique=False,
    )

    op.drop_constraint("uq_al_checkpoint", "al_model_checkpoints", type_="unique")
    op.alter_column(
        "al_model_checkpoints",
        "name",
        new_column_name="model_family_name",
        existing_type=sa.String(),
        existing_nullable=False,
    )
    op.create_unique_constraint(
        "uq_al_checkpoint",
        "al_model_checkpoints",
        ["dataset_id", "model_family_name", "version"],
    )


def downgrade() -> None:
    op.drop_constraint("uq_al_checkpoint", "al_model_checkpoints", type_="unique")
    op.alter_column(
        "al_model_checkpoints",
        "model_family_name",
        new_column_name="name",
        existing_type=sa.String(),
        existing_nullable=False,
    )
    op.create_unique_constraint(
        "uq_al_checkpoint",
        "al_model_checkpoints",
        ["dataset_id", "name", "version"],
    )

    op.drop_index(op.f("ix_al_model_family_state_model_family_name"), table_name="al_model_family_state")
    op.drop_index(op.f("ix_al_model_family_state_id"), table_name="al_model_family_state")
    op.drop_index(op.f("ix_al_model_family_state_dataset_id"), table_name="al_model_family_state")
    op.drop_index(op.f("ix_al_model_family_state_active_model_checkpoint_id"), table_name="al_model_family_state")
    op.drop_table("al_model_family_state")