"""model al_feedback_events model checkpoint nullable

Revision ID: 79932a6b9c63
Revises: 49352183e67f
Create Date: 2026-04-13 08:00:58.415753

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy import inspect


# revision identifiers, used by Alembic.
revision = '79932a6b9c63'
down_revision = '49352183e67f'
branch_labels = None
depends_on = None


def _get_fk_name(table_name: str, column_name: str, referred_table: str) -> str | None:
    bind = op.get_bind()
    inspector = inspect(bind)

    for fk in inspector.get_foreign_keys(table_name):
        constrained_columns = fk.get("constrained_columns") or []
        referred = fk.get("referred_table")
        name = fk.get("name")

        if constrained_columns == [column_name] and referred == referred_table:
            return name

    return None


def upgrade():
    fk_name = _get_fk_name(
        table_name="al_feedback_events",
        column_name="model_checkpoint_id",
        referred_table="al_model_checkpoints",
    )

    if fk_name:
        op.drop_constraint(fk_name, "al_feedback_events", type_="foreignkey")

    op.alter_column(
        "al_feedback_events",
        "model_checkpoint_id",
        existing_type=sa.Integer(),
        nullable=True,
    )

    op.create_foreign_key(
        "fk_al_feedback_events_model_checkpoint_id",
        "al_feedback_events",
        "al_model_checkpoints",
        ["model_checkpoint_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade():
    fk_name = _get_fk_name(
        table_name="al_feedback_events",
        column_name="model_checkpoint_id",
        referred_table="al_model_checkpoints",
    )

    if fk_name:
        op.drop_constraint(fk_name, "al_feedback_events", type_="foreignkey")

    op.alter_column(
        "al_feedback_events",
        "model_checkpoint_id",
        existing_type=sa.Integer(),
        nullable=False,
    )

    op.create_foreign_key(
        "fk_al_feedback_events_model_checkpoint_id",
        "al_feedback_events",
        "al_model_checkpoints",
        ["model_checkpoint_id"],
        ["id"],
        ondelete="CASCADE",
    )