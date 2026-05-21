"""refactor al_feedback_events to snippet-level feedback

Revision ID: 41e48fb512d0
Revises: 3a855f8ffec2
Create Date: 2026-03-26

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "41e48fb512d0"
down_revision = "3a855f8ffec2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. add new nullable columns first
    op.add_column("al_feedback_events", sa.Column("dataset_id", sa.Integer(), nullable=True))
    op.add_column("al_feedback_events", sa.Column("model_checkpoint_id", sa.Integer(), nullable=True))
    op.add_column("al_feedback_events", sa.Column("snippet_id", sa.Integer(), nullable=True))
    op.add_column("al_feedback_events", sa.Column("final_labels", sa.JSON(), nullable=True))

    op.create_index(op.f("ix_al_feedback_events_dataset_id"), "al_feedback_events", ["dataset_id"], unique=False)
    op.create_index(op.f("ix_al_feedback_events_model_checkpoint_id"), "al_feedback_events", ["model_checkpoint_id"], unique=False)
    op.create_index(op.f("ix_al_feedback_events_snippet_id"), "al_feedback_events", ["snippet_id"], unique=False)

    op.create_foreign_key(
        "fk_al_feedback_events_dataset_id",
        "al_feedback_events",
        "datasets",
        ["dataset_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "fk_al_feedback_events_model_checkpoint_id",
        "al_feedback_events",
        "al_model_checkpoints",
        ["model_checkpoint_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "fk_al_feedback_events_snippet_id",
        "al_feedback_events",
        "snippets",
        ["snippet_id"],
        ["id"],
        ondelete="CASCADE",
    )

    # 2. backfill new columns from old prediction_id
    op.execute("""
        UPDATE al_feedback_events AS f
        SET
            model_checkpoint_id = p.model_checkpoint_id,
            snippet_id = p.snippet_id,
            dataset_id = c.dataset_id,
            final_labels = f.modified_labels
        FROM al_predictions AS p
        JOIN al_model_checkpoints AS c
          ON c.id = p.model_checkpoint_id
        WHERE f.prediction_id = p.id
    """)

    # 3. only now make them NOT NULL
    op.alter_column("al_feedback_events", "dataset_id", nullable=False)
    op.alter_column("al_feedback_events", "model_checkpoint_id", nullable=False)
    op.alter_column("al_feedback_events", "snippet_id", nullable=False)

    # 4. drop old FK / index / columns
    op.drop_constraint("al_feedback_events_prediction_id_fkey", "al_feedback_events", type_="foreignkey")
    op.drop_index("ix_al_feedback_events_prediction_id", table_name="al_feedback_events")
    op.drop_column("al_feedback_events", "prediction_id")
    op.drop_column("al_feedback_events", "modified_labels")


def downgrade() -> None:
    # 1. add old columns back as nullable first
    op.add_column("al_feedback_events", sa.Column("prediction_id", sa.Integer(), nullable=True))
    op.add_column("al_feedback_events", sa.Column("modified_labels", sa.JSON(), nullable=True))

    op.create_index("ix_al_feedback_events_prediction_id", "al_feedback_events", ["prediction_id"], unique=False)
    op.create_foreign_key(
        "al_feedback_events_prediction_id_fkey",
        "al_feedback_events",
        "al_predictions",
        ["prediction_id"],
        ["id"],
        ondelete="CASCADE",
    )

    # 2. best-effort backfill old columns from new fields
    # Note: this assumes one prediction row per (checkpoint, snippet). If there are multiple,
    # Postgres may choose one arbitrarily. That is acceptable for downgrade fallback.
    op.execute("""
        UPDATE al_feedback_events AS f
        SET
            prediction_id = p.id,
            modified_labels = f.final_labels
        FROM al_predictions AS p
        WHERE p.model_checkpoint_id = f.model_checkpoint_id
          AND p.snippet_id = f.snippet_id
    """)

    # 3. remove new FKs / indexes / columns
    op.drop_constraint("fk_al_feedback_events_snippet_id", "al_feedback_events", type_="foreignkey")
    op.drop_constraint("fk_al_feedback_events_model_checkpoint_id", "al_feedback_events", type_="foreignkey")
    op.drop_constraint("fk_al_feedback_events_dataset_id", "al_feedback_events", type_="foreignkey")

    op.drop_index(op.f("ix_al_feedback_events_snippet_id"), table_name="al_feedback_events")
    op.drop_index(op.f("ix_al_feedback_events_model_checkpoint_id"), table_name="al_feedback_events")
    op.drop_index(op.f("ix_al_feedback_events_dataset_id"), table_name="al_feedback_events")

    op.drop_column("al_feedback_events", "final_labels")
    op.drop_column("al_feedback_events", "snippet_id")
    op.drop_column("al_feedback_events", "model_checkpoint_id")
    op.drop_column("al_feedback_events", "dataset_id")