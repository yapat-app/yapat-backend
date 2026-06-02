"""Add composite index on al_predictions (model_checkpoint_id, composite_score)

Speeds up get_top_prediction_suggestions queries which ORDER BY composite_score DESC
filtered by model_checkpoint_id on every "Generate Feed" call.

Revision ID: 2026_05_29_al_predictions_ckpt_score_idx
Revises: 2026_05_22_merge_heads
Create Date: 2026-05-29
"""

revision = "2026_05_29_al_predictions_ckpt_score_idx"
down_revision = "2026_05_22_merge_heads"
branch_labels = None
depends_on = None

from alembic import op


def upgrade() -> None:
    op.create_index(
        "ix_al_predictions_ckpt_score",
        "al_predictions",
        ["model_checkpoint_id", "composite_score"],
    )


def downgrade() -> None:
    op.drop_index("ix_al_predictions_ckpt_score", table_name="al_predictions")
