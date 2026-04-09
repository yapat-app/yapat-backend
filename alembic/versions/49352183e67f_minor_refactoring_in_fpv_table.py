"""minor refactoring in fpv table

Revision ID: 49352183e67f
Revises: f2f92a837226
Create Date: 2026-04-08 13:41:59.221389

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '49352183e67f'
down_revision = 'f2f92a837226'
branch_labels = None
depends_on = None


def upgrade():
    op.rename_table("al_vis", "fpv_vis")

    op.execute(
        "ALTER INDEX IF EXISTS ix_al_vis_id RENAME TO ix_fpv_vis_id"
    )
    op.execute(
        "ALTER INDEX IF EXISTS ix_al_vis_dataset_id RENAME TO ix_fpv_vis_dataset_id"
    )
    op.execute(
        "ALTER INDEX IF EXISTS ix_al_vis_model_checkpoint_id RENAME TO ix_fpv_vis_model_checkpoint_id"
    )
    op.execute(
        "ALTER INDEX IF EXISTS ix_al_vis_snippet_id RENAME TO ix_fpv_vis_snippet_id"
    )

    op.execute(
        "ALTER TABLE fpv_vis RENAME CONSTRAINT uq_al_vis_checkpoint_snippet TO uq_fpv_vis_checkpoint_snippet"
    )


def downgrade():
    op.execute(
        "ALTER TABLE fpv_vis RENAME CONSTRAINT uq_fpv_vis_checkpoint_snippet TO uq_al_vis_checkpoint_snippet"
    )

    op.execute(
        "ALTER INDEX IF EXISTS ix_fpv_vis_id RENAME TO ix_al_vis_id"
    )
    op.execute(
        "ALTER INDEX IF EXISTS ix_fpv_vis_dataset_id RENAME TO ix_al_vis_dataset_id"
    )
    op.execute(
        "ALTER INDEX IF EXISTS ix_fpv_vis_model_checkpoint_id RENAME TO ix_al_vis_model_checkpoint_id"
    )
    op.execute(
        "ALTER INDEX IF EXISTS ix_fpv_vis_snippet_id RENAME TO ix_al_vis_snippet_id"
    )

    op.rename_table("fpv_vis", "al_vis")


