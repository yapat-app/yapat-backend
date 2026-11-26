"""Recheck items for reviewed labels whose source label was deleted

- label_share_candidates.source_annotation_id may be empty: a recheck keeps no
  link to the deleted label.
- label_share_candidates.target_annotation_id names the reviewed label to keep
  or remove.

Revision ID: 2026_11_24_label_share_rechecks
Revises: 2026_11_13_label_sharing
Create Date: 2026-11-24
"""

import sqlalchemy as sa
from alembic import op

revision = "2026_11_24_label_share_rechecks"
down_revision = "2026_11_13_label_sharing"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column("label_share_candidates", "source_annotation_id", existing_type=sa.Integer(), nullable=True)
    op.add_column("label_share_candidates", sa.Column("target_annotation_id", sa.Integer(), nullable=True))
    op.create_foreign_key(
        "fk_label_share_candidates_target_annotation",
        "label_share_candidates",
        "annotations",
        ["target_annotation_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index(
        "ix_label_share_candidates_target_annotation_id", "label_share_candidates", ["target_annotation_id"]
    )


def downgrade() -> None:
    op.execute("DELETE FROM label_share_candidates WHERE source_annotation_id IS NULL")
    op.drop_index("ix_label_share_candidates_target_annotation_id", table_name="label_share_candidates")
    op.drop_constraint("fk_label_share_candidates_target_annotation", "label_share_candidates", type_="foreignkey")
    op.drop_column("label_share_candidates", "target_annotation_id")
    op.alter_column("label_share_candidates", "source_annotation_id", existing_type=sa.Integer(), nullable=False)
