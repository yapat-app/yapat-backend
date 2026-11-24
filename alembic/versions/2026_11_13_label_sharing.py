"""Label sharing between snippet sets (issue #93, part 2)

- annotations.origin and annotations.derived_from_id mark shared labels and
  link them to the label they were copied from.
- label_share_candidates is the review queue for labels that could not be
  copied automatically.
- al_snippet_annotation gets a DERIVED source for automatically shared labels.
- snippets gets an index to find the overlapping clips of other snippet sets.

Revision ID: 2026_11_13_label_sharing
Revises: 2026_11_13_dataset_default_model
Create Date: 2026-11-13
"""

import sqlalchemy as sa
from alembic import op

revision = "2026_11_13_label_sharing"
down_revision = "2026_11_13_dataset_default_model"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "annotations",
        sa.Column("origin", sa.String(20), nullable=False, server_default="manual"),
    )
    op.add_column("annotations", sa.Column("derived_from_id", sa.Integer(), nullable=True))
    op.create_foreign_key(
        "fk_annotations_derived_from",
        "annotations",
        "annotations",
        ["derived_from_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index("ix_annotations_derived_from_id", "annotations", ["derived_from_id"])

    op.create_table(
        "label_share_candidates",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("dataset_id", sa.Integer(), sa.ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False),
        sa.Column(
            "source_annotation_id",
            sa.Integer(),
            sa.ForeignKey("annotations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("source_snippet_id", sa.Integer(), sa.ForeignKey("snippets.id", ondelete="CASCADE"), nullable=False),
        sa.Column("target_snippet_id", sa.Integer(), sa.ForeignKey("snippets.id", ondelete="CASCADE"), nullable=False),
        sa.Column(
            "target_snippet_set_id",
            sa.Integer(),
            sa.ForeignKey("snippet_sets.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("taxon_id", sa.String(255), nullable=False),
        sa.Column("resolved_name_snapshot", sa.String(255), nullable=False),
        sa.Column("al_label", sa.String(), nullable=False),
        sa.Column("reason", sa.String(32), nullable=False),
        sa.Column("overlap_seconds", sa.Float(), nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="pending"),
        sa.Column("reviewed_by", sa.Integer(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "result_annotation_id",
            sa.Integer(),
            sa.ForeignKey("annotations.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("source_annotation_id", "target_snippet_id", name="uq_label_share_source_target"),
    )
    op.create_index(
        "ix_label_share_candidates_source_annotation_id", "label_share_candidates", ["source_annotation_id"]
    )
    op.create_index(
        "ix_label_share_candidates_target_snippet_id", "label_share_candidates", ["target_snippet_id"]
    )
    op.create_index(
        "ix_label_share_queue", "label_share_candidates", ["dataset_id", "status", "target_snippet_set_id"]
    )

    # IF NOT EXISTS: a hand-built index of the same name may already be there.
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_snippets_recording_set_start "
        "ON snippets (recording_id, snippet_set_id, start_time)"
    )

    if op.get_bind().dialect.name == "postgresql":
        # SQLAlchemy stores the enum member names. ADD VALUE cannot run inside a
        # transaction block that later uses the value, so commit it on its own.
        with op.get_context().autocommit_block():
            op.execute("ALTER TYPE al_annotation_source_enum ADD VALUE IF NOT EXISTS 'DERIVED'")


def downgrade() -> None:
    # Postgres cannot drop an enum value; remove the rows that use it instead.
    op.execute("DELETE FROM al_snippet_annotation WHERE source::text = 'DERIVED'")
    op.execute("DELETE FROM annotations WHERE origin = 'auto_shared'")

    op.execute("DROP INDEX IF EXISTS ix_snippets_recording_set_start")
    op.drop_index("ix_label_share_queue", table_name="label_share_candidates")
    op.drop_index("ix_label_share_candidates_target_snippet_id", table_name="label_share_candidates")
    op.drop_index("ix_label_share_candidates_source_annotation_id", table_name="label_share_candidates")
    op.drop_table("label_share_candidates")

    op.drop_index("ix_annotations_derived_from_id", table_name="annotations")
    op.drop_constraint("fk_annotations_derived_from", "annotations", type_="foreignkey")
    op.drop_column("annotations", "derived_from_id")
    op.drop_column("annotations", "origin")
