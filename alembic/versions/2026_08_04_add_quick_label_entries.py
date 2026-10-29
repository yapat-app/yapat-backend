"""Add quick_label_entries table for runtime-added quick labels

Backs the "GBIF pick becomes a quick label" flow. ``user_id`` NULL is the
dataset-wide lane reserved for Ontology Engineering label spaces.

Revision ID: 2026_08_04_quick_label_entries
Revises: 2026_07_24_ref_meta_path
Create Date: 2026-08-04
"""

from alembic import op
import sqlalchemy as sa

revision = "2026_08_04_quick_label_entries"
down_revision = "2026_07_24_ref_meta_path"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    if "quick_label_entries" in inspector.get_table_names():
        return

    op.create_table(
        "quick_label_entries",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("dataset_id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("taxon_id", sa.String(length=160), nullable=False),
        sa.Column("display_name", sa.String(length=255), nullable=False),
        sa.Column("rank", sa.String(length=32), nullable=True),
        sa.Column("source", sa.String(length=16), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_quick_label_entries_id", "quick_label_entries", ["id"])

    # Partial uniques: NULLs compare as distinct, so a single combined UNIQUE
    # would not stop duplicate dataset-wide rows.
    op.create_index(
        "uq_quick_label_entries_dataset_taxon",
        "quick_label_entries",
        ["dataset_id", "taxon_id"],
        unique=True,
        postgresql_where=sa.text("user_id IS NULL"),
        sqlite_where=sa.text("user_id IS NULL"),
    )
    op.create_index(
        "uq_quick_label_entries_user_dataset_taxon",
        "quick_label_entries",
        ["user_id", "dataset_id", "taxon_id"],
        unique=True,
        postgresql_where=sa.text("user_id IS NOT NULL"),
        sqlite_where=sa.text("user_id IS NOT NULL"),
    )
    op.create_index(
        "ix_quick_label_entries_scope",
        "quick_label_entries",
        ["dataset_id", "user_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_quick_label_entries_scope", table_name="quick_label_entries")
    op.drop_index(
        "uq_quick_label_entries_user_dataset_taxon", table_name="quick_label_entries"
    )
    op.drop_index(
        "uq_quick_label_entries_dataset_taxon", table_name="quick_label_entries"
    )
    op.drop_index("ix_quick_label_entries_id", table_name="quick_label_entries")
    op.drop_table("quick_label_entries")
