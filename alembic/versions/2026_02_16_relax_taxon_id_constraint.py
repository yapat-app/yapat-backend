"""relax annotations taxon_id check constraint and add unique (snippet_id, taxon_id)

Revision ID: 2026_02_16_taxon_constraint
Revises: 2026_01_30_custom_tax
Create Date: 2026-02-16

- Allow namespace:key where key is alphanumeric (e.g. local:vesperis_iridescentis).
- Add unique constraint on (snippet_id, taxon_id) to prevent duplicate annotations.
"""
from alembic import op


revision = "2026_02_16_taxon_constraint"
down_revision = "2026_01_30_custom_tax"
branch_labels = None
depends_on = None


def upgrade():
    # Drop existing check constraint and add relaxed pattern
    op.drop_constraint("valid_taxon_id_format", "annotations", type_="check")
    op.create_check_constraint(
        "valid_taxon_id_format",
        "annotations",
        "taxon_id ~ '^([a-z]+:[a-zA-Z0-9_-]+|custom:[a-f0-9-]+)$'",
    )
    # Remove duplicate annotations (keep one per snippet_id + taxon_id, keep lowest id)
    op.execute("""
        DELETE FROM annotations a
        USING annotations b
        WHERE a.snippet_id = b.snippet_id AND a.taxon_id = b.taxon_id AND a.id > b.id
    """)
    # Prevent duplicate annotations per snippet (one annotation per taxon_id per snippet)
    op.create_unique_constraint(
        "uq_annotations_snippet_taxon",
        "annotations",
        ["snippet_id", "taxon_id"],
    )


def downgrade():
    op.drop_constraint("uq_annotations_snippet_taxon", "annotations", type_="unique")
    op.drop_constraint("valid_taxon_id_format", "annotations", type_="check")
    op.create_check_constraint(
        "valid_taxon_id_format",
        "annotations",
        "taxon_id ~ '^([a-z]+:[0-9]+|custom:[a-f0-9-]+)$'",
    )
