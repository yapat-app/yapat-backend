"""Unique dataset-scoped label codes in custom_taxonomies

AL reuses one CustomTaxonomy row per label code so that `Annotation.taxon_id`
is a stable identifier. For a dataset with no team the row is scoped by
dataset_id with team_id NULL, and `uq_custom_taxonomy_team_dataset_name` cannot
police that: SQL treats NULLs as distinct, so (NULL, 4, 'SCIALT') never
collides with itself. Two concurrent annotations of a new code would each
insert a row, reintroducing per-annotation taxon ids.

Revision ID: 2026_09_02_custom_tax_dataset_uq
Revises: 2026_08_07_custom_tax_dataset_scope
Create Date: 2026-09-02
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "2026_09_02_custom_tax_dataset_uq"
down_revision = "2026_08_07_custom_tax_dataset_scope"
branch_labels = None
depends_on = None

INDEX_NAME = "uq_custom_taxonomy_dataset_name"


def upgrade() -> None:
    bind = op.get_bind()
    # Partial unique indexes exist on both dialects this project runs on, but
    # the WHERE clause has to be passed per-dialect.
    where = sa.text("team_id IS NULL")
    kwargs = (
        {"postgresql_where": where}
        if bind.dialect.name == "postgresql"
        else {"sqlite_where": where}
    )
    op.create_index(
        INDEX_NAME, "custom_taxonomies", ["dataset_id", "name"], unique=True, **kwargs
    )


def downgrade() -> None:
    op.drop_index(INDEX_NAME, table_name="custom_taxonomies")
