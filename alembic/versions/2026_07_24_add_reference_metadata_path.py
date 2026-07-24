"""add reference metadata path to datasets

Revision ID: 2026_07_24_ref_meta_path
Revises: d3e1f0a9b7c4
Create Date: 2026-07-24 00:00:00.000000

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "2026_07_24_ref_meta_path"
down_revision = "d3e1f0a9b7c4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "datasets",
        sa.Column("reference_metadata_path", sa.String(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("datasets", "reference_metadata_path")
