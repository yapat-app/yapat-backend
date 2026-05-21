from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import VECTOR

revision = "3a855f8ffec2"
down_revision = "00c4506074ad"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_column("al_predictions", "embedding")
    op.add_column("al_predictions", sa.Column("embedding", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("al_predictions", "embedding")
    op.add_column("al_predictions", sa.Column("embedding", VECTOR(dim=512), nullable=True))