"""drop ranking_score from pam_predictions

Revision ID: 2026_03_20_drop_ranking_score
Revises: 2026_03_10_pam_model_versioning
Create Date: 2026-03-20

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '2026_03_20_drop_ranking_score'
down_revision = '2026_03_10_pam_model_versioning'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_column('pam_predictions', 'ranking_score')


def downgrade() -> None:
    op.add_column(
        'pam_predictions',
        sa.Column('ranking_score', sa.Float(), nullable=True)
    )
