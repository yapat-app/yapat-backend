"""removed redundant columns from ALVis as that info is already present in other tables

Revision ID: f2f92a837226
Revises: 076da4e6bfaf
Create Date: 2026-04-02 14:13:07.486627

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'f2f92a837226'
down_revision = '076da4e6bfaf'
branch_labels = None
depends_on = None


def upgrade():
    # Remove redundant columns from al_vis
    op.drop_column("al_vis", "latent_embedding")
    op.drop_column("al_vis", "uncertainty")
    op.drop_column("al_vis", "diversity")
    op.drop_column("al_vis", "density")
    op.drop_column("al_vis", "composite_score")
    op.drop_column("al_vis", "model_predicted_labels")
    op.drop_column("al_vis", "model_predicted_probabilities")
    op.drop_column("al_vis", "trusted_labels")


def downgrade():
    # Recreate columns if rollback is needed

    op.add_column("al_vis", sa.Column("latent_embedding", sa.JSON(), nullable=True))
    op.add_column("al_vis", sa.Column("uncertainty", sa.Float(), nullable=True))
    op.add_column("al_vis", sa.Column("diversity", sa.Float(), nullable=True))
    op.add_column("al_vis", sa.Column("density", sa.Float(), nullable=True))
    op.add_column("al_vis", sa.Column("composite_score", sa.Float(), nullable=True))
    op.add_column("al_vis", sa.Column("model_predicted_labels", sa.JSON(), nullable=True))
    op.add_column("al_vis", sa.Column("model_predicted_probabilities", sa.JSON(), nullable=True))
    op.add_column("al_vis", sa.Column("trusted_labels", sa.JSON(), nullable=True))
