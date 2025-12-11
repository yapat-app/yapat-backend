"""add_foreign_keys_to_embedding_models

Revision ID: 13e73601d7de
Revises: 87b9dea72943
Create Date: 2025-12-11 

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '13e73601d7de'
down_revision = '87b9dea72943'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add foreign key from snippet_sets to embedding_models
    op.create_foreign_key(
        'snippet_sets_embedding_model_id_fkey',
        'snippet_sets',
        'embedding_models',
        ['embedding_model_id'],
        ['id'],
        ondelete='CASCADE'
    )
    
    # Add foreign key from embedding_jobs to embedding_models
    op.create_foreign_key(
        'embedding_jobs_embedding_model_id_fkey',
        'embedding_jobs',
        'embedding_models',
        ['embedding_model_id'],
        ['id'],
        ondelete='CASCADE'
    )


def downgrade() -> None:
    # Drop foreign keys
    op.drop_constraint('embedding_jobs_embedding_model_id_fkey', 'embedding_jobs', type_='foreignkey')
    op.drop_constraint('snippet_sets_embedding_model_id_fkey', 'snippet_sets', type_='foreignkey')

