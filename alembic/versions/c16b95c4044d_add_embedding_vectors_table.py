"""add embedding_vectors table

Revision ID: c16b95c4044d
Revises: 9a53fe5f5c9d
Create Date: 2025-12-11 16:11:04.047218

"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = 'c16b95c4044d'
down_revision = '9a53fe5f5c9d'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'embedding_vectors',
        sa.Column('id', sa.Integer(), primary_key=True),

        sa.Column('snippet_id', sa.Integer(),
                  sa.ForeignKey('snippets.id', ondelete="CASCADE"),
                  unique=True,
                  nullable=False),

        sa.Column('embedding_job_id', sa.Integer(),
                  sa.ForeignKey('embedding_jobs.id', ondelete="CASCADE"),
                  nullable=False),

        sa.Column('embedding_model_id', sa.Integer(),
                  sa.ForeignKey('embedding_models.id', ondelete="CASCADE"),
                  nullable=False),

        sa.Column('dim', sa.Integer(), nullable=False),

        sa.Column('vector', sa.ARRAY(sa.Float()), nullable=False),

        sa.Column('created_at', sa.DateTime(timezone=True),
                  server_default=sa.text('now()')),
    )
    op.create_index('ix_embedding_vectors_snippet_id', 'embedding_vectors', ['snippet_id'])
    op.create_index('ix_embedding_vectors_embedding_model_id', 'embedding_vectors', ['embedding_model_id'])
    op.create_index('ix_embedding_vectors_embedding_job_id', 'embedding_vectors', ['embedding_job_id'])


def downgrade() -> None:
    pass
