"""add_snippet_sets_and_embedding_jobs_tables

Revision ID: a35b50bd39de
Revises: j2k3l4m5n6o7
Create Date: 2025-12-10

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


# revision identifiers, used by Alembic.
revision = 'a35b50bd39de'
down_revision = 'j2k3l4m5n6o7'
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    
    # Check if enums exist, create only if they don't (using DO block to handle gracefully)
    conn.execute(sa.text("""
        DO $$ BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'snippetsetstatus') THEN
                CREATE TYPE snippetsetstatus AS ENUM ('pending', 'ready', 'failed');
            END IF;
        END $$;
    """))
    
    conn.execute(sa.text("""
        DO $$ BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'embeddingjobstatus') THEN
                CREATE TYPE embeddingjobstatus AS ENUM ('pending', 'running', 'completed', 'failed');
            END IF;
        END $$;
    """))
    
    # Create snippet_sets table using raw SQL to avoid enum creation issues
    conn.execute(sa.text("""
        CREATE TABLE IF NOT EXISTS snippet_sets (
            id SERIAL PRIMARY KEY,
            dataset_id INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
            embedding_model_id INTEGER NOT NULL,
            window_size DOUBLE PRECISION NOT NULL,
            step_size DOUBLE PRECISION NOT NULL,
            overlap DOUBLE PRECISION NOT NULL,
            status snippetsetstatus NOT NULL DEFAULT 'pending',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
        )
    """))
    op.create_index(op.f('ix_snippet_sets_id'), 'snippet_sets', ['id'], unique=False)
    
    # Create embedding_jobs table using raw SQL
    conn.execute(sa.text("""
        CREATE TABLE IF NOT EXISTS embedding_jobs (
            id SERIAL PRIMARY KEY,
            dataset_id INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
            embedding_model_id INTEGER NOT NULL,
            snippet_set_id INTEGER NOT NULL REFERENCES snippet_sets(id) ON DELETE CASCADE,
            status embeddingjobstatus NOT NULL DEFAULT 'pending',
            celery_task_id VARCHAR,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
            started_at TIMESTAMP WITH TIME ZONE,
            completed_at TIMESTAMP WITH TIME ZONE,
            error_message TEXT
        )
    """))
    op.create_index(op.f('ix_embedding_jobs_id'), 'embedding_jobs', ['id'], unique=False)
    
    # Update snippets table to add snippet_set_id if it doesn't exist
    inspector = inspect(conn)
    snippets_columns = [col['name'] for col in inspector.get_columns('snippets')]
    
    if 'snippet_set_id' not in snippets_columns:
        op.add_column('snippets', sa.Column('snippet_set_id', sa.Integer(), nullable=True))
        op.create_foreign_key('snippets_snippet_set_id_fkey', 'snippets', 'snippet_sets', ['snippet_set_id'], ['id'], ondelete='CASCADE')
        op.create_index(op.f('ix_snippets_snippet_set_id'), 'snippets', ['snippet_set_id'], unique=False)


def downgrade() -> None:
    # Drop foreign key and column from snippets table
    op.drop_index(op.f('ix_snippets_snippet_set_id'), table_name='snippets')
    op.drop_constraint('snippets_snippet_set_id_fkey', 'snippets', type_='foreignkey')
    op.drop_column('snippets', 'snippet_set_id')
    
    # Drop embedding_jobs table
    op.drop_index(op.f('ix_embedding_jobs_id'), table_name='embedding_jobs')
    op.drop_table('embedding_jobs')
    
    # Drop snippet_sets table
    op.drop_index(op.f('ix_snippet_sets_id'), table_name='snippet_sets')
    op.drop_table('snippet_sets')
    
    # Drop enums (only if no other tables use them)
    conn = op.get_bind()
    try:
        conn.execute(sa.text("DROP TYPE IF EXISTS embeddingjobstatus"))
    except Exception:
        pass
    try:
        conn.execute(sa.text("DROP TYPE IF EXISTS snippetsetstatus"))
    except Exception:
        pass
    conn.commit()

