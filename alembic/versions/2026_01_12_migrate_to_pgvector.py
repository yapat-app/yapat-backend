"""migrate to pgvector

Revision ID: 2026_01_12_pgvector
Revises: add_default_snippet_set
Create Date: 2026-01-12 

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '2026_01_12_pgvector'
down_revision = 'add_default_snippet_set'
branch_labels = None
depends_on = None


def upgrade():
    """
    Migrate from ARRAY(Float) to pgvector's native vector type.
    
    Steps:
    1. Enable pgvector extension
    2. Add new vector column with pgvector type
    3. Copy data from old column to new column
    4. Drop old column and rename new column
    5. Create HNSW index for fast similarity search
    """
    conn = op.get_bind()
    
    # Helper function to get current columns (avoiding inspector cache)
    def get_columns():
        result = conn.execute(sa.text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'embedding_vectors'
            ORDER BY ordinal_position
        """))
        return [row[0] for row in result]
    
    # Helper function to get current indexes
    def get_indexes():
        result = conn.execute(sa.text("""
            SELECT indexname 
            FROM pg_indexes 
            WHERE tablename = 'embedding_vectors'
        """))
        return [row[0] for row in result]
    
    # Step 1: Enable pgvector extension
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')
    
    # Step 2: Add new vector column (dimension 1024 for BirdNET) if it doesn't exist
    columns = get_columns()
    if 'vector_new' not in columns:
        op.execute('''
            ALTER TABLE embedding_vectors 
            ADD COLUMN vector_new vector(1024)
        ''')
    
    # Step 3: Copy data from ARRAY to vector type
    # PostgreSQL arrays use { } format, pgvector needs [ ] format
    # Convert: {0.1,0.2,...} -> [0.1,0.2,...] -> vector
    columns = get_columns()  # Refresh to get current state
    if 'vector' in columns and 'vector_new' in columns:
        # Only update rows where vector_new is NULL (in case of partial migration)
        op.execute('''
            UPDATE embedding_vectors 
            SET vector_new = (
                '[' || array_to_string(vector, ',') || ']'
            )::vector(1024)
            WHERE vector_new IS NULL
        ''')
    
    # Step 4: Drop old column and rename new one (only if both exist)
    columns = get_columns()  # Refresh again
    if 'vector' in columns and 'vector_new' in columns:
        op.drop_column('embedding_vectors', 'vector')
        op.execute('''
            ALTER TABLE embedding_vectors 
            RENAME COLUMN vector_new TO vector
        ''')
    
    # Step 5: Add NOT NULL constraint
    columns = get_columns()  # Refresh after rename
    vector_col_name = 'vector' if 'vector' in columns else 'vector_new'
    
    if vector_col_name in columns:
        op.execute(f'''
            ALTER TABLE embedding_vectors 
            ALTER COLUMN {vector_col_name} SET NOT NULL
        ''')
    
    # Step 6: Create HNSW index for fast cosine similarity search (if it doesn't exist)
    indexes = get_indexes()
    if 'embedding_vectors_vector_cosine_idx' not in indexes:
        # Verify column type before creating index
        result = conn.execute(sa.text("""
            SELECT t.typname 
            FROM pg_attribute a
            JOIN pg_class c ON a.attrelid = c.oid
            JOIN pg_type t ON a.atttypid = t.oid
            WHERE c.relname = 'embedding_vectors'
            AND a.attname = :col_name
        """), {'col_name': vector_col_name})
        row = result.fetchone()
        
        # Only create index if column type is 'vector' (pgvector type)
        if row and row[0] == 'vector':
            op.execute(f'''
                CREATE INDEX embedding_vectors_vector_cosine_idx 
                ON embedding_vectors 
                USING hnsw ({vector_col_name} vector_cosine_ops)
            ''')
        else:
            raise Exception(
                f"Cannot create index: column '{vector_col_name}' has type '{row[0] if row else 'unknown'}' "
                f"but needs 'vector' type. The drop/rename step may have failed. "
                f"Current columns: {get_columns()}"
            )



def downgrade():
    """
    Revert back to ARRAY(Float) from pgvector.
    """
    
    # Drop HNSW index
    op.execute('DROP INDEX IF EXISTS embedding_vectors_vector_cosine_idx')
    
    # Add back ARRAY column
    op.add_column('embedding_vectors', 
                  sa.Column('vector_old', sa.ARRAY(sa.Float()), nullable=True))
    
    # Copy data back
    op.execute('''
        UPDATE embedding_vectors 
        SET vector_old = vector::text::float[]
    ''')
    
    # Drop pgvector column
    op.drop_column('embedding_vectors', 'vector')
    
    # Rename old column back
    op.execute('''
        ALTER TABLE embedding_vectors 
        RENAME COLUMN vector_old TO vector
    ''')
    
    # Add NOT NULL constraint
    op.execute('''
        ALTER TABLE embedding_vectors 
        ALTER COLUMN vector SET NOT NULL
    ''')
    
