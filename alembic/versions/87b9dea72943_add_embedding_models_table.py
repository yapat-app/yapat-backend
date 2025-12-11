"""add_embedding_models_table

Revision ID: 87b9dea72943
Revises: a35b50bd39de
Create Date: 2025-12-11

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '87b9dea72943'
down_revision = 'a35b50bd39de'
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    
    # Create embedding_models table using raw SQL
    conn.execute(sa.text("""
        CREATE TABLE IF NOT EXISTS embedding_models (
            id SERIAL PRIMARY KEY,
            name VARCHAR NOT NULL,
            version VARCHAR,
            description VARCHAR,
            source_uri VARCHAR,
            window_size DOUBLE PRECISION NOT NULL,
            step_size DOUBLE PRECISION NOT NULL,
            overlap DOUBLE PRECISION NOT NULL,
            requires_fixed_window INTEGER DEFAULT 1,
            requires_fixed_step INTEGER DEFAULT 1,
            requires_fixed_overlap INTEGER DEFAULT 1,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
        )
    """))
    op.create_index(op.f('ix_embedding_models_id'), 'embedding_models', ['id'], unique=False)
    
    # Optionally seed with BirdNET model from config
    # This is optional since we're using config-based models, but good to have in DB too
    conn.execute(sa.text("""
        INSERT INTO embedding_models (id, name, version, description, window_size, step_size, overlap, requires_fixed_window, requires_fixed_step, requires_fixed_overlap)
        SELECT 1, 'birdnet', '2.4', 'BirdNET v2.4 - 3-second windows at 48 kHz, 1024-dim embedding', 3.0, 1.0, 0.0, 1, 1, 1
        WHERE NOT EXISTS (SELECT 1 FROM embedding_models WHERE id = 1)
    """))


def downgrade() -> None:
    # Drop embedding_models table
    op.drop_index(op.f('ix_embedding_models_id'), table_name='embedding_models')
    op.drop_table('embedding_models')

