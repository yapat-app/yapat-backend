"""add default_snippet_set_id to datasets

Revision ID: add_default_snippet_set
Revises: c16b95c4044d
Create Date: 2025-01-07 

"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

# revision identifiers, used by Alembic.
revision = 'add_default_snippet_set'
down_revision = 'c16b95c4044d'
branch_labels = None
depends_on = None


def upgrade():
    # Check if column already exists (in case it was added in initial schema)
    conn = op.get_bind()
    inspector = inspect(conn)
    columns = [col['name'] for col in inspector.get_columns('datasets')]
    
    if 'default_snippet_set_id' not in columns:
        # Add the column
        op.add_column(
            'datasets',
            sa.Column(
                'default_snippet_set_id',
                sa.Integer(),
                nullable=True
            )
        )
        
        # Add foreign key constraint
        op.create_foreign_key(
            'fk_datasets_default_snippet_set_id',
            'datasets', 'snippet_sets',
            ['default_snippet_set_id'], ['id'],
            ondelete='SET NULL'
        )
        
        # Add index
        op.create_index(
            'ix_datasets_default_snippet_set_id',
            'datasets',
            ['default_snippet_set_id']
        )
    else:
        # Column exists, but check if index and FK exist
        indexes = [idx['name'] for idx in inspector.get_indexes('datasets')]
        if 'ix_datasets_default_snippet_set_id' not in indexes:
            op.create_index(
                'ix_datasets_default_snippet_set_id',
                'datasets',
                ['default_snippet_set_id']
            )
        
        # Check foreign keys
        fks = [fk['name'] for fk in inspector.get_foreign_keys('datasets')]
        if 'fk_datasets_default_snippet_set_id' not in fks:
            op.create_foreign_key(
                'fk_datasets_default_snippet_set_id',
                'datasets', 'snippet_sets',
                ['default_snippet_set_id'], ['id'],
                ondelete='SET NULL'
            )


def downgrade():
    conn = op.get_bind()
    inspector = inspect(conn)
    columns = [col['name'] for col in inspector.get_columns('datasets')]
    
    if 'default_snippet_set_id' in columns:
        op.drop_index('ix_datasets_default_snippet_set_id', table_name='datasets')
        op.drop_constraint('fk_datasets_default_snippet_set_id', 'datasets', type_='foreignkey')
        op.drop_column('datasets', 'default_snippet_set_id')
