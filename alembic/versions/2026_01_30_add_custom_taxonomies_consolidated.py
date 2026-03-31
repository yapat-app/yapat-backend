"""add custom taxonomies and conversations 

This migration consolidates:
- Custom taxonomies tables creation
- Annotation taxon_id pattern update
- Label space and is_frozen columns
- Message metadata column rename

Revision ID: 2026_01_30_custom_tax
Revises: 2026_01_26_annotations_index
Create Date: 2026-01-30

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = '2026_01_30_custom_tax'
down_revision = '2026_01_26_annotations_index'
branch_labels = None
depends_on = None


def upgrade():
    # Create custom_taxonomies table
    op.create_table(
        'custom_taxonomies',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('taxonomy_id', sa.String(255), nullable=False, unique=True, index=True),
        sa.Column('team_id', sa.Integer(), sa.ForeignKey('teams.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('created_by_user_id', sa.Integer(), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('taxonomy_data', postgresql.JSONB(), nullable=False),
        sa.Column('status', sa.String(50), nullable=False, default='active', server_default='active'),
        sa.Column('is_global', sa.Boolean(), nullable=False, default=False, server_default='false'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.text('now()')),
    )
    
    # Create unique constraint on team_id and name
    op.create_unique_constraint(
        'uq_custom_taxonomy_team_name',
        'custom_taxonomies',
        ['team_id', 'name']
    )
    
    # Create index on status for efficient filtering
    op.create_index('ix_custom_taxonomies_status', 'custom_taxonomies', ['status'])
    
    # Create taxonomy_conversations table (with label_space and is_frozen included)
    op.create_table(
        'taxonomy_conversations',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('team_id', sa.Integer(), sa.ForeignKey('teams.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('custom_taxonomy_id', sa.Integer(), sa.ForeignKey('custom_taxonomies.id', ondelete='SET NULL'), nullable=True),
        sa.Column('status', sa.String(50), nullable=False, default='in_progress', server_default='in_progress'),
        sa.Column('label_space', postgresql.JSONB(), nullable=True, server_default='[]'),
        sa.Column('is_frozen', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.text('now()')),
    )
    
    # Create index on status for efficient filtering
    op.create_index('ix_taxonomy_conversations_status', 'taxonomy_conversations', ['status'])
    
    # Create taxonomy_messages table (with message_metadata instead of metadata)
    op.create_table(
        'taxonomy_messages',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('conversation_id', sa.Integer(), sa.ForeignKey('taxonomy_conversations.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('role', sa.String(50), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('message_metadata', postgresql.JSONB(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    )
    
    # Create composite index for efficient conversation message retrieval
    op.create_index(
        'ix_taxonomy_messages_conversation_created',
        'taxonomy_messages',
        ['conversation_id', 'created_at']
    )
    
    # Update annotation taxon_id constraint to accept both GBIF and custom taxonomy patterns
    # Pattern: ^([a-z]+:[0-9]+|custom:[a-f0-9-]+)$
    # Accepts: gbif:123456 or custom:uuid-format
    op.drop_constraint('valid_taxon_id_format', 'annotations', type_='check')
    op.create_check_constraint(
        'valid_taxon_id_format',
        'annotations',
        "taxon_id ~ '^([a-z]+:[0-9]+|custom:[a-f0-9-]+)$'"
    )


def downgrade():
    # Drop annotation constraint (restore old one)
    op.drop_constraint('valid_taxon_id_format', 'annotations', type_='check')
    op.create_check_constraint(
        'valid_taxon_id_format',
        'annotations',
        "taxon_id ~ '^[a-z]+:[0-9]+$'"
    )
    
    # Drop tables in reverse order (respecting foreign keys)
    op.drop_index('ix_taxonomy_messages_conversation_created', table_name='taxonomy_messages')
    op.drop_table('taxonomy_messages')
    
    op.drop_index('ix_taxonomy_conversations_status', table_name='taxonomy_conversations')
    op.drop_table('taxonomy_conversations')
    
    op.drop_index('ix_custom_taxonomies_status', table_name='custom_taxonomies')
    op.drop_constraint('uq_custom_taxonomy_team_name', 'custom_taxonomies', type_='unique')
    op.drop_table('custom_taxonomies')
