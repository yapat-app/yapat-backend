"""Add invitation links

Revision ID: b3c7d8e4f6a9
Revises: 92311d3a95a4
Create Date: 2025-11-20 14:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'b3c7d8e4f6a9'
down_revision = '92311d3a95a4'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create invitation_links table
    op.create_table('invitation_links',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('token', sa.String(), nullable=False),
        sa.Column('created_by', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('max_uses', sa.Integer(), nullable=True),
        sa.Column('uses_count', sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(['created_by'], ['users.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_invitation_links_id'), 'invitation_links', ['id'], unique=False)
    op.create_index(op.f('ix_invitation_links_token'), 'invitation_links', ['token'], unique=True)
    
    # Create invitation_datasets association table
    op.create_table('invitation_datasets',
        sa.Column('invitation_id', sa.Integer(), nullable=False),
        sa.Column('dataset_id', sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(['invitation_id'], ['invitation_links.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['dataset_id'], ['datasets.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('invitation_id', 'dataset_id')
    )


def downgrade() -> None:
    # Drop invitation_datasets association table
    op.drop_table('invitation_datasets')
    
    # Drop invitation_links table
    op.drop_index(op.f('ix_invitation_links_token'), table_name='invitation_links')
    op.drop_index(op.f('ix_invitation_links_id'), table_name='invitation_links')
    op.drop_table('invitation_links')

