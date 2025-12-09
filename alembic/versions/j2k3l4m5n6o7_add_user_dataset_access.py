"""Add user dataset access

Revision ID: j2k3l4m5n6o7
Revises: i1j2k3l4m5n6
Create Date: 2025-12-09 

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'j2k3l4m5n6o7'
down_revision = 'i1j2k3l4m5n6'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create user_datasets association table for direct dataset access
    # (datasets granted via invitation that don't belong to any team yet)
    op.create_table('user_datasets',
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('dataset_id', sa.Integer(), nullable=False),
        sa.Column('granted_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('granted_by_invitation_id', sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['dataset_id'], ['datasets.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['granted_by_invitation_id'], ['invitation_links.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('user_id', 'dataset_id')
    )
    op.create_index(op.f('ix_user_datasets_user_id'), 'user_datasets', ['user_id'], unique=False)
    op.create_index(op.f('ix_user_datasets_dataset_id'), 'user_datasets', ['dataset_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_user_datasets_dataset_id'), table_name='user_datasets')
    op.drop_index(op.f('ix_user_datasets_user_id'), table_name='user_datasets')
    op.drop_table('user_datasets')

