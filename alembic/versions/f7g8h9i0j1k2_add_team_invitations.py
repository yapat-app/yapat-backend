"""Add team invitations

Revision ID: f7g8h9i0j1k2
Revises: e6f7g8h9i0j1
Create Date: 2025-11-20 18:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = 'f7g8h9i0j1k2'
down_revision = 'e6f7g8h9i0j1'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create team_invitations table
    # Note: teamrole enum already exists from initial migration, so we use create_type=False
    teamrole_enum = postgresql.ENUM('OWNER', 'USER', name='teamrole', create_type=False)
    
    op.create_table('team_invitations',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('team_id', sa.Integer(), nullable=False),
        sa.Column('invited_by', sa.Integer(), nullable=True),
        sa.Column('token', sa.String(), nullable=False),
        sa.Column('email', sa.String(), nullable=True),
        sa.Column('target_role', teamrole_enum, nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('used_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('used_by', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['team_id'], ['teams.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['invited_by'], ['users.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['used_by'], ['users.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_team_invitations_id'), 'team_invitations', ['id'], unique=False)
    op.create_index(op.f('ix_team_invitations_token'), 'team_invitations', ['token'], unique=True)


def downgrade() -> None:
    # Drop team_invitations table
    op.drop_index(op.f('ix_team_invitations_token'), table_name='team_invitations')
    op.drop_index(op.f('ix_team_invitations_id'), table_name='team_invitations')
    op.drop_table('team_invitations')
    # Drop the enum type if it doesn't exist elsewhere
    # Note: We don't drop the enum as it's used by team_memberships table

