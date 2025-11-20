"""Update team invitations to support multiple uses

Revision ID: g8h9i0j1k2l3
Revises: f7g8h9i0j1k2
Create Date: 2025-11-20 19:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'g8h9i0j1k2l3'
down_revision = 'f7g8h9i0j1k2'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add max_uses and used_count columns
    op.add_column('team_invitations', sa.Column('max_uses', sa.Integer(), nullable=True))
    op.add_column('team_invitations', sa.Column('used_count', sa.Integer(), nullable=False, server_default='0'))
    
    # Migrate existing data: if used_at is set, set used_count to 1, otherwise 0
    op.execute("""
        UPDATE team_invitations 
        SET used_count = CASE 
            WHEN used_at IS NOT NULL THEN 1 
            ELSE 0 
        END
    """)
    
    # Drop old columns (used_at, used_by)
    op.drop_constraint('team_invitations_used_by_fkey', 'team_invitations', type_='foreignkey')
    op.drop_column('team_invitations', 'used_at')
    op.drop_column('team_invitations', 'used_by')


def downgrade() -> None:
    # Add back old columns
    op.add_column('team_invitations', sa.Column('used_at', sa.DateTime(timezone=True), nullable=True))
    op.add_column('team_invitations', sa.Column('used_by', sa.Integer(), nullable=True))
    op.create_foreign_key('team_invitations_used_by_fkey', 'team_invitations', 'users', ['used_by'], ['id'], ondelete='SET NULL')
    
    # Migrate data back: if used_count > 0, set used_at to created_at (approximation)
    op.execute("""
        UPDATE team_invitations 
        SET used_at = created_at 
        WHERE used_count > 0
    """)
    
    # Drop new columns
    op.drop_column('team_invitations', 'used_count')
    op.drop_column('team_invitations', 'max_uses')

