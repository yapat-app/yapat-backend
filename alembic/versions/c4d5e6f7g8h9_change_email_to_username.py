"""Change email to username

Revision ID: c4d5e6f7g8h9
Revises: b3c7d8e4f6a9
Create Date: 2025-11-20 15:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'c4d5e6f7g8h9'
down_revision = 'b3c7d8e4f6a9'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Drop the old email index
    op.drop_index('ix_users_email', table_name='users')
    
    # Rename email column to username using raw SQL
    op.execute('ALTER TABLE users RENAME COLUMN email TO username')
    
    # Create new username index
    op.create_index('ix_users_username', 'users', ['username'], unique=True)


def downgrade() -> None:
    # Drop the username index
    op.drop_index('ix_users_username', table_name='users')
    
    # Revert username back to email using raw SQL
    op.execute('ALTER TABLE users RENAME COLUMN username TO email')
    
    # Recreate email index
    op.create_index('ix_users_email', 'users', ['email'], unique=True)

