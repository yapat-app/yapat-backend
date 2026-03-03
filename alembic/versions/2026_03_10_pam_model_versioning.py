"""Add is_base and parent_checkpoint_id to pam_model_checkpoints

Revision ID: 2026_03_10_pam_model_versioning
Revises: 2026_03_02_pam_active_learning
Create Date: 2026-03-10 

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '2026_03_10_pam_model_versioning'
down_revision = '2026_03_02_pam_active_learning'
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    existing_cols = {c['name'] for c in inspector.get_columns('pam_model_checkpoints')}

    if 'is_base' not in existing_cols:
        op.add_column(
            'pam_model_checkpoints',
            sa.Column('is_base', sa.Integer(), nullable=False, server_default='0'),
        )

    if 'parent_checkpoint_id' not in existing_cols:
        op.add_column(
            'pam_model_checkpoints',
            sa.Column('parent_checkpoint_id', sa.Integer(), nullable=True),
        )
        op.create_foreign_key(
            'fk_pam_ckpt_parent',
            'pam_model_checkpoints',
            'pam_model_checkpoints',
            ['parent_checkpoint_id'],
            ['id'],
            ondelete='SET NULL',
        )
        op.create_index(
            'ix_pam_model_checkpoints_parent_id',
            'pam_model_checkpoints',
            ['parent_checkpoint_id'],
        )


def downgrade() -> None:
    op.drop_index('ix_pam_model_checkpoints_parent_id', table_name='pam_model_checkpoints')
    op.drop_constraint('fk_pam_ckpt_parent', 'pam_model_checkpoints', type_='foreignkey')
    op.drop_column('pam_model_checkpoints', 'parent_checkpoint_id')
    op.drop_column('pam_model_checkpoints', 'is_base')
