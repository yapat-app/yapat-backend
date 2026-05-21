"""Add updated_at to wssed_training_jobs

Revision ID: 2026_05_11_wssed_upd
Revises: 2026_04_28_merge
Create Date: 2026-05-11 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

revision = '2026_05_11_wssed_upd'
down_revision = '2026_04_28_merge_heads_dataset_and_fpv_vis'
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    result = conn.execute(sa.text(
        "SELECT EXISTS (SELECT 1 FROM information_schema.columns "
        "WHERE table_name = 'wssed_training_jobs' AND column_name = 'updated_at')"
    )).scalar()
    if not result:
        op.add_column(
            'wssed_training_jobs',
            sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        )


def downgrade() -> None:
    conn = op.get_bind()
    result = conn.execute(sa.text(
        "SELECT EXISTS (SELECT 1 FROM information_schema.columns "
        "WHERE table_name = 'wssed_training_jobs' AND column_name = 'updated_at')"
    )).scalar()
    if result:
        op.drop_column('wssed_training_jobs', 'updated_at')
