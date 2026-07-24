"""reference data pool: datasets.is_reference + dataset_reference_links

Revision ID: d3e1f0a9b7c4
Revises: a0c672de4d81
Create Date: 2026-07-24 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'd3e1f0a9b7c4'
down_revision = 'a0c672de4d81'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'datasets',
        sa.Column('is_reference', sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.create_index(op.f('ix_datasets_is_reference'), 'datasets', ['is_reference'], unique=False)
    op.create_table(
        'dataset_reference_links',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('reference_dataset_id', sa.Integer(), nullable=False),
        sa.Column('dataset_id', sa.Integer(), nullable=True),
        sa.Column('team_id', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['reference_dataset_id'], ['datasets.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['dataset_id'], ['datasets.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['team_id'], ['teams.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('reference_dataset_id', 'dataset_id', name='uq_ref_link_dataset'),
        sa.UniqueConstraint('reference_dataset_id', 'team_id', name='uq_ref_link_team'),
        sa.CheckConstraint(
            "(dataset_id IS NOT NULL AND team_id IS NULL) OR "
            "(dataset_id IS NULL AND team_id IS NOT NULL)",
            name='ck_dataset_reference_links_one_scope',
        ),
    )
    op.create_index(
        op.f('ix_dataset_reference_links_id'), 'dataset_reference_links', ['id'], unique=False
    )
    op.create_index(
        op.f('ix_dataset_reference_links_reference_dataset_id'),
        'dataset_reference_links', ['reference_dataset_id'], unique=False,
    )
    op.create_index(
        op.f('ix_dataset_reference_links_dataset_id'),
        'dataset_reference_links', ['dataset_id'], unique=False,
    )
    op.create_index(
        op.f('ix_dataset_reference_links_team_id'),
        'dataset_reference_links', ['team_id'], unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f('ix_dataset_reference_links_team_id'), table_name='dataset_reference_links')
    op.drop_index(op.f('ix_dataset_reference_links_dataset_id'), table_name='dataset_reference_links')
    op.drop_index(op.f('ix_dataset_reference_links_reference_dataset_id'), table_name='dataset_reference_links')
    op.drop_index(op.f('ix_dataset_reference_links_id'), table_name='dataset_reference_links')
    op.drop_table('dataset_reference_links')

    op.drop_index(op.f('ix_datasets_is_reference'), table_name='datasets')
    op.drop_column('datasets', 'is_reference')
