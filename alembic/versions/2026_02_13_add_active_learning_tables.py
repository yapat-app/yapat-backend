"""Add Active Learning tables for species models and snippet labels

Revision ID: 2026_02_13_active_learning
Revises: 2026_02_10_wssed
Create Date: 2026-02-13 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '2026_02_13_active_learning'
down_revision = '2026_02_10_wssed'
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    existing_tables = inspector.get_table_names()
    
    # Get enum types for column definitions (these were created in previous migration)
    training_status_enum = postgresql.ENUM(
        'PENDING', 'TRAINING', 'COMPLETED', 'FAILED',
        name='training_status_enum',
        create_type=False
    )
    feedback_enum = postgresql.ENUM(
        'ACCEPTED', 'REJECTED',
        name='feedback_enum',
        create_type=False
    )
    
    # Create wssed_species_models table
    if 'wssed_species_models' not in existing_tables:
        op.create_table(
            'wssed_species_models',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('species_name', sa.String(), nullable=False),
            sa.Column('dataset_id', sa.Integer(), nullable=False),
            sa.Column('model_directory', sa.String(), nullable=False),
            sa.Column('metric_type', sa.String(), nullable=False, server_default='macro'),
            sa.Column('prediction_level', sa.String(), nullable=False, server_default='segment'),
            sa.Column('model_version', sa.String(), nullable=True),
            sa.Column('hyperparameters', sa.JSON(), nullable=True),
            sa.Column('status', training_status_enum, nullable=False, server_default='COMPLETED'),
            sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
            sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
            sa.ForeignKeyConstraint(['dataset_id'], ['datasets.id'], ondelete='CASCADE'),
            sa.PrimaryKeyConstraint('id'),
            sa.UniqueConstraint('species_name', 'dataset_id', name='uq_species_model')
        )
        op.create_index(op.f('ix_wssed_species_models_id'), 'wssed_species_models', ['id'], unique=False)
        op.create_index(op.f('ix_wssed_species_models_species_name'), 'wssed_species_models', ['species_name'], unique=False)
        op.create_index(op.f('ix_wssed_species_models_dataset_id'), 'wssed_species_models', ['dataset_id'], unique=False)
    
    # Create wssed_snippet_labels table
    if 'wssed_snippet_labels' not in existing_tables:
        op.create_table(
            'wssed_snippet_labels',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('species_model_id', sa.Integer(), nullable=False),
            sa.Column('snippet_id', sa.Integer(), nullable=False),
            sa.Column('predicted_label', sa.Float(), nullable=False),
            sa.Column('confidence_score', sa.Float(), nullable=True),
            sa.Column('user_label', feedback_enum, nullable=True),
            sa.Column('labeled_at', sa.DateTime(timezone=True), nullable=True),
            sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
            sa.ForeignKeyConstraint(['species_model_id'], ['wssed_species_models.id'], ondelete='CASCADE'),
            sa.ForeignKeyConstraint(['snippet_id'], ['snippets.id'], ondelete='CASCADE'),
            sa.PrimaryKeyConstraint('id'),
            sa.UniqueConstraint('species_model_id', 'snippet_id', name='uq_species_snippet_label')
        )
        op.create_index(op.f('ix_wssed_snippet_labels_id'), 'wssed_snippet_labels', ['id'], unique=False)
        op.create_index(op.f('ix_wssed_snippet_labels_species_model_id'), 'wssed_snippet_labels', ['species_model_id'], unique=False)
        op.create_index(op.f('ix_wssed_snippet_labels_snippet_id'), 'wssed_snippet_labels', ['snippet_id'], unique=False)
        op.create_index(op.f('ix_wssed_snippet_labels_user_label'), 'wssed_snippet_labels', ['user_label'], unique=False)


def downgrade() -> None:
    # Drop tables in reverse order
    op.drop_table('wssed_snippet_labels')
    op.drop_table('wssed_species_models')
