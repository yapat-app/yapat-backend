"""Add WSSED tables and dataset_type field

Revision ID: 2026_02_10_wssed
Revises: 2026_01_30_custom_tax
Create Date: 2026-02-10 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '2026_02_10_wssed'
down_revision = '2026_01_30_custom_tax'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create enum types (check if they exist first)
    conn = op.get_bind()
    
    # Check and create training_status_enum
    result = conn.execute(sa.text(
        "SELECT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'training_status_enum')"
    )).scalar()
    if not result:
        training_status_enum = postgresql.ENUM(
            'PENDING', 'TRAINING', 'COMPLETED', 'FAILED',
            name='training_status_enum',
            create_type=True
        )
        training_status_enum.create(conn)
    
    # Check and create feedback_enum
    result = conn.execute(sa.text(
        "SELECT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'feedback_enum')"
    )).scalar()
    if not result:
        feedback_enum = postgresql.ENUM(
            'ACCEPTED', 'REJECTED',
            name='feedback_enum',
            create_type=True
        )
        feedback_enum.create(conn)
    
    # Check and create dataset_type_enum
    result = conn.execute(sa.text(
        "SELECT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'dataset_type_enum')"
    )).scalar()
    if not result:
        dataset_type_enum = postgresql.ENUM(
            'PAM', 'WEAKLY_LABELED',
            name='dataset_type_enum',
            create_type=True
        )
        dataset_type_enum.create(conn)
    
    # Check and create label_type_enum
    result = conn.execute(sa.text(
        "SELECT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'label_type_enum')"
    )).scalar()
    if not result:
        label_type_enum = postgresql.ENUM(
            'strong_positive', 'strong_negative',
            name='label_type_enum',
            create_type=True
        )
        label_type_enum.create(conn)
    
    # Get enum types for column definitions
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
    
    # Create wssed_training_jobs table (if it doesn't exist)
    inspector = sa.inspect(conn)
    existing_tables = inspector.get_table_names()
    
    if 'wssed_training_jobs' not in existing_tables:
        op.create_table(
            'wssed_training_jobs',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('dataset_id', sa.Integer(), nullable=False),
            sa.Column('model_name', sa.String(), nullable=False),
        sa.Column('hyperparameters', sa.JSON(), nullable=False),
        sa.Column('status', training_status_enum, nullable=False, server_default='PENDING'),
        sa.Column('model_path', sa.String(), nullable=True),
        sa.Column('training_metrics', sa.JSON(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['dataset_id'], ['datasets.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    # Create indexes if they don't exist
    result = conn.execute(sa.text(
        "SELECT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'ix_wssed_training_jobs_dataset_id')"
    )).scalar()
    if not result:
        op.create_index('ix_wssed_training_jobs_dataset_id', 'wssed_training_jobs', ['dataset_id'])
    
    result = conn.execute(sa.text(
        "SELECT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'ix_wssed_training_jobs_status')"
    )).scalar()
    if not result:
        op.create_index('ix_wssed_training_jobs_status', 'wssed_training_jobs', ['status'])
    
    # Create wssed_predictions table (if it doesn't exist)
    if 'wssed_predictions' not in existing_tables:
        op.create_table(
            'wssed_predictions',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('training_job_id', sa.Integer(), nullable=False),
            sa.Column('recording_id', sa.Integer(), nullable=False),
            sa.Column('species_name', sa.String(), nullable=False),
            sa.Column('start_time', sa.Float(), nullable=False),
            sa.Column('end_time', sa.Float(), nullable=False),
            sa.Column('confidence', sa.Float(), nullable=False),
            sa.Column('frame_probabilities', sa.JSON(), nullable=True),
            sa.Column('user_feedback', feedback_enum, nullable=True),
            sa.Column('feedback_at', sa.DateTime(timezone=True), nullable=True),
            sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
            sa.ForeignKeyConstraint(['training_job_id'], ['wssed_training_jobs.id'], ondelete='CASCADE'),
            sa.ForeignKeyConstraint(['recording_id'], ['recordings.id'], ondelete='CASCADE'),
            sa.PrimaryKeyConstraint('id')
        )
        op.create_index('ix_wssed_predictions_training_job_id', 'wssed_predictions', ['training_job_id'])
        op.create_index('ix_wssed_predictions_recording_id', 'wssed_predictions', ['recording_id'])
        op.create_index('ix_wssed_predictions_species_name', 'wssed_predictions', ['species_name'])
        op.create_index('ix_wssed_predictions_user_feedback', 'wssed_predictions', ['user_feedback'])
    else:
        # Table exists, check and create indexes if missing
        for idx_name, idx_column in [
            ('ix_wssed_predictions_training_job_id', 'training_job_id'),
            ('ix_wssed_predictions_recording_id', 'recording_id'),
            ('ix_wssed_predictions_species_name', 'species_name'),
            ('ix_wssed_predictions_user_feedback', 'user_feedback')
        ]:
            result = conn.execute(sa.text(f"SELECT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = '{idx_name}')")).scalar()
            if not result:
                op.create_index(idx_name, 'wssed_predictions', [idx_column])
    
    # Create wssed_strong_labels table (if it doesn't exist)
    if 'wssed_strong_labels' not in existing_tables:
        # Get label_type_enum for column definition
        label_type_enum = postgresql.ENUM(
            'strong_positive', 'strong_negative',
            name='label_type_enum',
            create_type=False
        )
        op.create_table(
            'wssed_strong_labels',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('prediction_id', sa.Integer(), nullable=False),
            sa.Column('recording_id', sa.Integer(), nullable=False),
            sa.Column('species_name', sa.String(), nullable=False),
            sa.Column('start_time', sa.Float(), nullable=False),
            sa.Column('end_time', sa.Float(), nullable=False),
            sa.Column('confidence', sa.Float(), nullable=False),
            sa.Column('label_type', label_type_enum, nullable=False, server_default='strong_positive'),
            sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
            sa.ForeignKeyConstraint(['prediction_id'], ['wssed_predictions.id'], ondelete='CASCADE'),
            sa.ForeignKeyConstraint(['recording_id'], ['recordings.id'], ondelete='CASCADE'),
            sa.PrimaryKeyConstraint('id')
        )
        op.create_index('ix_wssed_strong_labels_recording_id', 'wssed_strong_labels', ['recording_id'])
        op.create_index('ix_wssed_strong_labels_species_name', 'wssed_strong_labels', ['species_name'])
        op.create_index('ix_wssed_strong_labels_label_type', 'wssed_strong_labels', ['label_type'])
    else:
        # Table exists, check and add label_type column if missing
        result = conn.execute(sa.text(
            "SELECT EXISTS (SELECT 1 FROM information_schema.columns "
            "WHERE table_name = 'wssed_strong_labels' AND column_name = 'label_type')"
        )).scalar()
        
        if not result:
            label_type_enum = postgresql.ENUM(
                'strong_positive', 'strong_negative',
                name='label_type_enum',
                create_type=False
            )
            op.add_column(
                'wssed_strong_labels',
                sa.Column(
                    'label_type',
                    label_type_enum,
                    nullable=False,
                    server_default='strong_positive'
                )
            )
        
        # Table exists, check and create indexes if missing
        for idx_name, idx_column in [
            ('ix_wssed_strong_labels_recording_id', 'recording_id'),
            ('ix_wssed_strong_labels_species_name', 'species_name'),
            ('ix_wssed_strong_labels_label_type', 'label_type')
        ]:
            result = conn.execute(sa.text(f"SELECT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = '{idx_name}')")).scalar()
            if not result:
                op.create_index(idx_name, 'wssed_strong_labels', [idx_column])
    
    # Add dataset_type column to datasets table (if it doesn't exist)
    result = conn.execute(sa.text(
        "SELECT EXISTS (SELECT 1 FROM information_schema.columns "
        "WHERE table_name = 'datasets' AND column_name = 'dataset_type')"
    )).scalar()
    
    if not result:
        dataset_type_enum = postgresql.ENUM(
            'PAM', 'WEAKLY_LABELED',
            name='dataset_type_enum',
            create_type=False
        )
        op.add_column(
            'datasets',
            sa.Column(
                'dataset_type',
                dataset_type_enum,
                nullable=False,
                server_default='PAM'
            )
        )
        # Create index for dataset_type if it doesn't exist
        result = conn.execute(sa.text(
            "SELECT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'ix_datasets_dataset_type')"
        )).scalar()
        if not result:
            op.create_index('ix_datasets_dataset_type', 'datasets', ['dataset_type'])


def downgrade() -> None:
    # Drop dataset_type column and index
    inspector = sa.inspect(op.get_bind())
    existing_columns = [col['name'] for col in inspector.get_columns('datasets')]
    
    if 'dataset_type' in existing_columns:
        op.drop_index('ix_datasets_dataset_type', table_name='datasets')
        op.drop_column('datasets', 'dataset_type')
    
    # Drop tables
    op.drop_table('wssed_strong_labels')
    op.drop_table('wssed_predictions')
    op.drop_table('wssed_training_jobs')
    
    # Drop enums
    op.execute('DROP TYPE IF EXISTS label_type_enum')
    op.execute('DROP TYPE IF EXISTS feedback_enum')
    op.execute('DROP TYPE IF EXISTS training_status_enum')
    op.execute('DROP TYPE IF EXISTS dataset_type_enum')