"""Add PAM Active Learning tables

Revision ID: 2026_03_02_pam_active_learning
Revises: 2026_02_16_taxon_constraint
Create Date: 2026-03-02 

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '2026_03_02_pam_active_learning'
down_revision = '2026_02_16_relax_taxon_id_constraint'
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    existing_tables = inspector.get_table_names()

    # ── Enum types ──────────────────────────────────────────────────
    result = conn.execute(sa.text(
        "SELECT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'pam_model_status_enum')"
    )).scalar()
    if not result:
        postgresql.ENUM(
            'AVAILABLE', 'LOADING', 'ERROR',
            name='pam_model_status_enum', create_type=True
        ).create(conn)

    result = conn.execute(sa.text(
        "SELECT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'pam_feedback_action_enum')"
    )).scalar()
    if not result:
        postgresql.ENUM(
            'ACCEPT', 'REJECT', 'MODIFY',
            name='pam_feedback_action_enum', create_type=True
        ).create(conn)

    result = conn.execute(sa.text(
        "SELECT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'pam_retrain_status_enum')"
    )).scalar()
    if not result:
        postgresql.ENUM(
            'PENDING', 'RUNNING', 'COMPLETED', 'FAILED',
            name='pam_retrain_status_enum', create_type=True
        ).create(conn)

    # ── pam_model_checkpoints ───────────────────────────────────────
    if 'pam_model_checkpoints' not in existing_tables:
        op.create_table(
            'pam_model_checkpoints',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('dataset_id', sa.Integer(), nullable=False),
            sa.Column('name', sa.String(), nullable=False),
            sa.Column('version', sa.String(), nullable=False, server_default='v0'),
            sa.Column('checkpoint_path', sa.String(), nullable=True),
            sa.Column('model_type', sa.String(), nullable=False, server_default='pam_classifier'),
            sa.Column('hyperparameters', sa.JSON(), nullable=True),
            sa.Column('status',
                       postgresql.ENUM('AVAILABLE', 'LOADING', 'ERROR',
                                       name='pam_model_status_enum', create_type=False),
                       nullable=False, server_default='AVAILABLE'),
            sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
            sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
            sa.ForeignKeyConstraint(['dataset_id'], ['datasets.id'], ondelete='CASCADE'),
            sa.PrimaryKeyConstraint('id'),
            sa.UniqueConstraint('dataset_id', 'name', 'version', name='uq_pam_checkpoint'),
        )
        op.create_index('ix_pam_model_checkpoints_id', 'pam_model_checkpoints', ['id'])
        op.create_index('ix_pam_model_checkpoints_dataset_id', 'pam_model_checkpoints', ['dataset_id'])

    # ── pam_predictions ─────────────────────────────────────────────
    if 'pam_predictions' not in existing_tables:
        op.create_table(
            'pam_predictions',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('model_checkpoint_id', sa.Integer(), nullable=False),
            sa.Column('snippet_id', sa.Integer(), nullable=False),
            sa.Column('predicted_label', sa.String(), nullable=False),
            sa.Column('confidence', sa.Float(), nullable=False),
            sa.Column('ranking_score', sa.Float(), nullable=True),
            sa.Column('extra_scores', sa.JSON(), nullable=True),
            sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
            sa.ForeignKeyConstraint(['model_checkpoint_id'], ['pam_model_checkpoints.id'], ondelete='CASCADE'),
            sa.ForeignKeyConstraint(['snippet_id'], ['snippets.id'], ondelete='CASCADE'),
            sa.PrimaryKeyConstraint('id'),
            sa.UniqueConstraint('model_checkpoint_id', 'snippet_id', name='uq_pam_prediction'),
        )
        op.create_index('ix_pam_predictions_id', 'pam_predictions', ['id'])
        op.create_index('ix_pam_predictions_model_checkpoint_id', 'pam_predictions', ['model_checkpoint_id'])
        op.create_index('ix_pam_predictions_snippet_id', 'pam_predictions', ['snippet_id'])

    # ── pam_feedback_events ─────────────────────────────────────────
    if 'pam_feedback_events' not in existing_tables:
        op.create_table(
            'pam_feedback_events',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('prediction_id', sa.Integer(), nullable=False),
            sa.Column('user_id', sa.Integer(), nullable=True),
            sa.Column('action',
                       postgresql.ENUM('ACCEPT', 'REJECT', 'MODIFY',
                                       name='pam_feedback_action_enum', create_type=False),
                       nullable=False),
            sa.Column('modified_label', sa.String(), nullable=True),
            sa.Column('notes', sa.Text(), nullable=True),
            sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
            sa.ForeignKeyConstraint(['prediction_id'], ['pam_predictions.id'], ondelete='CASCADE'),
            sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='SET NULL'),
            sa.PrimaryKeyConstraint('id'),
        )
        op.create_index('ix_pam_feedback_events_id', 'pam_feedback_events', ['id'])
        op.create_index('ix_pam_feedback_events_prediction_id', 'pam_feedback_events', ['prediction_id'])
        op.create_index('ix_pam_feedback_events_user_id', 'pam_feedback_events', ['user_id'])

    # ── pam_retrain_jobs ────────────────────────────────────────────
    if 'pam_retrain_jobs' not in existing_tables:
        op.create_table(
            'pam_retrain_jobs',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('model_checkpoint_id', sa.Integer(), nullable=False),
            sa.Column('trigger', sa.String(), nullable=False, server_default='auto'),
            sa.Column('feedback_count', sa.Integer(), nullable=False, server_default='0'),
            sa.Column('status',
                       postgresql.ENUM('PENDING', 'RUNNING', 'COMPLETED', 'FAILED',
                                       name='pam_retrain_status_enum', create_type=False),
                       nullable=False, server_default='PENDING'),
            sa.Column('result_metrics', sa.JSON(), nullable=True),
            sa.Column('error_message', sa.Text(), nullable=True),
            sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
            sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
            sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
            sa.ForeignKeyConstraint(['model_checkpoint_id'], ['pam_model_checkpoints.id'], ondelete='CASCADE'),
            sa.PrimaryKeyConstraint('id'),
        )
        op.create_index('ix_pam_retrain_jobs_id', 'pam_retrain_jobs', ['id'])
        op.create_index('ix_pam_retrain_jobs_model_checkpoint_id', 'pam_retrain_jobs', ['model_checkpoint_id'])


def downgrade() -> None:
    op.drop_table('pam_retrain_jobs')
    op.drop_table('pam_feedback_events')
    op.drop_table('pam_predictions')
    op.drop_table('pam_model_checkpoints')

    conn = op.get_bind()
    for enum_name in ('pam_retrain_status_enum', 'pam_feedback_action_enum', 'pam_model_status_enum'):
        conn.execute(sa.text(f"DROP TYPE IF EXISTS {enum_name}"))
