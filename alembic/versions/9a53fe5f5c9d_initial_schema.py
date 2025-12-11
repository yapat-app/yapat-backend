"""initial_schema

Revision ID: 9a53fe5f5c9d
Revises: 
Create Date: 2025-12-11 14:26:32.640612

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '9a53fe5f5c9d'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
   
    op.create_table('embedding_models',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(), nullable=False),
    sa.Column('version', sa.String(), nullable=True),
    sa.Column('description', sa.String(), nullable=True),
    sa.Column('source_uri', sa.String(), nullable=True),
    sa.Column('window_size', sa.Float(), nullable=False),
    sa.Column('step_size', sa.Float(), nullable=False),
    sa.Column('overlap', sa.Float(), nullable=False),
    sa.Column('requires_fixed_window', sa.Integer(), nullable=True),
    sa.Column('requires_fixed_step', sa.Integer(), nullable=True),
    sa.Column('requires_fixed_overlap', sa.Integer(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_embedding_models_id'), 'embedding_models', ['id'], unique=False)
    op.create_table('teams',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(), nullable=False),
    sa.Column('description', sa.String(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_teams_id'), 'teams', ['id'], unique=False)
    op.create_index(op.f('ix_teams_name'), 'teams', ['name'], unique=False)
    op.create_table('users',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('username', sa.String(), nullable=False),
    sa.Column('hashed_password', sa.String(), nullable=False),
    sa.Column('full_name', sa.String(), nullable=True),
    sa.Column('role', sa.Enum('ADMIN', 'TEAM_OWNER', 'USER', name='userrole'), nullable=False),
    sa.Column('is_active', sa.Boolean(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_users_id'), 'users', ['id'], unique=False)
    op.create_index(op.f('ix_users_username'), 'users', ['username'], unique=True)
    op.create_table('datasets',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(), nullable=False),
    sa.Column('description', sa.Text(), nullable=True),
    sa.Column('source_uri', sa.String(), nullable=False),
    sa.Column('team_id', sa.Integer(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
    sa.ForeignKeyConstraint(['team_id'], ['teams.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('team_id', 'source_uri', name='uq_dataset_team_source')
    )
    op.create_index(op.f('ix_datasets_id'), 'datasets', ['id'], unique=False)
    op.create_index(op.f('ix_datasets_name'), 'datasets', ['name'], unique=False)
    op.create_index(op.f('ix_datasets_source_uri'), 'datasets', ['source_uri'], unique=False)
    op.create_table('invitation_links',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('token', sa.String(), nullable=False),
    sa.Column('created_by', sa.Integer(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('is_active', sa.Boolean(), nullable=False),
    sa.Column('max_uses', sa.Integer(), nullable=True),
    sa.Column('used_count', sa.Integer(), nullable=False),
    sa.ForeignKeyConstraint(['created_by'], ['users.id'], ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_invitation_links_id'), 'invitation_links', ['id'], unique=False)
    op.create_index(op.f('ix_invitation_links_token'), 'invitation_links', ['token'], unique=True)
    op.create_table('team_invitations',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('team_id', sa.Integer(), nullable=False),
    sa.Column('invited_by', sa.Integer(), nullable=True),
    sa.Column('token', sa.String(), nullable=False),
    sa.Column('email', sa.String(), nullable=True),
    sa.Column('target_role', sa.Enum('OWNER', 'USER', name='teamrole'), nullable=False),
    sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('is_active', sa.Boolean(), nullable=False),
    sa.Column('max_uses', sa.Integer(), nullable=True),
    sa.Column('used_count', sa.Integer(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.ForeignKeyConstraint(['invited_by'], ['users.id'], ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['team_id'], ['teams.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_team_invitations_id'), 'team_invitations', ['id'], unique=False)
    op.create_index(op.f('ix_team_invitations_token'), 'team_invitations', ['token'], unique=True)
    op.create_table('team_memberships',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('team_id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.Column('role', sa.Enum('OWNER', 'USER', name='teamrole'), nullable=False),
    sa.Column('joined_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.ForeignKeyConstraint(['team_id'], ['teams.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_team_memberships_id'), 'team_memberships', ['id'], unique=False)
    op.create_table('invitation_datasets',
    sa.Column('invitation_id', sa.Integer(), nullable=False),
    sa.Column('dataset_id', sa.Integer(), nullable=False),
    sa.ForeignKeyConstraint(['dataset_id'], ['datasets.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['invitation_id'], ['invitation_links.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('invitation_id', 'dataset_id')
    )
    op.create_table('recordings',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('dataset_id', sa.Integer(), nullable=False),
    sa.Column('file_path', sa.String(), nullable=False),
    sa.Column('file_name', sa.String(), nullable=False),
    sa.Column('duration', sa.Float(), nullable=True),
    sa.Column('sample_rate', sa.Float(), nullable=True),
    sa.Column('extra_metadata', sa.JSON(), nullable=True),
    sa.Column('audio_sha256', sa.String(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.ForeignKeyConstraint(['dataset_id'], ['datasets.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_recordings_id'), 'recordings', ['id'], unique=False)
    op.create_table('snippet_sets',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('dataset_id', sa.Integer(), nullable=False),
    sa.Column('embedding_model_id', sa.Integer(), nullable=False),
    sa.Column('window_size', sa.Float(), nullable=False),
    sa.Column('step_size', sa.Float(), nullable=False),
    sa.Column('overlap', sa.Float(), nullable=False),
    sa.Column('status', sa.String(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.ForeignKeyConstraint(['dataset_id'], ['datasets.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['embedding_model_id'], ['embedding_models.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_snippet_sets_id'), 'snippet_sets', ['id'], unique=False)
    op.create_table('user_datasets',
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.Column('dataset_id', sa.Integer(), nullable=False),
    sa.Column('granted_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.Column('granted_by_invitation_id', sa.Integer(), nullable=True),
    sa.ForeignKeyConstraint(['dataset_id'], ['datasets.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['granted_by_invitation_id'], ['invitation_links.id'], ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('user_id', 'dataset_id')
    )
    op.create_table('embedding_jobs',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('dataset_id', sa.Integer(), nullable=False),
    sa.Column('embedding_model_id', sa.Integer(), nullable=False),
    sa.Column('snippet_set_id', sa.Integer(), nullable=False),
    sa.Column('status', sa.String(), nullable=False),
    sa.Column('celery_task_id', sa.String(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('error_message', sa.Text(), nullable=True),
    sa.ForeignKeyConstraint(['dataset_id'], ['datasets.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['embedding_model_id'], ['embedding_models.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['snippet_set_id'], ['snippet_sets.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_embedding_jobs_id'), 'embedding_jobs', ['id'], unique=False)
    op.create_table('snippets',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('recording_id', sa.Integer(), nullable=False),
    sa.Column('snippet_set_id', sa.Integer(), nullable=False),
    sa.Column('start_time', sa.Float(), nullable=False),
    sa.Column('end_time', sa.Float(), nullable=False),
    sa.Column('duration', sa.Float(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.ForeignKeyConstraint(['recording_id'], ['recordings.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['snippet_set_id'], ['snippet_sets.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_snippets_id'), 'snippets', ['id'], unique=False)
    op.create_table('annotations',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('snippet_id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.Column('taxon_id', sa.String(length=255), nullable=False),
    sa.Column('resolved_name_snapshot', sa.String(length=255), nullable=False),
    sa.Column('confidence', sa.Float(), nullable=True),
    sa.Column('notes', sa.Text(), nullable=True),
    sa.Column('extra_metadata', sa.JSON(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.CheckConstraint("taxon_id ~ '^[a-z]+:[0-9]+$'", name='valid_taxon_id_format'),
    sa.ForeignKeyConstraint(['snippet_id'], ['snippets.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_annotations_id'), 'annotations', ['id'], unique=False)
    op.create_index(op.f('ix_annotations_taxon_id'), 'annotations', ['taxon_id'], unique=False)
    
    
    op.execute("""
        INSERT INTO embedding_models (name, version, description, window_size, step_size, overlap, requires_fixed_window, requires_fixed_step, requires_fixed_overlap)
        VALUES ('birdnet', '2.4', 'BirdNET v2.4 - 3-second windows at 48 kHz', 3.0, 1.0, 0.0, 1, 1, 1)
    """)
    # ### end Alembic commands ###


def downgrade() -> None:
    
    op.drop_index(op.f('ix_annotations_taxon_id'), table_name='annotations')
    op.drop_index(op.f('ix_annotations_id'), table_name='annotations')
    op.drop_table('annotations')
    op.drop_index(op.f('ix_snippets_id'), table_name='snippets')
    op.drop_table('snippets')
    op.drop_index(op.f('ix_embedding_jobs_id'), table_name='embedding_jobs')
    op.drop_table('embedding_jobs')
    op.drop_table('user_datasets')
    op.drop_index(op.f('ix_snippet_sets_id'), table_name='snippet_sets')
    op.drop_table('snippet_sets')
    op.drop_index(op.f('ix_recordings_id'), table_name='recordings')
    op.drop_table('recordings')
    op.drop_table('invitation_datasets')
    op.drop_index(op.f('ix_team_memberships_id'), table_name='team_memberships')
    op.drop_table('team_memberships')
    op.drop_index(op.f('ix_team_invitations_token'), table_name='team_invitations')
    op.drop_index(op.f('ix_team_invitations_id'), table_name='team_invitations')
    op.drop_table('team_invitations')
    op.drop_index(op.f('ix_invitation_links_token'), table_name='invitation_links')
    op.drop_index(op.f('ix_invitation_links_id'), table_name='invitation_links')
    op.drop_table('invitation_links')
    op.drop_index(op.f('ix_datasets_source_uri'), table_name='datasets')
    op.drop_index(op.f('ix_datasets_name'), table_name='datasets')
    op.drop_index(op.f('ix_datasets_id'), table_name='datasets')
    op.drop_table('datasets')
    op.drop_index(op.f('ix_users_username'), table_name='users')
    op.drop_index(op.f('ix_users_id'), table_name='users')
    op.drop_table('users')
    op.drop_index(op.f('ix_teams_name'), table_name='teams')
    op.drop_index(op.f('ix_teams_id'), table_name='teams')
    op.drop_table('teams')
    op.drop_index(op.f('ix_embedding_models_id'), table_name='embedding_models')
    op.drop_table('embedding_models')
    # ### end Alembic commands ###

