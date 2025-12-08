"""update annotations table taxonomy

Revision ID: i1j2k3l4m5n6
Revises: h9i0j1k2l3m4
Create Date: 2025-01-08 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'i1j2k3l4m5n6'
down_revision = 'h9i0j1k2l3m4'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """
    Update annotations table to use taxonomy system:
    - Add taxon_id (namespaced identifier like 'gbif:2420576')
    - Add resolved_name_snapshot (snapshot of resolved scientific name)
    - Remove species_name column
    - Add validation constraints
    """
    # Add new columns
    op.add_column('annotations', sa.Column('taxon_id', sa.String(255), nullable=True))
    op.add_column('annotations', sa.Column('resolved_name_snapshot', sa.String(255), nullable=True))
    
    # Migrate existing data: convert species_name to taxon_id
    op.execute("""
        UPDATE annotations
        SET taxon_id = 'gbif:0',
            resolved_name_snapshot = species_name
        WHERE species_name IS NOT NULL
    """)
    
    # Make columns non-nullable after migration
    op.alter_column('annotations', 'taxon_id', nullable=False)
    op.alter_column('annotations', 'resolved_name_snapshot', nullable=False)
    
    # Add index on taxon_id
    op.create_index('ix_annotations_taxon_id', 'annotations', ['taxon_id'])
    
    # Add check constraints
    op.create_check_constraint(
        'valid_confidence',
        'annotations',
        'confidence >= 0.0 AND confidence <= 1.0'
    )
    op.create_check_constraint(
        'valid_taxon_id_format',
        'annotations',
        "taxon_id ~ '^[a-z]+:[0-9]+$'"
    )
    
    # Drop old column
    op.drop_column('annotations', 'species_name')


def downgrade() -> None:
    """
    Revert annotations table changes
    """
    # Add back species_name column
    op.add_column('annotations', sa.Column('species_name', sa.String(), nullable=True))
    
    # Migrate data back
    op.execute("""
        UPDATE annotations
        SET species_name = resolved_name_snapshot
    """)
    
    # Make column non-nullable
    op.alter_column('annotations', 'species_name', nullable=False)
    
    # Drop constraints
    op.drop_constraint('valid_taxon_id_format', 'annotations')
    op.drop_constraint('valid_confidence', 'annotations')
    
    # Drop index
    op.drop_index('ix_annotations_taxon_id', 'annotations')
    
    # Drop new columns
    op.drop_column('annotations', 'resolved_name_snapshot')
    op.drop_column('annotations', 'taxon_id')

