"""Rename dataset type WEAKLY_LABELED to FOCAL_RECORDINGS.

Revision ID: 2026_04_01_rename_dataset_type
Revises: 2026_03_20_drop_ranking_score
Create Date: 2026-04-01
"""

from alembic import op


# revision identifiers, used by Alembic.
revision = "2026_04_01_rename_dataset_type"
down_revision = "2026_03_20_drop_ranking_score"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1
                FROM pg_enum e
                JOIN pg_type t ON t.oid = e.enumtypid
                WHERE t.typname = 'dataset_type_enum'
                  AND e.enumlabel = 'WEAKLY_LABELED'
            ) THEN
                ALTER TYPE dataset_type_enum RENAME VALUE 'WEAKLY_LABELED' TO 'FOCAL_RECORDINGS';
            END IF;
        END $$;
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1
                FROM pg_enum e
                JOIN pg_type t ON t.oid = e.enumtypid
                WHERE t.typname = 'dataset_type_enum'
                  AND e.enumlabel = 'FOCAL_RECORDINGS'
            ) THEN
                ALTER TYPE dataset_type_enum RENAME VALUE 'FOCAL_RECORDINGS' TO 'WEAKLY_LABELED';
            END IF;
        END $$;
        """
    )
