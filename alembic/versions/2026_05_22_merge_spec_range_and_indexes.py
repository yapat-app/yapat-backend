"""Follow-up marker after dataset spectrogram range migration.

Merge-only revision.

This previously depended on an indexes migration revision (`b5dd26a8d59c`) that
is not present in this repository, which breaks Alembic startup when the DB is
stamped past this point. Keep this as a no-op marker that follows the
spectrogram range migration.
"""

from alembic import op

revision = "2026_05_22_merge_heads"
down_revision = "2026_05_21_spec_range"
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
