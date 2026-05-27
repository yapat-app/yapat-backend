"""Follow-up marker after dataset spectrogram range migration.

Revision ID: 2026_05_22_merge_heads
Revises: 2026_05_21_spec_range
Create Date: 2026-05-22

No-op revision kept for continuity in environments that already reached
`2026_05_22_merge_heads`. The previous dependency on `b5dd26a8d59c` caused
startup failures where that missing revision never existed.
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
