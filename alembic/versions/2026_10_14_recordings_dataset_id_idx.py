"""Index recordings.dataset_id

Every per-dataset recording query filters on dataset_id, but the column had no
index, so Postgres read the whole table each time. GET /api/datasets/{id}
counts a dataset's recordings (and those with location/date/time metadata).

Created CONCURRENTLY so writes to recordings are not blocked while it builds.

Revision ID: 2026_10_14_recordings_dataset_idx
Revises: 2026_10_13_dataset_ingest_jobs
Create Date: 2026-10-14
"""

from alembic import op
import sqlalchemy as sa

revision = "2026_10_14_recordings_dataset_idx"
down_revision = "2026_10_13_dataset_ingest_jobs"
branch_labels = None
depends_on = None

INDEX_NAME = "ix_recordings_dataset_id"


def _index_exists(conn) -> bool:
    inspector = sa.inspect(conn)
    if "recordings" not in inspector.get_table_names():
        return True  # nothing to do
    return any(ix["name"] == INDEX_NAME for ix in inspector.get_indexes("recordings"))


def upgrade() -> None:
    conn = op.get_bind()
    if _index_exists(conn):
        return

    if conn.dialect.name != "postgresql":
        # SQLite (tests) has no CONCURRENTLY.
        op.create_index(INDEX_NAME, "recordings", ["dataset_id"])
        return

    # CONCURRENTLY cannot run inside a transaction block.
    with op.get_context().autocommit_block():
        op.execute(f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {INDEX_NAME} ON recordings (dataset_id)")


def downgrade() -> None:
    conn = op.get_bind()
    if conn.dialect.name != "postgresql":
        op.drop_index(INDEX_NAME, table_name="recordings")
        return
    with op.get_context().autocommit_block():
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {INDEX_NAME}")
