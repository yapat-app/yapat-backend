"""Index matching the suggestion query's ordering on al_predictions

The top-k suggestion query orders by ``composite_score DESC NULLS LAST, id ASC``.
The existing ix_al_predictions_ckpt_score is ``(model_checkpoint_id,
composite_score)`` ascending, and a backward scan of an ASC index yields
``DESC NULLS FIRST`` -- not what the query asks for. Postgres therefore could
not use it for ordering and fell back to reading every prediction row for the
checkpoint and top-N sorting it.

Measured on 3,000,000 rows (Nova, 2026-09-22): the planner scanned all 3M rows
at width=924 plus a sequential scan of ``snippets``, taking **1046 ms** to
return 5 rows. With this index it becomes an index scan feeding the Limit with
no Sort node: **28 ms**.

Created CONCURRENTLY -- ``al_predictions`` is large on real deployments and a
plain CREATE INDEX would hold a write lock for the duration.

Revision ID: 2026_09_22_al_pred_composite_idx
Revises: 2026_08_04_quick_label_entries
Create Date: 2026-09-22
"""

from alembic import op
import sqlalchemy as sa

revision = "2026_09_22_al_pred_composite_idx"
down_revision = "2026_08_04_quick_label_entries"
branch_labels = None
depends_on = None

INDEX_NAME = "ix_al_predictions_ckpt_composite_desc"


def _index_exists(conn) -> bool:
    inspector = sa.inspect(conn)
    if "al_predictions" not in inspector.get_table_names():
        return True  # nothing to do
    return any(ix["name"] == INDEX_NAME for ix in inspector.get_indexes("al_predictions"))


def upgrade() -> None:
    conn = op.get_bind()
    if _index_exists(conn):
        return

    if conn.dialect.name != "postgresql":
        # SQLite (tests) has no CONCURRENTLY and no NULLS LAST on index columns.
        op.create_index(
            INDEX_NAME,
            "al_predictions",
            ["model_checkpoint_id", "composite_score", "id"],
        )
        return

    # CONCURRENTLY cannot run inside a transaction block.
    with op.get_context().autocommit_block():
        op.execute(
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {INDEX_NAME} "
            "ON al_predictions "
            "(model_checkpoint_id, composite_score DESC NULLS LAST, id)"
        )


def downgrade() -> None:
    conn = op.get_bind()
    if conn.dialect.name != "postgresql":
        op.drop_index(INDEX_NAME, table_name="al_predictions")
        return
    with op.get_context().autocommit_block():
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {INDEX_NAME}")
