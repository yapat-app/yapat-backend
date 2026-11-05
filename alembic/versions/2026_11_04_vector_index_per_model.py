"""one partial HNSW index per embedding model

HNSW needs a fixed size, which the vector column no longer has, so each model
gets its own index over a cast. New models get theirs when their first embedding
job finishes (app/tasks/embedding_tasks.py: build_vector_index).

This builds the index for every model that already has vectors, which can take a
long time on a large table. It holds no lock that blocks reads or writes, and can
be run ahead of a deployment by hand: CREATE INDEX ... IF NOT EXISTS skips an index
that is already there. It is a separate revision so that a failed build only
repeats this step.

Revision ID: 2026_11_04_vector_indexes
Revises: 2026_11_04_multi_embedding
Create Date: 2026-11-04
"""

import sqlalchemy as sa
from alembic import op

revision = "2026_11_04_vector_indexes"
down_revision = "2026_11_04_multi_embedding"
branch_labels = None
depends_on = None

OLD_INDEX = "embedding_vectors_vector_cosine_idx"


def _index_name(model_id: int) -> str:
    return f"embedding_vectors_model_{int(model_id)}_hnsw"


def upgrade() -> None:
    conn = op.get_bind()
    models = conn.execute(
        sa.text(
            "SELECT m.id, m.dim FROM embedding_models m "
            "WHERE EXISTS (SELECT 1 FROM embedding_vectors ev WHERE ev.embedding_model_id = m.id)"
        )
    ).fetchall()
    with op.get_context().autocommit_block():
        # An interrupted earlier attempt leaves an invalid index that IF NOT EXISTS would keep.
        for model_id, _dim in models:
            invalid = conn.execute(
                sa.text(
                    "SELECT 1 FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid "
                    "WHERE c.relname = :name AND NOT i.indisvalid"
                ),
                {"name": _index_name(model_id)},
            ).first()
            if invalid is not None:
                op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {_index_name(model_id)}")
        for model_id, dim in models:
            op.execute(
                f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {_index_name(model_id)} ON embedding_vectors "
                f"USING hnsw ((vector::vector({int(dim)})) vector_cosine_ops) "
                f"WHERE embedding_model_id = {int(model_id)}"
            )
        # Kept until now so similarity search stayed fast while the new indexes were built.
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {OLD_INDEX}")


def downgrade() -> None:
    conn = op.get_bind()
    model_ids = [row[0] for row in conn.execute(sa.text("SELECT id FROM embedding_models")).fetchall()]
    # The previous revision's downgrade rebuilds the old index once the column has a size again.
    with op.get_context().autocommit_block():
        for model_id in model_ids:
            op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {_index_name(model_id)}")
