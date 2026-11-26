"""several embedding models per dataset (yapat-backend#74)

- embedding_models: dim, sample_rate, implementation_key, is_active; adds
  BirdNET+ V3.0 and Perch 2.0.
- snippet_sets: no longer tied to one model. Models with the same window, step
  and overlap share a set; EmbeddingJobs record which models a set has.
- embedding_vectors: one vector per snippet per model, in a column of no fixed
  size. The HNSW indexes move to the next revision.
- annotations, al_model_checkpoints, recording_embedding_failures: embedding_model_id.

Every existing snippet set has exactly one model, so each backfill below takes
the model from that set's embedding job.

Revision ID: 2026_11_04_multi_embedding
Revises: 2026_10_29_merge_heads
Create Date: 2026-11-04
"""

import sqlalchemy as sa
from alembic import op

revision = "2026_11_04_multi_embedding"
down_revision = "2026_10_29_merge_heads"
branch_labels = None
depends_on = None


NEW_MODELS = [
    {
        "name": "birdnet",
        "version": "3.0-preview3.1",
        "description": "BirdNET+ V3.0 developer preview 3.1 - 3-second windows at 32 kHz",
        "window_size": 3.0,
        "dim": 1280,
        "sample_rate": 32000,
        "implementation_key": "birdnet_v3",
    },
    {
        "name": "perch",
        "version": "2.0",
        "description": "Perch 2.0 - 5-second windows at 32 kHz",
        "window_size": 5.0,
        "dim": 1536,
        "sample_rate": 32000,
        "implementation_key": "perch_v2",
    },
]


def upgrade() -> None:
    # --- embedding_models ---------------------------------------------------
    op.add_column("embedding_models", sa.Column("dim", sa.Integer(), nullable=True))
    op.add_column("embedding_models", sa.Column("sample_rate", sa.Integer(), nullable=True))
    op.add_column("embedding_models", sa.Column("implementation_key", sa.String(), nullable=True))
    op.add_column(
        "embedding_models",
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
    )
    op.execute(
        "UPDATE embedding_models SET dim = 1024, sample_rate = 48000, implementation_key = 'birdnet_v2_4' "
        "WHERE name = 'birdnet' AND version = '2.4'"
    )
    # Any other row has no embedder: keep its vectors readable but do not offer it.
    op.execute(
        "UPDATE embedding_models m SET is_active = false, dim = COALESCE("
        "(SELECT max(ev.dim) FROM embedding_vectors ev WHERE ev.embedding_model_id = m.id), 1024) "
        "WHERE m.implementation_key IS NULL"
    )
    op.alter_column("embedding_models", "dim", nullable=False)
    op.create_unique_constraint(
        "uq_embedding_models_implementation_key", "embedding_models", ["implementation_key"]
    )
    models = sa.table(
        "embedding_models",
        sa.column("name", sa.String),
        sa.column("version", sa.String),
        sa.column("description", sa.String),
        sa.column("window_size", sa.Float),
        sa.column("step_size", sa.Float),
        sa.column("overlap", sa.Float),
        sa.column("requires_fixed_window", sa.Integer),
        sa.column("requires_fixed_step", sa.Integer),
        sa.column("requires_fixed_overlap", sa.Integer),
        sa.column("dim", sa.Integer),
        sa.column("sample_rate", sa.Integer),
        sa.column("implementation_key", sa.String),
    )
    op.bulk_insert(
        models,
        [
            {
                **model,
                "step_size": model["window_size"],
                "overlap": 0.0,
                "requires_fixed_window": 1,
                "requires_fixed_step": 1,
                "requires_fixed_overlap": 1,
            }
            for model in NEW_MODELS
        ],
    )

    # --- annotations --------------------------------------------------------
    op.add_column("annotations", sa.Column("embedding_model_id", sa.Integer(), nullable=True))
    op.execute(
        "UPDATE annotations a SET embedding_model_id = ss.embedding_model_id "
        "FROM snippets s JOIN snippet_sets ss ON ss.id = s.snippet_set_id "
        "WHERE s.id = a.snippet_id"
    )
    op.create_foreign_key(
        "fk_annotations_embedding_model", "annotations", "embedding_models",
        ["embedding_model_id"], ["id"], ondelete="SET NULL",
    )
    op.create_index("ix_annotations_embedding_model_id", "annotations", ["embedding_model_id"])

    # --- al_model_checkpoints -----------------------------------------------
    op.add_column("al_model_checkpoints", sa.Column("embedding_model_id", sa.Integer(), nullable=True))
    # The id in hyperparameters may name a model that no longer exists.
    op.execute(
        "UPDATE al_model_checkpoints c SET embedding_model_id = m.id "
        "FROM embedding_models m "
        "WHERE m.id::text = c.hyperparameters->>'embedding_model_id'"
    )
    # Older checkpoints did not record it; their dataset had one model.
    op.execute(
        "UPDATE al_model_checkpoints c SET embedding_model_id = ss.embedding_model_id "
        "FROM datasets d JOIN snippet_sets ss ON ss.id = d.default_snippet_set_id "
        "WHERE d.id = c.dataset_id AND c.embedding_model_id IS NULL "
        "AND c.model_type LIKE 'pam\\_%'"
    )
    op.create_foreign_key(
        "fk_al_model_checkpoints_embedding_model", "al_model_checkpoints", "embedding_models",
        ["embedding_model_id"], ["id"], ondelete="SET NULL",
    )
    op.create_index(
        "ix_al_model_checkpoints_embedding_model_id", "al_model_checkpoints", ["embedding_model_id"]
    )

    # --- recording_embedding_failures ---------------------------------------
    op.add_column(
        "recording_embedding_failures", sa.Column("embedding_model_id", sa.Integer(), nullable=True)
    )
    op.execute(
        "UPDATE recording_embedding_failures f SET embedding_model_id = ss.embedding_model_id "
        "FROM snippet_sets ss WHERE ss.id = f.snippet_set_id"
    )
    op.alter_column("recording_embedding_failures", "embedding_model_id", nullable=False)
    op.create_foreign_key(
        "fk_recording_embedding_failures_model", "recording_embedding_failures", "embedding_models",
        ["embedding_model_id"], ["id"], ondelete="CASCADE",
    )
    op.drop_constraint(
        "uq_recording_embedding_failure", "recording_embedding_failures", type_="unique"
    )
    op.create_unique_constraint(
        "uq_recording_embedding_failure", "recording_embedding_failures",
        ["recording_id", "snippet_set_id", "embedding_model_id"],
    )

    # --- embedding_vectors --------------------------------------------------
    op.execute("ALTER TABLE embedding_vectors DROP CONSTRAINT IF EXISTS embedding_vectors_snippet_id_key")
    op.execute("DROP INDEX IF EXISTS embedding_vectors_snippet_id_key")
    op.create_unique_constraint(
        "uq_embedding_vector_snippet_model", "embedding_vectors", ["snippet_id", "embedding_model_id"]
    )
    # Dropping the size does not rewrite the table or the existing index.
    op.execute("ALTER TABLE embedding_vectors ALTER COLUMN vector TYPE vector")

    # --- snippet_sets -------------------------------------------------------
    conn = op.get_bind()
    duplicates = conn.execute(
        sa.text(
            "SELECT dataset_id, window_size, step_size, overlap, array_agg(id ORDER BY id) "
            "FROM snippet_sets GROUP BY 1, 2, 3, 4 HAVING count(*) > 1"
        )
    ).fetchall()
    if duplicates:
        raise RuntimeError(
            "These snippet sets have identical segmentation, so they cannot be merged "
            f"automatically. Remove the unused ones before upgrading: {duplicates}"
        )
    op.create_unique_constraint(
        "uq_snippet_set_dataset_segmentation", "snippet_sets",
        ["dataset_id", "window_size", "step_size", "overlap"],
    )
    op.drop_column("snippet_sets", "embedding_model_id")


def downgrade() -> None:
    conn = op.get_bind()
    shared = conn.execute(
        sa.text(
            "SELECT count(*) FROM (SELECT snippet_id FROM embedding_vectors "
            "GROUP BY snippet_id HAVING count(*) > 1) AS shared"
        )
    ).scalar()
    extra = conn.execute(
        sa.text("SELECT count(*) FROM embedding_vectors WHERE dim <> 1024")
    ).scalar()
    if shared or extra:
        raise RuntimeError(
            f"Cannot downgrade: {shared} snippets have vectors from several models and "
            f"{extra} vectors are not 1024-dimensional. Delete them first."
        )

    op.add_column("snippet_sets", sa.Column("embedding_model_id", sa.Integer(), nullable=True))
    op.execute(
        "UPDATE snippet_sets ss SET embedding_model_id = ("
        "SELECT j.embedding_model_id FROM embedding_jobs j WHERE j.snippet_set_id = ss.id "
        "ORDER BY j.created_at, j.id LIMIT 1)"
    )
    op.execute(
        "UPDATE snippet_sets SET embedding_model_id = "
        "(SELECT id FROM embedding_models WHERE implementation_key = 'birdnet_v2_4') "
        "WHERE embedding_model_id IS NULL"
    )
    op.alter_column("snippet_sets", "embedding_model_id", nullable=False)
    op.create_foreign_key(
        "snippet_sets_embedding_model_id_fkey", "snippet_sets", "embedding_models",
        ["embedding_model_id"], ["id"], ondelete="CASCADE",
    )
    op.drop_constraint("uq_snippet_set_dataset_segmentation", "snippet_sets", type_="unique")

    op.execute("ALTER TABLE embedding_vectors ALTER COLUMN vector TYPE vector(1024)")
    op.execute(
        "CREATE INDEX embedding_vectors_vector_cosine_idx ON embedding_vectors "
        "USING hnsw (vector vector_cosine_ops)"
    )
    op.drop_constraint("uq_embedding_vector_snippet_model", "embedding_vectors", type_="unique")
    op.create_unique_constraint(
        "embedding_vectors_snippet_id_key", "embedding_vectors", ["snippet_id"]
    )

    op.drop_constraint(
        "uq_recording_embedding_failure", "recording_embedding_failures", type_="unique"
    )
    op.create_unique_constraint(
        "uq_recording_embedding_failure", "recording_embedding_failures",
        ["recording_id", "snippet_set_id"],
    )
    op.drop_column("recording_embedding_failures", "embedding_model_id")

    op.drop_index("ix_al_model_checkpoints_embedding_model_id", table_name="al_model_checkpoints")
    op.drop_column("al_model_checkpoints", "embedding_model_id")
    op.drop_index("ix_annotations_embedding_model_id", table_name="annotations")
    op.drop_column("annotations", "embedding_model_id")

    op.execute(
        "DELETE FROM embedding_models WHERE implementation_key IN ('birdnet_v3', 'perch_v2')"
    )
    op.drop_constraint("uq_embedding_models_implementation_key", "embedding_models", type_="unique")
    op.drop_column("embedding_models", "is_active")
    op.drop_column("embedding_models", "implementation_key")
    op.drop_column("embedding_models", "sample_rate")
    op.drop_column("embedding_models", "dim")
