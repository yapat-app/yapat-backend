"""
Embedding service: manages embedding models, snippet sets, and embedding jobs.
"""

import logging
from typing import Optional, List

from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from sqlalchemy.sql import func

from app.models.dataset import Dataset
from app.models.embedding import (
    EmbeddingModel,
    EmbeddingJob,
    EmbeddingJobStatus,
    EmbeddingVector,
    SnippetSet,
    SnippetSetStatus,
)

logger = logging.getLogger(__name__)


def vector_cast(dim: int) -> str:
    """SQL type a model's vectors are cast to. Queries must use it to match the model's index."""
    return f"vector({int(dim)})"


def vector_index_name(embedding_model_id: int) -> str:
    return f"embedding_vectors_model_{int(embedding_model_id)}_hnsw"


def ensure_vector_index(engine: Engine, embedding_model_id: int, dim: int) -> None:
    """Build a model's partial HNSW index if it does not exist yet (Postgres only).

    HNSW needs a fixed size and the vector column has none, so each model gets
    its own index over a cast. Built without blocking writes, so it can run
    while other models are embedding. A failed build leaves an invalid index
    behind, which is dropped here so the next attempt starts clean.
    """
    if engine.dialect.name != "postgresql":
        return
    name = vector_index_name(embedding_model_id)
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        leftover = conn.execute(
            text(
                "SELECT 1 FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid "
                "WHERE c.relname = :name AND NOT i.indisvalid "
                "AND NOT EXISTS (SELECT 1 FROM pg_stat_progress_create_index p "
                "                WHERE p.index_relid = i.indexrelid)"
            ),
            {"name": name},
        ).first()
        if leftover is not None:
            logger.warning("Dropping invalid vector index %s left by a failed build", name)
            conn.execute(text(f"DROP INDEX CONCURRENTLY IF EXISTS {name}"))
        conn.execute(
            text(
                f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {name} ON embedding_vectors "
                f"USING hnsw ((vector::{vector_cast(dim)}) vector_cosine_ops) "
                f"WHERE embedding_model_id = {int(embedding_model_id)}"
            )
        )


class EmbeddingService:
    def __init__(self, db: Session):
        self.db = db

    # ---------------------------------------------------------
    # Embedding Models (from database)
    # ---------------------------------------------------------

    def list_models(self) -> List[EmbeddingModel]:
        """List all available embedding models from database."""
        return self.db.query(EmbeddingModel).all()

    def get_model(self, model_id: int) -> EmbeddingModel:
        """Get embedding model from database by ID."""
        model = (
            self.db.query(EmbeddingModel)
            .filter(EmbeddingModel.id == model_id)
            .first()
        )
        if not model:
            raise ValueError(f"EmbeddingModel(id={model_id}) not found")
        return model

    # ---------------------------------------------------------
    # SnippetSet management
    # ---------------------------------------------------------

    def get_or_create_snippet_set(
            self,
            dataset: Dataset,
            model: EmbeddingModel,
            *,
            window_size: Optional[float] = None,
            step_size: Optional[float] = None,
            overlap: Optional[float] = None,
    ) -> SnippetSet:
        """
        Returns the dataset's SnippetSet with these parameters, whichever model
        created it, otherwise creates a new one. Models with the same window, step
        and overlap therefore share clips, and with them the clips' annotations.

        Strict models enforce their fixed parameters.
        """

        # --- Resolve parameters --------------------------------------------
        if model.requires_fixed_window:
            window = model.window_size
        else:
            window = window_size or model.window_size

        if model.requires_fixed_step:
            step = model.step_size
        else:
            step = step_size or model.step_size

        if model.requires_fixed_overlap:
            ov = model.overlap
        else:
            ov = overlap or model.overlap

        # --- Lookup existing SnippetSet ------------------------------------
        existing = (
            self.db.query(SnippetSet)
            .filter(
                SnippetSet.dataset_id == dataset.id,
                SnippetSet.window_size == window,
                SnippetSet.step_size == step,
                SnippetSet.overlap == ov,
            )
            .first()
        )
        if existing:
            return existing

        # --- Create new SnippetSet -----------------------------------------
        snippet_set = SnippetSet(
            dataset_id=dataset.id,
            window_size=window,
            step_size=step,
            overlap=ov,
            status=SnippetSetStatus.PENDING,
        )

        self.db.add(snippet_set)
        self.db.commit()
        self.db.refresh(snippet_set)
        return snippet_set

    # ---------------------------------------------------------
    # Embedding Jobs
    # ---------------------------------------------------------

    def create_embedding_job(
            self,
            dataset: Dataset,
            model: EmbeddingModel,
            *,
            window_size: Optional[float] = None,
            step_size: Optional[float] = None,
            overlap: Optional[float] = None,
    ) -> EmbeddingJob:
        """
        Create an EmbeddingJob for a dataset × model.
        Ensures a SnippetSet exists (or creates one).
        Prevents duplicate jobs for the same configuration.
        """

        snippet_set = self.get_or_create_snippet_set(
            dataset,
            model,
            window_size=window_size,
            step_size=step_size,
            overlap=overlap,
        )

        # Check if a job already exists for this configuration
        existing_job = (
            self.db.query(EmbeddingJob)
            .filter(
                EmbeddingJob.dataset_id == dataset.id,
                EmbeddingJob.embedding_model_id == model.id,
                EmbeddingJob.snippet_set_id == snippet_set.id,
            )
            .first()
        )

        if existing_job:
            status_msg = existing_job.status.value
            raise ValueError(
                f"An embedding job already exists for this configuration "
                f"(dataset_id={dataset.id}, model_id={model.id}, snippet_set_id={snippet_set.id}). "
                f"Existing job ID: {existing_job.id}, Status: {status_msg}. "
                f"To create a new job, please delete the existing one first."
            )

        job = EmbeddingJob(
            dataset_id=dataset.id,
            embedding_model_id=model.id,
            snippet_set_id=snippet_set.id,
            status=EmbeddingJobStatus.PENDING,
        )

        self.db.add(job)
        self.db.commit()
        self.db.refresh(job)
        return job

    def get_job(self, job_id: int) -> EmbeddingJob:
        job = (
            self.db.query(EmbeddingJob)
            .filter(EmbeddingJob.id == job_id)
            .first()
        )
        if not job:
            raise ValueError(f"EmbeddingJob(id={job_id}) not found")
        return job

    def list_jobs_for_dataset(self, dataset_id: int) -> List[EmbeddingJob]:
        return (
            self.db.query(EmbeddingJob)
            .filter(EmbeddingJob.dataset_id == dataset_id)
            .order_by(EmbeddingJob.created_at)
            .all()
        )

    # ---------------------------------------------------------
    # Job Status Updates
    # ---------------------------------------------------------

    def update_job_status(
            self,
            job_id: int,
            status: EmbeddingJobStatus,
            message: Optional[str] = None,
            celery_task_id: Optional[str] = None,
    ):
        job = self.get_job(job_id)

        job.status = status

        if status == EmbeddingJobStatus.RUNNING:
            job.started_at = func.now()

        if status in (EmbeddingJobStatus.COMPLETED, EmbeddingJobStatus.FAILED):
            job.completed_at = func.now()

        if celery_task_id:
            job.celery_task_id = celery_task_id

        if message:
            job.error_message = message

        self.db.commit()


class VectorStore:
    BULK_INSERT_BATCH_ROWS = 1000

    def __init__(self, db):
        self.db = db

    def insert(self, snippet_id: int, job_id: int, model_id: int, vector):
        ev = EmbeddingVector(
            snippet_id=snippet_id,
            embedding_job_id=job_id,
            embedding_model_id=model_id,
            dim=len(vector),
            vector=vector,
        )
        self.db.add(ev)
        self.db.commit()
        return ev

    def bulk_insert(self, embeddings: List[dict]) -> int:
        """
        Bulk insert multiple embedding vectors efficiently.
        
        Args:
            embeddings: List of dicts with keys:
                - snippet_id: int
                - job_id: int
                - model_id: int
                - vector: list[float]
        
        Returns:
            Number of embeddings inserted. Snippets that already have a vector
            from this model are skipped (ON CONFLICT DO NOTHING), so re-embedding
            a recording, or a retried chunk, never fails on the unique key.
        """
        if not embeddings:
            return 0

        rows = [
            {
                "snippet_id": emb["snippet_id"],
                "embedding_job_id": emb["job_id"],
                "embedding_model_id": emb["model_id"],
                "dim": len(emb["vector"]),
                "vector": emb["vector"],
            }
            for emb in embeddings
        ]

        dialect = self.db.get_bind().dialect.name
        if dialect == "postgresql":
            from sqlalchemy.dialects.postgresql import insert as dialect_insert
        elif dialect == "sqlite":
            from sqlalchemy.dialects.sqlite import insert as dialect_insert
        else:  # pragma: no cover - only Postgres (prod) and SQLite (tests) are used
            self.db.bulk_insert_mappings(EmbeddingVector, rows)
            self.db.commit()
            return len(rows)

        # A multi-row VALUES binds 5 parameters per row; batch so a 24 h recording
        # (~29k snippets) stays under Postgres' 65535 and SQLite's 32766 limits.
        inserted = 0
        for start in range(0, len(rows), self.BULK_INSERT_BATCH_ROWS):
            batch = rows[start:start + self.BULK_INSERT_BATCH_ROWS]
            stmt = dialect_insert(EmbeddingVector).values(batch).on_conflict_do_nothing(
                index_elements=["snippet_id", "embedding_model_id"]
            )
            result = self.db.execute(stmt)
            inserted += max(int(result.rowcount or 0), 0)
        self.db.commit()

        return inserted

    def get(self, snippet_id: int, model_id: int):
        row = (
            self.db.query(EmbeddingVector)
            .filter_by(snippet_id=snippet_id, embedding_model_id=model_id)
            .first()
        )
        return row

    def search(
        self, 
        model_id: int, 
        query_vector, 
        k: int = 10,
        snippet_set_id: Optional[int] = None,
        dataset_id: Optional[int] = None
    ):
        """
        Perform cosine similarity search using pgvector.
        
        Args:
            model_id: Embedding model ID to search within
            query_vector: Query embedding vector (list or array of floats)
            k: Number of most similar results to return (default 10)
            snippet_set_id: Optional filter to search only within a specific snippet set
            dataset_id: Optional filter to search only within a specific dataset
            
        Returns:
            List of (snippet_id, similarity_score) tuples, sorted by similarity (descending)
        """
        # Format vector for PostgreSQL: '[1,2,3]'
        vector_str = '[' + ','.join(str(x) for x in query_vector) + ']'
        cast = vector_cast(self.db.get(EmbeddingModel, model_id).dim)

        # Build SQL with pgvector's <=> cosine distance operator
        sql_parts = [
            "SELECT ev.snippet_id, ",
            f"       (ev.vector::{cast} <=> '{vector_str}'::{cast}) as distance ",
            "FROM embedding_vectors ev "
        ]
        
        params = {'model_id': model_id, 'limit': k}
        
        # Add joins if filtering by snippet_set or dataset
        if snippet_set_id is not None or dataset_id is not None:
            sql_parts.append("JOIN snippets s ON ev.snippet_id = s.id ")
            if dataset_id is not None:
                sql_parts.append("JOIN snippet_sets ss ON s.snippet_set_id = ss.id ")
        
        # Build WHERE clause
        where = ["ev.embedding_model_id = :model_id"]
        if snippet_set_id is not None:
            where.append("s.snippet_set_id = :snippet_set_id")
            params['snippet_set_id'] = snippet_set_id
        if dataset_id is not None:
            where.append("ss.dataset_id = :dataset_id")
            params['dataset_id'] = dataset_id
        
        sql_parts.append("WHERE " + " AND ".join(where) + " ")
        sql_parts.append("ORDER BY distance ASC, ev.snippet_id ASC LIMIT :limit")
        
        # Execute and return results
        result = self.db.execute(text("".join(sql_parts)), params)
        rows = result.fetchall()
        
        # Convert distance to similarity: similarity = 1 - distance
        return [(int(row[0]), float(1.0 - row[1])) for row in rows]
