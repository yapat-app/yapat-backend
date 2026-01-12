"""
Embedding service: manages embedding models, snippet sets, and embedding jobs.
"""

from typing import Optional, List

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
        Returns an existing SnippetSet if parameters match, otherwise creates a new one.

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
                SnippetSet.embedding_model_id == model.id,
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
            embedding_model_id=model.id,
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
        """

        snippet_set = self.get_or_create_snippet_set(
            dataset,
            model,
            window_size=window_size,
            step_size=step_size,
            overlap=overlap,
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
            Number of embeddings inserted
        """
        if not embeddings:
            return 0
        
        vectors = [
            EmbeddingVector(
                snippet_id=emb["snippet_id"],
                embedding_job_id=emb["job_id"],
                embedding_model_id=emb["model_id"],
                dim=len(emb["vector"]),
                vector=emb["vector"],
            )
            for emb in embeddings
        ]
        
        self.db.bulk_save_objects(vectors)
        self.db.commit()
        
        return len(vectors)

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
        from sqlalchemy import text
        
        # Format vector for PostgreSQL: '[1,2,3]'
        vector_str = '[' + ','.join(str(x) for x in query_vector) + ']'
        
        # Build SQL with pgvector's <=> cosine distance operator
        sql_parts = [
            "SELECT ev.snippet_id, ",
            f"       (ev.vector <=> '{vector_str}'::vector) as distance ",
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
        sql_parts.append("ORDER BY distance ASC LIMIT :limit")
        
        # Execute and return results
        result = self.db.execute(text("".join(sql_parts)), params)
        rows = result.fetchall()
        
        # Convert distance to similarity: similarity = 1 - distance
        return [(int(row[0]), float(1.0 - row[1])) for row in rows]
