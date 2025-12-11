"""
Embedding service: manages embedding models, snippet sets, and embedding jobs.
"""

from typing import Optional, List
from sqlalchemy.orm import Session
from sqlalchemy.sql import func

from app.models.embedding import (
    EmbeddingModel,
    EmbeddingJob,
    EmbeddingJobStatus,
    SnippetSet,
    SnippetSetStatus,
)
from app.models.dataset import Dataset


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
