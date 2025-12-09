"""
Embedding service: manages embedding models and embedding jobs.
"""

from typing import Optional, List
from sqlalchemy.orm import Session
from sqlalchemy.sql import func

from app.models.embedding import (
    EmbeddingModel,
    EmbeddingJob,
    EmbeddingJobStatus,
)
from app.models.dataset import Dataset
from app.models.snippet import SnippetConfig


class EmbeddingService:
    def __init__(self, db: Session):
        self.db = db

    # ---------------------------------------------------------
    # Embedding Models
    # ---------------------------------------------------------

    def list_models(self) -> List[EmbeddingModel]:
        """Return all available embedding models."""
        return self.db.query(EmbeddingModel).all()

    def get_model(self, model_id: int) -> EmbeddingModel:
        """Lookup model or raise ValueError."""
        model = self.db.query(EmbeddingModel).filter_by(id=model_id).first()
        if not model:
            raise ValueError(f"EmbeddingModel(id={model_id}) not found")
        return model

    # ---------------------------------------------------------
    # Embedding Jobs
    # ---------------------------------------------------------

    def create_embedding_job(
        self,
        dataset: Dataset,
        model: EmbeddingModel,
    ) -> EmbeddingJob:
        """
        Create an embedding job AND its snippet config (1:1).

        IMPORTANT:
        - Must create EmbeddingJob first to satisfy SnippetConfig.embedding_job_id FK.
        """

        # --------------------------------------------------
        # 1. Create embedding job (no snippet_config yet)
        # --------------------------------------------------
        job = EmbeddingJob(
            dataset_id=dataset.id,
            embedding_model_id=model.id,
            status=EmbeddingJobStatus.PENDING,
        )
        self.db.add(job)
        self.db.flush()   # <-- job.id is now available

        # --------------------------------------------------
        # 2. Create snippet config referencing job.id
        # --------------------------------------------------
        cfg = SnippetConfig(
            embedding_job_id=job.id,
            window_size=model.default_window_size,
            step_size=model.default_step_size,
            overlap=model.default_overlap,
        )
        self.db.add(cfg)
        self.db.flush()

        # Establish relationship (optional; SQLAlchemy will load it anyway)
        job.snippet_config = cfg

        self.db.commit()
        self.db.refresh(job)
        return job

    def get_job(self, job_id: int) -> EmbeddingJob:
        """Fetch embedding job or raise."""
        job = self.db.query(EmbeddingJob).filter_by(id=job_id).first()
        if job is None:
            raise ValueError(f"EmbeddingJob(id={job_id}) not found")
        return job

    def list_jobs_for_dataset(self, dataset_id: int) -> List[EmbeddingJob]:
        """Return all embedding jobs for a dataset."""
        return (
            self.db.query(EmbeddingJob)
            .filter_by(dataset_id=dataset_id)
            .order_by(EmbeddingJob.created_at)
            .all()
        )

    # ---------------------------------------------------------
    # Job Status Updates (used by Celery)
    # ---------------------------------------------------------

    def update_job_status(
        self,
        job_id: int,
        status: EmbeddingJobStatus,
        message: Optional[str] = None,
        celery_task_id: Optional[str] = None,
    ):
        """Update job status + bookkeeping fields."""
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
