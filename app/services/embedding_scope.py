"""
Which embedding models a dataset's snippet sets have, and which to use.

A snippet set is no longer tied to one model: its models are recorded by its
EmbeddingJobs. A model is usable on a set once its job has COMPLETED.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from sqlalchemy.orm import Session

from app.models.dataset import Dataset
from app.models.embedding import EmbeddingJob, EmbeddingJobStatus, EmbeddingModel, SnippetSet
from app.models.pam_active_learning import ALModelCheckpoint, ALModelFamilyState


def completed_jobs_for_set(db: Session, snippet_set_id: int) -> List[EmbeddingJob]:
    """Oldest first, so the first entry is the model that embedded the set first."""
    return (
        db.query(EmbeddingJob)
        .filter(
            EmbeddingJob.snippet_set_id == snippet_set_id,
            EmbeddingJob.status == EmbeddingJobStatus.COMPLETED,
        )
        .order_by(EmbeddingJob.created_at.asc(), EmbeddingJob.id.asc())
        .all()
    )


def models_for_set(db: Session, snippet_set_id: int) -> List[int]:
    return [job.embedding_model_id for job in completed_jobs_for_set(db, snippet_set_id)]


def default_model_for_set(db: Session, snippet_set_id: int) -> Optional[int]:
    """The model that embedded the set first, or None when none has finished."""
    models = models_for_set(db, snippet_set_id)
    return models[0] if models else None


def snippet_set_for_model(db: Session, dataset: Dataset, embedding_model_id: int) -> Optional[int]:
    """The dataset's snippet set for a model: its default set if the model has embedded it."""
    rows = (
        db.query(EmbeddingJob.snippet_set_id)
        .filter(
            EmbeddingJob.dataset_id == dataset.id,
            EmbeddingJob.embedding_model_id == embedding_model_id,
            EmbeddingJob.status == EmbeddingJobStatus.COMPLETED,
        )
        .order_by(EmbeddingJob.created_at.desc(), EmbeddingJob.id.desc())
        .all()
    )
    set_ids = [row[0] for row in rows]
    if dataset.default_snippet_set_id in set_ids:
        return dataset.default_snippet_set_id
    return set_ids[0] if set_ids else None


def family_name_for_model(model: EmbeddingModel) -> str:
    """Classifier family name for a model that has no classifier on a dataset yet."""
    raw = f"{model.name}_{model.version}" if model.version else model.name
    return "".join(ch if ch.isalnum() else "_" for ch in raw.lower())


def embedding_options(db: Session, dataset: Dataset) -> List[dict]:
    """One entry per model with an embedding job on this dataset, for the hub's model switcher."""
    jobs = (
        db.query(EmbeddingJob, EmbeddingModel, SnippetSet)
        .join(EmbeddingModel, EmbeddingModel.id == EmbeddingJob.embedding_model_id)
        .join(SnippetSet, SnippetSet.id == EmbeddingJob.snippet_set_id)
        .filter(EmbeddingJob.dataset_id == dataset.id)
        .order_by(EmbeddingJob.created_at.asc(), EmbeddingJob.id.asc())
        .all()
    )

    models_per_set: Dict[int, int] = {}
    for job, _model, snippet_set in jobs:
        models_per_set[snippet_set.id] = models_per_set.get(snippet_set.id, 0) + 1

    # Each classifier family is locked to one embedding model.
    active = (
        db.query(ALModelCheckpoint)
        .join(
            ALModelFamilyState,
            ALModelFamilyState.active_model_checkpoint_id == ALModelCheckpoint.id,
        )
        .filter(ALModelFamilyState.dataset_id == dataset.id)
        .order_by(ALModelCheckpoint.id.asc())
        .all()
    )
    checkpoint_by_model = {c.embedding_model_id: c for c in active if c.embedding_model_id is not None}

    default_found = False
    options: List[dict] = []
    for job, model, snippet_set in jobs:
        is_default = (
            not default_found
            and job.snippet_set_id == dataset.default_snippet_set_id
            and job.status == EmbeddingJobStatus.COMPLETED
        )
        default_found = default_found or is_default
        checkpoint = checkpoint_by_model.get(model.id)
        options.append({
            "embedding_model_id": model.id,
            "name": model.name,
            "version": model.version,
            "dim": model.dim,
            "snippet_set_id": snippet_set.id,
            "window_size": snippet_set.window_size,
            "step_size": snippet_set.step_size,
            "overlap": snippet_set.overlap,
            "embedding_job_id": job.id,
            "status": job.status.value,
            "is_default": is_default,
            "shares_snippet_set": models_per_set[snippet_set.id] > 1,
            "active_checkpoint_id": checkpoint.id if checkpoint else None,
            "model_family_name": checkpoint.model_family_name if checkpoint else family_name_for_model(model),
        })
    return options
