"""
Whether a dataset's reference pools can train with a given embedding model
(issue #93, part 3).

A classifier trains on the dataset's own labels plus every linked reference
pool (dataset-scoped and team-scoped links), all in the same embedding model's
vector space. A pool only contributes when it was embedded with that model.

Pool states for one model:

- ``ready``: embedded, with snippets.
- ``no_snippets``: embedded, but no snippets came out (for example every file
  is shorter than the model's window). Counts as ready: it can never get
  better, so it must not block the model. Training simply gets nothing from it.
- ``embedding``: an embedding job is still running.
- ``failed``: the last embedding job failed.
- ``not_embedded``: never embedded with this model.

The last three block switching a dataset to the model.
"""

from __future__ import annotations

from typing import Iterable, Optional

from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from app.models.dataset import Dataset
from app.models.embedding import EmbeddingJob, EmbeddingJobStatus, EmbeddingVector
from app.models.reference_link import DatasetReferenceLink

READY = "ready"
NO_SNIPPETS = "no_snippets"
EMBEDDING = "embedding"
FAILED = "failed"
NOT_EMBEDDED = "not_embedded"

BLOCKING = frozenset({EMBEDDING, FAILED, NOT_EMBEDDED})


def linked_pools(db: Session, dataset: Dataset) -> list[tuple[Dataset, str]]:
    """Reference datasets linked to this dataset, with how: ``dataset`` or ``team``."""
    filters = [DatasetReferenceLink.dataset_id == dataset.id]
    if dataset.team_id is not None:
        filters.append(DatasetReferenceLink.team_id == dataset.team_id)
    scope_by_id: dict[int, str] = {}
    for ref_id, linked_dataset_id in (
        db.query(DatasetReferenceLink.reference_dataset_id, DatasetReferenceLink.dataset_id)
        .filter(or_(*filters))
        .all()
    ):
        scope = "dataset" if linked_dataset_id is not None else "team"
        # A pool linked both ways is shown as linked to the dataset.
        if scope_by_id.get(ref_id) != "dataset":
            scope_by_id[ref_id] = scope
    if not scope_by_id:
        return []
    pools = db.query(Dataset).filter(Dataset.id.in_(scope_by_id)).order_by(Dataset.name).all()
    return [(pool, scope_by_id[pool.id]) for pool in pools]


def pool_states(
    db: Session, dataset: Dataset, embedding_model_ids: Iterable[int]
) -> dict[int, list[dict]]:
    """For each model: one entry per linked pool with its state and snippet count."""
    model_ids = list(dict.fromkeys(embedding_model_ids))
    pools = linked_pools(db, dataset)
    result: dict[int, list[dict]] = {model_id: [] for model_id in model_ids}
    if not pools or not model_ids:
        return result

    pool_ids = [pool.id for pool, _ in pools]
    # Latest job per (pool, model); a finished job wins over a later failed retry.
    jobs: dict[tuple[int, int], EmbeddingJob] = {}
    for job in (
        db.query(EmbeddingJob)
        .filter(EmbeddingJob.dataset_id.in_(pool_ids), EmbeddingJob.embedding_model_id.in_(model_ids))
        .order_by(EmbeddingJob.created_at, EmbeddingJob.id)
        .all()
    ):
        key = (job.dataset_id, job.embedding_model_id)
        kept = jobs.get(key)
        if kept is not None and kept.status == EmbeddingJobStatus.COMPLETED and job.status != EmbeddingJobStatus.COMPLETED:
            continue
        jobs[key] = job

    finished_ids = [job.id for job in jobs.values() if job.status == EmbeddingJobStatus.COMPLETED]
    vectors = dict(
        db.query(EmbeddingVector.embedding_job_id, func.count(EmbeddingVector.id))
        .filter(EmbeddingVector.embedding_job_id.in_(finished_ids))
        .group_by(EmbeddingVector.embedding_job_id)
        .all()
    ) if finished_ids else {}

    for model_id in model_ids:
        for pool, scope in pools:
            job = jobs.get((pool.id, model_id))
            snippets = 0
            if job is None:
                state = NOT_EMBEDDED
            elif job.status == EmbeddingJobStatus.COMPLETED:
                snippets = int(vectors.get(job.id, 0))
                state = READY if snippets > 0 else NO_SNIPPETS
            elif job.status == EmbeddingJobStatus.FAILED:
                state = FAILED
            else:
                state = EMBEDDING
            result[model_id].append({
                "dataset_id": pool.id,
                "name": pool.name,
                "scope": scope,
                "state": state,
                "snippets": snippets,
            })
    return result


def blocking_pools(db: Session, dataset: Dataset, embedding_model_id: int) -> list[dict]:
    """Pools that keep this model from being chosen for the dataset."""
    return [p for p in pool_states(db, dataset, [embedding_model_id])[embedding_model_id] if p["state"] in BLOCKING]


class ReferencePoolsNotReady(ValueError):
    """The chosen model is missing from some linked reference pools."""

    def __init__(self, pools: list[dict], model_name: Optional[str] = None):
        self.pools = pools
        names = ", ".join(p["name"] for p in pools)
        model = model_name or "this model"
        super().__init__(
            f"These reference pools have no {model} embeddings yet: {names}. "
            "Generate their embeddings first, or unlink them."
        )
