"""
Prediction retention for PAM active learning.

Every retrain writes one prediction row per snippet in the dataset (about
1.6 KB each in Postgres), and nothing removed the rows of superseded
checkpoints. On a 360k-snippet dataset that is ~0.6 GB per retrain, so
`al_predictions` grew without bound.

Per dataset and model family, only these checkpoints keep their predictions:

* the active checkpoint -- what the feed and explore views read;
* the previous checkpoint -- the newest other checkpoint that still has
  predictions, so a page opened just before a retrain keeps working and the
  last change can be compared;
* the first trained checkpoint -- the earliest non-base checkpoint, a stable
  baseline for "how much did annotation improve the model".

Only prediction rows are deleted. Checkpoint rows, model files, annotations
and feedback are untouched (no table references `al_predictions`), and the
predictions of a pruned checkpoint can be recomputed by running inference for
it again.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.models.pam_active_learning import ALModelCheckpoint, ALModelFamilyState

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 50_000


@dataclass
class RetentionPlan:
    dataset_id: int
    model_family_name: str
    active_id: Optional[int]
    previous_id: Optional[int]
    first_trained_id: Optional[int]
    prune_ids: list[int] = field(default_factory=list)

    @property
    def keep_ids(self) -> set[int]:
        return {i for i in (self.active_id, self.previous_id, self.first_trained_id) if i is not None}


def _has_predictions(db: Session, checkpoint_id: int) -> bool:
    # Index lookup on model_checkpoint_id; avoids a DISTINCT over the whole table.
    row = db.execute(
        text("SELECT 1 FROM al_predictions WHERE model_checkpoint_id = :cid LIMIT 1"),
        {"cid": checkpoint_id},
    ).first()
    return row is not None


def plan_retention(db: Session, dataset_id: int, model_family_name: str) -> RetentionPlan:
    """Decide which checkpoints of one dataset/family keep their predictions."""
    checkpoints = (
        db.query(ALModelCheckpoint.id, ALModelCheckpoint.is_base)
        .filter(
            ALModelCheckpoint.dataset_id == dataset_id,
            ALModelCheckpoint.model_family_name == model_family_name,
        )
        .order_by(ALModelCheckpoint.id.asc())
        .all()
    )
    with_predictions = [cid for cid, _ in checkpoints if _has_predictions(db, cid)]

    state = (
        db.query(ALModelFamilyState)
        .filter(
            ALModelFamilyState.dataset_id == dataset_id,
            ALModelFamilyState.model_family_name == model_family_name,
        )
        .first()
    )
    # Without a recorded active checkpoint, treat the newest scored one as active.
    active_id = state.active_model_checkpoint_id if state else None
    if active_id is None and with_predictions:
        active_id = with_predictions[-1]

    older = [cid for cid in with_predictions if cid != active_id]
    previous_id = older[-1] if older else None
    first_trained_id = next((cid for cid, is_base in checkpoints if not is_base), None)

    plan = RetentionPlan(dataset_id, model_family_name, active_id, previous_id, first_trained_id)
    plan.prune_ids = [cid for cid in with_predictions if cid not in plan.keep_ids]
    return plan


def count_predictions(db: Session, checkpoint_ids: list[int]) -> int:
    if not checkpoint_ids:
        return 0
    total = 0
    for cid in checkpoint_ids:
        total += db.execute(
            text("SELECT count(*) FROM al_predictions WHERE model_checkpoint_id = :cid"),
            {"cid": cid},
        ).scalar() or 0
    return total


def delete_predictions(db: Session, checkpoint_ids: list[int], batch_size: int = DEFAULT_BATCH_SIZE) -> int:
    """Delete prediction rows of the given checkpoints in committed batches.

    Small batches keep each transaction short, so concurrent feed queries and
    inference writes are never blocked behind one multi-million-row delete.
    """
    deleted = 0
    for cid in checkpoint_ids:
        while True:
            result = db.execute(
                text(
                    "DELETE FROM al_predictions WHERE id IN ("
                    "SELECT id FROM al_predictions WHERE model_checkpoint_id = :cid LIMIT :n)"
                ),
                {"cid": cid, "n": batch_size},
            )
            db.commit()
            if not result.rowcount:
                break
            deleted += result.rowcount
    return deleted


def prune_family_predictions(
    db: Session,
    dataset_id: int,
    model_family_name: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> int:
    """Apply the retention rule to one dataset/family; returns rows deleted."""
    plan = plan_retention(db, dataset_id, model_family_name)
    if not plan.prune_ids:
        return 0
    deleted = delete_predictions(db, plan.prune_ids, batch_size)
    logger.info(
        "Pruned predictions dataset_id=%d family=%s checkpoints=%s rows=%d "
        "(kept active=%s previous=%s first_trained=%s)",
        dataset_id, model_family_name, plan.prune_ids, deleted,
        plan.active_id, plan.previous_id, plan.first_trained_id,
    )
    return deleted
