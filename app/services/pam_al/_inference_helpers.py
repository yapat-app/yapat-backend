"""
Inference helpers: scoring, prediction CRUD, and suggestion ranking.
"""

from __future__ import annotations

import logging
import time
from typing import Iterable, List, Sequence

import numpy as np
import torch
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.models.snippet import Snippet
from app.models.pam_active_learning import ALPrediction
from app.schemas.pam_active_learning import ALInferenceRow

from active_learning.samplers import uncertainty, density, diversity, composite
from active_learning.config import (
    DEFAULT_INFERENCE_THRESHOLD,
    DEFAULT_DENSITY_K,
    DEFAULT_COMPOSITE_WU,
    DEFAULT_COMPOSITE_WD,
    DEFAULT_COMPOSITE_WR,
)

logger = logging.getLogger(__name__)


def resolve_inference_params(
    threshold: float | None,
    density_k: int | None,
    wu: float | None,
    wd: float | None,
    wr: float | None,
) -> tuple[float, int, float, float, float]:
    return (
        threshold if threshold is not None else DEFAULT_INFERENCE_THRESHOLD,
        density_k if density_k is not None else DEFAULT_DENSITY_K,
        wu if wu is not None else DEFAULT_COMPOSITE_WU,
        wd if wd is not None else DEFAULT_COMPOSITE_WD,
        wr if wr is not None else DEFAULT_COMPOSITE_WR,
    )


def build_inference_rows(
    probs: torch.Tensor,
    preds: torch.Tensor,
    embeddings: torch.Tensor,
    snippet_ids: Sequence[int],
    labeled_snippet_ids: set[int],
    label_order: List[str],
    density_k: int,
    wu: float,
    wd: float,
    wr: float,
) -> list[ALInferenceRow]:
    """
    Compute prediction rows for all snippets and attach acquisition scores
    for unlabeled snippets.
    """

    unlabeled_indices = [i for i, sid in enumerate(snippet_ids) if sid not in labeled_snippet_ids]
    labeled_indices = [i for i, sid in enumerate(snippet_ids) if sid in labeled_snippet_ids]

    z_u = embeddings[unlabeled_indices] if unlabeled_indices else torch.empty(
        (0, embeddings.shape[1]), device=embeddings.device
    )
    z_l = embeddings[labeled_indices] if labeled_indices else torch.empty(
        (0, embeddings.shape[1]), device=embeddings.device
    )

    uncertainty_scores_u = uncertainty(probs[unlabeled_indices]) if unlabeled_indices else torch.empty(
        0, device=embeddings.device
    )

    start = time.perf_counter()
    diversity_scores_u = diversity(z_u, z_l)
    mid = time.perf_counter()
    density_scores_u = density(z_u, k=density_k)
    end = time.perf_counter()
    logger.info(
        "pam-al inference: acquisition scoring diversity=%.4fs density=%.4fs total=%.4fs",
        mid - start,
        end - mid,
        end - start,
    )
    composite_scores_u = composite(
        uncertainty_scores=uncertainty_scores_u,
        diversity_scores=diversity_scores_u,
        density_scores=density_scores_u,
        wu=wu,
        wd=wd,
        wr=wr,
    )

    uncertainty_full = [None] * len(snippet_ids)
    diversity_full = [None] * len(snippet_ids)
    density_full = [None] * len(snippet_ids)
    composite_full = [None] * len(snippet_ids)

    if unlabeled_indices:
        uncertainty_values = uncertainty_scores_u.detach().cpu().numpy()
        diversity_values = diversity_scores_u.detach().cpu().numpy()
        density_values = density_scores_u.detach().cpu().numpy()
        composite_values = composite_scores_u.detach().cpu().numpy()

        for pos, idx in enumerate(unlabeled_indices):
            uncertainty_full[idx] = float(uncertainty_values[pos])
            diversity_full[idx] = float(diversity_values[pos])
            density_full[idx] = float(density_values[pos])
            composite_full[idx] = float(composite_values[pos])

    rows: list[ALInferenceRow] = []
    probs_np = probs.detach().cpu().numpy()
    preds_np = preds.detach().cpu().numpy()

    for i, snippet_id in enumerate(snippet_ids):
        pred_indices = np.flatnonzero(preds_np[i] > 0)
        pred_labels = [label_order[j] for j in pred_indices]
        prob_dict = dict(zip(label_order, map(float, probs_np[i])))

        rows.append(
            ALInferenceRow(
                snippet_id=snippet_id,
                # Embedding vectors are stored in the dedicated embedding store.
                # Avoid duplicating large vectors into the predictions table.
                embedding=None,
                predicted_labels=pred_labels,
                predicted_probabilities=prob_dict,
                uncertainty=uncertainty_full[i],
                diversity=diversity_full[i],
                density=density_full[i],
                composite_score=composite_full[i],
            )
        )

    return rows


def save_prediction_rows(
    db: Session,
    model_checkpoint_id: int,
    rows,
) -> None:
    """
    Persist model predictions using chunked bulk upserts.

    PostgreSQL uses the checkpoint/snippet unique constraint for conflict
    resolution, avoiding ORM hydration of existing prediction rows.
    """

    # Keep each statement large enough for throughput while bounding payload size.
    chunk_size = 5000
    rows = list(rows)

    total = len(rows)
    total_chunks = (total + chunk_size - 1) // chunk_size
    logger.info(
        "pam-al inference: saving %s prediction rows for checkpoint_id=%s in %s chunks (chunk_size=%s)",
        total,
        model_checkpoint_id,
        total_chunks,
        chunk_size,
    )

    bind = db.get_bind()
    dialect_name = bind.dialect.name if bind is not None else ""

    for chunk_idx, start in enumerate(range(0, len(rows), chunk_size), start=1):
        chunk = rows[start : start + chunk_size]
        values = [
            {
                "model_checkpoint_id": model_checkpoint_id,
                "snippet_id": row.snippet_id,
                "predicted_labels": row.predicted_labels,
                "predicted_probabilities": row.predicted_probabilities,
                "uncertainty": row.uncertainty,
                "diversity": row.diversity,
                "density": row.density,
                "composite_score": row.composite_score,
            }
            for row in chunk
        ]

        if dialect_name == "postgresql":
            stmt = pg_insert(ALPrediction).values(values)
            stmt = stmt.on_conflict_do_update(
                constraint="uq_al_prediction",
                set_={
                    "predicted_labels": stmt.excluded.predicted_labels,
                    "predicted_probabilities": stmt.excluded.predicted_probabilities,
                    "uncertainty": stmt.excluded.uncertainty,
                    "diversity": stmt.excluded.diversity,
                    "density": stmt.excluded.density,
                    "composite_score": stmt.excluded.composite_score,
                },
            )
            db.execute(stmt)
        else:
            # Preserve compatibility with non-Postgres engines used in local tests.
            snippet_ids = [row.snippet_id for row in chunk]
            existing_rows = (
                db.query(ALPrediction)
                .filter(
                    ALPrediction.model_checkpoint_id == model_checkpoint_id,
                    ALPrediction.snippet_id.in_(snippet_ids),
                )
                .all()
            )
            existing_by_sid = {p.snippet_id: p for p in existing_rows}

            to_add: list[ALPrediction] = []
            for row in chunk:
                pred = existing_by_sid.get(row.snippet_id)
                if pred is None:
                    pred = ALPrediction(model_checkpoint_id=model_checkpoint_id, snippet_id=row.snippet_id)
                    to_add.append(pred)

                pred.predicted_labels = row.predicted_labels
                pred.predicted_probabilities = row.predicted_probabilities
                pred.uncertainty = row.uncertainty
                pred.diversity = row.diversity
                pred.density = row.density
                pred.composite_score = row.composite_score

            if to_add:
                db.add_all(to_add)

        db.flush()

        logger.info(
            "pam-al inference: upsert chunk %s/%s (rows=%s, dialect=%s)",
            chunk_idx,
            total_chunks,
            len(chunk),
            dialect_name,
        )


def _iter_batches(n: int, batch_size: int) -> Iterable[tuple[int, int]]:
    for start in range(0, n, batch_size):
        yield start, min(n, start + batch_size)


def run_and_store_inference(
    db: Session,
    dataset_id: int,
    model_ckpt,
    model,
    X,
    snippet_rows,
    label_order: list[str],
    labeled_snippet_ids: set[int],
    threshold: float | None = None,
    density_k: int | None = None,
    wu: float | None = None,
    wd: float | None = None,
    wr: float | None = None,
) -> dict:
    threshold, density_k, wu, wd, wr = resolve_inference_params(
        threshold=threshold, density_k=density_k, wu=wu, wd=wd, wr=wr,
    )

    device = next(model.parameters()).device

    snippet_ids = [row["snippet_id"] for row in snippet_rows]

    # Batch size keeps GPU/CPU memory bounded. If the checkpoint stored a batch
    # size, prefer that; otherwise use a conservative default.
    h = getattr(model_ckpt, "hyperparameters", None) or {}
    batch_size = int(h.get("batch_size") or 256)
    batch_size = max(1, batch_size)

    n = int(X.shape[0])
    num_batches = (n + batch_size - 1) // batch_size
    logger.info(
        "pam-al inference: running inference dataset_id=%s checkpoint_id=%s on device=%s (n=%s, batch_size=%s, num_batches=%s)",
        dataset_id,
        getattr(model_ckpt, "id", None),
        device,
        n,
        batch_size,
        num_batches,
    )

    # Acquisition scoring needs features/probabilities for the full snippet set.
    # Keep intermediate tensors on CPU to reduce VRAM, and process GPU batches
    # sequentially.
    t0 = time.perf_counter()
    features: torch.Tensor | None = None
    probs: torch.Tensor | None = None
    preds: torch.Tensor | None = None

    with torch.inference_mode():
        for batch_idx, (start, end) in enumerate(_iter_batches(n, batch_size), start=1):
            x_batch = torch.as_tensor(X[start:end], dtype=torch.float32, device=device)
            feat_b = model.extract_features(x_batch).detach().cpu()
            prob_b, pred_b = model.predict(x_batch, threshold=threshold)
            prob_b = prob_b.detach().cpu()
            pred_b = pred_b.detach().cpu()

            # Preallocate output tensors once we know the feature/prob dimensions.
            if features is None:
                features = torch.empty((n, feat_b.shape[1]), dtype=feat_b.dtype)
            if probs is None:
                probs = torch.empty((n, prob_b.shape[1]), dtype=prob_b.dtype)
            if preds is None:
                preds = torch.empty((n, pred_b.shape[1]), dtype=pred_b.dtype)

            features[start:end] = feat_b
            probs[start:end] = prob_b
            preds[start:end] = pred_b

            if batch_idx == 1 or batch_idx == num_batches or (batch_idx % 10 == 0):
                logger.info(
                    "pam-al inference: batch %s/%s (snippets %s..%s)",
                    batch_idx,
                    num_batches,
                    start,
                    end,
                )


    if features is None or probs is None or preds is None:
        raise ValueError("Inference input is empty; no predictions generated.")

    logger.info(
        "pam-al inference: forward pass done in %.2fs (n=%s)",
        time.perf_counter() - t0,
        n,
    )

    t1 = time.perf_counter()

    rows = build_inference_rows(
        probs=probs,
        preds=preds,
        embeddings=features,
        snippet_ids=snippet_ids,
        labeled_snippet_ids=labeled_snippet_ids,
        label_order=label_order,
        density_k=density_k,
        wu=wu,
        wd=wd,
        wr=wr,
    )

    logger.info(
        "pam-al inference: scoring/row materialization done in %.2fs (rows=%s)",
        time.perf_counter() - t1,
        len(rows),
    )

    t2 = time.perf_counter()

    save_prediction_rows(db=db, model_checkpoint_id=model_ckpt.id, rows=rows)

    logger.info(
        "pam-al inference: DB upsert done in %.2fs",
        time.perf_counter() - t2,
    )

    logger.info(
        "pam-al inference: completed checkpoint_id=%s (rows=%s, batch_size=%s, num_batches=%s)",
        getattr(model_ckpt, "id", None),
        len(rows),
        batch_size,
        num_batches,
    )

    return {
        "num_predictions": len(rows),
        "num_labeled_snippets": len(labeled_snippet_ids),
        "threshold": threshold,
        "density_k": density_k,
        "composite_wu": wu,
        "composite_wd": wd,
        "composite_wr": wr,
        "batch_size": batch_size,
    }


def get_predictions_for_checkpoint_and_snippet_set(
    db: Session,
    model_checkpoint_id: int,
    snippet_set_id: int,
) -> list[ALPrediction]:
    return (
        db.query(ALPrediction)
        .join(Snippet, Snippet.id == ALPrediction.snippet_id)
        .filter(
            ALPrediction.model_checkpoint_id == model_checkpoint_id,
            Snippet.snippet_set_id == snippet_set_id,
        )
        .order_by(ALPrediction.composite_score.desc().nullslast(), ALPrediction.id.asc())
        .all()
    )


def rank_prediction_suggestions(
    db: Session,
    dataset_id: int,
    snippet_set_id: int,
    predictions: list[ALPrediction],
    strategy: str,
    annotated_ids: set[int],
) -> list[ALPrediction]:
    candidates = [p for p in predictions if p.snippet_id not in annotated_ids]

    if strategy == "random":
        import random
        candidates = candidates[:]
        random.shuffle(candidates)
        return candidates

    key_map = {
        "uncertainty": lambda p: p.uncertainty if p.uncertainty is not None else float("-inf"),
        "diversity": lambda p: p.diversity if p.diversity is not None else float("-inf"),
        "density": lambda p: p.density if p.density is not None else float("-inf"),
        "composite": lambda p: p.composite_score if p.composite_score is not None else float("-inf"),
    }

    if strategy not in key_map:
        raise ValueError(f"Unsupported suggestion strategy '{strategy}'.")

    return sorted(candidates, key=key_map[strategy], reverse=True)
