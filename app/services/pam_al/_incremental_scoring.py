"""
Score only the snippets a checkpoint has no prediction for yet (yapat-backend#70).

Used after "add recordings": the active checkpoint already scored every old
snippet, so re-running full inference (~18 min at 3M snippets) would recompute
values that cannot change. Instead, for the unscored snippets only:

* probabilities, predicted labels and uncertainty come from a forward pass over
  those rows. They depend on nothing but the snippet, so they are identical to
  what a full inference would produce.
* diversity is the distance to the nearest *labelled* snippet in the full set,
  as in full inference, minus its greedy redundancy correction (which lowers the
  score of the ~update_k neighbours of DIVERSITY_NUM_CENTERS picked centres, a
  tiny fraction of millions of rows). The next retrain applies it again.
* density uses the k nearest *unlabelled* neighbours among *all* snippets, so a
  new snippet is compared with the old ones, not just with its own recording.
* composite z-scores each component against the population statistics of the
  checkpoint's stored scores, so a new composite is on the same scale as the
  existing ones in the ranked feed.

Existing rows are left as they are (their density/composite go slightly stale
near the new recordings until the next retrain, which recomputes everything).
"""

from __future__ import annotations

import logging
import math
from typing import Callable, Optional

import numpy as np
import torch
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models.embedding import EmbeddingVector
from app.models.pam_active_learning import ALPrediction
from app.models.snippet import Snippet
from app.schemas.pam_active_learning import ALInferenceRow
from app.services.pam_al import (
    _annotation_helpers as ann_h,
    _checkpoint_helpers as ckpt_h,
    _data_helpers as data_h,
    _inference_helpers as inf_h,
    _scoring,
)
from app.services.pam_al._ann_index_cache import (
    build_transient_index,
    load_or_build_index,
    normalize_rows,
)
from app.services.pam_al._embedding_cache import compute_embedding_fingerprint

logger = logging.getLogger(__name__)

SCORE_FIELDS = ("uncertainty", "diversity", "density")


def unscored_snippet_ids(db: Session, checkpoint_id: int, snippet_set_id: int, embedding_model_id: int) -> list[int]:
    """Snippets of the set that have a vector but no prediction for this checkpoint."""
    rows = (
        db.query(Snippet.id)
        .join(
            EmbeddingVector,
            (EmbeddingVector.snippet_id == Snippet.id)
            & (EmbeddingVector.embedding_model_id == embedding_model_id),
        )
        .outerjoin(
            ALPrediction,
            (ALPrediction.snippet_id == Snippet.id)
            & (ALPrediction.model_checkpoint_id == checkpoint_id),
        )
        .filter(Snippet.snippet_set_id == snippet_set_id, ALPrediction.id.is_(None))
        .order_by(Snippet.id)
        .all()
    )
    return [r[0] for r in rows]


def population_stats(db: Session, checkpoint_id: int) -> dict[str, tuple[int, float, float]]:
    """{score: (n, mean, sample std)} over the checkpoint's scored (unlabelled) rows.

    Sums rather than STDDEV so it runs on SQLite (tests) as well as Postgres.
    """
    cols = [getattr(ALPrediction, name) for name in SCORE_FIELDS]
    aggregates = [func.count(ALPrediction.id)]
    for col in cols:
        aggregates += [func.sum(col), func.sum(col * col)]
    row = (
        db.query(*aggregates)
        .filter(
            ALPrediction.model_checkpoint_id == checkpoint_id,
            ALPrediction.composite_score.isnot(None),
        )
        .one()
    )
    n = int(row[0] or 0)
    stats: dict[str, tuple[int, float, float]] = {}
    for i, name in enumerate(SCORE_FIELDS):
        total = float(row[1 + 2 * i] or 0.0)
        total_sq = float(row[2 + 2 * i] or 0.0)
        if n == 0:
            stats[name] = (0, 0.0, 0.0)
            continue
        mean = total / n
        var = (total_sq - n * mean * mean) / (n - 1) if n > 1 else 0.0
        stats[name] = (n, mean, math.sqrt(max(var, 0.0)))
    return stats


def z_against(values: np.ndarray, stat: tuple[int, float, float]) -> np.ndarray:
    """Standardise with the stored population's mean/std (same guards as _scoring.zscore).

    Without a population to compare against (checkpoint never scored anything),
    fall back to standardising the new values among themselves.
    """
    n, mean, std = stat
    values = np.asarray(values, dtype=np.float32)
    if n < 2:
        return _scoring.zscore(values)
    if not np.isfinite(std) or std < 1e-8:
        return np.zeros_like(values)
    return ((values - mean) / std).astype(np.float32)


def _forward(model, X: np.ndarray, rows: np.ndarray, threshold: float, batch_size: int, device):
    """(features, probs, preds) for X[rows], batched, on CPU."""
    feats, probs, preds = [], [], []
    predict_fn = getattr(model, "predict_with_features", None)
    with torch.inference_mode():
        for start in range(0, rows.shape[0], batch_size):
            x = torch.as_tensor(np.asarray(X[rows[start:start + batch_size]]), dtype=torch.float32, device=device)
            if predict_fn is not None:
                f, p, d = predict_fn(x, threshold=threshold)
            else:
                f = model.extract_features(x)
                p, d = model.predict(x, threshold=threshold)
            feats.append(f.detach().cpu().numpy())
            probs.append(p.detach().cpu().numpy())
            preds.append(d.detach().cpu().numpy())
    return np.concatenate(feats), np.concatenate(probs), np.concatenate(preds)


def _all_features(model, X: np.ndarray, batch_size: int, device) -> np.ndarray:
    """Feature matrix for every row (MLP: hidden activations, not the embeddings)."""
    out = None
    with torch.inference_mode():
        for start in range(0, X.shape[0], batch_size):
            x = torch.as_tensor(np.asarray(X[start:start + batch_size]), dtype=torch.float32, device=device)
            f = model.extract_features(x).detach().cpu().numpy().astype(np.float32)
            if out is None:
                out = np.empty((X.shape[0], f.shape[1]), dtype=np.float32)
            out[start:start + f.shape[0]] = f
    return out


def _density(index, F: np.ndarray, rows: np.ndarray, is_labeled: np.ndarray, k: int) -> np.ndarray:
    """1 / mean distance to the k nearest unlabelled neighbours in the *whole* set.

    _scoring.compute_density caps k by the number of rows it is given, which
    would be wrong here (a handful of new rows against millions), so the cap is
    the size of the full unlabelled pool instead.
    """
    if rows.size == 0:
        return np.empty(0, dtype=np.float32)
    n_unlabeled_total = int((~is_labeled).sum())
    k_eff = min(k, n_unlabeled_total - 1)
    if k_eff < 1:
        return np.zeros(rows.shape[0], dtype=np.float32)
    avg = np.empty(rows.shape[0], dtype=np.float32)
    for start in range(0, rows.shape[0], _scoring.QUERY_CHUNK):
        chunk = rows[start:start + _scoring.QUERY_CHUNK]
        avg[start:start + chunk.shape[0]] = _scoring._mean_unlabeled_distance(
            index, normalize_rows(F[chunk]), chunk, is_labeled, k_eff
        )
    return (1.0 / (avg + 1e-8)).astype(np.float32)


def score_unscored_snippets(
    db: Session,
    model_ckpt,
    snippet_set_id: int,
    *,
    device: Optional[str] = None,
    on_total: Optional[Callable[[int], None]] = None,
    on_scored: Optional[Callable[[int], None]] = None,
) -> int:
    """Predict + score the checkpoint's unscored snippets in the set. Returns rows written."""
    from app.services.pam_al.service import _resolve_device

    hyper = model_ckpt.hyperparameters or {}
    embedding_model_id = hyper.get("embedding_model_id")
    if embedding_model_id is None:
        raise ValueError(f"Checkpoint {model_ckpt.id} has no embedding_model_id")
    embedding_model_id = int(embedding_model_id)

    todo_ids = unscored_snippet_ids(db, model_ckpt.id, snippet_set_id, embedding_model_id)
    if on_total is not None:
        on_total(len(todo_ids))
    if not todo_ids:
        return 0

    X, snippet_rows = data_h.load_embeddings(db, snippet_set_id, embedding_model_id)
    all_ids = np.asarray([row["snippet_id"] for row in snippet_rows], dtype=np.int64)
    fingerprint = compute_embedding_fingerprint(db, snippet_set_id, embedding_model_id)
    labeled_ids = ann_h.get_labeled_snippet_ids_for_dataset(db, model_ckpt.dataset_id)
    stats = population_stats(db, model_ckpt.id)
    # Long CPU work follows: don't sit idle in the read transaction.
    db.commit()

    todo = np.asarray(todo_ids, dtype=np.int64)
    pos = np.searchsorted(all_ids, todo)
    present = (pos < all_ids.size) & (all_ids[np.minimum(pos, all_ids.size - 1)] == todo)
    rows_new = pos[present].astype(np.int64)
    new_ids = all_ids[rows_new]
    if rows_new.size == 0:
        return 0

    device = _resolve_device(device)
    label_order = ckpt_h.resolve_checkpoint_label_order(model_ckpt)
    model = ckpt_h.load_model_from_checkpoint(model_ckpt, device=device)
    threshold, density_k, wu, wd, wr = inf_h.resolve_inference_params(None, None, None, None, None)
    batch_size = max(1, int(hyper.get("batch_size") or 256))

    feats_new, probs_new, preds_new = _forward(model, X, rows_new, threshold, batch_size, device)

    # Neighbour space: the stored embeddings for the linear head (cached index
    # over the whole set), hidden activations for the MLP (transient index).
    x_new = np.asarray(X[rows_new], dtype=np.float32)
    linear = feats_new.shape == x_new.shape and np.allclose(feats_new[:64], x_new[:64])
    if linear:
        F = X
        if fingerprint["count"] == X.shape[0]:
            index = load_or_build_index(X, snippet_set_id, embedding_model_id, fingerprint)
        else:
            index = build_transient_index(X)
    else:
        F = _all_features(model, X, batch_size, device)
        index = build_transient_index(F)

    is_labeled = (
        np.isin(all_ids, np.fromiter(labeled_ids, dtype=np.int64, count=len(labeled_ids)))
        if labeled_ids else np.zeros(all_ids.size, dtype=bool)
    )
    labeled_rows = np.flatnonzero(is_labeled).astype(np.int64)
    unl = ~is_labeled[rows_new]
    rows_u = rows_new[unl]

    uncertainty = _scoring.compute_uncertainty(probs_new[unl])
    if labeled_rows.size == 0:
        diversity = np.ones(rows_u.shape[0], dtype=np.float32)
    else:
        diversity = _scoring.compute_nearest_labeled(F, rows_u, labeled_rows)
    density = _density(index, F, rows_u, is_labeled, density_k)
    composite = _scoring.compute_composite(
        z_against(uncertainty, stats["uncertainty"]),
        z_against(diversity, stats["diversity"]),
        z_against(density, stats["density"]),
        wu=wu, wd=wd, wr=wr,
    )

    scores = np.full((rows_new.shape[0], 4), np.nan, dtype=np.float32)
    scores[unl, 0] = uncertainty
    scores[unl, 1] = diversity
    scores[unl, 2] = density
    scores[unl, 3] = composite

    def _opt(v):
        return None if np.isnan(v) else float(v)

    rows = [
        ALInferenceRow(
            snippet_id=int(sid),
            predicted_labels=[label_order[j] for j in np.flatnonzero(preds_new[i] > 0)],
            predicted_probabilities=dict(zip(label_order, map(float, probs_new[i]))),
            uncertainty=_opt(scores[i, 0]),
            diversity=_opt(scores[i, 1]),
            density=_opt(scores[i, 2]),
            composite_score=_opt(scores[i, 3]),
        )
        for i, sid in enumerate(new_ids)
    ]
    inf_h.save_prediction_rows(db=db, model_checkpoint_id=model_ckpt.id, rows=rows)
    if on_scored is not None:
        on_scored(len(rows))

    try:
        from app.services.inference_feed_cache import invalidate_inference_feed

        invalidate_inference_feed(model_ckpt.id)
    except Exception:
        pass

    from app.services.explore.hooks import append_to_model_layer

    append_to_model_layer(
        checkpoint_id=model_ckpt.id,
        snippet_set_id=snippet_set_id,
        snippet_ids=new_ids,
        probs=probs_new,
        preds=preds_new,
        score_matrix=scores,
        label_order=label_order,
    )

    logger.info(
        "pam-al incremental scoring: checkpoint=%s snippet_set=%s scored=%s (unlabelled=%s)",
        model_ckpt.id, snippet_set_id, len(rows), int(unl.sum()),
    )
    return len(rows)
