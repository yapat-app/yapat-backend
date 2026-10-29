"""
Inference helpers: scoring, prediction CRUD, and suggestion ranking.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Iterable, List, Sequence

import numpy as np
import torch
from sqlalchemy import func
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session, selectinload

from app.models.snippet import Snippet
from app.models.pam_active_learning import ALPrediction, ALSnippetAnnotation
from app.schemas.pam_active_learning import ALInferenceRow

from active_learning.config import (
    DEFAULT_INFERENCE_THRESHOLD,
    DEFAULT_DENSITY_K,
    DEFAULT_COMPOSITE_WU,
    DEFAULT_COMPOSITE_WD,
    DEFAULT_COMPOSITE_WR,
    DIVERSITY_NUM_CENTERS,
    DIVERSITY_UPDATE_K,
)

logger = logging.getLogger(__name__)


def aggregate_confidence(
    predicted_probabilities: dict[str, float],
    label_scope: list[str] | None = None,
) -> float:
    """
    Noisy-OR aggregate confidence: P(at least one label in scope is present).

        c = 1 - prod(1 - p(x_i)  for i in label_scope)

    label_scope: subset of label names to consider.
        - If None or empty, falls back to max(predicted_probabilities.values())
          to avoid the inflation artefact that occurs when many low-probability
          labels are combined without a meaningful scope.
    """
    if not label_scope:
        # No scope → avoid noisy-OR inflation; use max as a conservative fallback.
        return max(predicted_probabilities.values(), default=0.0)

    return 1.0 - math.prod(
        1.0 - predicted_probabilities.get(label, 0.0)
        for label in label_scope
    )


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
    density_k: int, #TODO delete this k. It is not configurable from config file
    wu: float,
    wd: float,
    wr: float,
    ann_index=None,
) -> list[ALInferenceRow]:
    """
    Compute prediction rows for all snippets and attach acquisition scores
    for unlabeled snippets.

    Neighbour lookups go through ``ann_index``, an HNSW index over *every*
    snippet in the set (labelled included) -- see ``_ann_index_cache``. Scores
    are unchanged: labelled hits are dropped from each result so density and
    diversity still range over the unlabelled set only. Working in matrix-row
    space also avoids the ``embeddings[unlabeled_indices]`` copy, which was
    12.3 GB at 3M snippets and the largest single contributor to peak memory.

    ``ann_index=None`` builds one for this call -- the same cost as before,
    just not reused.
    """
    from app.services.pam_al import _ann_index_cache, _scoring

    features = embeddings.detach().cpu().numpy()
    n = features.shape[0]

    ids_array = np.asarray(snippet_ids)
    is_labeled = np.isin(ids_array, np.fromiter(labeled_snippet_ids, dtype=ids_array.dtype, count=len(labeled_snippet_ids))) \
        if labeled_snippet_ids else np.zeros(n, dtype=bool)
    unlabeled_rows = np.flatnonzero(~is_labeled).astype(np.int64)
    labeled_rows = np.flatnonzero(is_labeled).astype(np.int64)

    if ann_index is None:
        ann_index = _ann_index_cache.build_transient_index(features)

    probs_np = probs.detach().cpu().numpy()
    preds_np = preds.detach().cpu().numpy()

    uncertainty_raw = _scoring.compute_uncertainty(probs_np[unlabeled_rows])
    uncertainty_z = _scoring.zscore(uncertainty_raw)
    if uncertainty_raw.size:
        logger.info(
            "pam-al inference: uncertainty min value = %.4f max value = %.4f",
            float(uncertainty_raw.min()), float(uncertainty_raw.max()),
        )

    start = time.perf_counter()
    nearest_labeled = _scoring.compute_nearest_labeled(features, unlabeled_rows, labeled_rows)
    if labeled_rows.size == 0:
        # Cold start: nothing to be diverse *from*, so every point is equally
        # uninformative on this axis and the greedy redundancy pass would only
        # invent differences. Matches ALQueryScorer.diversity's n_l == 0 branch.
        diversity_raw = np.ones(unlabeled_rows.shape[0], dtype=np.float32)
    else:
        diversity_raw = _scoring.compute_diversity(
            ann_index,
            features,
            unlabeled_rows,
            is_labeled,
            nearest_labeled,
            num_centers=DIVERSITY_NUM_CENTERS,
            update_k=DIVERSITY_UPDATE_K,
        )
    diversity_z = _scoring.zscore(diversity_raw)
    if diversity_raw.size:
        logger.info(
            "pam-al inference: diversity min value = %.4f max value = %.4f",
            float(diversity_raw.min()), float(diversity_raw.max()),
        )

    mid = time.perf_counter()
    density_raw = _scoring.compute_density(ann_index, features, unlabeled_rows, is_labeled, density_k)
    density_z = _scoring.zscore(density_raw)
    if density_raw.size:
        logger.info(
            "pam-al inference: density min value = %.4f max value = %.4f",
            float(density_raw.min()), float(density_raw.max()),
        )
    end = time.perf_counter()
    logger.info(
        "pam-al inference: acquisition scoring diversity=%.4fs density=%.4fs total=%.4fs",
        mid - start,
        end - mid,
        end - start,
    )

    composite_scores_u = _scoring.compute_composite(
        uncertainty_z, diversity_z, density_z, wu=wu, wd=wd, wr=wr
    )
    if composite_scores_u.size:
        logger.info(
            "pam-al inference: composite min value = %.4f max value = %.4f",
            float(composite_scores_u.min()), float(composite_scores_u.max()),
        )

    uncertainty_full = np.full(n, np.nan, dtype=np.float64)
    diversity_full = np.full(n, np.nan, dtype=np.float64)
    density_full = np.full(n, np.nan, dtype=np.float64)
    composite_full = np.full(n, np.nan, dtype=np.float64)
    uncertainty_full[unlabeled_rows] = uncertainty_raw
    diversity_full[unlabeled_rows] = diversity_raw
    density_full[unlabeled_rows] = density_raw
    composite_full[unlabeled_rows] = composite_scores_u

    def _score(values, i):
        value = values[i]
        return None if np.isnan(value) else float(value)

    rows: list[ALInferenceRow] = []
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
                uncertainty=_score(uncertainty_full, i),
                diversity=_score(diversity_full, i),
                density=_score(density_full, i),
                composite_score=_score(composite_full, i),
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

    Opens a fresh DB session for writes so that a stale/dead connection from
    the caller's long-running session (idle during forward pass + scoring) never
    causes OperationalError.  Each chunk is committed independently so partial
    progress survives a mid-run failure.
    """
    from app.database import SessionLocal

    rows = list(rows)
    total = len(rows)
    chunk_size = 5000
    total_chunks = (total + chunk_size - 1) // chunk_size

    logger.info(
        "pam-al inference: saving %s prediction rows for checkpoint_id=%s in %s chunks (chunk_size=%s)",
        total,
        model_checkpoint_id,
        total_chunks,
        chunk_size,
    )

    write_db = SessionLocal()
    try:
        bind = write_db.get_bind()
        dialect_name = bind.dialect.name if bind is not None else ""

        for chunk_idx, start in enumerate(range(0, total, chunk_size), start=1):
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
                write_db.execute(stmt)
            else:
                # Preserve compatibility with non-Postgres engines used in local tests.
                snippet_ids = [row.snippet_id for row in chunk]
                existing_rows = (
                    write_db.query(ALPrediction)
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
                    write_db.add_all(to_add)

            write_db.commit()

            logger.info(
                "pam-al inference: upsert chunk %s/%s (rows=%s)",
                chunk_idx,
                total_chunks,
                len(chunk),
            )

    except Exception:
        write_db.rollback()
        raise
    finally:
        write_db.close()

        logger.info(
            "pam-al inference: upsert chunk %s/%s (rows=%s, dialect=%s)",
            chunk_idx,
            total_chunks,
            len(chunk),
            dialect_name,
        )


def _score_matrix(rows: Sequence[ALInferenceRow]) -> np.ndarray:
    """[n, 4] float32 (uncertainty, diversity, density, composite); NaN = unscored."""
    out = np.full((len(rows), 4), np.nan, dtype=np.float32)
    for i, row in enumerate(rows):
        if row.composite_score is None and row.uncertainty is None:
            continue
        for j, value in enumerate((row.uncertainty, row.diversity, row.density, row.composite_score)):
            if value is not None:
                out[i, j] = value
    return out


def _iter_batches(n: int, batch_size: int) -> Iterable[tuple[int, int]]:
    for start in range(0, n, batch_size):
        yield start, min(n, start + batch_size)


def _acquisition_index(db: Session, model_ckpt, X, features, snippet_rows):
    """The cached HNSW index for acquisition scoring, or None for a transient one.

    The index is only cacheable when the space its neighbours live in is stable
    across retrains. That holds when ``extract_features`` returns the input
    embeddings unchanged (the linear classifier) and not when it returns
    hidden-layer activations that move with the weights (the MLP). Rather than
    branching on model_type, compare the two arrays -- exact, cheap, and still
    correct if another architecture is added later.
    """
    from app.services.pam_al._ann_index_cache import load_or_build_index
    from app.services.pam_al._embedding_cache import compute_embedding_fingerprint

    try:
        feats = features.detach().cpu().numpy()
        if feats.shape != tuple(X.shape):
            logger.info("ann index: feature space differs from embeddings; using transient index")
            return None
        probe = min(64, feats.shape[0])
        if not np.array_equal(feats[:probe], np.asarray(X[:probe], dtype=feats.dtype)):
            logger.info("ann index: features are not the stored embeddings; using transient index")
            return None

        hyper = getattr(model_ckpt, "hyperparameters", None) or {}
        embedding_model_id = hyper.get("embedding_model_id")
        if embedding_model_id is None or not snippet_rows:
            return None

        # All snippets in one inference run belong to one set, so the first row
        # identifies it. Cheaper than threading the id through five call sites.
        snippet_set_id = db.query(Snippet.snippet_set_id).filter(
            Snippet.id == int(snippet_rows[0]["snippet_id"])
        ).scalar()
        if snippet_set_id is None:
            return None

        fingerprint = compute_embedding_fingerprint(db, int(snippet_set_id), int(embedding_model_id))
        if fingerprint["count"] != int(X.shape[0]):
            # The matrix is not the whole snippet set (a subset run, or the
            # embeddings changed under us) -- a cached index would not line up
            # with these row indices.
            logger.info(
                "ann index: matrix rows=%s != embeddings count=%s; using transient index",
                X.shape[0], fingerprint["count"],
            )
            return None

        return load_or_build_index(X, int(snippet_set_id), int(embedding_model_id), fingerprint)
    except Exception:
        logger.warning("ann index: could not resolve a cached index; using transient", exc_info=True)
        return None


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

    # End any read transaction before CPU-bound forward pass; callers often load
    # embeddings in the same session, which would otherwise sit idle-in-transaction
    # for minutes and hit idle_in_transaction_session_timeout on Postgres.
    db.commit()

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

    predict_fn = getattr(model, "predict_with_features", None)

    with torch.inference_mode():
        for batch_idx, (start, end) in enumerate(_iter_batches(n, batch_size), start=1):
            x_batch = torch.as_tensor(X[start:end], dtype=torch.float32, device=device)
            if predict_fn is not None:
                feat_b, prob_b, pred_b = predict_fn(x_batch, threshold=threshold)
            else:
                feat_b = model.extract_features(x_batch)
                prob_b, pred_b = model.predict(x_batch, threshold=threshold)
            feat_b = feat_b.detach().cpu()
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

    ann_index = _acquisition_index(db, model_ckpt, X, features, snippet_rows)

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
        ann_index=ann_index,
    )

    logger.info(
        "pam-al inference: scoring/row materialization done in %.2fs (rows=%s)",
        time.perf_counter() - t1,
        len(rows),
    )

    t2 = time.perf_counter()

    save_prediction_rows(db=db, model_checkpoint_id=model_ckpt.id, rows=rows)

    # Invalidate any cached confidence rankings for this checkpoint so the next
    # validate-mode request recomputes from the fresh predictions.
    try:
        from app.services.inference_feed_cache import invalidate_inference_feed
        invalidate_inference_feed(model_ckpt.id)
    except Exception:
        pass

    # Publish the explore-store model layer straight from the tensors, so the
    # Annotation Hub never has to parse millions of JSON prediction rows.
    from app.services.explore.hooks import publish_model_layer_from_inference

    publish_model_layer_from_inference(
        db,
        checkpoint_id=model_ckpt.id,
        snippet_ids=snippet_ids,
        probs=probs.detach().cpu().numpy(),
        preds=preds.detach().cpu().numpy(),
        score_matrix=_score_matrix(rows),
        label_order=label_order,
    )

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
        .options(
            selectinload(ALPrediction.snippet).load_only(
                Snippet.start_time, Snippet.end_time, Snippet.recording_id
            )
        )
        .order_by(ALPrediction.composite_score.desc().nullslast(), ALPrediction.id.asc())
        .all()
    )


def predictions_exist_for_checkpoint_and_snippet_set(
    db: Session,
    model_checkpoint_id: int,
    snippet_set_id: int,
) -> bool:
    return (
        db.query(ALPrediction.id)
        .join(Snippet, Snippet.id == ALPrediction.snippet_id)
        .filter(
            ALPrediction.model_checkpoint_id == model_checkpoint_id,
            Snippet.snippet_set_id == snippet_set_id,
        )
        .first()
        is not None
    )


def count_predictions_for_checkpoint_and_snippet_set(
    db: Session,
    model_checkpoint_id: int,
    snippet_set_id: int,
) -> int:
    count = (
        db.query(func.count(ALPrediction.id))
        .join(Snippet, Snippet.id == ALPrediction.snippet_id)
        .filter(
            ALPrediction.model_checkpoint_id == model_checkpoint_id,
            Snippet.snippet_set_id == snippet_set_id,
        )
        .scalar()
    )
    return int(count or 0)


def _noisy_or_confidence(
    predicted_probabilities: dict | None,
    label_scope: list[str] | None,
) -> float:
    """
    Aggregate confidence over a label scope using noisy-OR:
        P(any species present) = 1 - prod(1 - p_i  for i in scope)

    Falls back to max probability when no scope is given, or 0 if no
    probabilities are available.
    """
    probs = predicted_probabilities or {}
    if not probs:
        return 0.0
    if label_scope:
        scope_probs = [probs.get(s, 0.0) for s in label_scope]
    else:
        scope_probs = list(probs.values())
    if not scope_probs:
        return 0.0
    result = 1.0
    for p in scope_probs:
        result *= 1.0 - max(0.0, min(1.0, float(p)))
    return 1.0 - result


def get_top_prediction_suggestions(
    db: Session,
    dataset_id: int,
    model_checkpoint_id: int,
    snippet_set_id: int,
    strategy: str,
    k: int,
    label_scope: list[str] | None = None,
) -> list[ALPrediction]:
    annotated_exists = (
        db.query(ALSnippetAnnotation.id)
        .filter(
            ALSnippetAnnotation.dataset_id == dataset_id,
            ALSnippetAnnotation.snippet_id == ALPrediction.snippet_id,
        )
        .exists()
    )

    query = (
        db.query(ALPrediction)
        .join(Snippet, Snippet.id == ALPrediction.snippet_id)
        .filter(
            ALPrediction.model_checkpoint_id == model_checkpoint_id,
            Snippet.snippet_set_id == snippet_set_id,
            ~annotated_exists,
        )
    )

    if strategy == "random":
        return (
            query.order_by(func.random())
            .limit(k)
            .options(
                selectinload(ALPrediction.snippet).load_only(
                    Snippet.start_time, Snippet.end_time, Snippet.recording_id
                )
            )
            .all()
        )

    # confidence: noisy-OR over label_scope — must be computed in Python since
    # predicted_probabilities is a JSON column.
    #
    # Cache the full ranked list (ignoring per-request annotation filter) so
    # repeat calls (e.g. every mode-switch to validate) are served in <50ms
    # instead of loading 100k+ rows into Python on every request.
    if strategy == "confidence":
        from app.services.inference_feed_cache import (
            get_cached_confidence_ranking,
            set_cached_confidence_ranking,
        )

        ranked_triples = get_cached_confidence_ranking(
            model_checkpoint_id, snippet_set_id, label_scope
        )

        if ranked_triples is None:
            # Cache miss: load all predictions for this checkpoint+snippet_set (no
            # annotation filter — the filter is applied cheaply after sorting).
            all_preds = (
                db.query(ALPrediction)
                .join(Snippet, Snippet.id == ALPrediction.snippet_id)
                .filter(
                    ALPrediction.model_checkpoint_id == model_checkpoint_id,
                    Snippet.snippet_set_id == snippet_set_id,
                )
                .options(
                    selectinload(ALPrediction.snippet).load_only(
                        Snippet.start_time, Snippet.end_time, Snippet.recording_id
                    )
                )
                .all()
            )
            scored = [
                (p, _noisy_or_confidence(p.predicted_probabilities, label_scope))
                for p in all_preds
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            all_preds = [p for p, _ in scored]
            ranked_triples = [(p.id, p.snippet_id, s) for p, s in scored]
            set_cached_confidence_ranking(model_checkpoint_id, snippet_set_id, label_scope, ranked_triples)

            # Filter out already-annotated snippets using the pre-sorted objects.
            annotated_exists_set = {
                row[0]
                for row in db.query(ALSnippetAnnotation.snippet_id).filter(
                    ALSnippetAnnotation.dataset_id == dataset_id,
                ).all()
            }
            candidates = [p for p in all_preds if p.snippet_id not in annotated_exists_set]
            return candidates[:k]

        # Cache hit: filter annotated IDs and fetch only the top-k full objects.
        annotated_snippet_ids = {
            row[0]
            for row in db.query(ALSnippetAnnotation.snippet_id).filter(
                ALSnippetAnnotation.dataset_id == dataset_id,
            ).all()
        }
        top_k_pred_ids = [
            pred_id
            for pred_id, snippet_id, _ in ranked_triples
            if snippet_id not in annotated_snippet_ids
        ][:k]

        if not top_k_pred_ids:
            return []

        id_to_rank = {pred_id: rank for rank, pred_id in enumerate(top_k_pred_ids)}
        objs = (
            db.query(ALPrediction)
            .filter(ALPrediction.id.in_(top_k_pred_ids))
            .options(
                selectinload(ALPrediction.snippet).load_only(
                    Snippet.start_time, Snippet.end_time, Snippet.recording_id
                )
            )
            .all()
        )
        objs.sort(key=lambda p: id_to_rank[p.id])
        return objs

    if strategy == "confidence":
        # Noisy-OR is label_scope-specific and cannot be pushed into a JSONB SQL
        # expression, so we always sort in Python.
        candidates = query.all()
        candidates.sort(
            key=lambda p: aggregate_confidence(p.predicted_probabilities or {}, label_scope),
            reverse=True,
        )
        return candidates[:k]

    score_columns = {
        "uncertainty": ALPrediction.uncertainty,
        "diversity": ALPrediction.diversity,
        "density": ALPrediction.density,
        "composite": ALPrediction.composite_score,
    }
    if strategy not in score_columns:
        raise ValueError(f"Unsupported suggestion strategy '{strategy}'.")

    score_column = score_columns[strategy]
    return (
        query.order_by(score_column.desc().nullslast(), ALPrediction.id.asc())
        .limit(k)
        .options(
            selectinload(ALPrediction.snippet).load_only(
                Snippet.start_time, Snippet.end_time, Snippet.recording_id
            )
        )
        .all()
    )


def rank_prediction_suggestions(
    db: Session,
    dataset_id: int,
    snippet_set_id: int,
    predictions: list[ALPrediction],
    strategy: str,
    annotated_ids: set[int],
    label_scope: list[str] | None = None,
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
        "confidence": lambda p: aggregate_confidence(
            p.predicted_probabilities or {},
            label_scope,
        ),
    }

    if strategy not in key_map:
        raise ValueError(f"Unsupported suggestion strategy '{strategy}'.")

    return sorted(candidates, key=key_map[strategy], reverse=True)
