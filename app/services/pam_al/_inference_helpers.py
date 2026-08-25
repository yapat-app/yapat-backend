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

from active_learning.samplers import composite, zscore, ALQueryScorer
from active_learning.config import (
    DEFAULT_INFERENCE_THRESHOLD,
    DEFAULT_DENSITY_K,
    DEFAULT_COMPOSITE_WU,
    DEFAULT_COMPOSITE_WD,
    DEFAULT_COMPOSITE_WR,
)

logger = logging.getLogger(__name__)


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def aggregate_confidence(
    predicted_probabilities: dict[str, float] | None,
    label_scope: list[str] | None = None,
) -> float:
    """
    Noisy-OR aggregate confidence: P(at least one label in scope is present).

        c = 1 - prod(1 - p(x_i)  for i in label_scope)

    label_scope: subset of label names to consider.
        - If None or empty, falls back to max(predicted_probabilities.values())
          to avoid the inflation artefact that occurs when many low-probability
          labels are combined without a meaningful scope.

    This is the single definition of confidence: ranking, min_confidence
    filtering and the frontend mirror (utils/aggregateConfidence.ts) must all
    agree, otherwise a threshold means different things on different paths.
    """
    probs = predicted_probabilities or {}
    if not label_scope:
        # No scope → avoid noisy-OR inflation; use max as a conservative fallback.
        return _clamp01(max(probs.values(), default=0.0))

    return 1.0 - math.prod(
        1.0 - _clamp01(probs.get(label, 0.0))
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

    # One scorer per cycle: caches z_u_np/z_l_np and the shared HNSW indices
    # so uncertainty()/diversity()/density() below don't each redo the same
    # L2-normalize and index-build work. Each method is called once, raw
    # (the scorer's default), and the z-scored version composite() needs is
    # derived from that same raw tensor via zscore() rather than calling the
    # method again -- avoids recomputing entropy (uncertainty isn't cached).
    scorer = ALQueryScorer(z_u, z_l)

    uncertainty_raw = (
        scorer.uncertainty(probs[unlabeled_indices])
        if unlabeled_indices
        else torch.empty(0, device=embeddings.device)
    )
    uncertainty_z = zscore(uncertainty_raw)
    logger.info(
        "pam-al inference: uncertainty min value = %.4f max value = %.4f",
        uncertainty_raw.min().item(), uncertainty_raw.max().item(),
    )

    start = time.perf_counter()
    diversity_raw = scorer.diversity()
    diversity_z = zscore(diversity_raw)
    logger.info(
        "pam-al inference: diversity min value = %.4f max value = %.4f",
        diversity_raw.min().item(), diversity_raw.max().item(),
    )
    mid = time.perf_counter()
    density_raw = scorer.density()
    density_z = zscore(density_raw)
    logger.info(
        "pam-al inference: density min value = %.4f max value = %.4f",
        density_raw.min().item(), density_raw.max().item(),
    )
    end = time.perf_counter()
    logger.info(
        "pam-al inference: acquisition scoring diversity=%.4fs density=%.4fs total=%.4fs",
        mid - start,
        end - mid,
        end - start,
    )
    composite_scores_u = composite(
        uncertainty_scores=uncertainty_z,
        diversity_scores=diversity_z,
        density_scores=density_z,
        wu=wu,
        wd=wd,
        wr=wr,
    )
    logger.info(
        "pam-al inference: composite min value = %.4f max value = %.4f",
        composite_scores_u.min().item(), composite_scores_u.max().item(),
    )

    uncertainty_full = [None] * len(snippet_ids)
    diversity_full = [None] * len(snippet_ids)
    density_full = [None] * len(snippet_ids)
    composite_full = [None] * len(snippet_ids)

    if unlabeled_indices:
        uncertainty_values = uncertainty_raw.detach().cpu().numpy()
        diversity_values = diversity_raw.detach().cpu().numpy()
        density_values = density_raw.detach().cpu().numpy()
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

    rows = build_inference_rows(
        probs=probs,
        preds=preds,
        embeddings=features,
        snippet_ids=snippet_ids,
        labeled_snippet_ids=labeled_snippet_ids,
        label_order=label_order,
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

    # Invalidate any cached confidence rankings for this checkpoint so the next
    # validate-mode request recomputes from the fresh predictions.
    try:
        from app.services.inference_feed_cache import invalidate_inference_feed
        invalidate_inference_feed(model_ckpt.id)
    except Exception:
        pass

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


def _snippet_load_options():
    return selectinload(ALPrediction.snippet).load_only(
        Snippet.start_time, Snippet.end_time, Snippet.recording_id
    )


def _passes_min_confidence(
    prediction: ALPrediction,
    label_scope: list[str] | None,
    min_confidence: float | None,
) -> bool:
    if min_confidence is None:
        return True
    return (
        aggregate_confidence(prediction.predicted_probabilities, label_scope)
        >= min_confidence
    )


# Page size for the confidence-filtered scan below. Large enough that a loose
# threshold is satisfied by a single round trip, small enough that a strict one
# doesn't pull the whole checkpoint into memory at once.
_CONFIDENCE_SCAN_PAGE = 2000


def _take_k_passing(
    ordered_query,
    k: int,
    label_scope: list[str] | None,
    min_confidence: float,
) -> list[ALPrediction]:
    """
    Walk a deterministically-ordered query in pages, keeping rows that clear
    min_confidence, until k of them are collected.

    Confidence is a noisy-OR over a JSON column, so it cannot be expressed in
    SQL and cannot join the ORDER BY ... LIMIT. Filtering has to happen in
    Python, but it must happen *before* the cut to k — otherwise a threshold
    silently truncates the result instead of reaching further down the ranking.
    Paging keeps that scan bounded.

    The caller's ordering must be total (score plus an id tiebreak), or OFFSET
    paging can repeat and skip rows between round trips.
    """
    collected: list[ALPrediction] = []
    offset = 0
    while len(collected) < k:
        page = ordered_query.limit(_CONFIDENCE_SCAN_PAGE).offset(offset).all()
        if not page:
            break
        offset += len(page)
        for pred in page:
            if _passes_min_confidence(pred, label_scope, min_confidence):
                collected.append(pred)
                if len(collected) == k:
                    break
        if len(page) < _CONFIDENCE_SCAN_PAGE:
            break  # exhausted the candidate set
    return collected


def get_top_prediction_suggestions(
    db: Session,
    dataset_id: int,
    model_checkpoint_id: int,
    snippet_set_id: int,
    strategy: str,
    k: int,
    label_scope: list[str] | None = None,
    min_confidence: float | None = None,
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
        random_query = query.order_by(func.random()).options(_snippet_load_options())
        if min_confidence is None:
            return random_query.limit(k).all()

        # ORDER BY random() re-rolls on every query, so OFFSET paging would both
        # repeat and skip rows. Draw one oversampled batch instead and filter it:
        # a uniform random subset, filtered, is still a uniform sample of the
        # passing rows. Only fall back to an unbounded draw if that came up short
        # of k while candidates remained.
        draw_size = max(k * 8, 200)
        rows = random_query.limit(draw_size).all()
        passing = [
            p for p in rows if _passes_min_confidence(p, label_scope, min_confidence)
        ]
        if len(passing) >= k or len(rows) < draw_size:
            return passing[:k]

        rows = random_query.all()
        return [
            p for p in rows if _passes_min_confidence(p, label_scope, min_confidence)
        ][:k]

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
                .options(_snippet_load_options())
                .all()
            )
            scored = [
                (p, aggregate_confidence(p.predicted_probabilities, label_scope))
                for p in all_preds
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            ranked_triples = [(p.id, p.snippet_id, s) for p, s in scored]
            # Cache the *unfiltered* ranking: min_confidence is not part of the
            # cache key, so the stored list has to serve every threshold.
            set_cached_confidence_ranking(model_checkpoint_id, snippet_set_id, label_scope, ranked_triples)

            # Filter out already-annotated snippets using the pre-sorted objects.
            annotated_exists_set = {
                row[0]
                for row in db.query(ALSnippetAnnotation.snippet_id).filter(
                    ALSnippetAnnotation.dataset_id == dataset_id,
                ).all()
            }
            candidates = [
                p
                for p, score in scored
                if p.snippet_id not in annotated_exists_set
                and (min_confidence is None or score >= min_confidence)
            ]
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
            for pred_id, snippet_id, score in ranked_triples
            if snippet_id not in annotated_snippet_ids
            and (min_confidence is None or score >= min_confidence)
        ][:k]

        if not top_k_pred_ids:
            return []

        id_to_rank = {pred_id: rank for rank, pred_id in enumerate(top_k_pred_ids)}
        objs = (
            db.query(ALPrediction)
            .filter(ALPrediction.id.in_(top_k_pred_ids))
            .options(_snippet_load_options())
            .all()
        )
        objs.sort(key=lambda p: id_to_rank[p.id])
        return objs

    score_columns = {
        "uncertainty": ALPrediction.uncertainty,
        "diversity": ALPrediction.diversity,
        "density": ALPrediction.density,
        "composite": ALPrediction.composite_score,
    }
    if strategy not in score_columns:
        raise ValueError(f"Unsupported suggestion strategy '{strategy}'.")

    score_column = score_columns[strategy]
    ordered = (
        query.order_by(score_column.desc().nullslast(), ALPrediction.id.asc())
        .options(_snippet_load_options())
    )
    if min_confidence is None:
        return ordered.limit(k).all()
    return _take_k_passing(ordered, k, label_scope, min_confidence)
