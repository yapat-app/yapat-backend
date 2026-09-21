"""
Build explore layers from Postgres (or from in-memory inference output).

Every builder streams its source rows in bounded chunks so a multi-million-row
snippet set never has to fit in Python objects at once. Each returns
``(arrays, meta)`` ready for :func:`storage.write_layer`.
"""

from __future__ import annotations

import datetime as _dt
import logging
import math
from typing import Any, Iterable, Sequence

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.embedding import SnippetSet
from app.models.pam_active_learning import ALModelCheckpoint, ALPrediction
from app.models.recording import Recording
from app.models.snippet import Snippet
from app.models.visualisation import FPVVis
from app.services.explore.query import MISSING_DAY, SCORE_COLUMNS, compute_default_confidence, packbits_rows
from app.services.explore.sampling import (
    bounds_of,
    priority_permutation,
    stable_seed,
    stratified_sample,
)

logger = logging.getLogger(__name__)

STREAM_CHUNK = 50_000
PROJECTION_METHODS: tuple[str, ...] = ("pca", "umap", "tsne", "isomap")
_EPOCH = _dt.date(1970, 1, 1)


def base_key(snippet_set_id: int) -> str:
    return f"ss{int(snippet_set_id)}"


def model_key(checkpoint_id: int, snippet_set_id: int) -> str:
    return f"ckpt{int(checkpoint_id)}_ss{int(snippet_set_id)}"


def projection_key(dataset_id: int, embedding_model_id: int) -> str:
    return f"ds{int(dataset_id)}_em{int(embedding_model_id)}"


def stream_rows(db: Session, stmt) -> Iterable[Sequence[Any]]:
    """Yield result partitions without buffering the whole result client-side."""
    options: dict[str, Any] = {"yield_per": STREAM_CHUNK}
    if db.get_bind().dialect.name == "postgresql":
        options["stream_results"] = True
    result = db.execute(stmt.execution_options(**options))
    for part in result.partitions(STREAM_CHUNK):
        yield part


# ── Base layer ────────────────────────────────────────────────────────────────


def parse_recording_metadata(meta: Any) -> tuple[str | None, int, int, float]:
    """(location, epoch_day, month, time_seconds) mirroring useRecordingMetadata.

    A recording only has a date/time when both ``recorded_date`` (YYYY-MM-DD)
    and a numeric ``recorded_time`` are present — useRecordingDateTimes drops
    recordings missing either.
    """
    if not isinstance(meta, dict):
        return None, MISSING_DAY, 0, math.nan
    location = meta.get("location")
    location = location if isinstance(location, str) and location else None

    date_str = meta.get("recorded_date")
    time_val = meta.get("recorded_time")
    if (
        isinstance(date_str, str)
        and date_str
        and isinstance(time_val, (int, float))
        and not isinstance(time_val, bool)
        and math.isfinite(float(time_val))
    ):
        try:
            parsed = _dt.date.fromisoformat(date_str[:10])
        except ValueError:
            return location, MISSING_DAY, 0, math.nan
        return location, (parsed - _EPOCH).days, parsed.month, float(time_val)
    return location, MISSING_DAY, 0, math.nan


def build_base_arrays(db: Session, snippet_set_id: int) -> tuple[dict[str, np.ndarray], dict]:
    snippet_set = db.get(SnippetSet, snippet_set_id)
    if snippet_set is None:
        raise ValueError(f"SnippetSet {snippet_set_id} not found")

    id_chunks: list[np.ndarray] = []
    rec_chunks: list[np.ndarray] = []
    dur_chunks: list[np.ndarray] = []
    stmt = (
        select(Snippet.id, Snippet.recording_id, Snippet.start_time, Snippet.end_time)
        .where(Snippet.snippet_set_id == snippet_set_id)
        .order_by(Snippet.id)
    )
    for part in stream_rows(db, stmt):
        block = np.asarray(part, dtype=np.float64).reshape(-1, 4)
        id_chunks.append(block[:, 0].astype(np.int64))
        rec_chunks.append(block[:, 1].astype(np.int64))
        dur_chunks.append((block[:, 3] - block[:, 2]).astype(np.float32))
    snippet_ids = np.concatenate(id_chunks) if id_chunks else np.empty(0, np.int64)
    snippet_rec = np.concatenate(rec_chunks) if rec_chunks else np.empty(0, np.int64)
    duration = np.concatenate(dur_chunks) if dur_chunks else np.empty(0, np.float32)

    rec_ids_list: list[int] = []
    rec_loc_names: list[str | None] = []
    rec_day: list[int] = []
    rec_month: list[int] = []
    rec_time: list[float] = []
    rec_stmt = (
        select(Recording.id, Recording.extra_metadata)
        .where(Recording.dataset_id == snippet_set.dataset_id)
        .order_by(Recording.id)
    )
    for part in stream_rows(db, rec_stmt):
        for rec_id, meta in part:
            location, day, month, time_s = parse_recording_metadata(meta)
            rec_ids_list.append(int(rec_id))
            rec_loc_names.append(location)
            rec_day.append(day)
            rec_month.append(month)
            rec_time.append(time_s)

    locations = sorted({name for name in rec_loc_names if name is not None})
    loc_index = {name: i for i, name in enumerate(locations)}
    rec_ids = np.asarray(rec_ids_list, dtype=np.int64)
    rec_location = np.asarray(
        [loc_index[name] if name is not None else -1 for name in rec_loc_names], dtype=np.int32
    )

    recording_idx = np.full(snippet_ids.shape[0], -1, dtype=np.int32)
    if rec_ids.size and snippet_rec.size:
        pos = np.searchsorted(rec_ids, snippet_rec)
        clipped = np.minimum(pos, rec_ids.size - 1)
        found = rec_ids[clipped] == snippet_rec
        recording_idx = np.where(found, clipped, -1).astype(np.int32)

    arrays = {
        "snippet_ids": snippet_ids,
        "recording_idx": recording_idx,
        "duration": duration,
        "rec_ids": rec_ids,
        "rec_location": rec_location,
        "rec_epoch_day": np.asarray(rec_day, dtype=np.int32),
        "rec_month": np.asarray(rec_month, dtype=np.int8),
        "rec_time": np.asarray(rec_time, dtype=np.float32),
    }
    meta = {
        "snippet_set_id": int(snippet_set_id),
        "dataset_id": int(snippet_set.dataset_id),
        "embedding_model_id": int(snippet_set.embedding_model_id),
        "locations": locations,
        "n": int(snippet_ids.shape[0]),
    }
    logger.info(
        "explore: built base arrays snippet_set=%s snippets=%s recordings=%s",
        snippet_set_id, snippet_ids.shape[0], rec_ids.shape[0],
    )
    return arrays, meta


# ── Model layer ───────────────────────────────────────────────────────────────


def model_arrays_from_matrices(
    snippet_ids: Sequence[int] | np.ndarray,
    probs: np.ndarray,
    preds: np.ndarray,
    scores: np.ndarray,
    label_order: Sequence[str],
    *,
    checkpoint_id: int,
    snippet_set_id: int,
) -> tuple[dict[str, np.ndarray], dict]:
    """Model layer straight from inference output (no JSON round-trip)."""
    ids = np.asarray(snippet_ids, dtype=np.int64)
    order = np.argsort(ids, kind="stable")
    ids = ids[order]
    if ids.size and np.any(ids[1:] == ids[:-1]):
        raise ValueError("duplicate snippet ids in inference output")
    probs16 = np.asarray(probs, dtype=np.float32)[order].astype(np.float16)
    bits = packbits_rows(np.asarray(preds)[order] > 0)
    score_mat = np.asarray(scores, dtype=np.float32)[order]
    if score_mat.shape != (ids.shape[0], len(SCORE_COLUMNS)):
        raise ValueError(f"scores must have shape (n, {len(SCORE_COLUMNS)})")
    arrays = {
        "snippet_ids": ids,
        "scores": score_mat,
        "probs": probs16,
        "pred_bits": bits,
        "default_confidence": compute_default_confidence(probs16),
    }
    meta = {
        "checkpoint_id": int(checkpoint_id),
        "snippet_set_id": int(snippet_set_id),
        "label_order": [str(label) for label in label_order],
        "n": int(ids.shape[0]),
        "source": "inference",
    }
    return arrays, meta


def build_model_arrays(
    db: Session, checkpoint_id: int, snippet_set_id: int
) -> tuple[dict[str, np.ndarray], dict]:
    """Fallback: stream persisted al_predictions rows for a checkpoint."""
    checkpoint = db.get(ALModelCheckpoint, checkpoint_id)
    if checkpoint is None:
        raise ValueError(f"Checkpoint {checkpoint_id} not found")
    hyper = checkpoint.hyperparameters or {}
    label_order: list[str] = [str(x) for x in (hyper.get("label_order") or [])]
    label_pos = {name: i for i, name in enumerate(label_order)}

    ids_chunks: list[np.ndarray] = []
    score_chunks: list[np.ndarray] = []
    prob_chunks: list[np.ndarray] = []
    pred_chunks: list[np.ndarray] = []

    stmt = (
        select(
            ALPrediction.snippet_id,
            ALPrediction.predicted_labels,
            ALPrediction.predicted_probabilities,
            ALPrediction.uncertainty,
            ALPrediction.diversity,
            ALPrediction.density,
            ALPrediction.composite_score,
        )
        .join(Snippet, Snippet.id == ALPrediction.snippet_id)
        .where(ALPrediction.model_checkpoint_id == checkpoint_id)
        .where(Snippet.snippet_set_id == snippet_set_id)
        .order_by(ALPrediction.snippet_id)
    )
    for part in stream_rows(db, stmt):
        m = len(part)
        # Discover labels first so the chunk matrices have their final width.
        for row in part:
            for name in list((row[2] or {}).keys()) + list(row[1] or []):
                if name not in label_pos:
                    label_pos[name] = len(label_order)
                    label_order.append(name)
        width = len(label_order)
        probs = np.full((m, width), np.nan, dtype=np.float32)
        preds = np.zeros((m, width), dtype=bool)
        ids = np.empty(m, dtype=np.int64)
        scores = np.empty((m, 4), dtype=np.float64)
        for i, row in enumerate(part):
            ids[i] = row[0]
            for name, value in (row[2] or {}).items():
                try:
                    probs[i, label_pos[name]] = float(value)
                except (TypeError, ValueError):
                    continue
            for name in row[1] or []:
                preds[i, label_pos[name]] = True
            scores[i] = [
                np.nan if row[3] is None else row[3],
                np.nan if row[4] is None else row[4],
                np.nan if row[5] is None else row[5],
                np.nan if row[6] is None else row[6],
            ]
        ids_chunks.append(ids)
        score_chunks.append(scores.astype(np.float32))
        prob_chunks.append(probs)
        pred_chunks.append(preds)

    width = len(label_order)
    if ids_chunks:
        ids_all = np.concatenate(ids_chunks)
        probs_all = np.concatenate(
            [np.pad(p, ((0, 0), (0, width - p.shape[1])), constant_values=np.nan) for p in prob_chunks]
        )
        preds_all = np.concatenate(
            [np.pad(p, ((0, 0), (0, width - p.shape[1]))) for p in pred_chunks]
        )
        scores_all = np.concatenate(score_chunks)
    else:
        ids_all = np.empty(0, np.int64)
        probs_all = np.empty((0, width), np.float32)
        preds_all = np.empty((0, width), bool)
        scores_all = np.empty((0, 4), np.float32)

    arrays, meta = model_arrays_from_matrices(
        ids_all, probs_all, preds_all, scores_all, label_order,
        checkpoint_id=checkpoint_id, snippet_set_id=snippet_set_id,
    )
    meta["source"] = "database"
    logger.info(
        "explore: built model arrays checkpoint=%s snippet_set=%s rows=%s labels=%s",
        checkpoint_id, snippet_set_id, ids_all.shape[0], width,
    )
    return arrays, meta


# ── Projection layer ──────────────────────────────────────────────────────────


def build_projection_arrays(
    db: Session, dataset_id: int, embedding_model_id: int, max_points: int
) -> tuple[dict[str, np.ndarray], dict]:
    columns = [FPVVis.snippet_id]
    for method in PROJECTION_METHODS:
        columns += [getattr(FPVVis, f"{method}_2d_x"), getattr(FPVVis, f"{method}_2d_y")]
    stmt = (
        select(*columns)
        .where(FPVVis.dataset_id == dataset_id)
        .where(FPVVis.embedding_model_id == embedding_model_id)
        .where(FPVVis.model_checkpoint_id.is_(None))
        .order_by(FPVVis.snippet_id)
    )
    chunks: list[np.ndarray] = []
    for part in stream_rows(db, stmt):
        chunks.append(np.asarray(part, dtype=np.float64).reshape(-1, 1 + 2 * len(PROJECTION_METHODS)))
    data = np.concatenate(chunks) if chunks else np.empty((0, 1 + 2 * len(PROJECTION_METHODS)))
    return projection_arrays_from_matrix(
        data, dataset_id=dataset_id, embedding_model_id=embedding_model_id, max_points=max_points
    )


def projection_arrays_from_matrix(
    data: np.ndarray, *, dataset_id: int, embedding_model_id: int, max_points: int
) -> tuple[dict[str, np.ndarray], dict]:
    snippet_ids = data[:, 0].astype(np.int64)
    n = snippet_ids.shape[0]
    seed = stable_seed(projection_key(dataset_id, embedding_model_id))
    priority = priority_permutation(n, seed)
    arrays: dict[str, np.ndarray] = {"snippet_ids": snippet_ids, "priority": priority}
    methods_meta: dict[str, dict] = {}
    for i, method in enumerate(PROJECTION_METHODS):
        coords = data[:, 1 + 2 * i : 3 + 2 * i].astype(np.float32)
        x, y = coords[:, 0], coords[:, 1]
        finite = int(np.count_nonzero(np.isfinite(x) & np.isfinite(y)))
        bounds = bounds_of(x, y)
        sample = stratified_sample(x, y, max_points, priority) if finite else np.empty(0, np.int64)
        arrays[f"coords_{method}"] = coords
        arrays[f"sample_{method}"] = sample.astype(np.int64)
        methods_meta[method] = {
            "finite": finite,
            "bounds": list(bounds) if bounds else None,
            "sampled": finite > sample.shape[0],
        }
    meta = {
        "dataset_id": int(dataset_id),
        "embedding_model_id": int(embedding_model_id),
        "n": int(n),
        "max_points": int(max_points),
        "methods": methods_meta,
    }
    logger.info(
        "explore: built projection arrays dataset=%s embedding_model=%s points=%s",
        dataset_id, embedding_model_id, n,
    )
    return arrays, meta
