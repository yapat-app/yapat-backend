"""
Per-process access to explore layers: loading, alignment, caching and builds.

Every API worker keeps small bounded caches. Nothing here is required for
correctness across workers — all shared state lives in the versioned layers on
disk and in Postgres — so any worker can serve any request.
"""

from __future__ import annotations

import fcntl
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Hashable

import numpy as np
from sqlalchemy.orm import Session

from app.config import settings
from app.services.explore import storage
from app.services.explore.builders import (
    base_key,
    build_base_arrays,
    build_model_arrays,
    build_projection_arrays,
    model_key,
    projection_key,
)
from app.services.explore.labels import build_labels, labels_fingerprint
from app.services.explore.query import (
    BaseData,
    LabelsData,
    ModelData,
    compute_native_order,
)

logger = logging.getLogger(__name__)

BUILD_GUARD_TTL_SECONDS = 900
BUILD_ERROR_TTL_SECONDS = 60
RETRY_AFTER_MS = 1500


# ── Errors surfaced to the API layer ──────────────────────────────────────────


class ExploreError(Exception):
    """Base class for explore errors that map to HTTP responses."""


class LayerBuilding(ExploreError):
    def __init__(self, kind: str, key: str):
        super().__init__(f"{kind} layer {key} is being built")
        self.kind = kind
        self.key = key


class LayerBuildFailed(ExploreError):
    pass


class NoPredictions(ExploreError):
    pass


class ProjectionMissing(ExploreError):
    pass


# ── Caches ────────────────────────────────────────────────────────────────────


class KeyedLRU:
    """Thread-safe LRU whose misses are computed once per key (no stampede)."""

    def __init__(self, maxsize: int):
        self.maxsize = maxsize
        self._data: OrderedDict[Hashable, Any] = OrderedDict()
        self._lock = threading.Lock()
        self._key_locks: dict[Hashable, threading.Lock] = {}

    def get_or_compute(self, key: Hashable, compute: Callable[[], Any]) -> Any:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                return self._data[key]
            key_lock = self._key_locks.setdefault(key, threading.Lock())
        with key_lock:
            with self._lock:
                if key in self._data:
                    self._data.move_to_end(key)
                    return self._data[key]
            try:
                value = compute()
                with self._lock:
                    self._data[key] = value
                    self._data.move_to_end(key)
                    while len(self._data) > self.maxsize:
                        self._data.popitem(last=False)
                return value
            finally:
                with self._lock:
                    self._key_locks.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


@dataclass
class _LayerEntry:
    version: str
    layer: storage.Layer


class LayerCache:
    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], _LayerEntry] = {}
        self._lock = threading.Lock()

    def get(self, kind: str, key: str) -> storage.Layer | None:
        for attempt in range(2):
            pointer = storage.read_pointer(kind, key)
            if pointer is None:
                return None
            with self._lock:
                entry = self._entries.get((kind, key))
            if entry is not None and entry.version == pointer.version:
                return entry.layer
            try:
                layer = storage.load_layer(kind, key, pointer.version)
            except FileNotFoundError:
                # The version was pruned between reading the pointer and
                # opening it; the pointer now names a newer version.
                if attempt == 0:
                    continue
                return None
            with self._lock:
                self._entries[(kind, key)] = _LayerEntry(pointer.version, layer)
            return layer
        return None

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


layers = LayerCache()
base_cache = KeyedLRU(16)
model_cache = KeyedLRU(8)
projection_map_cache = KeyedLRU(32)
population_cache = KeyedLRU(8)
non_score_cache = KeyedLRU(16)
filter_cache = KeyedLRU(16)
order_cache = KeyedLRU(6)

_labels_lock = threading.Lock()
_labels_key_locks: dict[tuple[int, int], threading.Lock] = {}


@dataclass
class _LabelsEntry:
    base_version: str
    fingerprint: tuple[int, int]
    checked_at: float
    data: LabelsData


_labels_entries: "OrderedDict[tuple[int, int], _LabelsEntry]" = OrderedDict()
_LABELS_MAX_ENTRIES = 16


def reset_caches() -> None:
    """Drop every in-process cache (tests / after invalidation in-process)."""
    layers.clear()
    for cache in (
        base_cache, model_cache, projection_map_cache,
        population_cache, non_score_cache, filter_cache, order_cache,
    ):
        cache.clear()
    with _labels_lock:
        _labels_entries.clear()


# ── Builds ────────────────────────────────────────────────────────────────────


def run_build(db: Session, kind: str, params: dict) -> str:
    """Build and publish a layer. Used by the Celery task and inline mode."""
    if kind == "base":
        arrays, meta = build_base_arrays(db, int(params["snippet_set_id"]))
        key = base_key(params["snippet_set_id"])
    elif kind == "model":
        arrays, meta = build_model_arrays(
            db, int(params["checkpoint_id"]), int(params["snippet_set_id"])
        )
        key = model_key(params["checkpoint_id"], params["snippet_set_id"])
    elif kind == "projection":
        arrays, meta = build_projection_arrays(
            db,
            int(params["dataset_id"]),
            int(params["embedding_model_id"]),
            settings.EXPLORE_PROJECTION_MAX_POINTS,
        )
        key = projection_key(params["dataset_id"], params["embedding_model_id"])
    else:
        raise ValueError(f"Unknown explore layer kind {kind!r}")
    return storage.write_layer(kind, key, arrays, meta)


def _redis_client():
    try:
        import redis

        return redis.Redis.from_url(settings.CELERY_BROKER_URL, socket_timeout=2)
    except Exception:  # pragma: no cover - defensive
        return None


def guard_key(kind: str, key: str) -> str:
    return f"explore:enqueued:{kind}:{key}"


def error_key(kind: str, key: str) -> str:
    return f"explore:error:{kind}:{key}"


_inline_locks: dict[str, threading.Lock] = {}
_inline_locks_guard = threading.Lock()


def build_inline(db: Session, kind: str, key: str, params: dict) -> None:
    """Build in this process, serialised per layer across threads and processes."""
    with _inline_locks_guard:
        thread_lock = _inline_locks.setdefault(f"{kind}:{key}", threading.Lock())
    with thread_lock:
        lock_dir = storage.layer_dir(kind, key)
        lock_dir.mkdir(parents=True, exist_ok=True)
        with open(lock_dir / ".build.lock", "w") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                if storage.read_pointer(kind, key) is not None:
                    return  # somebody else finished it while we waited
                run_build(db, kind, params)
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def request_build(db: Session, kind: str, key: str, params: dict) -> None:
    """Make sure a build is under way. Returns only after an inline build."""
    if settings.EXPLORE_BUILD_INLINE:
        build_inline(db, kind, key, params)
        return

    client = _redis_client()
    try:
        if client is None:
            raise ConnectionError("redis unavailable")
        failure = client.get(error_key(kind, key))
        if failure:
            raise LayerBuildFailed(failure.decode("utf-8", "replace"))
        if client.set(guard_key(kind, key), "1", nx=True, ex=BUILD_GUARD_TTL_SECONDS):
            from app.tasks.explore_tasks import build_explore_layer

            try:
                build_explore_layer.delay(kind, params)
            except Exception:
                client.delete(guard_key(kind, key))
                raise
    except LayerBuildFailed:
        raise
    except Exception as exc:
        logger.warning("explore: cannot enqueue %s/%s (%s); building inline", kind, key, exc)
        build_inline(db, kind, key, params)
        return
    raise LayerBuilding(kind, key)


def _get_or_build(db: Session, kind: str, key: str, params: dict) -> storage.Layer:
    layer = layers.get(kind, key)
    if layer is not None:
        return layer
    request_build(db, kind, key, params)
    layer = layers.get(kind, key)
    if layer is None:
        raise LayerBuilding(kind, key)
    return layer


# ── Typed accessors ───────────────────────────────────────────────────────────


def get_base(db: Session, snippet_set_id: int) -> BaseData:
    key = base_key(snippet_set_id)
    layer = _get_or_build(db, "base", key, {"snippet_set_id": int(snippet_set_id)})

    def load() -> BaseData:
        return BaseData(
            version=layer.version,
            snippet_ids=layer["snippet_ids"],
            recording_idx=layer["recording_idx"],
            duration=layer["duration"],
            rec_ids=layer["rec_ids"],
            rec_location=layer["rec_location"],
            rec_epoch_day=layer["rec_epoch_day"],
            rec_month=layer["rec_month"],
            rec_time=layer["rec_time"],
            locations=list(layer.meta.get("locations", [])),
        )

    return base_cache.get_or_compute(("base", layer.version), load)


def get_model(
    db: Session,
    base: BaseData,
    checkpoint_id: int,
    snippet_set_id: int,
    predictions_exist: Callable[[], bool],
) -> ModelData:
    key = model_key(checkpoint_id, snippet_set_id)
    layer = layers.get("model", key)
    if layer is None:
        if not predictions_exist():
            raise NoPredictions(
                f"No predictions exist for checkpoint {checkpoint_id} on snippet set {snippet_set_id}."
            )
        layer = _get_or_build(
            db, "model", key,
            {"checkpoint_id": int(checkpoint_id), "snippet_set_id": int(snippet_set_id)},
        )
    return model_cache.get_or_compute(
        ("model", base.version, layer.version), lambda: align_model(base, layer)
    )


def align_model(base: BaseData, layer: storage.Layer) -> ModelData:
    ids = layer["snippet_ids"]
    rows = base.index_of(np.asarray(ids))
    valid = rows >= 0
    n = base.n
    has_prediction = np.zeros(n, dtype=bool)
    model_row = np.full(n, -1, dtype=np.int64)
    scores = np.full((n, 4), np.nan, dtype=np.float32)
    default_conf = np.full(n, np.nan, dtype=np.float32)
    target = rows[valid]
    source = np.flatnonzero(valid)
    has_prediction[target] = True
    model_row[target] = source
    if source.size:
        scores[target] = np.asarray(layer["scores"])[source]
        default_conf[target] = np.asarray(layer["default_confidence"])[source]
    if not valid.all():
        logger.warning(
            "explore: %s rows of model layer %s are not in base %s",
            int((~valid).sum()), layer.key, base.version,
        )
    return ModelData(
        version=layer.version,
        has_prediction=has_prediction,
        model_row=model_row,
        scores=scores,
        probs=layer["probs"],
        pred_bits=layer["pred_bits"],
        label_order=list(layer.meta.get("label_order", [])),
        default_confidence=default_conf,
        native_order=compute_native_order(has_prediction, scores[:, 3]),
    )


def get_labels(db: Session, base: BaseData, dataset_id: int, snippet_set_id: int) -> LabelsData:
    cache_key = (int(dataset_id), int(snippet_set_id))
    now = time.monotonic()
    with _labels_lock:
        entry = _labels_entries.get(cache_key)
        key_lock = _labels_key_locks.setdefault(cache_key, threading.Lock())
    if (
        entry is not None
        and entry.base_version == base.version
        and now - entry.checked_at < settings.EXPLORE_LABELS_RECHECK_SECONDS
    ):
        return entry.data

    with key_lock:
        fingerprint = labels_fingerprint(db, dataset_id)
        with _labels_lock:
            entry = _labels_entries.get(cache_key)
        if entry is not None and entry.base_version == base.version and entry.fingerprint == fingerprint:
            entry.checked_at = time.monotonic()
            return entry.data
        data = build_labels(db, base, dataset_id, snippet_set_id, fingerprint)
        with _labels_lock:
            _labels_entries[cache_key] = _LabelsEntry(base.version, fingerprint, time.monotonic(), data)
            _labels_entries.move_to_end(cache_key)
            while len(_labels_entries) > _LABELS_MAX_ENTRIES:
                _labels_entries.popitem(last=False)
        return data


def get_projection(
    db: Session,
    dataset_id: int,
    embedding_model_id: int,
    projection_exists: Callable[[], bool],
) -> storage.Layer:
    key = projection_key(dataset_id, embedding_model_id)
    layer = layers.get("projection", key)
    if layer is not None:
        return layer
    if not projection_exists():
        raise ProjectionMissing(
            f"No dataset-level feature projection rows found for dataset_id={dataset_id}, "
            f"embedding_model_id={embedding_model_id}. Generate projections first."
        )
    return _get_or_build(
        db, "projection", key,
        {"dataset_id": int(dataset_id), "embedding_model_id": int(embedding_model_id)},
    )


@dataclass
class ProjectionMaps:
    proj_to_base: np.ndarray  # int64 [m], -1 when the point is not in the base set
    base_to_proj: np.ndarray  # int64 [n], -1 when the snippet has no projection row


def get_projection_maps(base: BaseData, layer: storage.Layer) -> ProjectionMaps:
    def compute() -> ProjectionMaps:
        proj_ids = np.asarray(layer["snippet_ids"])
        proj_to_base = base.index_of(proj_ids)
        base_to_proj = np.full(base.n, -1, dtype=np.int64)
        ok = proj_to_base >= 0
        base_to_proj[proj_to_base[ok]] = np.flatnonzero(ok)
        return ProjectionMaps(proj_to_base=proj_to_base, base_to_proj=base_to_proj)

    return projection_map_cache.get_or_compute(("maps", base.version, layer.version), compute)


def invalidate(kind: str, key: str) -> None:
    storage.invalidate_layer(kind, key)
    client = _redis_client()
    if client is not None:
        try:
            client.delete(guard_key(kind, key), error_key(kind, key))
        except Exception:
            pass
