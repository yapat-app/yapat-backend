"""
Pure-numpy query engine for the explore store.

Every function here mirrors a piece of the Annotation Hub's former client-side
pipeline (PredictionFeed.filteredAndSorted, scoreVisibility.isPointVisible,
useScoreHistogramData, HistogramSlider.computeBins, alSlice.withDisplayFields).
The docstrings name the JS behaviour being reproduced; the unit tests in
tests/unit/explore pin it down. Nothing in this module touches the database or
the filesystem.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

# Model score columns, in storage order.
SCORE_COLUMNS: tuple[str, ...] = ("uncertainty", "diversity", "density", "composite")
SCORE_KEYS: tuple[str, ...] = SCORE_COLUMNS + ("confidence",)

# alProperties.ts declared ranges — used when the data provides no domain.
PROPERTY_RANGES: dict[str, tuple[float, float]] = {
    "uncertainty": (0.0, 1.0),
    "diversity": (0.0, 1.0),
    "density": (0.0, 1.0),
    "composite": (-2.0, 2.0),
    "confidence": (0.0, 1.0),
}
# computeScoreDomains only derives domains for these (confidence keeps [0,1]).
DOMAIN_KEYS: tuple[str, ...] = SCORE_COLUMNS

SCORE_UPPER_EPS = 1e-9
MISSING_DAY = np.iinfo(np.int32).min

SORT_FIELDS = ("confidence", "composite", "uncertainty", "diversity", "density", "date", "time")


# ── Aligned data ──────────────────────────────────────────────────────────────


@dataclass
class BaseData:
    """Snippet-set rows (sorted by snippet id) plus their recording metadata."""

    version: str
    snippet_ids: np.ndarray  # int64 [n], ascending
    recording_idx: np.ndarray  # int32 [n], index into rec_* (-1 unknown)
    duration: np.ndarray  # float32 [n]
    rec_ids: np.ndarray  # int64 [R]
    rec_location: np.ndarray  # int32 [R], index into locations (-1 none)
    rec_epoch_day: np.ndarray  # int32 [R], MISSING_DAY when no date/time
    rec_month: np.ndarray  # int8 [R], 0 when no date/time
    rec_time: np.ndarray  # float32 [R], NaN when no date/time
    locations: list[str]

    @property
    def n(self) -> int:
        return int(self.snippet_ids.shape[0])

    def index_of(self, snippet_ids: Iterable[int] | np.ndarray) -> np.ndarray:
        """Base row index for each id (-1 when the id is not in the set)."""
        if isinstance(snippet_ids, np.ndarray):
            ids = snippet_ids.astype(np.int64, copy=False)
        else:
            ids = np.asarray(list(snippet_ids), dtype=np.int64)
        if ids.size == 0 or self.n == 0:
            return np.full(ids.shape, -1, dtype=np.int64)
        pos = np.searchsorted(self.snippet_ids, ids)
        pos_clipped = np.minimum(pos, self.n - 1)
        found = self.snippet_ids[pos_clipped] == ids
        return np.where(found, pos_clipped, -1).astype(np.int64)


@dataclass
class ModelData:
    """Checkpoint predictions aligned to BaseData rows."""

    version: str
    has_prediction: np.ndarray  # bool [n]
    model_row: np.ndarray  # int64 [n], row in probs/pred_bits (-1 none)
    scores: np.ndarray  # float32 [n, 4] (SCORE_COLUMNS), NaN missing
    probs: np.ndarray  # float16 [m, L], NaN = label absent from the row
    pred_bits: np.ndarray  # uint8 [m, ceil(L/8)], little bit order
    label_order: list[str]
    default_confidence: np.ndarray  # float32 [n], NaN = undefined
    native_order: np.ndarray  # int64 [n_pred], base row indices

    @property
    def label_positions(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.label_order)}


@dataclass
class LabelsData:
    """Ground-truth/user labels aligned to BaseData rows (CSR)."""

    version: str
    labeled: np.ndarray  # bool [n]
    indptr: np.ndarray  # int64 [n+1]
    label_idx: np.ndarray  # int32 [nnz], sorted by label name within a row
    first_label: np.ndarray  # int32 [n], -1 unlabeled
    vocab: list[str]

    def labels_for(self, row: int) -> list[str]:
        lo, hi = int(self.indptr[row]), int(self.indptr[row + 1])
        return [self.vocab[i] for i in self.label_idx[lo:hi]]


@dataclass(frozen=True)
class Filters:
    annotation_status: str = "any"
    annotated_species: tuple[str, ...] = ()
    predicted_species: tuple[str, ...] = ()
    label_scope: tuple[str, ...] = ()
    locations: tuple[str, ...] = ()
    date_range: tuple[float, float] | None = None
    months: tuple[int, ...] = ()
    time_range: tuple[float, float] | None = None
    score_ranges: tuple[tuple[str, float, float], ...] = ()
    sticky_ids: tuple[int, ...] = ()

    def population_key(self) -> tuple:
        return (self.predicted_species,)

    def non_score_key(self) -> tuple:
        return (
            self.predicted_species,
            self.annotation_status,
            self.annotated_species,
            self.label_scope,
            self.locations,
            self.date_range,
            self.months,
            self.time_range,
            self.sticky_ids if self.annotation_status != "any" else (),
        )

    def visible_key(self) -> tuple:
        return self.non_score_key() + (self.active_score_ranges(),)

    def active_score_ranges(self) -> tuple[tuple[str, float, float], ...]:
        """hasActiveScoreVisibilityFilters: ranges only apply when one is narrowed."""
        if any(lo > 0 or hi < 1 for _, lo, hi in self.score_ranges):
            return self.score_ranges
        return ()


@dataclass
class Population:
    mask: np.ndarray  # bool [n]
    confidence: np.ndarray  # float32 [n], NaN undefined
    domains: dict[str, tuple[float, float]]


@dataclass
class FilterResult:
    population: Population
    non_score: np.ndarray  # bool [n]
    visible: np.ndarray  # bool [n]
    extra: dict = field(default_factory=dict)


# ── Model helpers ─────────────────────────────────────────────────────────────


def packbits_rows(pred: np.ndarray) -> np.ndarray:
    """[m, L] 0/1 → [m, ceil(L/8)] uint8 (little bit order)."""
    pred = np.asarray(pred)
    if pred.ndim != 2:
        raise ValueError("pred must be 2-D")
    if pred.shape[1] == 0:
        return np.zeros((pred.shape[0], 0), dtype=np.uint8)
    return np.packbits(pred.astype(bool), axis=1, bitorder="little")


def unpack_labels(bits_row: np.ndarray, label_order: Sequence[str]) -> list[str]:
    if len(label_order) == 0:
        return []
    flags = np.unpackbits(np.asarray(bits_row, dtype=np.uint8), bitorder="little")[: len(label_order)]
    return [label_order[i] for i in np.flatnonzero(flags)]


def compute_default_confidence(probs: np.ndarray) -> np.ndarray:
    """withDisplayFields: max probability, undefined (NaN) when ≤ 0 or absent."""
    m = probs.shape[0]
    out = np.full(m, np.nan, dtype=np.float32)
    if m == 0 or probs.shape[1] == 0:
        return out
    chunk = 262_144
    for start in range(0, m, chunk):
        block = np.asarray(probs[start : start + chunk], dtype=np.float32)
        with np.errstate(invalid="ignore"):
            all_nan = np.all(np.isnan(block), axis=1)
            filled = np.where(np.isnan(block), -np.inf, block)
            best = filled.max(axis=1)
        best = np.where(all_nan | ~(best > 0), np.nan, best)
        out[start : start + chunk] = best
    return out


def compute_native_order(has_prediction: np.ndarray, composite: np.ndarray) -> np.ndarray:
    """Composite desc (nulls last), then snippet id asc — over rows with predictions."""
    rows = np.flatnonzero(has_prediction)
    if rows.size == 0:
        return rows.astype(np.int64)
    comp = composite[rows].astype(np.float64)
    missing = np.isnan(comp)
    key = np.where(missing, 0.0, -comp)
    # lexsort: last key is primary. Base rows are id-ascending already.
    order = np.lexsort((rows, key, missing))
    return rows[order].astype(np.int64)


def _species_positions(names: Sequence[str], label_order: Sequence[str]) -> np.ndarray:
    positions = {n: i for i, n in enumerate(label_order)}
    return np.asarray(sorted({positions[n] for n in names if n in positions}), dtype=np.int64)


def _rows_with_any_bit(bits: np.ndarray, rows: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """For model rows, True when any of the label positions is set."""
    result = np.zeros(rows.shape[0], dtype=bool)
    if positions.size == 0 or rows.size == 0 or bits.shape[1] == 0:
        return result
    for pos in positions:
        byte, bit = divmod(int(pos), 8)
        if byte >= bits.shape[1]:
            continue
        result |= (bits[rows, byte] & np.uint8(1 << bit)) != 0
    return result


# ── Filtering ─────────────────────────────────────────────────────────────────


def compute_population(
    base: BaseData,
    model: ModelData | None,
    filters: Filters,
) -> Population:
    n = base.n
    if model is None:
        mask = np.ones(n, dtype=bool)
        confidence = np.full(n, np.nan, dtype=np.float32)
    else:
        mask = model.has_prediction.copy()
        confidence = model.default_confidence

    if filters.predicted_species:
        if model is None:
            # No predicted labels exist, so nothing can match the scope.
            mask[:] = False
        else:
            positions = _species_positions(filters.predicted_species, model.label_order)
            rows = np.flatnonzero(mask)
            mrows = model.model_row[rows]
            matches = _rows_with_any_bit(model.pred_bits, mrows, positions)
            mask[:] = False
            kept = rows[matches]
            mask[kept] = True
            if positions.size and kept.size:
                confidence = confidence.copy()
                scoped = _noisy_or(model.probs, model.model_row[kept], positions)
                valid = ~np.isnan(scoped)
                confidence[kept[valid]] = scoped[valid]

    domains: dict[str, tuple[float, float]] = {}
    if model is not None:
        rows = np.flatnonzero(mask)
        for col, key in enumerate(SCORE_COLUMNS):
            if key not in DOMAIN_KEYS:
                continue
            vals = model.scores[rows, col]
            finite = vals[np.isfinite(vals)]
            if finite.size:
                domains[key] = (float(finite.min()), float(finite.max()))
    return Population(mask=mask, confidence=confidence, domains=domains)


def _noisy_or(probs: np.ndarray, model_rows: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """aggregateConfidence: 1 - Π(1 - clamp(p)), skipping absent labels; NaN if none."""
    out = np.empty(model_rows.shape[0], dtype=np.float32)
    chunk = 262_144
    for start in range(0, model_rows.shape[0], chunk):
        sub_rows = model_rows[start : start + chunk]
        block = np.asarray(probs[sub_rows][:, positions], dtype=np.float64)
        present = ~np.isnan(block)
        clipped = np.clip(np.where(present, block, 0.0), 0.0, 1.0)
        inverse = np.prod(np.where(present, 1.0 - clipped, 1.0), axis=1)
        value = 1.0 - inverse
        value[~present.any(axis=1)] = np.nan
        out[start : start + chunk] = value
    return out


def compute_non_score_mask(
    base: BaseData,
    model: ModelData | None,
    labels: LabelsData,
    population: Population,
    filters: Filters,
) -> np.ndarray:
    mask = population.mask.copy()

    if filters.annotation_status in ("annotated", "unannotated"):
        want = filters.annotation_status == "annotated"
        status_ok = labels.labeled == want
        if filters.sticky_ids:
            sticky = base.index_of(filters.sticky_ids)
            sticky = sticky[sticky >= 0]
            status_ok = status_ok.copy()
            status_ok[sticky] = True
        mask &= status_ok

    if filters.annotated_species:
        vocab_pos = {name: i for i, name in enumerate(labels.vocab)}
        wanted = np.asarray(
            sorted({vocab_pos[s] for s in filters.annotated_species if s in vocab_pos}),
            dtype=np.int32,
        )
        has_species = np.zeros(base.n, dtype=bool)
        if wanted.size and labels.label_idx.size:
            hits = np.isin(labels.label_idx, wanted)
            row_of_entry = np.repeat(
                np.arange(base.n, dtype=np.int64), np.diff(labels.indptr).astype(np.int64)
            )
            has_species[row_of_entry[hits]] = True
        mask &= has_species

    if filters.label_scope:
        if model is None:
            mask[:] = False
        else:
            positions = _species_positions(filters.label_scope, model.label_order)
            rows = np.flatnonzero(mask)
            ok = _rows_with_any_bit(model.pred_bits, model.model_row[rows], positions)
            mask[:] = False
            mask[rows[ok]] = True

    rec = base.recording_idx
    has_rec = rec >= 0
    safe_rec = np.where(has_rec, rec, 0)

    if filters.locations:
        loc_pos = {name: i for i, name in enumerate(base.locations)}
        codes = np.asarray(
            sorted({loc_pos[l] for l in filters.locations if l in loc_pos}), dtype=np.int32
        )
        if base.rec_location.size:
            row_loc = base.rec_location[safe_rec]
            mask &= has_rec & np.isin(row_loc, codes)
        else:
            mask[:] = False

    needs_dt = filters.date_range is not None or filters.months or filters.time_range is not None
    if needs_dt:
        if base.rec_epoch_day.size:
            row_day = base.rec_epoch_day[safe_rec]
            has_dt = has_rec & (row_day != MISSING_DAY)
        else:
            row_day = np.zeros(base.n, dtype=np.int32)
            has_dt = np.zeros(base.n, dtype=bool)
        mask &= has_dt
        if filters.date_range is not None:
            lo, hi = filters.date_range
            mask &= (row_day >= lo) & (row_day <= hi)
        if filters.months:
            row_month = base.rec_month[safe_rec] if base.rec_month.size else np.zeros(base.n, np.int8)
            mask &= np.isin(row_month, np.asarray(filters.months, dtype=np.int8))
        if filters.time_range is not None:
            lo, hi = filters.time_range
            row_time = base.rec_time[safe_rec] if base.rec_time.size else np.full(base.n, np.nan)
            with np.errstate(invalid="ignore"):
                mask &= (row_time >= lo) & (row_time <= hi)

    return mask


def score_column(
    key: str,
    model: ModelData | None,
    population: Population,
    n: int,
) -> np.ndarray:
    if key == "confidence":
        return population.confidence
    if model is None:
        return np.full(n, np.nan, dtype=np.float32)
    return model.scores[:, SCORE_COLUMNS.index(key)]


def domain_for(key: str, population: Population) -> tuple[float, float]:
    return population.domains.get(key) or PROPERTY_RANGES[key]


def compute_score_pass(
    model: ModelData | None,
    population: Population,
    filters: Filters,
    n: int,
) -> np.ndarray | None:
    """isPointVisible(multi): missing scores pass; clamp; +eps on the upper bound."""
    ranges = filters.active_score_ranges()
    if not ranges:
        return None
    ok = np.ones(n, dtype=bool)
    for key, norm_lo, norm_hi in ranges:
        if key not in PROPERTY_RANGES:
            continue
        p_min, p_max = domain_for(key, population)
        lo = p_min + norm_lo * (p_max - p_min)
        hi = p_min + norm_hi * (p_max - p_min)
        vals = score_column(key, model, population, n).astype(np.float64)
        missing = np.isnan(vals)
        clamped = np.clip(np.where(missing, p_min, vals), p_min, p_max)
        ok &= missing | ((clamped >= lo) & (clamped <= hi + SCORE_UPPER_EPS))
    return ok


def apply_filters(
    base: BaseData,
    model: ModelData | None,
    labels: LabelsData,
    filters: Filters,
    population: Population | None = None,
    non_score: np.ndarray | None = None,
) -> FilterResult:
    if population is None:
        population = compute_population(base, model, filters)
    if non_score is None:
        non_score = compute_non_score_mask(base, model, labels, population, filters)
    score_ok = compute_score_pass(model, population, filters, base.n)
    visible = non_score if score_ok is None else (non_score & score_ok)
    return FilterResult(population=population, non_score=non_score, visible=visible)


# ── Histograms ────────────────────────────────────────────────────────────────


def histogram(values: np.ndarray, bins: int, lo: float, hi: float) -> list[int]:
    """HistogramSlider.computeBins: clamp into [lo,hi], span||1, last bin inclusive."""
    finite = values[np.isfinite(values)].astype(np.float64)
    if finite.size == 0:
        return [0] * bins
    span = (hi - lo) or 1.0
    clamped = np.clip(finite, lo, hi)
    idx = np.minimum(bins - 1, np.floor((clamped - lo) / span * bins)).astype(np.int64)
    idx = np.maximum(idx, 0)
    return np.bincount(idx, minlength=bins)[:bins].astype(np.int64).tolist()


def score_histograms(
    model: ModelData | None,
    result: FilterResult,
    bins: int,
    n: int,
) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for key in SCORE_KEYS:
        lo, hi = domain_for(key, result.population)
        col = score_column(key, model, result.population, n)
        out[key] = {
            "total": histogram(col[result.non_score], bins, lo, hi),
            "visible": histogram(col[result.visible], bins, lo, hi),
        }
    return out


# ── Ordering ──────────────────────────────────────────────────────────────────


def _sort_values(
    field_name: str,
    rows: np.ndarray,
    base: BaseData,
    model: ModelData | None,
    population: Population,
) -> np.ndarray:
    """PredictionFeed.getSortValue for the given base rows (float64)."""
    if field_name in ("date", "time"):
        rec = base.recording_idx[rows]
        has_rec = rec >= 0
        safe = np.where(has_rec, rec, 0)
        if base.rec_epoch_day.size == 0:
            return np.full(rows.shape[0], -np.inf)
        day = base.rec_epoch_day[safe]
        has_dt = has_rec & (day != MISSING_DAY)
        if field_name == "date":
            vals = day.astype(np.float64)
        else:
            vals = base.rec_time[safe].astype(np.float64)
        return np.where(has_dt, vals, -np.inf)
    if field_name == "confidence":
        vals = population.confidence[rows].astype(np.float64)
        # prediction.confidence is 0 (not undefined) when no probability > 0.
        return np.where(np.isnan(vals), 0.0, vals)
    if model is None:
        return np.full(rows.shape[0], -np.inf)
    vals = model.scores[rows, SCORE_COLUMNS.index(field_name)].astype(np.float64)
    return np.where(np.isnan(vals), -np.inf, vals)


def native_order(base: BaseData, model: ModelData | None) -> np.ndarray:
    if model is None:
        return np.arange(base.n, dtype=np.int64)
    return model.native_order


def order_rows(
    base: BaseData,
    model: ModelData | None,
    result: FilterResult,
    sort: Sequence[tuple[str, str]],
) -> np.ndarray:
    """Visible base rows in feed order (int64)."""
    native = native_order(base, model)
    rows = native[result.visible[native]]
    if not sort or rows.size == 0:
        return rows
    keys: list[np.ndarray] = [np.arange(rows.size)]  # tie-break: native order
    for field_name, direction in reversed(list(sort)):
        vals = _sort_values(field_name, rows, base, model, result.population)
        keys.append(-vals if direction == "desc" else vals)
    return rows[np.lexsort(keys)]


def resolve_anchor(
    order: np.ndarray,
    anchor_row: int,
    labels: LabelsData,
    prefer_unlabeled: bool,
) -> int | None:
    """alSlice.resumeFromAnchor: anchor, or the next unlabeled row after it."""
    if order.size == 0:
        return None
    hits = np.flatnonzero(order == anchor_row) if anchor_row >= 0 else np.empty(0, np.int64)
    start = int(hits[0]) if hits.size else 0
    if not prefer_unlabeled:
        return start if hits.size else None
    unlabeled = ~labels.labeled[order[start:]]
    nxt = np.flatnonzero(unlabeled)
    if nxt.size:
        return start + int(nxt[0])
    return start


# ── Date / time facets ────────────────────────────────────────────────────────


def date_time_counts(base: BaseData) -> dict:
    """Recording-level counts for the Date range / Time of day histograms.

    Mirrors useDateTimeFilterData: a recording counts only when it has both a
    date and a time; counts cover recordings present in the snippet set.
    """
    if base.n == 0 or base.rec_ids.size == 0:
        return {"has_date_time": False, "date_domain": None, "date_counts": [], "time_counts": []}
    used = np.zeros(base.rec_ids.shape[0], dtype=bool)
    rec = base.recording_idx
    used[rec[rec >= 0]] = True
    has_dt = used & (base.rec_epoch_day != MISSING_DAY)
    if not has_dt.any():
        return {"has_date_time": False, "date_domain": None, "date_counts": [], "time_counts": []}
    days = base.rec_epoch_day[has_dt].astype(np.int64)
    uniq_days, day_counts = np.unique(days, return_counts=True)
    times = base.rec_time[has_dt].astype(np.float64)
    minutes = np.clip(np.floor(times / 60.0), 0, 1440).astype(np.int64)
    uniq_min, min_counts = np.unique(minutes, return_counts=True)
    return {
        "has_date_time": True,
        "date_domain": [int(uniq_days[0]), int(uniq_days[-1])],
        "date_counts": np.stack([uniq_days, day_counts], axis=1).tolist(),
        "time_counts": np.stack([uniq_min, min_counts], axis=1).tolist(),
    }


def finite_or_none(value: float) -> float | None:
    return None if value is None or not math.isfinite(value) else float(value)
