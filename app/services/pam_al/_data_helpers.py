"""
Data loading helpers: embeddings, ground-truth metadata CSV, and alignment.
"""

from __future__ import annotations
import logging
import csv
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.models.recording import Recording
from app.models.snippet import Snippet
from app.models.embedding import EmbeddingVector
from app.services.pam_al._embedding_cache import load_embeddings_cached
from app.utils.pam_training_paths import resolve_pam_metadata_path, resolve_pam_training_paths

logger = logging.getLogger(__name__)


def _load_embeddings_from_db(
    db: Session,
    snippet_set_id: int,
    embedding_model_id: int,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Uncached DB load (fallback when cache is disabled)."""
    rows = (
        db.query(
            Snippet.id,
            Snippet.recording_id,
            Snippet.start_time,
            Snippet.end_time,
            Recording.file_name,
            Recording.file_path,
            EmbeddingVector.vector,
            EmbeddingVector.dim,
        )
        .join(Recording, Snippet.recording_id == Recording.id)
        .join(EmbeddingVector, Snippet.id == EmbeddingVector.snippet_id)
        .filter(Snippet.snippet_set_id == snippet_set_id)
        .filter(EmbeddingVector.embedding_model_id == embedding_model_id)
        .order_by(Snippet.id)
        .all()
    )

    if not rows:
        raise ValueError(
            f"No embeddings found for snippet_set_id={snippet_set_id}, "
            f"embedding_model_id={embedding_model_id}"
        )

    dims = {row[7] for row in rows}
    if len(dims) != 1:
        raise ValueError(f"Inconsistent embedding dimensions found: {dims}")

    X = np.asarray([row[6] for row in rows], dtype=np.float32)

    snippet_rows = [
        {
            "snippet_id": row[0],
            "recording_id": row[1],
            "start_time": float(row[2]),
            "end_time": float(row[3]),
            "file_name": row[4],
            "file_path": row[5],
        }
        for row in rows
    ]

    return X, snippet_rows


def load_embeddings(
    db: Session,
    snippet_set_id: int,
    embedding_model_id: int,
    *,
    use_cache: bool = True,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Load embeddings for a snippet set.

    Returns (X [N, D], snippet_rows). Uses on-disk cache by default.
    """
    if use_cache:
        return load_embeddings_cached(db, snippet_set_id, embedding_model_id)
    return _load_embeddings_from_db(db, snippet_set_id, embedding_model_id)


def align_embeddings_and_labels(
    X: np.ndarray,
    snippet_rows: List[Dict[str, Any]],
    gt_index: Dict[str, List[Dict[str, Any]]],
    species_list: List[str],
    min_overlap_seconds: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Align snippet embeddings with ground truth.

    Matching is always filename-based first (snippet's file_name, falling
    back to file_path). Within a matched file, an event's labels apply to a
    snippet in one of two ways depending on what the metadata CSV provided:

    - Time-aware: if the event carries real start_time/end_time (i.e. the
      CSV had onset/offset, start_time/end_time, or min_t/max_t columns),
      the event only applies to snippets whose own [start_time, end_time)
      window overlaps the event's by more than ``min_overlap_seconds``.
      This is what supports arbitrary-length recordings that get
      auto-windowed into many snippets, with a metadata CSV giving precise
      per-event onset/offset times against the *whole recording's* clock
      (which is what a snippet's own start_time/end_time is measured
      against too, so the two are directly comparable).
    - Whole-file: if the event has no time info at all, its labels apply to
      every snippet of that file. This is the historical behavior, and
      remains correct for chunk-level copied files named like
      ``originalfname_min_t_max_t.wav`` where each file *is* already a
      single labeled clip (each snippet row then has
      start_time=0.0, end_time=<window>, matching the one event for that
      file's key).
    """

    keep_indices: List[int] = []
    y_rows: List[np.ndarray] = []
    used_snippet_ids: List[int] = []

    matched_by_file_name = 0
    matched_by_file_path = 0
    no_gt_key_match = 0
    positive_aligned = 0
    negative_aligned = 0

    logger.info("========== ALIGNMENT START ==========")
    logger.info("X shape: %s", getattr(X, "shape", None))
    logger.info("Number of snippet rows: %d", len(snippet_rows))
    logger.info("GT index size: %d", len(gt_index))
    logger.info("Species list: %s", species_list)

    time_filtered_events = 0

    for i, snippet in enumerate(snippet_rows):
        snippet_id = snippet.get("snippet_id")
        snippet_file_name = snippet.get("file_name")
        snippet_file_path = snippet.get("file_path")
        snippet_start = snippet.get("start_time")
        snippet_end = snippet.get("end_time")

        events = gt_index.get(snippet_file_name)
        matched_key = snippet_file_name

        if events is not None:
            matched_by_file_name += 1
        else:
            events = gt_index.get(snippet_file_path)
            matched_key = snippet_file_path

            if events is not None:
                matched_by_file_path += 1

        if not events:
            no_gt_key_match += 1

            if i < 30:
                logger.warning(
                    "[NO GT KEY MATCH] i=%d snippet_id=%s file_name=%s file_path=%s",
                    i,
                    snippet_id,
                    snippet_file_name,
                    snippet_file_path,
                )

            continue

        y = np.zeros(len(species_list), dtype=np.float32)

        for event_idx, event in enumerate(events):
            event_start = event.get("start_time")
            event_end = event.get("end_time")

            if (
                event_start is not None and event_end is not None
                and snippet_start is not None and snippet_end is not None
            ):
                # Time-aware: only apply this event's labels if it actually
                # overlaps this snippet's own window (both measured against
                # the same recording's clock).
                overlap = min(snippet_end, event_end) - max(snippet_start, event_start)
                if overlap <= min_overlap_seconds:
                    time_filtered_events += 1
                    continue

            event_labels = event["labels"]
            y = np.maximum(y, event_labels)

            if i < 30:
                logger.info(
                    "[LABEL MATCH] i=%d event=%d key=%s snippet_id=%s "
                    "file_name=%s snippet=[%s,%s) event=[%s,%s) labels=%s",
                    i,
                    event_idx,
                    matched_key,
                    snippet_id,
                    snippet_file_name,
                    snippet_start, snippet_end, event_start, event_end,
                    event_labels.astype(int).tolist(),
                )

        # Keep every matched row, positive or confirmed-negative (all-zero).
        # A snippet whose window overlapped a matched recording but no actual
        # event is a genuine "no target species here" sample, not noise --
        # callers that need to route negatives around min/max-samples-per-class
        # filtering (e.g. split_filter_reattach_negatives) rely on the zero
        # rows still being present here rather than silently dropped.
        keep_indices.append(i)
        y_rows.append(y)
        used_snippet_ids.append(snippet_id)
        if y.sum() > 0:
            positive_aligned += 1
        else:
            negative_aligned += 1

    logger.info("========== ALIGNMENT SUMMARY ==========")
    logger.info("Matched by file_name: %d", matched_by_file_name)
    logger.info("Matched by file_path: %d", matched_by_file_path)
    logger.info("No GT key match: %d", no_gt_key_match)
    logger.info("Time-filtered (non-overlapping) events skipped: %d", time_filtered_events)
    logger.info("Positive aligned samples: %d", positive_aligned)
    logger.info("Negative (confirmed no-event) aligned samples: %d", negative_aligned)

    if not keep_indices:
        raise ValueError(
            "No matching labels found between snippet embeddings and ground-truth metadata."
        )

    X_aligned = X[keep_indices]
    y_aligned = np.stack(y_rows, axis=0).astype(np.float32)

    logger.info("Final X_aligned shape: %s", X_aligned.shape)
    logger.info("Final y_aligned shape: %s", y_aligned.shape)
    logger.info("Final class support: %s", y_aligned.sum(axis=0).astype(int).tolist())
    logger.info("========== ALIGNMENT END ==========")

    return X_aligned, y_aligned, used_snippet_ids

# TODO: This function is suitable for AnuraSet and will need adaptation in future
def load_ground_truth_metadata(
    metadata_path: str,
    species_list: List[str],
    allowed_subsets: Optional[List[str]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    if not os.path.isfile(metadata_path):
        raise ValueError(f"Metadata file not found: {metadata_path}")

    species_to_idx = {species: i for i, species in enumerate(species_list)}
    gt_index: Dict[str, List[Dict[str, Any]]] = {}

    with open(metadata_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        subset_col = "subset" if "subset" in fieldnames else None
        has_fname_clip_key = all(col in fieldnames for col in ["fname", "min_t", "max_t"])

        id_col = None
        if not has_fname_clip_key:
            for candidate in [
                "file_name", "recording_file", "recording_name",
                "file_path", "fname", "sample_name",
            ]:
                if candidate in fieldnames:
                    id_col = candidate
                    break

        if not has_fname_clip_key and id_col is None:
            raise ValueError(
                "Metadata must contain either:\n"
                "- fname + min_t + max_t for snippet-level matching, or\n"
                "- one of sample_name, fname, file_name, recording_file, recording_name, file_path."
            )

        start_col = None
        end_col = None
        if "min_t" in fieldnames and "max_t" in fieldnames:
            start_col, end_col = "min_t", "max_t"
        elif "start_time" in fieldnames and "end_time" in fieldnames:
            start_col, end_col = "start_time", "end_time"
        elif "onset" in fieldnames and "offset" in fieldnames:
            start_col, end_col = "onset", "offset"

        has_species_columns = all(sp in fieldnames for sp in species_list)
        species_col = None
        for candidate in ["species", "label"]:
            if candidate in fieldnames:
                species_col = candidate
                break

        if not has_species_columns and species_col is None:
            raise ValueError(
                "Metadata must contain either:\n"
                "- one binary column per species in species_list, or\n"
                "- a 'species' / 'label' column."
            )

        for row in reader:
            if subset_col and allowed_subsets is not None:
                subset_value = str(row.get(subset_col, "")).strip().lower()
                if subset_value not in allowed_subsets:
                    continue

            start_time = None
            end_time = None
            if start_col is not None and end_col is not None:
                raw_start = row.get(start_col)
                raw_end = row.get(end_col)
                if raw_start not in (None, "") and raw_end not in (None, ""):
                    start_time = float(raw_start)
                    end_time = float(raw_end)

            if has_fname_clip_key:
                fname = str(row["fname"]).strip()
                if not fname:
                    continue
                if start_time is None or end_time is None:
                    raise ValueError("fname-based snippet metadata requires min_t/max_t or equivalent times.")

                def _fmt_time(t: float) -> str:
                    return str(int(t)) if float(t).is_integer() else str(t)

                recording_key = f"{fname}_{_fmt_time(start_time)}_{_fmt_time(end_time)}.wav"
            else:
                recording_key = str(row[id_col]).strip()
                if not recording_key:
                    continue

            y = np.zeros(len(species_list), dtype=np.float32)

            if has_species_columns:
                for sp in species_list:
                    value = str(row.get(sp, "0")).strip().lower()
                    y[species_to_idx[sp]] = 1.0 if value in {"1", "true", "yes"} else 0.0
            else:
                species_value = str(row[species_col]).strip()
                if species_value in species_to_idx:
                    y[species_to_idx[species_value]] = 1.0

            # Rows with all-zero labels (relative to species_list) are kept as
            # confirmed negatives -- e.g. a wide-format row asserting "none of
            # these species" for a clip/time-window, or an arbitrary-length
            # recording's onset/offset row for a non-target event. Previously
            # dropped here, which silently excluded such rows from gt_index
            # entirely (bad for reference-pool file filtering too: a recording
            # mentioned only via all-zero rows never got registered).
            gt_index.setdefault(recording_key, []).append(
                {
                    "labels": y,
                    "start_time": start_time,
                    "end_time": end_time,
                }
            )

    if not gt_index:
        raise ValueError(f"No usable ground-truth rows found in metadata file: {metadata_path}")

    return gt_index


def split_filter_reattach_negatives(
    X: np.ndarray,
    y_full: np.ndarray,
    snippet_ids: List[Optional[int]],
    species_candidates: List[str],
    model,
    min_samples_per_class: int,
    max_samples_per_class: Optional[int],
    is_negative_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[Optional[int]], List[str], List[str], Dict[str, int]]:
    """
    Run filter_and_balance_classes on positive rows only, then reattach
    negative rows as explicit all-zero targets sized to the final label_order.

    filter_and_balance_classes drops all-zero rows internally
    (``y.sum(axis=1) > 0``), which would otherwise discard confirmed
    negatives (e.g. rows for the reserved "no event" sentinel label, or
    structurally all-zero ground-truth CSV rows) as if they were noise. This
    splits negatives out first, runs min/max-samples-per-class filtering only
    on the positive rows -- so a large negative pool can never distort which
    species clear min_samples_per_class or how max_samples_per_class caps
    them -- then adds the negatives back in once num_classes/label_order is
    finalized.

    ``is_negative_mask`` is caller-defined: pass ``y_full.sum(axis=1) == 0``
    for a structural check (ground-truth CSV rows, reference-pool rows), or
    an annotation-derived mask (e.g. a snippet's only trusted label is the
    NO_EVENT_LABEL sentinel) when y_full was built by excluding that sentinel
    from species_candidates in the first place -- in both cases the negative
    rows are already all-zero in y_full by construction.

    Returns the same 6-tuple shape as model.filter_and_balance_classes.
    """
    is_negative_mask = np.asarray(is_negative_mask, dtype=bool)
    pos_mask = ~is_negative_mask

    X_pos = X[pos_mask]
    y_pos = y_full[pos_mask]
    pos_sids = [sid for sid, keep in zip(snippet_ids, pos_mask) if keep]

    X_neg = X[is_negative_mask]
    neg_sids = [sid for sid, keep in zip(snippet_ids, is_negative_mask) if keep]

    X_train, y_train, labeled_sids, used_species, excluded_species, class_counts = (
        model.filter_and_balance_classes(
            X=X_pos, y=y_pos, snippet_ids=pos_sids,
            species_list=species_candidates,
            min_samples_per_class=min_samples_per_class,
            max_samples_per_class=max_samples_per_class,
        )
    )

    if neg_sids and len(used_species) > 0:
        y_neg = np.zeros((len(neg_sids), len(used_species)), dtype=np.float32)
        X_train = np.concatenate([X_train, X_neg], axis=0)
        y_train = np.concatenate([y_train, y_neg], axis=0)
        labeled_sids = list(labeled_sids) + neg_sids
        logger.info(
            "split_filter_reattach_negatives: reattached %d confirmed-negative rows "
            "(positive rows=%d, classes=%d)",
            len(neg_sids), len(pos_sids), len(used_species),
        )

    return X_train, y_train, labeled_sids, used_species, excluded_species, class_counts


# ── Reference data pool ──────────────────────────────────────────────────
#
# Reference datasets are ordinary Datasets (Dataset.is_reference=True) that
# went through the normal scan/snippet/embed pipeline and carry their own
# pam_metadata.csv (same format load_ground_truth_metadata already parses).
# A target dataset opts into one or more reference datasets via
# DatasetReferenceLink (direct dataset_id link, or team_id link shared by
# every dataset under that team). See docs/reference-data-pool-design.md.

_METADATA_STRUCTURAL_COLUMNS = {
    "fname", "min_t", "max_t", "start_time", "end_time",
    "onset", "offset", "subset", "file_name", "recording_file",
    "recording_name", "file_path", "sample_name",
}


def scan_metadata_species(metadata_path: str) -> set:
    """
    Read a pam_metadata.csv-format file and return the set of species it
    references, without needing a species_list upfront (unlike
    load_ground_truth_metadata, which requires one to build its label
    index). Used to discover reference-only species before mixing reference
    data into a training run.
    """
    if not os.path.isfile(metadata_path):
        raise ValueError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        species_col = None
        for candidate in ["species", "label"]:
            if candidate in fieldnames:
                species_col = candidate
                break

        if species_col is not None:
            species: set = set()
            for row in reader:
                value = str(row.get(species_col, "")).strip()
                if value:
                    species.add(value)
            return species

        # Wide format: one binary column per species -- everything that
        # isn't a known structural column is treated as a species column.
        return {c for c in fieldnames if c not in _METADATA_STRUCTURAL_COLUMNS}


def get_referenced_filenames(metadata_path: str) -> set:
    """
    Return the exact set of recording-key filenames a pam_metadata.csv-format
    file references -- i.e. the same keys load_ground_truth_metadata would
    build in gt_index (either ``{fname}_{min_t}_{max_t}.wav`` for chunk-level
    CSVs, or the plain filename column value otherwise).

    Used to filter which on-disk audio files actually get registered/scanned
    for a reference dataset, so scan/embed compute isn't spent on files the
    CSV will never use as ground truth -- lets ``source_uri`` point at a
    larger directory (e.g. the full upstream corpus) while only the files
    the metadata CSV mentions get turned into Recordings.

    Reuses load_ground_truth_metadata directly (via a species list
    discovered from the CSV itself) so the filter can never drift out of
    sync with what training-time matching actually does.
    """
    species_list = sorted(scan_metadata_species(metadata_path))
    if not species_list:
        return set()
    gt_index = load_ground_truth_metadata(metadata_path, species_list, allowed_subsets=None)
    return set(gt_index.keys())


def get_reference_dataset_ids(db: Session, dataset) -> List[int]:
    """
    Resolve the effective reference-dataset ids for a target dataset: the
    union of dataset-scoped links (dataset_id == dataset.id) and
    team-scoped links (team_id == dataset.team_id).
    """
    from app.models.reference_link import DatasetReferenceLink

    filters = [DatasetReferenceLink.dataset_id == dataset.id]
    if dataset.team_id is not None:
        filters.append(DatasetReferenceLink.team_id == dataset.team_id)

    rows = (
        db.query(DatasetReferenceLink.reference_dataset_id)
        .filter(or_(*filters))
        .distinct()
        .all()
    )
    return [r[0] for r in rows]


def load_reference_pool_training_data(
    db: Session,
    dataset,
    embedding_model_id: int,
    species_list: List[str],
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
    """
    Load and concatenate training data from every reference dataset linked
    to ``dataset`` (directly, or via its team).

    Returns (X_ref, y_ref, unified_species_list, info):
      - unified_species_list is ``species_list`` with any reference-only
        species appended at the end. If it differs from ``species_list``,
        the caller must zero-pad its own y matrix's columns to match before
        concatenating with y_ref.
      - info carries per-reference-dataset sample counts and a list of
        skipped datasets (with reasons) for checkpoint provenance / debugging.

    Reference datasets missing a metadata CSV, a default snippet set, or
    embeddings for ``embedding_model_id`` are skipped with a logged reason
    rather than failing the whole training run -- reference data is
    supplementary, not required.
    """
    from app.config import settings
    from app.models.dataset import Dataset as DatasetModel

    empty_info = {"reference_dataset_ids": [], "reference_sample_count": 0, "skipped": []}

    ref_ids = get_reference_dataset_ids(db, dataset)
    if not ref_ids:
        return (
            np.empty((0, 0), dtype=np.float32),
            np.empty((0, len(species_list)), dtype=np.float32),
            list(species_list),
            empty_info,
        )

    ref_datasets = db.query(DatasetModel).filter(DatasetModel.id.in_(ref_ids)).all()
    DATA_ROOT = settings.DATA_ROOT or "/data"

    skipped: List[Dict[str, Any]] = []

    # Pass 1: resolve each reference dataset's metadata path and discover its
    # own species vocabulary, so the unified label space is known before any
    # y matrix gets built.
    ref_meta: Dict[int, Dict[str, Any]] = {}
    extra_species: set = set()

    for ref_ds in ref_datasets:
        if ref_ds.default_snippet_set_id is None:
            skipped.append({
                "dataset_id": ref_ds.id, "name": ref_ds.name,
                "reason": "no default_snippet_set_id",
            })
            continue
        try:
            meta_rel = resolve_pam_metadata_path(
                DATA_ROOT, ref_ds.source_uri, metadata_path=ref_ds.reference_metadata_path,
            )
            meta_path = os.path.join(DATA_ROOT, meta_rel)
            own_species = scan_metadata_species(meta_path)
        except ValueError as e:
            skipped.append({"dataset_id": ref_ds.id, "name": ref_ds.name, "reason": str(e)})
            continue

        if not own_species:
            skipped.append({
                "dataset_id": ref_ds.id, "name": ref_ds.name,
                "reason": "metadata file has no usable species",
            })
            continue

        ref_meta[ref_ds.id] = {
            "dataset": ref_ds, "metadata_path": meta_path, "own_species": sorted(own_species),
        }
        extra_species |= own_species

    unified_species = list(species_list) + [sp for sp in sorted(extra_species) if sp not in species_list]
    species_to_unified_idx = {sp: i for i, sp in enumerate(unified_species)}

    # Pass 2: load embeddings + labels per reference dataset (against its own
    # species vocabulary, since a single CSV rarely covers every species
    # across every linked reference dataset), then scatter into the unified
    # column space.
    X_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    per_dataset_counts: Dict[int, int] = {}

    for ref_id, meta in ref_meta.items():
        ref_ds = meta["dataset"]
        try:
            X, snippet_rows = load_embeddings(db, ref_ds.default_snippet_set_id, embedding_model_id)
            gt_index = load_ground_truth_metadata(meta["metadata_path"], meta["own_species"], allowed_subsets=None)
            X_own, y_own, _ = align_embeddings_and_labels(X, snippet_rows, gt_index, meta["own_species"])
        except ValueError as e:
            skipped.append({"dataset_id": ref_ds.id, "name": ref_ds.name, "reason": str(e)})
            continue

        if X_own.shape[0] == 0:
            continue

        y_scattered = np.zeros((y_own.shape[0], len(unified_species)), dtype=np.float32)
        for local_idx, sp in enumerate(meta["own_species"]):
            y_scattered[:, species_to_unified_idx[sp]] = y_own[:, local_idx]

        X_parts.append(X_own)
        y_parts.append(y_scattered)
        per_dataset_counts[ref_id] = int(X_own.shape[0])

    if not X_parts:
        return (
            np.empty((0, 0), dtype=np.float32),
            np.empty((0, len(unified_species)), dtype=np.float32),
            unified_species,
            {"reference_dataset_ids": ref_ids, "reference_sample_count": 0, "skipped": skipped},
        )

    X_ref = np.concatenate(X_parts, axis=0)
    y_ref = np.concatenate(y_parts, axis=0)

    info = {
        "reference_dataset_ids": ref_ids,
        "reference_sample_count": int(X_ref.shape[0]),
        "reference_counts_by_dataset": per_dataset_counts,
        "skipped": skipped,
    }
    return X_ref, y_ref, unified_species, info
