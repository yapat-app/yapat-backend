"""
Data loading helpers: embeddings, ground-truth metadata CSV, and alignment.
"""

from __future__ import annotations
import logging
import csv
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy.orm import Session

from app.models.recording import Recording
from app.models.snippet import Snippet
from app.models.embedding import EmbeddingVector
from app.services.pam_al._embedding_cache import load_embeddings_cached

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
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Align snippet embeddings with ground truth.

    For chunk-level copied files, matching is filename-based.
    If a GT event matches the snippet filename, its labels are applied directly,
    regardless of original min_t/max_t values.

    This is appropriate when files are already named like:
        originalfname_min_t_max_t.wav

    and each snippet row has:
        start_time=0.0, end_time=3.0
    """

    keep_indices: List[int] = []
    y_rows: List[np.ndarray] = []
    used_snippet_ids: List[int] = []

    matched_by_file_name = 0
    matched_by_file_path = 0
    no_gt_key_match = 0
    positive_aligned = 0
    negative_aligned = 0

    logger.info("========== CHUNK-LEVEL ALIGNMENT START ==========")
    logger.info("X shape: %s", getattr(X, "shape", None))
    logger.info("Number of snippet rows: %d", len(snippet_rows))
    logger.info("GT index size: %d", len(gt_index))
    logger.info("Species list: %s", species_list)

    for i, snippet in enumerate(snippet_rows):
        snippet_id = snippet.get("snippet_id")
        snippet_file_name = snippet.get("file_name")
        snippet_file_path = snippet.get("file_path")

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
            event_labels = event["labels"]
            y = np.maximum(y, event_labels)

            if i < 30:
                logger.info(
                    "[CHUNK LABEL MATCH] i=%d event=%d key=%s snippet_id=%s "
                    "file_name=%s labels=%s",
                    i,
                    event_idx,
                    matched_key,
                    snippet_id,
                    snippet_file_name,
                    event_labels.astype(int).tolist(),
                )

        keep_indices.append(i)
        y_rows.append(y)
        used_snippet_ids.append(snippet_id)
        if y.sum() > 0:
            positive_aligned += 1
        else:
            negative_aligned += 1

    logger.info("========== CHUNK-LEVEL ALIGNMENT SUMMARY ==========")
    logger.info("Matched by file_name: %d", matched_by_file_name)
    logger.info("Matched by file_path: %d", matched_by_file_path)
    logger.info("No GT key match: %d", no_gt_key_match)
    logger.info("Positive aligned samples: %d", positive_aligned)
    logger.info("Negative aligned samples: %d", negative_aligned)

    if not keep_indices:
        raise ValueError(
            "No matching labels found between snippet embeddings and ground-truth metadata."
        )

    X_aligned = X[keep_indices]
    y_aligned = np.stack(y_rows, axis=0).astype(np.float32)

    logger.info("Final X_aligned shape: %s", X_aligned.shape)
    logger.info("Final y_aligned shape: %s", y_aligned.shape)
    logger.info("Final class support: %s", y_aligned.sum(axis=0).astype(int).tolist())
    logger.info("========== CHUNK-LEVEL ALIGNMENT END ==========")

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

    with open(metadata_path, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            # Sniffer can't always confidently detect (e.g. very short files,
            # single-column files) -- fall back to the standard comma dialect.
            dialect = csv.excel

        reader = csv.DictReader(f, dialect=dialect)
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
    snippet_ids: List[int],
    species_candidates: List[str],
    model,
    min_samples_per_class: int,
    max_samples_per_class: Optional[int],
    is_negative_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[int], List[str], List[str], Dict[str, int]]:
    """
    Run filter_and_balance_classes on positive rows only, then reattach
    negative rows as explicit all-zero targets sized to the final label_order.

    filter_and_balance_classes drops all-zero rows internally
    (`y.sum(axis=1) > 0`), which would otherwise discard confirmed negatives
    as if they were noise. This function splits negatives out first, filters
    only the positive rows, then adds the negatives back in as true-negative
    training examples once num_classes/label_order is finalized.

    `is_negative_mask` defines what counts as a negative row for the caller:
    - metadata-CSV cold start: rows where every species column is 0
      (structural, no sentinel involved -- a CSV has no NO_EVENT_LABEL concept)
    - annotation/feedback-derived training: rows whose only label is the
      reserved NO_EVENT_LABEL sentinel

    Returns the same 6-tuple shape as model.filter_and_balance_classes.
    """
    is_negative_mask = np.asarray(is_negative_mask, dtype=bool)
    pos_mask = ~is_negative_mask

    X_pos = X[pos_mask]
    y_pos = y_full[pos_mask]
    pos_sids = [sid for sid, m in zip(snippet_ids, pos_mask) if m]

    X_neg = X[is_negative_mask]
    neg_sids = [sid for sid, m in zip(snippet_ids, is_negative_mask) if m]

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
            "split_filter_reattach_negatives: reattached %d confirmed-negative rows",
            len(neg_sids),
        )

    return X_train, y_train, labeled_sids, used_species, excluded_species, class_counts
