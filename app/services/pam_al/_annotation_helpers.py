"""
Annotation helpers: read/write ALSnippetAnnotation rows and build multi-hot
label matrices.
"""

from __future__ import annotations

import re
from typing import List, Optional

import numpy as np
from sqlalchemy.orm import Session

from app.models.snippet import Snippet
from app.models.user import User
from app.models.dataset import Dataset
from app.models.pam_active_learning import (
    ALSnippetAnnotation,
    ALAnnotationSource,
)
from active_learning.config import NO_EVENT_LABEL, NO_EVENT_LABELS

_SLUG_WHITESPACE_RE = re.compile(r"\s+")
_SLUG_INVALID_CHARS_RE = re.compile(r"[^a-z0-9_-]")


def _quick_label_slug(display_name: str) -> str:
    """
    Mirror the frontend's local-label slugify exactly (annotationCreatePayload.ts /
    DatasetQuickLabelsModal.tsx: `display.trim().toLowerCase().replace(/\\s+/g,
    "_").replace(/[^a-z0-9_-]/g, "").slice(0, 120)`), so a taxon_id derived here
    from arbitrary label text matches what the frontend would have generated for
    the same text.
    """
    slug = display_name.strip().lower()
    slug = _SLUG_WHITESPACE_RE.sub("_", slug)
    slug = _SLUG_INVALID_CHARS_RE.sub("", slug)
    return slug[:120]


def resolve_no_event_labels(db: Session, dataset_id: int) -> frozenset[str]:
    """
    Per-dataset confirmed-negative sentinel set: NO_EVENT_LABELS (taxon_id
    form, e.g. "local:no_biophony") plus the dataset's own quick-label
    display_name for any quick label whose taxon_id is one of those
    sentinels.

    The blind-annotation UI persists a quick label's display_name -- not its
    taxon_id -- as ALSnippetAnnotation.label (see useQuickLabelList.ts on the
    frontend, which maps stored quick labels to display_name before they ever
    reach the label picker). A "No biophony" quick label with
    taxon_id="local:no_biophony" therefore round-trips through training data
    as the literal string "No biophony", which NO_EVENT_LABELS alone (taxon_id
    form) would never match -- silently letting the sentinel leak back in as
    a trainable species class. Resolving the dataset's quick_labels config
    here closes that gap without having to change what gets persisted.

    This is an exact-match set; callers that need to tolerate casing/whitespace
    drift in previously-submitted annotation labels (e.g. the quick label's
    display text was retyped after annotations already used the old spelling)
    should check via `is_no_event_label` instead of membership in this set.
    """
    labels: set[str] = set(NO_EVENT_LABELS)
    quick_labels = db.query(Dataset.quick_labels).filter(Dataset.id == dataset_id).scalar() or []
    for item in quick_labels:
        if isinstance(item, dict) and item.get("taxon_id") in NO_EVENT_LABELS:
            display_name = item.get("display_name")
            if display_name:
                labels.add(display_name)
    return frozenset(labels)


def is_no_event_label(label: str, no_event_labels: frozenset[str]) -> bool:
    """
    True if `label` (an ALSnippetAnnotation.label value) is a confirmed-negative
    sentinel.

    Checks, in order:
      1. Exact match against `no_event_labels` (NO_EVENT_LABELS taxon_ids plus
         the dataset's currently-configured quick-label display_name(s)).
      2. A case/whitespace-insensitive slug match against the canonical
         NO_EVENT_LABELS taxon_ids -- covers a quick label whose display text
         was typed with different casing or extra whitespace than whatever is
         configured right now (e.g. "No Biophony", "no  biophony", trailing
         spaces), including historical annotations from before the display
         text was last edited.
    """
    if label in no_event_labels:
        return True
    return f"local:{_quick_label_slug(label)}" in NO_EVENT_LABELS


def store_snippet_annotations(
    db: Session,
    dataset_id: int,
    snippet_ids: list[int],
    y: np.ndarray,
    label_order: list[str],
    source: ALAnnotationSource,
    model_checkpoint_id: int | None = None,
    user_id: int | None = None,
) -> None:
    """
    Store one annotation row per positive label per snippet; a snippet with
    no positive label (a confirmed negative -- see
    split_filter_reattach_negatives) is stored as a single NO_EVENT_LABEL
    row instead of being silently skipped, so it round-trips as "labeled,
    no event" rather than looking unlabeled on reload.
    """
    if len(snippet_ids) != y.shape[0]:
        raise ValueError(f"Mismatch: {len(snippet_ids)=} but y has {y.shape[0]} rows.")
    if len(label_order) != y.shape[1]:
        raise ValueError(f"Mismatch: {len(label_order)=} but y has {y.shape[1]} columns.")

    # Track (snippet_id, label) pairs added within this call so that duplicate
    # input rows don't produce two pending session adds before the flush.
    seen: set[tuple[int, str]] = set()

    for row_idx, snippet_id in enumerate(snippet_ids):
        positive_indices = np.where(y[row_idx] > 0)[0]
        labels_to_store = (
            [label_order[i] for i in positive_indices]
            if len(positive_indices) > 0
            else [NO_EVENT_LABEL]
        )

        for label in labels_to_store:
            if (snippet_id, label) in seen:
                continue

            exists = (
                db.query(ALSnippetAnnotation)
                .filter(
                    ALSnippetAnnotation.snippet_id == snippet_id,
                    ALSnippetAnnotation.label == label,
                    ALSnippetAnnotation.source == source,
                    ALSnippetAnnotation.user_id == user_id,
                    ALSnippetAnnotation.model_checkpoint_id == model_checkpoint_id,
                )
                .first()
            )

            if exists is None:
                db.add(
                    ALSnippetAnnotation(
                        dataset_id=dataset_id,
                        snippet_id=snippet_id,
                        label=label,
                        source=source,
                        user_id=user_id,
                        model_checkpoint_id=model_checkpoint_id,
                    )
                )
                seen.add((snippet_id, label))


def replace_user_labels_for_snippet(
    db: Session,
    dataset_id: int,
    snippet_id: int,
    labels: list[str],
    model_checkpoint_id: int | None,
    user_id: int | None = None,
) -> None:
    """Replace the user's labels for a snippet (removes stale labels first)."""
    delete_user_labels_for_snippet(
        db,
        dataset_id=dataset_id,
        snippet_id=snippet_id,
        model_checkpoint_id=None,
        user_id=user_id,
    )
    if labels:
        store_user_labels_for_snippet(
            db,
            dataset_id=dataset_id,
            snippet_id=snippet_id,
            labels=labels,
            model_checkpoint_id=model_checkpoint_id,
            user_id=user_id,
        )


def store_user_labels_for_snippet(
    db: Session,
    dataset_id: int,
    snippet_id: int,
    labels: list[str],
    model_checkpoint_id: int | None,
    user_id: int | None = None,
) -> None:
    for label in labels:
        exists = (
            db.query(ALSnippetAnnotation)
            .filter(
                ALSnippetAnnotation.dataset_id == dataset_id,
                ALSnippetAnnotation.snippet_id == snippet_id,
                ALSnippetAnnotation.label == label,
                ALSnippetAnnotation.source == ALAnnotationSource.USER,
                ALSnippetAnnotation.user_id == user_id,
                ALSnippetAnnotation.model_checkpoint_id == model_checkpoint_id,
            )
            .one_or_none()
        )

        if exists is None:
            db.add(
                ALSnippetAnnotation(
                    dataset_id=dataset_id,
                    snippet_id=snippet_id,
                    label=label,
                    source=ALAnnotationSource.USER,
                    user_id=user_id,
                    model_checkpoint_id=model_checkpoint_id,
                )
            )


def delete_user_labels_for_snippet(
    db: Session,
    dataset_id: int,
    snippet_id: int,
    model_checkpoint_id: int | None = None,
    user_id: int | None = None,
) -> int:
    """
    Delete user-provided AL annotations for a snippet.

    Returns number of deleted rows.
    """
    query = db.query(ALSnippetAnnotation).filter(
        ALSnippetAnnotation.dataset_id == dataset_id,
        ALSnippetAnnotation.snippet_id == snippet_id,
        ALSnippetAnnotation.source == ALAnnotationSource.USER,
    )
    if user_id is not None:
        query = query.filter(ALSnippetAnnotation.user_id == user_id)
    if model_checkpoint_id is not None:
        query = query.filter(ALSnippetAnnotation.model_checkpoint_id == model_checkpoint_id)
    deleted = query.delete(synchronize_session=False)
    return int(deleted or 0)


def get_trusted_annotations(
    db: Session,
    dataset_id: int,
) -> dict[int, set[str]]:
    rows = (
        db.query(ALSnippetAnnotation.snippet_id, ALSnippetAnnotation.label)
        .filter(
            ALSnippetAnnotation.dataset_id == dataset_id,
            ALSnippetAnnotation.source.in_([
                ALAnnotationSource.GROUND_TRUTH,
                ALAnnotationSource.USER,
            ]),
        )
        .all()
    )

    out: dict[int, set[str]] = {}
    for snippet_id, label in rows:
        out.setdefault(snippet_id, set()).add(label)
    return out


def get_labeled_snippet_ids_for_dataset(db: Session, dataset_id: int) -> set[int]:
    rows = (
        db.query(ALSnippetAnnotation.snippet_id)
        .filter(ALSnippetAnnotation.dataset_id == dataset_id)
        .distinct()
        .all()
    )
    return {row[0] for row in rows}


def get_user_labeled_snippet_ids_for_dataset(
    db: Session,
    dataset_id: int,
    user_id: int,
) -> set[int]:
    rows = (
        db.query(ALSnippetAnnotation.snippet_id)
        .filter(
            ALSnippetAnnotation.dataset_id == dataset_id,
            ALSnippetAnnotation.source == ALAnnotationSource.USER,
            ALSnippetAnnotation.user_id == user_id,
        )
        .distinct()
        .all()
    )
    return {row[0] for row in rows}


def get_labeled_snippet_ids_for_snippet_set(
    db: Session,
    dataset_id: int,
    snippet_set_id: int,
) -> set[int]:
    """Like get_labeled_snippet_ids_for_dataset, but scoped to one snippet set."""
    rows = (
        db.query(ALSnippetAnnotation.snippet_id)
        .join(Snippet, Snippet.id == ALSnippetAnnotation.snippet_id)
        .filter(
            ALSnippetAnnotation.dataset_id == dataset_id,
            Snippet.snippet_set_id == snippet_set_id,
        )
        .distinct()
        .all()
    )
    return {row[0] for row in rows}


def get_user_labeled_snippet_ids_for_snippet_set(
    db: Session,
    dataset_id: int,
    snippet_set_id: int,
    user_id: int,
) -> set[int]:
    rows = (
        db.query(ALSnippetAnnotation.snippet_id)
        .join(Snippet, Snippet.id == ALSnippetAnnotation.snippet_id)
        .filter(
            ALSnippetAnnotation.dataset_id == dataset_id,
            Snippet.snippet_set_id == snippet_set_id,
            ALSnippetAnnotation.source == ALAnnotationSource.USER,
            ALSnippetAnnotation.user_id == user_id,
        )
        .distinct()
        .all()
    )
    return {row[0] for row in rows}


def get_labels_by_snippet(
    db: Session,
    dataset_id: int,
    snippet_set_id: int | None = None,
) -> dict[int, list[str]]:
    """
    Map snippet_id -> sorted list of trusted labels (ground-truth or user).

    Optionally restrict to a single snippet_set so the response stays compact
    when used to colour an FPV by `actual_label`.
    """
    query = (
        db.query(ALSnippetAnnotation.snippet_id, ALSnippetAnnotation.label)
        .filter(
            ALSnippetAnnotation.dataset_id == dataset_id,
            ALSnippetAnnotation.source.in_([
                ALAnnotationSource.GROUND_TRUTH,
                ALAnnotationSource.USER,
            ]),
        )
    )
    if snippet_set_id is not None:
        query = (
            query.join(Snippet, Snippet.id == ALSnippetAnnotation.snippet_id)
            .filter(Snippet.snippet_set_id == snippet_set_id)
        )

    grouped: dict[int, set[str]] = {}
    for snippet_id, label in query.all():
        grouped.setdefault(snippet_id, set()).add(label)
    return {sid: sorted(labels) for sid, labels in grouped.items()}


def get_label_details_by_snippet(
    db: Session,
    dataset_id: int,
    snippet_set_id: int | None = None,
    ground_truth_can_edit: bool = False,
    user_label_can_edit: bool = True,
) -> dict[int, list[dict]]:
    """
    Map snippet_id -> trusted labels with source and attribution metadata.

    Ground-truth labels are imported data and intentionally have no user row,
    so expose a stable display name instead of leaving clients to show Unknown.
    """
    query = (
        db.query(
            ALSnippetAnnotation.snippet_id,
            ALSnippetAnnotation.label,
            ALSnippetAnnotation.source,
            ALSnippetAnnotation.user_id,
            User.username,
        )
        .outerjoin(User, User.id == ALSnippetAnnotation.user_id)
        .filter(
            ALSnippetAnnotation.dataset_id == dataset_id,
            ALSnippetAnnotation.source.in_([
                ALAnnotationSource.GROUND_TRUTH,
                ALAnnotationSource.USER,
            ]),
        )
    )
    if snippet_set_id is not None:
        query = (
            query.join(Snippet, Snippet.id == ALSnippetAnnotation.snippet_id)
            .filter(Snippet.snippet_set_id == snippet_set_id)
        )

    grouped: dict[int, dict[tuple[str, str, int | None], dict]] = {}
    for snippet_id, label, source, user_id, username in query.all():
        source_value = source.value if hasattr(source, "value") else str(source)
        is_ground_truth = source_value == ALAnnotationSource.GROUND_TRUTH.value
        labeled_by = "Ground truth" if is_ground_truth else username
        key = (label, source_value, user_id)
        grouped.setdefault(snippet_id, {})[key] = {
            "label": label,
            "source": source_value,
            "user_id": user_id,
            "username": username,
            "labeled_by": labeled_by or "Unknown",
            "can_edit": ground_truth_can_edit if is_ground_truth else user_label_can_edit,
        }

    return {
        sid: sorted(details.values(), key=lambda item: (item["label"], item["source"], item["user_id"] or 0))
        for sid, details in grouped.items()
    }


def get_annotated_snippet_ids_for_snippet_set(
    db: Session,
    dataset_id: int,
    snippet_set_id: int,
) -> set[int]:
    rows = (
        db.query(ALSnippetAnnotation.snippet_id)
        .join(Snippet, Snippet.id == ALSnippetAnnotation.snippet_id)
        .filter(
            ALSnippetAnnotation.dataset_id == dataset_id,
            Snippet.snippet_set_id == snippet_set_id,
        )
        .distinct()
        .all()
    )
    return {row[0] for row in rows}


def build_multihot_from_annotations(
    snippet_ids: list[int],
    label_order: list[str],
    annotations_by_snippet: dict[int, set[str]],
) -> np.ndarray:
    label_to_idx = {label: i for i, label in enumerate(label_order)}
    y = np.zeros((len(snippet_ids), len(label_order)), dtype=np.float32)

    for row_idx, snippet_id in enumerate(snippet_ids):
        for label in annotations_by_snippet.get(snippet_id, set()):
            if label in label_to_idx:
                y[row_idx, label_to_idx[label]] = 1.0

    return y
