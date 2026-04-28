"""
Annotation helpers: read/write ALSnippetAnnotation rows and build multi-hot
label matrices.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from sqlalchemy.orm import Session

from app.models.snippet import Snippet
from app.models.pam_active_learning import (
    ALSnippetAnnotation,
    ALAnnotationSource,
)


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
    """Store one annotation row per positive label per snippet."""
    if len(snippet_ids) != y.shape[0]:
        raise ValueError(f"Mismatch: {len(snippet_ids)=} but y has {y.shape[0]} rows.")
    if len(label_order) != y.shape[1]:
        raise ValueError(f"Mismatch: {len(label_order)=} but y has {y.shape[1]} columns.")

    for row_idx, snippet_id in enumerate(snippet_ids):
        positive_indices = np.where(y[row_idx] > 0)[0]

        for class_idx in positive_indices:
            label = label_order[class_idx]

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


def store_user_labels_for_snippet(
    db: Session,
    dataset_id: int,
    snippet_id: int,
    labels: list[str],
    model_checkpoint_id: int,
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
