"""
Ground-truth / user labels aligned to a base layer.

Labels change whenever anyone annotates, so this layer is not persisted. It is
rebuilt from ``al_snippet_annotation`` and cached per worker, keyed by a cheap
table fingerprint ``(count, max(id))`` for the dataset. Any insert raises
``max(id)``; any delete lowers ``count``; so every write path (feedback,
label deletion, ground-truth import, other users) is picked up without having
to hook each of them.
"""

from __future__ import annotations

import hashlib

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.pam_active_learning import ALAnnotationSource, ALSnippetAnnotation
from app.models.snippet import Snippet
from app.services.explore.builders import stream_rows
from app.services.explore.query import BaseData, LabelsData

TRUSTED_SOURCES = (ALAnnotationSource.GROUND_TRUTH, ALAnnotationSource.USER)


def labels_fingerprint(db: Session, dataset_id: int) -> tuple[int, int]:
    count, max_id = db.execute(
        select(func.count(ALSnippetAnnotation.id), func.max(ALSnippetAnnotation.id)).where(
            ALSnippetAnnotation.dataset_id == dataset_id
        )
    ).one()
    return int(count or 0), int(max_id or 0)


def build_labels(
    db: Session,
    base: BaseData,
    dataset_id: int,
    snippet_set_id: int,
    fingerprint: tuple[int, int],
) -> LabelsData:
    snippet_chunks: list[np.ndarray] = []
    names: list[str] = []
    stmt = (
        select(ALSnippetAnnotation.snippet_id, ALSnippetAnnotation.label)
        .join(Snippet, Snippet.id == ALSnippetAnnotation.snippet_id)
        .where(ALSnippetAnnotation.dataset_id == dataset_id)
        .where(Snippet.snippet_set_id == snippet_set_id)
        .where(ALSnippetAnnotation.source.in_(TRUSTED_SOURCES))
    )
    for part in stream_rows(db, stmt):
        snippet_chunks.append(np.fromiter((row[0] for row in part), dtype=np.int64, count=len(part)))
        names.extend(str(row[1]) for row in part)

    version = hashlib.sha1(
        f"{base.version}|{dataset_id}|{snippet_set_id}|{fingerprint[0]}|{fingerprint[1]}".encode()
    ).hexdigest()[:16]
    return labels_from_pairs(
        base,
        np.concatenate(snippet_chunks) if snippet_chunks else np.empty(0, np.int64),
        names,
        version=version,
    )


def labels_from_pairs(
    base: BaseData,
    snippet_ids: np.ndarray,
    names: list[str],
    *,
    version: str,
) -> LabelsData:
    n = base.n
    rows_all = base.index_of(snippet_ids) if snippet_ids.size else np.empty(0, np.int64)
    in_set = rows_all >= 0
    names = [name for name, keep in zip(names, in_set) if keep]
    snippet_ids = snippet_ids[in_set]
    vocab = sorted(set(names))
    if not vocab or snippet_ids.size == 0:
        return LabelsData(
            version=version,
            labeled=np.zeros(n, dtype=bool),
            indptr=np.zeros(n + 1, dtype=np.int64),
            label_idx=np.empty(0, dtype=np.int32),
            first_label=np.full(n, -1, dtype=np.int32),
            vocab=vocab,
        )
    code_of = {name: i for i, name in enumerate(vocab)}
    codes = np.fromiter((code_of[name] for name in names), dtype=np.int64, count=len(names))
    rows = rows_all[in_set]

    # Unique (row, code) pairs sorted by row then label name.
    pair = np.unique(rows * (len(vocab) + 1) + codes)
    rows_u = pair // (len(vocab) + 1)
    codes_u = (pair % (len(vocab) + 1)).astype(np.int32)

    counts = np.bincount(rows_u, minlength=n)
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    labeled = counts > 0
    first_label = np.full(n, -1, dtype=np.int32)
    first_label[labeled] = codes_u[indptr[:-1][labeled]]
    return LabelsData(
        version=version,
        labeled=labeled,
        indptr=indptr,
        label_idx=codes_u,
        first_label=first_label,
        vocab=vocab,
    )
