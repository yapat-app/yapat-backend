"""
Share labels between the snippet sets of a dataset (issue #93, part 2).

Models with different windows cut the same recording into different clips, for
example 3 s clips for BirdNET and 5 s clips for Perch. A label made on one clip
is shared with every clip of another set that overlaps it:

- Copied automatically when it must hold for the target clip:
  - a species label, when the source clip lies fully inside the target clip
    (the call is in the target too);
  - a no-event label, when the target clip lies fully inside the source clip
    (if the whole source is silent, so is every part of it).
- Every other overlap goes to the review queue (``label_share_candidates``).
- A label already on the target is never overwritten. A target that has a label
  of the opposite kind (species vs no event) gets a queue item, not a copy.

Copied labels have ``origin`` ``auto_shared`` and ``derived_from_id`` pointing to
their source. Only ``manual`` labels are shared, so copies are never copied
again. A copy is removed with its source, and a copy someone deletes is
remembered as rejected so it does not come back. A reviewed label stays when
its source is deleted; an optional recheck item (reason ``source_removed``)
lets someone keep or remove it.

Both label stores are kept in sync (see the yapat-codebase skill, invariant 5):
an automatic copy is mirrored into ``al_snippet_annotation`` with source
DERIVED, and a reviewed one with source USER under the reviewer.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Optional

from sqlalchemy import inspect as sa_inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from active_learning.config import NO_EVENT_LABELS
from app.config import settings
from app.models.annotation import Annotation
from app.models.dataset import Dataset
from app.models.label_share import LabelShareCandidate, LabelShareReason, LabelShareStatus
from app.models.pam_active_learning import ALAnnotationSource, ALSnippetAnnotation
from app.models.recording import Recording
from app.models.snippet import Snippet
from app.services.pam_al._annotation_helpers import is_no_event_label, resolve_no_event_labels

logger = logging.getLogger(__name__)

ORIGIN_MANUAL = "manual"
ORIGIN_AUTO = "auto_shared"
ORIGIN_REVIEWED = "reviewed_shared"

# Clip times are floats computed from window and step; allow for rounding.
_EPS = 1e-3


# ── Small helpers ───────────────────────────────────────────────────────

@dataclass
class _DatasetContext:
    dataset_id: int
    no_event_labels: frozenset
    # taxon_id -> quick label text, which is what the hub stores as the AL label.
    quick_label_text: dict = field(default_factory=dict)


def _context(db: Session, dataset_id: int) -> _DatasetContext:
    quick_labels = db.query(Dataset.quick_labels).filter(Dataset.id == dataset_id).scalar() or []
    quick_text = {}
    for item in quick_labels:
        if isinstance(item, dict) and item.get("taxon_id") and item.get("display_name"):
            quick_text.setdefault(str(item["taxon_id"]), str(item["display_name"]).strip())
    return _DatasetContext(
        dataset_id=dataset_id,
        no_event_labels=resolve_no_event_labels(db, dataset_id),
        quick_label_text=quick_text,
    )


def _overlap(a: Snippet, b: Snippet) -> float:
    return min(a.end_time, b.end_time) - max(a.start_time, b.start_time)


def _contains(outer: Snippet, inner: Snippet) -> bool:
    return outer.start_time <= inner.start_time + _EPS and inner.end_time <= outer.end_time + _EPS


def _label_text(annotation: Annotation) -> str:
    """The label as the hub shows it, for comparing labels across taxon ids."""
    meta = annotation.extra_metadata or {}
    return str(meta.get("al_label") or annotation.resolved_name_snapshot or "").strip().lower()


def _has_label(existing: Iterable[Annotation], label: Annotation, al_label: str) -> bool:
    """Whether a clip already carries this label.

    Matches the taxon id, or the label text: the same name can exist under
    several custom taxon ids (one per code created on the fly), and a clip
    showing "Frog A" should not be offered "Frog A" again.
    """
    texts = {al_label.strip().lower(), (label.resolved_name_snapshot or "").strip().lower()} - {""}
    return any(a.taxon_id == label.taxon_id or _label_text(a) in texts for a in existing)


def _drop(db: Session, candidate: LabelShareCandidate) -> None:
    """Delete a queue item, unless an earlier step of this pass already did."""
    state = sa_inspect(candidate)
    if state.deleted or state.was_deleted or candidate in db.deleted:
        return
    db.delete(candidate)


def _close_pending_for(
    db: Session,
    target_id: int,
    label: Annotation,
    al_label: str,
    keep: Optional[LabelShareCandidate] = None,
) -> None:
    """A clip just got this label: pending suggestions of it have nothing left to decide.

    ``keep`` is the suggestion being reviewed. Sessions do not autoflush, so the
    database can still list it as pending, and its record must survive.
    """
    query = db.query(LabelShareCandidate).filter(
        LabelShareCandidate.target_snippet_id == target_id,
        LabelShareCandidate.status == LabelShareStatus.PENDING,
    )
    if keep is not None:
        query = query.filter(LabelShareCandidate.id != keep.id)
    for candidate in query:
        same_text = (candidate.al_label or "").strip().lower() == al_label.strip().lower()
        if candidate.taxon_id == label.taxon_id or same_text:
            _drop(db, candidate)


def _same_label(candidate: LabelShareCandidate, label: Annotation, al_label: str) -> bool:
    texts = {al_label.strip().lower(), (label.resolved_name_snapshot or "").strip().lower()} - {""}
    return candidate.taxon_id == label.taxon_id or (candidate.al_label or "").strip().lower() in texts


def _is_no_event(annotation: Annotation, ctx: _DatasetContext) -> bool:
    if annotation.taxon_id in NO_EVENT_LABELS:
        return True
    meta = annotation.extra_metadata or {}
    for text in (meta.get("al_label"), annotation.resolved_name_snapshot):
        if text and is_no_event_label(text, ctx.no_event_labels):
            return True
    return False


def al_label_for(
    annotation: Annotation,
    ctx: _DatasetContext,
    labels_on_source: Optional[set[str]] = None,
) -> str:
    """The al_snippet_annotation label that stands for this annotation.

    The two stores do not link rows, and the label text differs by write path:
    the hub stores quick-label text, /api/annotations the resolved name. Prefer a
    text the source clip actually has in al_snippet_annotation, so the copy trains
    as the same class.
    """
    meta = annotation.extra_metadata or {}
    options = [
        meta.get("al_label"),
        ctx.quick_label_text.get(annotation.taxon_id),
        annotation.resolved_name_snapshot,
        annotation.taxon_id,
    ]
    options = [str(o).strip() for o in options if o and str(o).strip()]
    if labels_on_source:
        for option in options:
            if option in labels_on_source:
                return option
    return options[0]


def add_al_row(
    db: Session,
    dataset_id: int,
    snippet_id: int,
    label: str,
    source: ALAnnotationSource,
    user_id: Optional[int],
) -> None:
    exists = (
        db.query(ALSnippetAnnotation.id)
        .filter(
            ALSnippetAnnotation.snippet_id == snippet_id,
            ALSnippetAnnotation.label == label,
            ALSnippetAnnotation.source == source,
            ALSnippetAnnotation.user_id == user_id if user_id is not None else ALSnippetAnnotation.user_id.is_(None),
            ALSnippetAnnotation.model_checkpoint_id.is_(None),
        )
        .first()
    )
    if exists is None:
        db.add(ALSnippetAnnotation(
            dataset_id=dataset_id,
            snippet_id=snippet_id,
            label=label,
            source=source,
            user_id=user_id,
            model_checkpoint_id=None,
        ))


def _remove_derived_al_row(db: Session, annotation: Annotation) -> None:
    label = (annotation.extra_metadata or {}).get("al_label") or annotation.resolved_name_snapshot
    (
        db.query(ALSnippetAnnotation)
        .filter(
            ALSnippetAnnotation.snippet_id == annotation.snippet_id,
            ALSnippetAnnotation.label == label,
            ALSnippetAnnotation.source == ALAnnotationSource.DERIVED,
        )
        .delete(synchronize_session=False)
    )


def _dataset_id_for_snippet(db: Session, snippet: Snippet) -> Optional[int]:
    return db.query(Recording.dataset_id).filter(Recording.id == snippet.recording_id).scalar()


def _overlapping_snippets(
    db: Session, snippet: Snippet, only_set_id: Optional[int] = None
) -> list[Snippet]:
    """Clips of the recording's other snippet sets that overlap this one.

    ``only_set_id`` limits them to one snippet set (a newly embedded one).
    """
    min_overlap = settings.LABEL_SHARE_MIN_OVERLAP_S
    query = db.query(Snippet).filter(
        Snippet.recording_id == snippet.recording_id,
        Snippet.snippet_set_id != snippet.snippet_set_id,
        Snippet.start_time < snippet.end_time,
        Snippet.end_time > snippet.start_time,
    )
    if only_set_id is not None:
        query = query.filter(Snippet.snippet_set_id == only_set_id)
    rows = query.all()
    # A clip that lies inside the source counts even when shorter than the minimum.
    return [
        t for t in rows
        if _overlap(snippet, t) >= min_overlap - _EPS or _contains(snippet, t)
    ]


# ── Sharing ─────────────────────────────────────────────────────────────

def _share_snippet(
    db: Session,
    ctx: _DatasetContext,
    source: Snippet,
    only_set_id: Optional[int] = None,
) -> dict:
    """Bring the copies and queue items of one clip's manual labels up to date.

    ``only_set_id`` shares onto that snippet set's clips only.
    """
    counts = {"auto_shared": 0, "queued": 0}
    targets = _overlapping_snippets(db, source, only_set_id)
    if not targets:
        return counts
    target_ids = [t.id for t in targets]

    labels_on_targets: dict[int, list[Annotation]] = defaultdict(list)
    for ann in db.query(Annotation).filter(Annotation.snippet_id.in_(target_ids)).all():
        labels_on_targets[ann.snippet_id].append(ann)

    # Copies whose source was deleted elsewhere (the FK set derived_from_id to NULL).
    for snippet_id, anns in labels_on_targets.items():
        for ann in list(anns):
            if ann.origin == ORIGIN_AUTO and ann.derived_from_id is None:
                _remove_derived_al_row(db, ann)
                db.delete(ann)
                anns.remove(ann)

    sources = (
        db.query(Annotation)
        .filter(Annotation.snippet_id == source.id, Annotation.origin == ORIGIN_MANUAL)
        .all()
    )
    if not sources:
        db.flush()
        return counts

    known_pairs = {
        (row.source_annotation_id, row.target_snippet_id): row
        for row in db.query(LabelShareCandidate).filter(
            LabelShareCandidate.source_annotation_id.in_([a.id for a in sources]),
        )
    }
    labels_on_source = {
        row[0]
        for row in db.query(ALSnippetAnnotation.label).filter(ALSnippetAnnotation.snippet_id == source.id)
    }
    # Rechecks of labels that lost their source: a label shared again supports them.
    rechecks: dict[int, list[LabelShareCandidate]] = defaultdict(list)
    for row in db.query(LabelShareCandidate).filter(
        LabelShareCandidate.target_snippet_id.in_(target_ids),
        LabelShareCandidate.status == LabelShareStatus.PENDING,
        LabelShareCandidate.reason == LabelShareReason.SOURCE_REMOVED,
    ):
        rechecks[row.target_snippet_id].append(row)

    for label in sources:
        no_event = _is_no_event(label, ctx)
        al_label = al_label_for(label, ctx, labels_on_source)
        for target in targets:
            existing = labels_on_targets[target.id]
            known = known_pairs.get((label.id, target.id))
            if _has_label(existing, label, al_label):
                # Already labelled: a pending queue item has nothing left to decide.
                if known is not None and known.status == LabelShareStatus.PENDING:
                    _drop(db, known)
                for recheck in rechecks[target.id]:
                    if _same_label(recheck, label, al_label):
                        _drop(db, recheck)
                continue
            if known is not None:
                continue  # queued already, or reviewed

            source_inside = _contains(target, source)
            target_inside = _contains(source, target)
            certain = target_inside if no_event else source_inside
            conflict = any(_is_no_event(a, ctx) != no_event for a in existing)

            if certain and not conflict:
                copy = Annotation(
                    snippet_id=target.id,
                    user_id=label.user_id,
                    embedding_model_id=None,
                    taxon_id=label.taxon_id,
                    resolved_name_snapshot=label.resolved_name_snapshot,
                    confidence=label.confidence,
                    origin=ORIGIN_AUTO,
                    derived_from_id=label.id,
                    extra_metadata={
                        "source": "label_sharing",
                        "dataset_id": ctx.dataset_id,
                        "al_label": al_label,
                    },
                )
                db.add(copy)
                existing.append(copy)
                add_al_row(db, ctx.dataset_id, target.id, al_label, ALAnnotationSource.DERIVED, label.user_id)
                # Another source may have queued this label for this clip first.
                _close_pending_for(db, target.id, label, al_label)
                counts["auto_shared"] += 1
                continue

            if conflict:
                reason = LabelShareReason.CONFLICT
            elif not (source_inside or target_inside):
                reason = LabelShareReason.PARTIAL_OVERLAP
            elif no_event:
                reason = LabelShareReason.TARGET_LONGER
            else:
                reason = LabelShareReason.TARGET_SHORTER
            candidate = LabelShareCandidate(
                dataset_id=ctx.dataset_id,
                source_annotation_id=label.id,
                source_snippet_id=source.id,
                target_snippet_id=target.id,
                target_snippet_set_id=target.snippet_set_id,
                taxon_id=label.taxon_id,
                resolved_name_snapshot=label.resolved_name_snapshot,
                al_label=al_label,
                reason=reason,
                overlap_seconds=round(_overlap(source, target), 3),
            )
            db.add(candidate)
            known_pairs[(label.id, target.id)] = candidate
            counts["queued"] += 1

    db.flush()
    return counts


def _resync_targets(db: Session, ctx: _DatasetContext, target_ids: Iterable[int]) -> None:
    """Re-share onto clips that lost a copy: another overlapping label may still apply."""
    target_ids = set(target_ids)
    if not target_ids:
        return
    resynced: set[int] = set()
    for target in db.query(Snippet).filter(Snippet.id.in_(target_ids)).all():
        for neighbour in _overlapping_snippets(db, target):
            if neighbour.id in resynced:
                continue
            has_manual = (
                db.query(Annotation.id)
                .filter(Annotation.snippet_id == neighbour.id, Annotation.origin == ORIGIN_MANUAL)
                .first()
            )
            if has_manual:
                _share_snippet(db, ctx, neighbour)
            resynced.add(neighbour.id)


def share_after_change(
    db: Session,
    snippet_ids: Iterable[int],
    lost_copies_on: Iterable[int] = (),
) -> None:
    """Call after labels on these clips were added or removed, before the commit.

    Runs in a savepoint: a failure here (for example two requests sharing onto
    the same clip at once) must not lose the label the user just made. Anything
    skipped is picked up by the next change on the clip or a dataset backfill.
    """
    if not settings.LABEL_SHARING_ENABLED:
        return
    snippet_ids = set(snippet_ids)
    lost_copies_on = set(lost_copies_on)
    if not snippet_ids and not lost_copies_on:
        return
    try:
        with db.begin_nested():
            db.flush()
            contexts: dict[int, _DatasetContext] = {}
            for snippet in db.query(Snippet).filter(Snippet.id.in_(snippet_ids | lost_copies_on)).all():
                dataset_id = _dataset_id_for_snippet(db, snippet)
                if dataset_id is None:
                    continue
                ctx = contexts.get(dataset_id) or contexts.setdefault(dataset_id, _context(db, dataset_id))
                if snippet.id in snippet_ids:
                    _share_snippet(db, ctx, snippet)
                if snippet.id in lost_copies_on:
                    _resync_targets(db, ctx, [snippet.id])
    except IntegrityError:
        logger.warning("Label sharing skipped for snippets %s after a concurrent write", sorted(snippet_ids))


def _queue_recheck(db: Session, label: Annotation, source: Annotation) -> None:
    """The source of this reviewed label is being deleted: offer to keep or remove it."""
    exists = (
        db.query(LabelShareCandidate.id)
        .filter(
            LabelShareCandidate.target_annotation_id == label.id,
            LabelShareCandidate.status == LabelShareStatus.PENDING,
        )
        .first()
    )
    if exists is not None:
        return
    target = db.get(Snippet, label.snippet_id)
    source_snippet = db.get(Snippet, source.snippet_id)
    if target is None or source_snippet is None:
        return
    meta = label.extra_metadata or {}
    db.add(LabelShareCandidate(
        dataset_id=meta.get("dataset_id") or _dataset_id_for_snippet(db, target),
        source_annotation_id=None,
        source_snippet_id=source_snippet.id,
        target_snippet_id=target.id,
        target_snippet_set_id=target.snippet_set_id,
        target_annotation_id=label.id,
        taxon_id=label.taxon_id,
        resolved_name_snapshot=label.resolved_name_snapshot or label.taxon_id,
        al_label=meta.get("al_label") or label.resolved_name_snapshot or label.taxon_id,
        reason=LabelShareReason.SOURCE_REMOVED,
        overlap_seconds=round(_overlap(source_snippet, target), 3),
    ))


def before_annotations_deleted(
    db: Session,
    annotations: Iterable[Annotation],
    deleted_by: Optional[int] = None,
) -> set[int]:
    """Call before deleting these annotations. Returns clips that lost a copy.

    - Automatic copies of a deleted label are deleted with it.
    - Reviewed copies stay (a person confirmed them) but lose the link, and get
      an optional recheck item to keep or remove them.
    - Pending rechecks of a deleted label close.
    - Deleting an automatic copy itself is remembered as a rejected suggestion,
      so the next share does not bring it back.
    - The DERIVED mirror row of a deleted automatic copy is removed.

    Pass the result to ``share_after_change(lost_copies_on=...)`` after the delete.
    """
    annotations = list(annotations)
    if not annotations:
        return set()
    deleted = {a.id: a for a in annotations if a.id is not None}
    ids = list(deleted)
    lost: set[int] = set()

    if ids:
        (
            db.query(LabelShareCandidate)
            .filter(
                LabelShareCandidate.target_annotation_id.in_(ids),
                LabelShareCandidate.status == LabelShareStatus.PENDING,
            )
            .delete(synchronize_session=False)
        )
        (
            db.query(LabelShareCandidate)
            .filter(LabelShareCandidate.target_annotation_id.in_(ids))
            .update({LabelShareCandidate.target_annotation_id: None}, synchronize_session=False)
        )
        for child in db.query(Annotation).filter(Annotation.derived_from_id.in_(ids)).all():
            if child.origin == ORIGIN_AUTO:
                _remove_derived_al_row(db, child)
                db.delete(child)
                lost.add(child.snippet_id)
                continue
            if child.origin == ORIGIN_REVIEWED and child.id not in deleted:
                _queue_recheck(db, child, deleted[child.derived_from_id])
                # Another overlapping label may still support it: re-sharing closes the recheck.
                lost.add(child.snippet_id)
            child.derived_from_id = None
        (
            db.query(LabelShareCandidate)
            .filter(LabelShareCandidate.source_annotation_id.in_(ids))
            .delete(synchronize_session=False)
        )
        (
            db.query(LabelShareCandidate)
            .filter(LabelShareCandidate.result_annotation_id.in_(ids))
            .update({LabelShareCandidate.result_annotation_id: None}, synchronize_session=False)
        )

    for ann in annotations:
        if ann.origin != ORIGIN_AUTO:
            continue
        _remove_derived_al_row(db, ann)
        source = db.get(Annotation, ann.derived_from_id) if ann.derived_from_id else None
        if source is None or source.id in ids:
            continue
        exists = (
            db.query(LabelShareCandidate.id)
            .filter(
                LabelShareCandidate.source_annotation_id == source.id,
                LabelShareCandidate.target_snippet_id == ann.snippet_id,
            )
            .first()
        )
        if exists is None:
            target = db.get(Snippet, ann.snippet_id)
            source_snippet = db.get(Snippet, source.snippet_id)
            if target is None or source_snippet is None:
                continue
            db.add(LabelShareCandidate(
                dataset_id=(ann.extra_metadata or {}).get("dataset_id")
                or _dataset_id_for_snippet(db, target),
                source_annotation_id=source.id,
                source_snippet_id=source.snippet_id,
                target_snippet_id=ann.snippet_id,
                target_snippet_set_id=target.snippet_set_id,
                taxon_id=ann.taxon_id,
                resolved_name_snapshot=ann.resolved_name_snapshot,
                al_label=(ann.extra_metadata or {}).get("al_label") or ann.resolved_name_snapshot,
                reason=LabelShareReason.PARTIAL_OVERLAP
                if not (_contains(target, source_snippet) or _contains(source_snippet, target))
                else LabelShareReason.TARGET_SHORTER,
                overlap_seconds=round(_overlap(source_snippet, target), 3),
                status=LabelShareStatus.REJECTED,
                reviewed_by=deleted_by,
                reviewed_at=datetime.now(timezone.utc),
            ))
    return lost


def confirm_existing_label(db: Session, annotation: Annotation, user_id: int, **fields) -> Annotation:
    """A person labelled a clip with a taxon it has as an automatic copy: make it theirs.

    The label keeps its derived_from_id as a record, but is now manual, so it is
    shared on and stays when its source is deleted.
    """
    _remove_derived_al_row(db, annotation)
    annotation.origin = ORIGIN_MANUAL
    annotation.user_id = user_id
    for key, value in fields.items():
        if value is not None:
            setattr(annotation, key, value)
    return annotation


def share_dataset(
    db: Session,
    dataset_id: int,
    commit_every: int = 200,
    target_snippet_set_id: Optional[int] = None,
    source_snippet_set_id: Optional[int] = None,
) -> dict:
    """Share every manual label of a dataset. Safe to run again: it only fills gaps.

    Used to backfill existing labels (every set onto every other set), and after
    a snippet set is embedded with ``target_snippet_set_id``: then only labels
    from the other sets are shared, and only onto that set's clips. Labels on
    the other sets were already shared among themselves as they were made.

    With ``source_snippet_set_id`` too, it is a one-way transfer: only labels on
    that set's clips, only onto the target set's clips.
    """
    totals = {"source_clips": 0, "auto_shared": 0, "queued": 0}
    if not settings.LABEL_SHARING_ENABLED:
        return totals
    ctx = _context(db, dataset_id)
    sources = (
        db.query(Annotation.snippet_id)
        .join(Snippet, Snippet.id == Annotation.snippet_id)
        .join(Recording, Recording.id == Snippet.recording_id)
        .filter(Recording.dataset_id == dataset_id, Annotation.origin == ORIGIN_MANUAL)
    )
    if target_snippet_set_id is not None:
        sources = sources.filter(Snippet.snippet_set_id != target_snippet_set_id)
    if source_snippet_set_id is not None:
        sources = sources.filter(Snippet.snippet_set_id == source_snippet_set_id)
    source_ids = [row[0] for row in sources.distinct().order_by(Annotation.snippet_id).all()]
    for start in range(0, len(source_ids), commit_every):
        chunk = source_ids[start:start + commit_every]
        for snippet in db.query(Snippet).filter(Snippet.id.in_(chunk)).all():
            counts = _share_snippet(db, ctx, snippet, target_snippet_set_id)
            totals["source_clips"] += 1
            totals["auto_shared"] += counts["auto_shared"]
            totals["queued"] += counts["queued"]
        db.commit()
    logger.info(
        "Label sharing for dataset %s (from set %s onto set %s): %s",
        dataset_id,
        source_snippet_set_id if source_snippet_set_id is not None else "all",
        target_snippet_set_id if target_snippet_set_id is not None else "all",
        totals,
    )
    return totals


# ── Review ──────────────────────────────────────────────────────────────

def _mark_reviewed(candidate: LabelShareCandidate, status: str, user_id: int, result_id: Optional[int]) -> None:
    candidate.status = status
    candidate.reviewed_by = user_id
    candidate.reviewed_at = datetime.now(timezone.utc)
    candidate.result_annotation_id = result_id


def _label_target(
    db: Session,
    candidate: LabelShareCandidate,
    user_id: int,
    taxon_id: str,
    name: str,
    al_label: str,
    embedding_model_id: Optional[int],
) -> Annotation:
    # Same taxon id, or the same name under another id (custom taxa are
    # created per code, so one species can have several): either way the clip
    # already has this label and must not get a second row.
    wanted = {al_label.strip().lower(), (name or "").strip().lower()} - {""}
    on_clip = db.query(Annotation).filter(Annotation.snippet_id == candidate.target_snippet_id).all()
    # Prefer a row a person made over an automatic copy.
    on_clip.sort(key=lambda a: a.origin == ORIGIN_AUTO)
    existing = next((a for a in on_clip if a.taxon_id == taxon_id), None) or next(
        (a for a in on_clip if _label_text(a) in wanted), None
    )
    if existing is not None:
        if existing.origin == ORIGIN_AUTO:
            _remove_derived_al_row(db, existing)
            existing.origin = ORIGIN_REVIEWED
            existing.user_id = user_id
            add_al_row(db, candidate.dataset_id, existing.snippet_id, al_label, ALAnnotationSource.USER, user_id)
        return existing
    annotation = Annotation(
        snippet_id=candidate.target_snippet_id,
        user_id=user_id,
        embedding_model_id=embedding_model_id,
        taxon_id=taxon_id,
        resolved_name_snapshot=name,
        origin=ORIGIN_REVIEWED,
        derived_from_id=candidate.source_annotation_id,
        extra_metadata={
            "source": "label_sharing_review",
            "dataset_id": candidate.dataset_id,
            "al_label": al_label,
            "candidate_id": candidate.id,
        },
    )
    db.add(annotation)
    add_al_row(db, candidate.dataset_id, candidate.target_snippet_id, al_label, ALAnnotationSource.USER, user_id)
    db.flush()
    return annotation


def accept(db: Session, candidate: LabelShareCandidate, user_id: int,
           embedding_model_id: Optional[int] = None) -> Annotation:
    annotation = _label_target(
        db, candidate, user_id, candidate.taxon_id, candidate.resolved_name_snapshot,
        candidate.al_label, embedding_model_id,
    )
    _mark_reviewed(candidate, LabelShareStatus.ACCEPTED, user_id, annotation.id)
    _close_pending_for(db, candidate.target_snippet_id, annotation, candidate.al_label, keep=candidate)
    return annotation


def relabel(db: Session, candidate: LabelShareCandidate, user_id: int, taxon_id: str, name: str,
            embedding_model_id: Optional[int] = None) -> Annotation:
    if taxon_id == candidate.taxon_id:
        return accept(db, candidate, user_id, embedding_model_id)
    ctx = _context(db, candidate.dataset_id)
    al_label = ctx.quick_label_text.get(taxon_id) or name
    annotation = _label_target(db, candidate, user_id, taxon_id, name, al_label, embedding_model_id)
    _mark_reviewed(candidate, LabelShareStatus.RELABELLED, user_id, annotation.id)
    _close_pending_for(db, candidate.target_snippet_id, annotation, al_label, keep=candidate)
    return annotation


def reject(db: Session, candidate: LabelShareCandidate, user_id: int) -> None:
    _mark_reviewed(candidate, LabelShareStatus.REJECTED, user_id, None)


def keep_rechecked_label(db: Session, candidate: LabelShareCandidate, user_id: int) -> Optional[int]:
    """A recheck: the label still holds for its clip. Nothing on the clip changes."""
    _mark_reviewed(candidate, LabelShareStatus.ACCEPTED, user_id, candidate.target_annotation_id)
    return candidate.target_annotation_id


def _remove_user_al_row(db: Session, annotation: Annotation) -> None:
    labels = {(annotation.extra_metadata or {}).get("al_label"), annotation.resolved_name_snapshot} - {None, ""}
    (
        db.query(ALSnippetAnnotation)
        .filter(
            ALSnippetAnnotation.snippet_id == annotation.snippet_id,
            ALSnippetAnnotation.label.in_(labels),
            ALSnippetAnnotation.source == ALAnnotationSource.USER,
            ALSnippetAnnotation.user_id == annotation.user_id,
            ALSnippetAnnotation.model_checkpoint_id.is_(None),
        )
        .delete(synchronize_session=False)
    )


def remove_rechecked_label(db: Session, candidate: LabelShareCandidate, user_id: int) -> None:
    """A recheck: the label does not hold without its source. Delete it from the clip."""
    annotation = db.get(Annotation, candidate.target_annotation_id) if candidate.target_annotation_id else None
    _mark_reviewed(candidate, LabelShareStatus.REJECTED, user_id, None)
    # Sessions do not autoflush: the record must read as reviewed before the
    # delete below closes pending rechecks of this label.
    db.flush()
    if annotation is None:
        return
    lost = before_annotations_deleted(db, [annotation], deleted_by=user_id)
    _remove_user_al_row(db, annotation)
    db.delete(annotation)
    db.flush()
    share_after_change(db, [], lost_copies_on=lost)
