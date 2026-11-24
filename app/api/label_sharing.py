"""
Label sharing between snippet sets: review queue and backfill (issue #93, part 2).

Mounted under /api/label-sharing. Rules in app/services/label_sharing.py.
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import case, func
from sqlalchemy.orm import Session

from app.api.deps import get_current_active_user, get_db
from app.core import taxonomy
from app.models.annotation import Annotation
from app.models.dataset import Dataset
from app.models.embedding import SnippetSet
from app.models.label_share import LabelShareCandidate, LabelShareReason, LabelShareStatus
from app.models.recording import Recording
from app.models.snippet import Snippet
from app.models.user import User
from app.schemas.label_share import (
    AcceptCandidateRequest,
    ClipInfo,
    LabelShareCandidateList,
    LabelShareCandidateOut,
    LabelShareSummary,
    RelabelCandidateRequest,
    ReviewResult,
    ShareDatasetRequest,
    ShareDatasetResponse,
    SnippetBreakdown,
    SnippetSetBreakdown,
)
from app.services import label_sharing
from app.services.dataset_service import DatasetService
from app.services.embedding_scope import models_for_set, snippet_set_for_model

router = APIRouter()


def _require_access(db: Session, user: User, dataset_id: int) -> Dataset:
    service = DatasetService(db)
    dataset = service.get_dataset(dataset_id)
    if dataset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
    if not service.user_can_access_dataset(user, dataset_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized for this dataset")
    return dataset


def _get_candidate(db: Session, user: User, candidate_id: int) -> LabelShareCandidate:
    candidate = db.get(LabelShareCandidate, candidate_id)
    if candidate is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Suggestion not found")
    _require_access(db, user, candidate.dataset_id)
    return candidate


def _require_pending(candidate: LabelShareCandidate) -> None:
    if candidate.status != LabelShareStatus.PENDING:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"This suggestion was already reviewed ({candidate.status}).",
        )


def _check_model(db: Session, candidate: LabelShareCandidate, embedding_model_id: Optional[int]) -> None:
    if embedding_model_id is not None and embedding_model_id not in models_for_set(db, candidate.target_snippet_set_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Embedding model {embedding_model_id} has not embedded this snippet's set.",
        )


def _pending_by_set(db: Session, dataset_id: int) -> dict[int, int]:
    return dict(
        db.query(LabelShareCandidate.target_snippet_set_id, func.count(LabelShareCandidate.id))
        .filter(
            LabelShareCandidate.dataset_id == dataset_id,
            LabelShareCandidate.status == LabelShareStatus.PENDING,
        )
        .group_by(LabelShareCandidate.target_snippet_set_id)
        .all()
    )


def _serialize(db: Session, candidates: list[LabelShareCandidate]) -> list[LabelShareCandidateOut]:
    if not candidates:
        return []
    snippet_ids = {c.source_snippet_id for c in candidates} | {c.target_snippet_id for c in candidates}
    clips = {
        snippet.id: (snippet, file_name)
        for snippet, file_name in (
            db.query(Snippet, Recording.file_name)
            .join(Recording, Recording.id == Snippet.recording_id)
            .filter(Snippet.id.in_(snippet_ids))
            .all()
        )
    }
    target_labels: dict[int, list[str]] = {}
    for snippet_id, name in (
        db.query(Annotation.snippet_id, Annotation.resolved_name_snapshot)
        .filter(Annotation.snippet_id.in_({c.target_snippet_id for c in candidates}))
        .order_by(Annotation.resolved_name_snapshot)
        .all()
    ):
        # One chip per name: older data can hold the same name twice on a
        # clip under two custom taxon ids.
        names = target_labels.setdefault(snippet_id, [])
        if name and name.strip().lower() not in {n.strip().lower() for n in names}:
            names.append(name)
    source_users = dict(
        db.query(Annotation.id, User.username)
        .join(User, User.id == Annotation.user_id)
        .filter(Annotation.id.in_({c.source_annotation_id for c in candidates} - {None}))
        .all()
    )

    def clip(snippet_id: int) -> ClipInfo:
        snippet, file_name = clips[snippet_id]
        return ClipInfo(
            snippet_id=snippet.id,
            snippet_set_id=snippet.snippet_set_id,
            recording_id=snippet.recording_id,
            recording_file_name=file_name,
            start_time=snippet.start_time,
            end_time=snippet.end_time,
        )

    return [
        LabelShareCandidateOut(
            id=c.id,
            dataset_id=c.dataset_id,
            status=c.status,
            reason=c.reason,
            taxon_id=c.taxon_id,
            resolved_name_snapshot=c.resolved_name_snapshot,
            label=c.al_label,
            overlap_seconds=c.overlap_seconds,
            source_annotation_id=c.source_annotation_id,
            target_annotation_id=c.target_annotation_id,
            source_labeled_by=source_users.get(c.source_annotation_id),
            source=clip(c.source_snippet_id),
            target=clip(c.target_snippet_id),
            target_labels=target_labels.get(c.target_snippet_id, []),
            reviewed_by=c.reviewed_by,
            reviewed_at=c.reviewed_at,
            result_annotation_id=c.result_annotation_id,
            created_at=c.created_at,
        )
        for c in candidates
    ]


@router.get("/datasets/{dataset_id}/summary", response_model=LabelShareSummary)
def get_summary(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """How many labels were shared automatically, and the state of the review queue."""
    _require_access(db, current_user, dataset_id)
    by_status = dict(
        db.query(LabelShareCandidate.status, func.count(LabelShareCandidate.id))
        .filter(LabelShareCandidate.dataset_id == dataset_id)
        .group_by(LabelShareCandidate.status)
        .all()
    )
    pending_by_set = _pending_by_set(db, dataset_id)
    # Pending per snippet set and reason, for the review's reason filter.
    pending_by_set_reason: dict[int, dict[str, int]] = {}
    for set_id, reason, count in (
        db.query(
            LabelShareCandidate.target_snippet_set_id,
            LabelShareCandidate.reason,
            func.count(LabelShareCandidate.id),
        )
        .filter(
            LabelShareCandidate.dataset_id == dataset_id,
            LabelShareCandidate.status == LabelShareStatus.PENDING,
        )
        .group_by(LabelShareCandidate.target_snippet_set_id, LabelShareCandidate.reason)
        .all()
    ):
        pending_by_set_reason.setdefault(set_id, {})[str(reason)] = count
    # Labelled clips per set, for choosing where a transfer comes from.
    labelled_by_set = dict(
        db.query(Snippet.snippet_set_id, func.count(func.distinct(Annotation.snippet_id)))
        .join(Snippet, Snippet.id == Annotation.snippet_id)
        .join(Recording, Recording.id == Snippet.recording_id)
        .filter(Recording.dataset_id == dataset_id, Annotation.origin == label_sharing.ORIGIN_MANUAL)
        .group_by(Snippet.snippet_set_id)
        .all()
    )
    auto_shared = (
        db.query(func.count(Annotation.id))
        .join(Snippet, Snippet.id == Annotation.snippet_id)
        .join(Recording, Recording.id == Snippet.recording_id)
        .filter(Recording.dataset_id == dataset_id, Annotation.origin == label_sharing.ORIGIN_AUTO)
        .scalar()
    )
    return LabelShareSummary(
        dataset_id=dataset_id,
        auto_shared=auto_shared or 0,
        pending=by_status.get(LabelShareStatus.PENDING, 0),
        accepted=by_status.get(LabelShareStatus.ACCEPTED, 0),
        rejected=by_status.get(LabelShareStatus.REJECTED, 0),
        relabelled=by_status.get(LabelShareStatus.RELABELLED, 0),
        pending_by_snippet_set=pending_by_set,
        pending_by_snippet_set_reason=pending_by_set_reason,
        labelled_by_snippet_set=labelled_by_set,
    )


@router.get("/datasets/{dataset_id}/snippet-breakdown", response_model=SnippetBreakdown)
def get_snippet_breakdown(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Snippets of each snippet set by label status, for the dataset card.

    A snippet with both a confirmed label and an automatic copy counts as labelled,
    so labelled + auto-copied is the number of snippets with any label.
    """
    _require_access(db, current_user, dataset_id)
    labelled_snippets = (
        db.query(
            Annotation.snippet_id.label("snippet_id"),
            func.max(case((Annotation.origin != label_sharing.ORIGIN_AUTO, 1), else_=0)).label("confirmed"),
        )
        .join(Snippet, Snippet.id == Annotation.snippet_id)
        .join(Recording, Recording.id == Snippet.recording_id)
        .filter(Recording.dataset_id == dataset_id)
        .group_by(Annotation.snippet_id)
        .subquery()
    )
    counts = {
        row.snippet_set_id: row
        for row in (
            db.query(
                Snippet.snippet_set_id.label("snippet_set_id"),
                func.count(Snippet.id).label("total"),
                func.count(func.distinct(Snippet.recording_id)).label("recordings"),
                func.sum(case((labelled_snippets.c.confirmed == 1, 1), else_=0)).label("labelled"),
                func.sum(case((labelled_snippets.c.confirmed == 0, 1), else_=0)).label("auto_copied"),
            )
            .join(Recording, Recording.id == Snippet.recording_id)
            .outerjoin(labelled_snippets, labelled_snippets.c.snippet_id == Snippet.id)
            .filter(Recording.dataset_id == dataset_id)
            .group_by(Snippet.snippet_set_id)
            .all()
        )
    }
    pending = _pending_by_set(db, dataset_id)
    snippet_sets = (
        db.query(SnippetSet).filter(SnippetSet.dataset_id == dataset_id).order_by(SnippetSet.id).all()
    )

    def breakdown(snippet_set: SnippetSet) -> SnippetSetBreakdown:
        row = counts.get(snippet_set.id)
        return SnippetSetBreakdown(
            snippet_set_id=snippet_set.id,
            window_size=snippet_set.window_size,
            status=str(getattr(snippet_set.status, "value", snippet_set.status)),
            total_snippets=row.total if row else 0,
            recording_count=row.recordings if row else 0,
            labelled=int(row.labelled or 0) if row else 0,
            auto_copied=int(row.auto_copied or 0) if row else 0,
            to_review=pending.get(snippet_set.id, 0),
        )

    return SnippetBreakdown(dataset_id=dataset_id, snippet_sets=[breakdown(s) for s in snippet_sets])


@router.get("/datasets/{dataset_id}/candidates", response_model=LabelShareCandidateList)
def list_candidates(
    dataset_id: int,
    status_filter: str = Query(LabelShareStatus.PENDING, alias="status"),
    snippet_set_id: Optional[int] = Query(None, description="Only suggestions for clips of this set"),
    embedding_model_id: Optional[int] = Query(
        None, description="Only suggestions for clips of this model's snippet set"
    ),
    reason: Optional[str] = Query(None),
    taxon_id: Optional[str] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """The review queue, in recording order so a reviewer can listen along a recording."""
    dataset = _require_access(db, current_user, dataset_id)
    if status_filter not in LabelShareStatus.ALL:
        raise HTTPException(status_code=400, detail=f"status must be one of {', '.join(LabelShareStatus.ALL)}")
    if embedding_model_id is not None:
        model_set = snippet_set_for_model(db, dataset, embedding_model_id)
        if model_set is None:
            return LabelShareCandidateList(items=[], total=0, skip=skip, limit=limit)
        if snippet_set_id is not None and snippet_set_id != model_set:
            return LabelShareCandidateList(items=[], total=0, skip=skip, limit=limit)
        snippet_set_id = model_set

    query = db.query(LabelShareCandidate).filter(
        LabelShareCandidate.dataset_id == dataset_id,
        LabelShareCandidate.status == status_filter,
    )
    if snippet_set_id is not None:
        query = query.filter(LabelShareCandidate.target_snippet_set_id == snippet_set_id)
    if reason:
        query = query.filter(LabelShareCandidate.reason == reason)
    if taxon_id:
        query = query.filter(LabelShareCandidate.taxon_id == taxon_id)

    total = query.count()
    rows = (
        query.join(Snippet, Snippet.id == LabelShareCandidate.target_snippet_id)
        .order_by(Snippet.recording_id, Snippet.start_time, LabelShareCandidate.id)
        .offset(skip)
        .limit(limit)
        .all()
    )
    return LabelShareCandidateList(items=_serialize(db, rows), total=total, skip=skip, limit=limit)


@router.post("/candidates/{candidate_id}/accept", response_model=ReviewResult)
def accept_candidate(
    candidate_id: int,
    body: Optional[AcceptCandidateRequest] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """The label holds for the target clip: add it there, under the reviewer."""
    candidate = _get_candidate(db, current_user, candidate_id)
    _require_pending(candidate)
    if candidate.reason == LabelShareReason.SOURCE_REMOVED:
        # A recheck: keep the label the clip already has.
        annotation_id = label_sharing.keep_rechecked_label(db, candidate, current_user.id)
        db.commit()
        db.refresh(candidate)
        return ReviewResult(candidate=_serialize(db, [candidate])[0], annotation_id=annotation_id)
    embedding_model_id = body.embedding_model_id if body else None
    _check_model(db, candidate, embedding_model_id)
    annotation = label_sharing.accept(db, candidate, current_user.id, embedding_model_id)
    db.commit()
    db.refresh(candidate)
    return ReviewResult(candidate=_serialize(db, [candidate])[0], annotation_id=annotation.id)


@router.post("/candidates/{candidate_id}/reject", response_model=ReviewResult)
def reject_candidate(
    candidate_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """The label does not hold for the target clip. It is not suggested again.

    For a recheck (its source label was deleted), this removes the label from the clip.
    """
    candidate = _get_candidate(db, current_user, candidate_id)
    _require_pending(candidate)
    if candidate.reason == LabelShareReason.SOURCE_REMOVED:
        label_sharing.remove_rechecked_label(db, candidate, current_user.id)
    else:
        label_sharing.reject(db, candidate, current_user.id)
    db.commit()
    db.refresh(candidate)
    return ReviewResult(candidate=_serialize(db, [candidate])[0])


@router.post("/candidates/{candidate_id}/relabel", response_model=ReviewResult)
def relabel_candidate(
    candidate_id: int,
    body: RelabelCandidateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """The target clip has a different label: add that one instead."""
    candidate = _get_candidate(db, current_user, candidate_id)
    _require_pending(candidate)
    if candidate.reason == LabelShareReason.SOURCE_REMOVED:
        raise HTTPException(status_code=400, detail="A recheck can only keep or remove the label.")
    _check_model(db, candidate, body.embedding_model_id)
    resolved = taxonomy.resolve_taxon_id(body.taxon_id, db_session=db)
    if not resolved:
        raise HTTPException(status_code=400, detail=f"Invalid or unresolvable taxon_id: {body.taxon_id}")
    name = (
        body.display_name
        or resolved.get("canonical_name")
        or resolved.get("scientific_name")
        or body.taxon_id
    )
    annotation = label_sharing.relabel(
        db, candidate, current_user.id, body.taxon_id, name, body.embedding_model_id
    )
    db.commit()
    db.refresh(candidate)
    return ReviewResult(candidate=_serialize(db, [candidate])[0], annotation_id=annotation.id)


@router.post(
    "/datasets/{dataset_id}/share",
    response_model=ShareDatasetResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def share_dataset_labels(
    dataset_id: int,
    body: Optional[ShareDatasetRequest] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Share the dataset's existing labels now, in the background.

    With ``from_embedding_model_id`` and ``to_embedding_model_id``: a one-way
    transfer of the labels on the first model's clips onto the second's.
    Without: every model's labels onto every other model's clips.
    """
    dataset = _require_access(db, current_user, dataset_id)
    if not DatasetService(db).user_can_manage_dataset(current_user, dataset):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins and team owners can share a whole dataset's labels",
        )
    source_set = target_set = None
    if body and (body.from_embedding_model_id is not None or body.to_embedding_model_id is not None):
        if body.from_embedding_model_id is None or body.to_embedding_model_id is None:
            raise HTTPException(status_code=400, detail="Give both the model to transfer from and the one to transfer to.")
        source_set = snippet_set_for_model(db, dataset, body.from_embedding_model_id)
        target_set = snippet_set_for_model(db, dataset, body.to_embedding_model_id)
        if source_set is None or target_set is None:
            raise HTTPException(status_code=400, detail="Both models must have finished embedding this dataset.")
        if source_set == target_set:
            raise HTTPException(
                status_code=400,
                detail="These models use the same snippets, so they already share every label.",
            )
    from app.tasks.label_sharing_tasks import share_labels_for_dataset

    result = share_labels_for_dataset.delay(
        dataset_id, target_snippet_set_id=target_set, source_snippet_set_id=source_set
    )
    return ShareDatasetResponse(dataset_id=dataset_id, task_id=getattr(result, "id", None))
