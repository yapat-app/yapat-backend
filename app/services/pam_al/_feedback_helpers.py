"""
Feedback helpers: counting, syncing events to annotations, label normalization.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models.pam_active_learning import (
    ALFeedbackAction,
    ALFeedbackEvent,
    ALPrediction,
    ALRetrainJob,
    ALRetrainStatus,
)

from app.services.pam_al._annotation_helpers import replace_user_labels_for_snippet

from active_learning.config import NO_EVENT_LABEL

# Trigger values that represent actual model-retrain jobs.
# "inference" jobs use the same ALRetrainJob table but are NOT retrains and
# must never block or influence retrain-gating logic.
_RETRAIN_TRIGGERS = frozenset({"manual", "auto_feedback"})


def feedback_count_since_retrain(db:Session,
        checkpoint_id: int | None,
        dataset_id: int | None = None,
) -> int:
    # Only consider real retrain jobs (not inference jobs) when determining the
    # "last completed retrain" cutoff.
    query_last_retrain = db.query(ALRetrainJob.completed_at).filter(
        ALRetrainJob.model_checkpoint_id == checkpoint_id,
        ALRetrainJob.status == ALRetrainStatus.COMPLETED,
        ALRetrainJob.trigger.in_(_RETRAIN_TRIGGERS),
    )

    if dataset_id is not None:
        query_last_retrain = query_last_retrain.filter(ALRetrainJob.dataset_id == dataset_id)

    last_retrain = query_last_retrain.order_by(ALRetrainJob.completed_at.desc()).first()

    cutoff = (
        last_retrain[0]
        if last_retrain and last_retrain[0] is not None
        else datetime.min.replace(tzinfo=timezone.utc)
    )

    query_count = db.query(
        func.count(func.distinct(ALFeedbackEvent.snippet_id))
    ).filter(
        ALFeedbackEvent.model_checkpoint_id == checkpoint_id,
        ALFeedbackEvent.created_at > cutoff,
    )

    if dataset_id is not None:
        query_count = query_count.filter(ALFeedbackEvent.dataset_id == dataset_id)

    count = query_count.scalar()
    return int(count or 0)


def get_last_completed_retrain_cutoff(db: Session, checkpoint_id: int) -> datetime:
    last_retrain = (
        db.query(ALRetrainJob.completed_at)
        .filter(
            ALRetrainJob.model_checkpoint_id == checkpoint_id,
            ALRetrainJob.status == ALRetrainStatus.COMPLETED,
            ALRetrainJob.trigger.in_(_RETRAIN_TRIGGERS),
        )
        .order_by(ALRetrainJob.completed_at.desc())
        .first()
    )
    if last_retrain and last_retrain[0] is not None:
        return last_retrain[0]
    return datetime.min.replace(tzinfo=timezone.utc)


def get_feedback_events_since_last_retrain(
    db: Session,
    checkpoint_id: int,
) -> list[ALFeedbackEvent]:
    cutoff = get_last_completed_retrain_cutoff(db, checkpoint_id)
    return (
        db.query(ALFeedbackEvent)
        .filter(
            ALFeedbackEvent.model_checkpoint_id == checkpoint_id,
            ALFeedbackEvent.created_at > cutoff,
        )
        .order_by(ALFeedbackEvent.created_at.asc())
        .all()
    )


def sync_feedback_events_to_annotations(db: Session, checkpoint_id: int) -> int:
    """
    Sync recent ACCEPT / MODIFY feedback since the last retrain into
    ALSnippetAnnotation.  Returns the number of events processed.
    """
    events = get_feedback_events_since_last_retrain(db, checkpoint_id)
    if not events:
        return 0

    # Events are ordered by created_at asc, so iterating in order means the
    # last assignment per (snippet_id, user_id) wins — correct for re-feedback.
    latest_by_snippet_user: dict[tuple[int, int | None], ALFeedbackEvent] = {}
    for event in events:
        if event.action not in {ALFeedbackAction.ACCEPT, ALFeedbackAction.MODIFY}:
            continue
        if not (event.final_labels or []):
            continue
        latest_by_snippet_user[(event.snippet_id, event.user_id)] = event

    for event in latest_by_snippet_user.values():
        replace_user_labels_for_snippet(
            db=db,
            dataset_id=event.dataset_id,
            snippet_id=event.snippet_id,
            labels=event.final_labels or [],
            model_checkpoint_id=event.model_checkpoint_id,
            user_id=event.user_id,
        )
        # Flush per snippet to avoid a single massive INSERT batch that can
        # exceed SQLAlchemy's insertmanyvalues compile limit.
        db.flush()

    return len(events)


def has_active_retrain_job(db: Session, checkpoint_id: int) -> bool:
    """Return True if a retrain job (not an inference job) is PENDING/RUNNING."""
    return (
        db.query(ALRetrainJob)
        .filter(
            ALRetrainJob.model_checkpoint_id == checkpoint_id,
            ALRetrainJob.status.in_([ALRetrainStatus.PENDING, ALRetrainStatus.RUNNING]),
            ALRetrainJob.trigger.in_(_RETRAIN_TRIGGERS),
        )
        .first()
        is not None
    )


def has_pending_child_retrain(db: Session, parent_checkpoint_id: int) -> bool:
    """Return True if any child checkpoint has a PENDING/RUNNING retrain job."""
    from app.models.pam_active_learning import ALModelCheckpoint

    return (
        db.query(ALRetrainJob)
        .join(ALModelCheckpoint, ALRetrainJob.model_checkpoint_id == ALModelCheckpoint.id)
        .filter(
            ALModelCheckpoint.parent_checkpoint_id == parent_checkpoint_id,
            ALRetrainJob.status.in_([ALRetrainStatus.PENDING, ALRetrainStatus.RUNNING]),
            ALRetrainJob.trigger.in_(_RETRAIN_TRIGGERS),
        )
        .first()
        is not None
    )


def has_failed_child_retrain(db: Session, parent_checkpoint_id: int) -> bool:
    """Return True if the most recent child *retrain* job (not inference) ended FAILED."""
    from app.models.pam_active_learning import ALModelCheckpoint

    latest_child_job = (
        db.query(ALRetrainJob)
        .join(ALModelCheckpoint, ALRetrainJob.model_checkpoint_id == ALModelCheckpoint.id)
        .filter(
            ALModelCheckpoint.parent_checkpoint_id == parent_checkpoint_id,
            ALRetrainJob.trigger.in_(_RETRAIN_TRIGGERS),
        )
        .order_by(ALRetrainJob.created_at.desc())
        .first()
    )
    return latest_child_job is not None and latest_child_job.status == ALRetrainStatus.FAILED


def collect_predicted_labels_for_snippet(
    predictions: list[ALPrediction],
) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()

    for prediction in predictions:
        if prediction.predicted_labels:
            for label in prediction.predicted_labels:
                value = str(label).strip()
                if value and value not in seen:
                    seen.add(value)
                    labels.append(value)

    return labels


def normalize_feedback_labels(labels: list[str] | None) -> list[str]:
    """Clean incoming labels: remove None, empty, Swagger placeholder, duplicates."""
    if not labels:
        return []

    cleaned: list[str] = []
    seen: set[str] = set()

    for label in labels:
        if label is None:
            continue
        value = str(label).strip()
        if not value or value.lower() == "string":
            continue
        if value not in seen:
            seen.add(value)
            cleaned.append(value)

    return cleaned


def resolve_feedback_labels(
    action: str,
    predicted_labels: list[str],
    labels: list[str] | None,
) -> list[str]:
    incoming_labels = normalize_feedback_labels(labels)

    if action == "ACCEPT":
        return predicted_labels
    if action == "MODIFY":
        return incoming_labels
    return []




def is_no_event_feedback(labels: list[str] | None) -> bool:

    if not labels or labels == [NO_EVENT_LABEL]:
        return True
