"""
Study interaction event endpoints

Batch ingest for the per-participant interaction event stream captured during
the YAPAT user study. Fire-and-forget from the client; this endpoint stamps the
authenticated user server-side and bulk-inserts.

Admin read endpoints (GET /admin/...) let the study administrator browse the
collected events by user → session → event, and export per-session CSVs.
"""

import csv
import io
import json
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import func, distinct
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_current_admin_user
from app.core.security import verify_token
from app.models.study_event import StudyEvent as StudyEventModel
from app.models.user import User
from app.schemas.study_event import (
    StudyEventBatchCreate,
    StudyEventBatchResponse,
    StudyLogUserSummary,
    StudyLogSessionSummary,
    StudyLogEventRow,
    StudyLogEventsPage,
)

router = APIRouter()

# auto_error=False so a missing header does not 401 — we fall back to the
# in-body token used by the navigator.sendBeacon unload path.
_optional_bearer = HTTPBearer(auto_error=False)


def _resolve_user(db: Session, token: Optional[str]) -> User:
    """Validate a JWT (from header or body) and return the active user."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if not token:
        raise credentials_exception

    payload = verify_token(token)
    if payload is None:
        raise credentials_exception

    sub = payload.get("sub")
    if sub is None:
        raise credentials_exception
    try:
        user_id = int(sub)
    except (ValueError, TypeError):
        raise credentials_exception

    user = db.query(User).filter(User.id == user_id).first()
    if user is None or not user.is_active:
        raise credentials_exception
    return user


@router.post(
    "/batch",
    response_model=StudyEventBatchResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def ingest_study_events(
    batch_in: StudyEventBatchCreate,
    db: Session = Depends(get_db),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_optional_bearer),
):
    """
    Bulk-ingest a batch of study interaction events.

    Auth: bearer header (normal path) OR a `token` field in the request body
    (navigator.sendBeacon unload path, which cannot set headers). The user_id is
    always taken from the validated token server-side — the client envelope's
    userId is never trusted.
    """
    token = credentials.credentials if credentials else batch_in.token
    user = _resolve_user(db, token)

    if not batch_in.events:
        return StudyEventBatchResponse(accepted=0, session_id=None)

    rows = [
        StudyEventModel(
            session_id=ev.session_id,
            user_id=user.id,
            phase_id=ev.phase_id,
            dataset_id=ev.dataset_id,
            snippet_set_id=ev.snippet_set_id,
            snippet_id=ev.snippet_id,
            event_type=ev.event_type,
            payload=ev.payload,
            duration_ms=ev.duration_ms,
            client_ts=ev.timestamp,
        )
        for ev in batch_in.events
    ]
    db.add_all(rows)
    db.commit()

    return StudyEventBatchResponse(
        accepted=len(rows),
        session_id=batch_in.events[0].session_id,
    )


# ---------------------------------------------------------------------------
# Admin read endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/admin/users",
    response_model=List[StudyLogUserSummary],
)
def list_study_users(
    db: Session = Depends(get_db),
    _admin: User = Depends(get_current_admin_user),
):
    """Return all users who have at least one study event, with aggregate counts."""
    rows = (
        db.query(
            StudyEventModel.user_id,
            User.username,
            func.count(distinct(StudyEventModel.session_id)).label("session_count"),
            func.count(StudyEventModel.id).label("event_count"),
            func.max(StudyEventModel.client_ts).label("last_seen"),
        )
        .join(User, User.id == StudyEventModel.user_id)
        .group_by(StudyEventModel.user_id, User.username)
        .order_by(func.max(StudyEventModel.client_ts).desc())
        .all()
    )
    return [
        StudyLogUserSummary(
            user_id=r.user_id,
            username=r.username,
            session_count=r.session_count,
            event_count=r.event_count,
            last_seen=r.last_seen,
        )
        for r in rows
    ]


@router.get(
    "/admin/users/{user_id}/sessions",
    response_model=List[StudyLogSessionSummary],
)
def list_user_sessions(
    user_id: int,
    db: Session = Depends(get_db),
    _admin: User = Depends(get_current_admin_user),
):
    """Return all sessions for a given user, with aggregate metadata."""
    # Verify user exists
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    rows = (
        db.query(
            StudyEventModel.session_id,
            func.count(StudyEventModel.id).label("event_count"),
            func.min(StudyEventModel.client_ts).label("first_event_at"),
            func.max(StudyEventModel.client_ts).label("last_event_at"),
        )
        .filter(StudyEventModel.user_id == user_id)
        .group_by(StudyEventModel.session_id)
        .order_by(func.min(StudyEventModel.client_ts).desc())
        .all()
    )

    summaries = []
    for r in rows:
        # Collect distinct phase_ids for this session
        phase_rows = (
            db.query(distinct(StudyEventModel.phase_id))
            .filter(
                StudyEventModel.user_id == user_id,
                StudyEventModel.session_id == r.session_id,
                StudyEventModel.phase_id.isnot(None),
            )
            .all()
        )
        phase_ids = [p[0] for p in phase_rows]

        duration_minutes: Optional[float] = None
        if r.first_event_at and r.last_event_at:
            delta = r.last_event_at - r.first_event_at
            duration_minutes = round(delta.total_seconds() / 60, 1)

        summaries.append(
            StudyLogSessionSummary(
                session_id=r.session_id,
                event_count=r.event_count,
                first_event_at=r.first_event_at,
                last_event_at=r.last_event_at,
                phase_ids=phase_ids,
                duration_minutes=duration_minutes,
            )
        )
    return summaries


@router.get(
    "/admin/sessions/{session_id}/events",
    response_model=StudyLogEventsPage,
)
def get_session_events(
    session_id: str,
    event_type: Optional[str] = Query(None),
    phase_id: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    _admin: User = Depends(get_current_admin_user),
):
    """Return paginated events for a session, optionally filtered by type/phase."""
    q = db.query(StudyEventModel).filter(StudyEventModel.session_id == session_id)
    if event_type:
        q = q.filter(StudyEventModel.event_type == event_type)
    if phase_id:
        q = q.filter(StudyEventModel.phase_id == phase_id)

    total = q.count()
    # Secondary sort by id ensures insertion order when timestamps tie
    # (e.g. session_start and panel_enter logged within the same millisecond).
    events = q.order_by(StudyEventModel.client_ts, StudyEventModel.id).offset(offset).limit(limit).all()

    return StudyLogEventsPage(
        session_id=session_id,
        total=total,
        events=[
            StudyLogEventRow(
                id=ev.id,
                client_ts=ev.client_ts,
                event_type=ev.event_type,
                phase_id=ev.phase_id,
                dataset_id=ev.dataset_id,
                snippet_id=ev.snippet_id,
                payload=ev.payload,
                duration_ms=ev.duration_ms,
            )
            for ev in events
        ],
    )


@router.get("/admin/sessions/{session_id}/export")
def export_session_csv(
    session_id: str,
    db: Session = Depends(get_db),
    _admin: User = Depends(get_current_admin_user),
):
    """Stream all events for a session as a CSV file download."""
    events = (
        db.query(StudyEventModel)
        .filter(StudyEventModel.session_id == session_id)
        .order_by(StudyEventModel.client_ts, StudyEventModel.id)
        .all()
    )

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "id", "session_id", "user_id", "client_ts", "created_at",
        "event_type", "phase_id", "dataset_id", "snippet_set_id",
        "snippet_id", "duration_ms", "payload",
    ])
    for ev in events:
        writer.writerow([
            ev.id,
            ev.session_id,
            ev.user_id,
            ev.client_ts.isoformat() if ev.client_ts else "",
            ev.created_at.isoformat() if ev.created_at else "",
            ev.event_type,
            ev.phase_id or "",
            ev.dataset_id or "",
            ev.snippet_set_id or "",
            ev.snippet_id or "",
            ev.duration_ms or "",
            json.dumps(ev.payload) if ev.payload else "",
        ])

    output.seek(0)
    filename = f"study_events_{session_id[:8]}.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
