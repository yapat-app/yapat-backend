"""
Server-side explore API: feed pages, score histograms, facets and projection
data computed over an entire snippet set.

Endpoints are synchronous so FastAPI runs them in its threadpool; the numpy
work releases the GIL. Layer builds never block a request: a missing layer
returns 202 and the client retries.
"""

from __future__ import annotations

from typing import Callable

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.api.deps import get_current_active_user, get_db
from app.models.user import User
from app.schemas.explore import (
    FacetsRequest,
    FeedRequest,
    ProjectionCoordsRequest,
    ProjectionRequest,
    ProjectionStateRequest,
    RowsRequest,
    SummaryRequest,
    ViewportRequest,
)
from app.services.explore import registry
from app.services.explore.service import ExploreService, ScopeError

router = APIRouter()


def _run(db: Session, user: User, handler: Callable[[ExploreService], dict]):
    service = ExploreService(db, user)
    try:
        return handler(service)
    except ScopeError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail)
    except registry.LayerBuilding as exc:
        return JSONResponse(
            status_code=202,
            content={
                "status": "building",
                "layer": exc.kind,
                "retry_after_ms": registry.RETRY_AFTER_MS,
                "detail": str(exc),
            },
        )
    except registry.NoPredictions as exc:
        return JSONResponse(status_code=409, content={"status": "no_predictions", "detail": str(exc)})
    except registry.ProjectionMissing as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except registry.LayerBuildFailed as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/summary")
def explore_summary(
    body: SummaryRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Counts, score domains and per-score histogram bins for a filter set."""
    return _run(db, current_user, lambda svc: svc.summary(body))


@router.post("/facets")
def explore_facets(
    body: FacetsRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Filter options: locations, annotated species, date/time histograms."""
    return _run(db, current_user, lambda svc: svc.facets(body.scope))


@router.post("/feed")
def explore_feed(
    body: FeedRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """One page of the filtered, sorted feed."""
    return _run(db, current_user, lambda svc: svc.feed(body))


@router.post("/rows")
def explore_rows(
    body: RowsRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Feed rows for specific snippets (overlays, multi-select, label refresh)."""
    return _run(db, current_user, lambda svc: svc.rows(body))


@router.post("/projection")
def explore_projection(
    body: ProjectionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Static (filter-independent) projection points for one method."""
    return _run(db, current_user, lambda svc: svc.projection(body))


@router.post("/projection/state")
def explore_projection_state(
    body: ProjectionStateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Filter-dependent visibility/labels for the projection sample."""
    return _run(db, current_user, lambda svc: svc.projection_state(body))


@router.post("/projection/coords")
def explore_projection_coords(
    body: ProjectionCoordsRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Projection coordinates for specific snippets (e.g. the current selection)."""
    return _run(db, current_user, lambda svc: svc.projection_coords(body))


@router.post("/projection/viewport")
def explore_projection_viewport(
    body: ViewportRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Full-detail points inside a zoomed-in bounding box."""
    return _run(db, current_user, lambda svc: svc.viewport(body))
