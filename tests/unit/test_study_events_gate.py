"""Tests for the STUDY_EVENTS_ENABLED gate on study-event ingest."""

import pytest
from fastapi import HTTPException

from app.api.study_events import ingest_study_events
from app.config import settings
from app.schemas.study_event import StudyEventBatchCreate


def test_ingest_rejected_when_study_events_disabled(monkeypatch):
    monkeypatch.setattr(settings, "STUDY_EVENTS_ENABLED", False)

    # The gate runs before auth and before the database is touched, so no
    # token or session is needed to prove nothing gets stored.
    with pytest.raises(HTTPException) as exc:
        ingest_study_events(StudyEventBatchCreate(events=[]), db=None, credentials=None)

    assert exc.value.status_code == 403


def test_ingest_reaches_auth_when_study_events_enabled(monkeypatch):
    monkeypatch.setattr(settings, "STUDY_EVENTS_ENABLED", True)

    # With the gate open, a request without a token fails on auth (401)
    # instead of the gate (403).
    with pytest.raises(HTTPException) as exc:
        ingest_study_events(StudyEventBatchCreate(events=[]), db=None, credentials=None)

    assert exc.value.status_code == 401


def test_study_events_disabled_by_default():
    assert type(settings).model_fields["STUDY_EVENTS_ENABLED"].default is False
