"""
force_refresh must not re-run model inference.

The annotation hub sends ``force_refresh: true`` after every retrain (its
checkpoint poller, useHubALSession), meaning "the payload I cached is stale".
It used to reach the inference path, so each retrain triggered a second full
pass over the snippet set -- measured at ~18 minutes on 3M snippets, for
predictions the retrain had just written.

``force_inference`` is now the flag that re-runs the model.
See docs/superpowers/plans/2026-09-22-retrain-scaling-fixes.md (R1).
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.pool import StaticPool

from app.api import pam_active_learning as pam_api
from app.database import Base
from app.api.deps import get_current_active_user, get_db
from app.models.dataset import Dataset, DatasetType
from app.models.embedding import EmbeddingModel, SnippetSet, SnippetSetStatus
from app.models.pam_active_learning import (
    ALModelCheckpoint,
    ALModelFamilyState,
    ALModelStatus,
    ALPrediction,
)
from app.models.recording import Recording
from app.models.snippet import Snippet
from app.models.user import User, UserRole

FAMILY = "TestFamily"


@pytest.fixture(scope="module")
def engine():
    """Override the root conftest engine: StaticPool so the one in-memory DB is
    shared with the thread TestClient serves the app on. Without it, the commit
    inside the endpoint releases the connection and the next one lands on a
    fresh, empty database."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def enforce_fk(dbapi_conn, conn_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(bind=engine)
    return engine


@pytest.fixture
def world(db_session):
    admin = User(username="admin", hashed_password="x", role=UserRole.ADMIN)
    ds = Dataset(name="d", source_uri="/tmp/d", dataset_type=DatasetType.PAM)
    em = EmbeddingModel(name="birdnet", window_size=3, step_size=3, overlap=0)
    db_session.add_all([admin, ds, em])
    db_session.flush()

    ss = SnippetSet(dataset_id=ds.id, embedding_model_id=em.id, window_size=3,
                    step_size=3, overlap=0, status=SnippetSetStatus.READY)
    db_session.add(ss)
    db_session.flush()

    rec = Recording(dataset_id=ds.id, file_path="a.wav", file_name="a.wav")
    db_session.add(rec)
    db_session.flush()

    ckpt = ALModelCheckpoint(
        dataset_id=ds.id,
        model_family_name=FAMILY,
        version="v1",
        checkpoint_path="/nonexistent/model.pt",
        label_config_path="",
        hyperparameters={
            "embedding_model_id": em.id,
            "label_order": ["AAA", "BBB"],
            "used_species": ["AAA", "BBB"],
        },
        is_base=1,
        status=ALModelStatus.AVAILABLE,
    )
    db_session.add(ckpt)
    db_session.flush()
    db_session.add(
        ALModelFamilyState(
            dataset_id=ds.id, model_family_name=FAMILY, active_model_checkpoint_id=ckpt.id
        )
    )

    # Predictions already exist for this checkpoint -- the state right after a
    # retrain that ran inference.
    for i in range(4):
        snippet = Snippet(recording_id=rec.id, snippet_set_id=ss.id, start_time=i * 3.0,
                          end_time=i * 3.0 + 3.0, duration=3.0)
        db_session.add(snippet)
        db_session.flush()
        db_session.add(
            ALPrediction(
                model_checkpoint_id=ckpt.id,
                snippet_id=snippet.id,
                predicted_labels=["AAA"],
                predicted_probabilities={"AAA": 0.9 - i * 0.1, "BBB": 0.1},
                uncertainty=0.5,
                diversity=0.5,
                density=0.5,
                composite_score=1.0 - i * 0.1,
            )
        )
    db_session.commit()
    return {"admin": admin, "dataset_id": ds.id, "snippet_set_id": ss.id, "checkpoint_id": ckpt.id}


@pytest.fixture
def client(db_session, world):
    app = FastAPI()
    app.include_router(pam_api.router, prefix="/api/pam-al")
    app.dependency_overrides[get_db] = lambda: db_session
    app.dependency_overrides[get_current_active_user] = lambda: world["admin"]
    return TestClient(app)


@pytest.fixture
def no_inference(monkeypatch):
    """Make any attempt to run the model a loud failure."""
    from app.services.pam_al import _data_helpers as data_h

    def _boom(*args, **kwargs):
        raise AssertionError("inference ran: load_embeddings must not be reached")

    monkeypatch.setattr(data_h, "load_embeddings", _boom)


@pytest.fixture
def dispatched(monkeypatch):
    """Record Celery dispatches instead of sending them."""
    calls = []
    from app.tasks import pam_al_tasks

    monkeypatch.setattr(
        pam_al_tasks.pam_al_create_predictions,
        "delay",
        lambda **kwargs: calls.append(kwargs),
    )
    return calls


def _body(world, **overrides):
    body = {
        "model_family_name": FAMILY,
        "dataset_id": world["dataset_id"],
        "snippet_set_id": world["snippet_set_id"],
        "sample_suggestion": True,
        "suggestion_strategy": "composite",
        "k": 3,
    }
    body.update(overrides)
    return body


def test_force_refresh_serves_existing_predictions(client, world, no_inference, dispatched):
    """The regression this whole item is about: no second inference pass."""
    response = client.post(
        "/api/pam-al/inference/get-or-create", json=_body(world, force_refresh=True)
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["mode"] == "suggestions"
    assert payload["used_checkpoint_id"] == world["checkpoint_id"]
    assert len(payload["rows"]) == 3
    assert dispatched == [], "force_refresh must not dispatch an inference job"


def test_force_refresh_drops_the_cached_payload(client, world, no_inference, monkeypatch):
    """It still does what its name says -- invalidates the cache."""
    from app.services import inference_feed_cache

    invalidated = []
    monkeypatch.setattr(
        inference_feed_cache, "invalidate_inference_feed", lambda ckpt: invalidated.append(ckpt)
    )
    client.post("/api/pam-al/inference/get-or-create", json=_body(world, force_refresh=True))
    assert invalidated == [world["checkpoint_id"]]


def test_without_force_refresh_cache_is_not_invalidated(client, world, no_inference, monkeypatch):
    from app.services import inference_feed_cache

    invalidated = []
    monkeypatch.setattr(
        inference_feed_cache, "invalidate_inference_feed", lambda ckpt: invalidated.append(ckpt)
    )
    client.post("/api/pam-al/inference/get-or-create", json=_body(world))
    assert invalidated == []


def test_force_inference_still_dispatches(client, world, dispatched):
    """The escape hatch works: callers that really want the model re-run get it."""
    response = client.post(
        "/api/pam-al/inference/get-or-create", json=_body(world, force_inference=True)
    )
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "PENDING"
    assert len(dispatched) == 1
    assert dispatched[0]["inference_body"]["force_inference"] is True


def test_missing_predictions_still_dispatches(client, world, db_session, dispatched):
    """Nothing changes when predictions genuinely are absent."""
    db_session.query(ALPrediction).delete()
    db_session.commit()

    response = client.post("/api/pam-al/inference/get-or-create", json=_body(world))
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "PENDING"
    assert len(dispatched) == 1


def test_service_does_not_infer_on_force_refresh(db_session, world, no_inference):
    """The same guard at the service layer, which the worker task also goes through."""
    from app.schemas.pam_active_learning import ALRunInferenceRequest
    from app.services.pam_al.service import PAMActiveLearningService

    request = ALRunInferenceRequest(**_body(world, force_refresh=True))
    result = PAMActiveLearningService(db_session).get_or_create_predictions(request)
    assert result["mode"] == "suggestions"
    assert len(result["rows"]) == 3
