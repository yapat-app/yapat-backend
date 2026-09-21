"""
End-to-end tests for /api/explore with inline layer builds on SQLite.

Only the explore router is mounted, so the test does not depend on the heavy
ML imports pulled in by app.main.
"""

import base64

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import explore as explore_api
from app.api.deps import get_current_active_user, get_db
from app.models.dataset import Dataset, DatasetType
from app.models.embedding import EmbeddingModel, SnippetSet, SnippetSetStatus
from app.models.pam_active_learning import (
    ALAnnotationSource,
    ALModelCheckpoint,
    ALPrediction,
    ALSnippetAnnotation,
)
from app.models.recording import Recording
from app.models.snippet import Snippet
from app.models.user import User, UserRole
from app.models.visualisation import FPVVis
from app.services.explore import registry, storage


@pytest.fixture(autouse=True)
def explore_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(storage.settings, "EXPLORE_CACHE_DIR", str(tmp_path / "explore"))
    monkeypatch.setattr(storage.settings, "EXPLORE_BUILD_INLINE", True)
    monkeypatch.setattr(storage.settings, "EXPLORE_LABELS_RECHECK_SECONDS", 0.0)
    registry.reset_caches()
    yield
    registry.reset_caches()


@pytest.fixture
def world(db_session):
    admin = User(username="admin", hashed_password="x", role=UserRole.ADMIN)
    outsider = User(username="outsider", hashed_password="x", role=UserRole.USER)
    ds = Dataset(name="d", source_uri="/tmp/d", dataset_type=DatasetType.PAM)
    other_ds = Dataset(name="o", source_uri="/tmp/o", dataset_type=DatasetType.PAM)
    em = EmbeddingModel(name="birdnet", window_size=3, step_size=3, overlap=0)
    db_session.add_all([admin, outsider, ds, other_ds, em])
    db_session.flush()

    ss = SnippetSet(dataset_id=ds.id, embedding_model_id=em.id, window_size=3, step_size=3,
                    overlap=0, status=SnippetSetStatus.READY)
    other_ss = SnippetSet(dataset_id=other_ds.id, embedding_model_id=em.id, window_size=3,
                          step_size=3, overlap=0, status=SnippetSetStatus.READY)
    db_session.add_all([ss, other_ss])
    db_session.flush()

    rec_a = Recording(dataset_id=ds.id, file_path="a.wav", file_name="a.wav",
                      extra_metadata={"location": "North", "recorded_date": "2024-05-15",
                                      "recorded_time": 3600})
    rec_b = Recording(dataset_id=ds.id, file_path="b.wav", file_name="b.wav",
                      extra_metadata={"location": "South"})
    db_session.add_all([rec_a, rec_b])
    db_session.flush()

    snippets = []
    for i in range(6):
        rec = rec_a if i < 3 else rec_b
        s = Snippet(recording_id=rec.id, snippet_set_id=ss.id, start_time=i * 3.0,
                    end_time=i * 3.0 + 3.0, duration=3.0)
        snippets.append(s)
    db_session.add_all(snippets)
    db_session.flush()

    ckpt = ALModelCheckpoint(dataset_id=ds.id, model_family_name="fam", version="v1",
                             checkpoint_path="x.pt", label_config_path="labels.json",
                             hyperparameters={"label_order": ["owl", "wren"]})
    empty_ckpt = ALModelCheckpoint(dataset_id=ds.id, model_family_name="fam", version="v2",
                                   checkpoint_path="y.pt", label_config_path="labels.json")
    db_session.add_all([ckpt, empty_ckpt])
    db_session.flush()

    composite = [1.5, 0.2, None, -0.4, 0.9, 0.0]
    owl = [0.9, 0.1, 0.0, 0.6, 0.3, 0.2]
    for i, s in enumerate(snippets):
        db_session.add(ALPrediction(
            model_checkpoint_id=ckpt.id, snippet_id=s.id,
            predicted_labels=["owl"] if owl[i] >= 0.5 else [],
            predicted_probabilities={"owl": owl[i], "wren": 1 - owl[i]},
            uncertainty=None if composite[i] is None else 0.1 * i,
            diversity=None if composite[i] is None else 0.5,
            density=None if composite[i] is None else 0.5,
            composite_score=composite[i],
        ))
    db_session.add(ALSnippetAnnotation(dataset_id=ds.id, snippet_id=snippets[2].id, label="owl",
                                       source=ALAnnotationSource.USER, user_id=None))
    for i, s in enumerate(snippets):
        db_session.add(FPVVis(dataset_id=ds.id, embedding_model_id=em.id, snippet_id=s.id,
                              pca_2d_x=float(i), pca_2d_y=float(i % 2)))
    db_session.commit()
    return {
        "admin": admin, "outsider": outsider, "ds": ds, "ss": ss, "other_ss": other_ss,
        "em": em, "ckpt": ckpt, "empty_ckpt": empty_ckpt, "snippets": snippets,
    }


@pytest.fixture
def client_for(db_session):
    app = FastAPI()
    app.include_router(explore_api.router, prefix="/api/explore")

    def _as(user):
        app.dependency_overrides[get_db] = lambda: db_session
        app.dependency_overrides[get_current_active_user] = lambda: user
        return TestClient(app)

    yield _as
    app.dependency_overrides.clear()


def scope(world, **overrides):
    body = {
        "dataset_id": world["ds"].id,
        "snippet_set_id": world["ss"].id,
        "checkpoint_id": world["ckpt"].id,
        "embedding_model_id": world["em"].id,
    }
    body.update(overrides)
    return body


def ids_of(rows):
    return [r["snippet_id"] for r in rows]


def test_feed_native_order_and_row_shape(world, client_for):
    client = client_for(world["admin"])
    sid = [s.id for s in world["snippets"]]
    resp = client.post("/api/explore/feed", json={"scope": scope(world), "limit": 4})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total"] == 6
    # composite desc (nulls last): 1.5, 0.9, 0.2, 0.0, -0.4, None
    assert ids_of(body["rows"]) == [sid[0], sid[4], sid[1], sid[5]]
    first = body["rows"][0]
    assert first["predicted_labels"] == ["owl"]
    assert first["predicted_label"] == "owl"
    assert first["confidence"] == pytest.approx(0.9, abs=1e-3)
    assert first["recording_id"] is not None and first["duration_sec"] == 3.0
    assert set(first["scores"]) >= {"composite", "confidence"}

    page2 = client.post("/api/explore/feed", json={"scope": scope(world), "offset": 4, "limit": 4}).json()
    assert ids_of(page2["rows"]) == [sid[3], sid[2]]
    assert page2["rows"][1]["labels"] == ["owl"]


def test_filters_sort_and_anchor(world, client_for):
    client = client_for(world["admin"])
    sid = [s.id for s in world["snippets"]]
    body = {
        "scope": scope(world),
        "filters": {"annotation_status": "unannotated", "locations": ["North"]},
        "sort": [{"field": "confidence", "direction": "asc"}],
    }
    resp = client.post("/api/explore/feed", json=body).json()
    # North = snippets 0..2; 2 is labeled; confidence asc: 1 (0.9 wren), 0 (0.9 owl) tie → native
    assert ids_of(resp["rows"]) == [sid[0], sid[1]]

    anchored = client.post(
        "/api/explore/feed",
        json={"scope": scope(world), "limit": 2, "anchor_snippet_id": sid[2], "prefer_unlabeled": True},
    ).json()
    # Snippet 2 is last in native order and labeled; stays on it.
    assert anchored["anchor_index"] == 5
    assert anchored["offset"] == 4
    assert anchored["anchor_snippet_id"] == sid[2]


def test_summary_counts_and_histograms(world, client_for):
    client = client_for(world["admin"])
    body = {
        "scope": scope(world),
        "filters": {"score_ranges": {"composite": [0.5, 1.0]}},
        "bins": 8,
    }
    resp = client.post("/api/explore/summary", json=body)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["counts"]["total_snippets"] == 6
    assert data["counts"]["population"] == 6
    assert data["counts"]["labeled"] == 1
    assert data["domains"]["composite"] == pytest.approx([-0.4, 1.5], abs=1e-5)
    # composite ≥ -0.4 + 0.5*1.9 = 0.55 → 1.5, 0.9, plus the unscored row passes.
    assert data["counts"]["visible"] == 3
    assert len(data["histograms"]["composite"]["total"]) == 8
    assert sum(data["histograms"]["composite"]["total"]) == 5
    assert sum(data["histograms"]["composite"]["visible"]) == 2


def test_labels_refresh_without_restart(world, client_for, db_session):
    client = client_for(world["admin"])
    body = {"scope": scope(world), "filters": {"annotation_status": "annotated"}}
    assert client.post("/api/explore/summary", json=body).json()["counts"]["visible"] == 1
    db_session.add(ALSnippetAnnotation(dataset_id=world["ds"].id, snippet_id=world["snippets"][0].id,
                                       label="wren", source=ALAnnotationSource.USER))
    db_session.commit()
    assert client.post("/api/explore/summary", json=body).json()["counts"]["visible"] == 2
    facets = client.post("/api/explore/facets", json={"scope": scope(world)}).json()
    assert facets["annotated_species"] == ["owl", "wren"]


def test_facets(world, client_for):
    client = client_for(world["admin"])
    facets = client.post("/api/explore/facets", json={"scope": scope(world)}).json()
    assert facets["locations"] == ["North", "South"]
    assert facets["has_date_time"] is True
    assert facets["date_counts"] == [[19858, 1]]  # 2024-05-15
    assert facets["time_counts"] == [[60, 1]]
    assert facets["label_order"] == ["owl", "wren"]


def test_projection_and_state(world, client_for):
    client = client_for(world["admin"])
    proj = client.post("/api/explore/projection", json={"scope": scope(world), "method": "pca"}).json()
    assert proj["available"] is True and proj["sampled"] is False
    ids = np.frombuffer(base64.b64decode(proj["ids"]), dtype="<i4")
    xs = np.frombuffer(base64.b64decode(proj["x"]), dtype="<f4")
    assert ids.tolist() == [s.id for s in world["snippets"]]
    assert xs.tolist() == [0, 1, 2, 3, 4, 5]

    umap = client.post("/api/explore/projection", json={"scope": scope(world), "method": "umap"}).json()
    assert umap["available"] is False and umap["point_count"] == 0

    state = client.post(
        "/api/explore/projection/state",
        json={"scope": scope(world), "method": "pca", "filters": {"locations": ["South"]}},
    ).json()
    bits = np.unpackbits(np.frombuffer(base64.b64decode(state["visible"]), np.uint8), bitorder="little")[:6]
    assert bits.tolist() == [0, 0, 0, 1, 1, 1]
    labels = np.frombuffer(base64.b64decode(state["label_idx"]), dtype="<i2")
    assert labels.tolist() == [-1, -1, 0, -1, -1, -1]
    assert state["label_vocab"] == ["owl"]
    assert state["visible_points"] == 3 and state["density"] is None

    coords = client.post(
        "/api/explore/projection/coords",
        json={"scope": scope(world), "method": "pca", "snippet_ids": [world["snippets"][4].id, 12345]},
    ).json()
    assert coords["count"] == 1
    assert np.frombuffer(base64.b64decode(coords["x"]), dtype="<f4").tolist() == [4.0]

    viewport = client.post(
        "/api/explore/projection/viewport",
        json={"scope": scope(world), "method": "pca", "bbox": [0.5, 3.5, -1, 2], "max_points": 2},
    ).json()
    assert viewport["complete"] is False and viewport["count"] == 2


def test_rows_endpoint_ignores_foreign_ids(world, client_for):
    client = client_for(world["admin"])
    sid = world["snippets"][3].id
    resp = client.post("/api/explore/rows", json={"scope": scope(world), "snippet_ids": [sid, 999999]}).json()
    assert ids_of(resp["rows"]) == [sid]


def test_no_model_scope(world, client_for):
    client = client_for(world["admin"])
    resp = client.post("/api/explore/feed", json={"scope": scope(world, checkpoint_id=None)}).json()
    assert ids_of(resp["rows"]) == [s.id for s in world["snippets"]]
    assert resp["rows"][0]["predicted_labels"] is None


def test_validation_errors(world, client_for):
    admin = client_for(world["admin"])
    r = admin.post("/api/explore/feed", json={"scope": scope(world, snippet_set_id=world["other_ss"].id)})
    assert r.status_code == 400
    r = admin.post("/api/explore/feed", json={"scope": scope(world, checkpoint_id=world["empty_ckpt"].id)})
    assert r.status_code == 409 and r.json()["status"] == "no_predictions"
    r = admin.post("/api/explore/feed", json={"scope": scope(world, embedding_model_id=world["em"].id + 99)})
    assert r.status_code == 400
    r = admin.post("/api/explore/feed", json={"scope": scope(world), "limit": 5000})
    assert r.status_code == 422
    r = admin.post("/api/explore/feed", json={"scope": scope(world), "filters": {"months": [13]}})
    assert r.status_code == 422
    outsider = client_for(world["outsider"])
    assert outsider.post("/api/explore/feed", json={"scope": scope(world)}).status_code == 403


def test_building_returns_202_when_not_inline(world, client_for, monkeypatch):
    monkeypatch.setattr(storage.settings, "EXPLORE_BUILD_INLINE", False)

    class FakeRedis:
        def __init__(self):
            self.store = {}

        def get(self, key):
            return self.store.get(key)

        def set(self, key, value, nx=False, ex=None):
            if nx and key in self.store:
                return False
            self.store[key] = value
            return True

        def delete(self, *keys):
            for key in keys:
                self.store.pop(key, None)

    fake = FakeRedis()
    delayed = []
    monkeypatch.setattr(registry, "_redis_client", lambda: fake)
    import app.tasks.explore_tasks as tasks

    monkeypatch.setattr(tasks.build_explore_layer, "delay", lambda *a, **k: delayed.append(a))
    client = client_for(world["admin"])
    r1 = client.post("/api/explore/summary", json={"scope": scope(world)})
    r2 = client.post("/api/explore/summary", json={"scope": scope(world)})
    assert r1.status_code == 202 and r1.json()["status"] == "building"
    assert r2.status_code == 202
    assert len(delayed) == 1  # de-duplicated by the Redis guard

    fake.store[registry.error_key("base", f"ss{world['ss'].id}")] = b"boom"
    fake.delete(registry.guard_key("base", f"ss{world['ss'].id}"))
    r3 = client.post("/api/explore/summary", json={"scope": scope(world)})
    assert r3.status_code == 500 and "boom" in r3.json()["detail"]


def test_inference_hook_publishes_model_layer(world, db_session):
    from app.services.explore.builders import model_key
    from app.services.explore.hooks import publish_model_layer_from_inference

    snippets = world["snippets"]
    publish_model_layer_from_inference(
        db_session,
        checkpoint_id=world["ckpt"].id,
        snippet_ids=[s.id for s in reversed(snippets)],
        probs=np.tile([[0.2, 0.8]], (6, 1)),
        preds=np.tile([[0, 1]], (6, 1)),
        score_matrix=np.zeros((6, 4), dtype=np.float32),
        label_order=["owl", "wren"],
    )
    pointer = storage.read_pointer("model", model_key(world["ckpt"].id, world["ss"].id))
    assert pointer is not None
    layer = storage.load_layer("model", model_key(world["ckpt"].id, world["ss"].id), pointer.version)
    assert layer["snippet_ids"].tolist() == [s.id for s in snippets]
    assert layer.meta["source"] == "inference"
