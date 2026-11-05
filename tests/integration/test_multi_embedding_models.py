"""Several embedding models on one dataset (yapat-backend#74)."""

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_current_active_user, get_db
from app.main import app
from app.models.annotation import Annotation
from app.models.dataset import Dataset, DatasetType
from app.models.embedding import (
    EmbeddingJob,
    EmbeddingJobStatus,
    EmbeddingModel,
    EmbeddingVector,
    SnippetSetStatus,
)
from app.models.pam_active_learning import (
    ALFeedbackEvent,
    ALModelCheckpoint,
    ALModelStatus,
    ALRetrainJob,
    ALRetrainStatus,
)
from app.models.recording import Recording
from app.models.snippet import Snippet
from app.models.user import User, UserRole
from app.services.embedders import get_embedder, register_embedder, unregister_embedder, weights_available
from app.services.embedders.base import AudioEmbedder
from app.services.embedding_scope import (
    default_model_for_set,
    embedding_options,
    models_for_set,
    snippet_set_for_model,
)
from app.services.embedding_service import EmbeddingService, VectorStore
from app.schemas.pam_active_learning import ALFeedbackSubmit
from app.services.pam_al import _checkpoint_helpers as ckpt_h
from app.services.pam_al.service import PAMActiveLearningService


def _model(db, name, window, dim):
    model = EmbeddingModel(
        name=name, window_size=window, step_size=window, overlap=0.0, dim=dim,
    )
    db.add(model)
    db.flush()
    return model


@pytest.fixture
def world(db_session):
    db = db_session
    dataset = Dataset(name="multi", source_uri="site", dataset_type=DatasetType.PAM)
    db.add(dataset)
    db.flush()
    three_a = _model(db, "three_a", 3.0, 8)
    three_b = _model(db, "three_b", 3.0, 16)
    five = _model(db, "five", 5.0, 32)
    db.commit()
    return dataset, three_a, three_b, five


def _finish(db, job):
    job.status = EmbeddingJobStatus.COMPLETED
    job.snippet_set.status = SnippetSetStatus.READY
    if job.dataset.default_snippet_set_id is None:
        job.dataset.default_snippet_set_id = job.snippet_set_id
    db.commit()


def _user(db):
    user = User(username="u", hashed_password="x", role=UserRole.ADMIN)
    db.add(user)
    db.commit()
    return user


def _snippet(db, dataset, snippet_set_id):
    recording = Recording(dataset_id=dataset.id, file_path="site/a.wav", file_name="a.wav", duration=6.0, sample_rate=48000)
    db.add(recording)
    db.flush()
    snippet = Snippet(recording_id=recording.id, snippet_set_id=snippet_set_id, start_time=0.0, end_time=3.0, duration=3.0)
    db.add(snippet)
    db.commit()
    return snippet


def test_same_window_shares_a_snippet_set_and_other_window_does_not(db_session, world):
    dataset, three_a, three_b, five = world
    service = EmbeddingService(db_session)

    job_a = service.create_embedding_job(dataset, three_a)
    job_b = service.create_embedding_job(dataset, three_b)
    job_five = service.create_embedding_job(dataset, five)

    assert job_a.snippet_set_id == job_b.snippet_set_id
    assert job_five.snippet_set_id != job_a.snippet_set_id


def test_different_step_gets_its_own_snippet_set(db_session, world):
    dataset, three_a, three_b, _ = world
    three_b.step_size = 1.5
    db_session.commit()
    service = EmbeddingService(db_session)

    job_a = service.create_embedding_job(dataset, three_a)
    job_b = service.create_embedding_job(dataset, three_b)

    assert job_a.snippet_set_id != job_b.snippet_set_id


def test_snippet_holds_one_vector_per_model_of_different_sizes(db_session, world):
    dataset, three_a, three_b, _ = world
    service = EmbeddingService(db_session)
    job_a = service.create_embedding_job(dataset, three_a)
    job_b = service.create_embedding_job(dataset, three_b)
    snippet = _snippet(db_session, dataset, job_a.snippet_set_id)
    store = VectorStore(db_session)

    for _ in range(2):  # the second round must be ignored by the unique key
        store.bulk_insert([
            {"snippet_id": snippet.id, "job_id": job_a.id, "model_id": three_a.id, "vector": [0.1] * 8},
            {"snippet_id": snippet.id, "job_id": job_b.id, "model_id": three_b.id, "vector": [0.2] * 16},
        ])

    rows = db_session.query(EmbeddingVector).filter_by(snippet_id=snippet.id).all()
    assert sorted((r.embedding_model_id, r.dim) for r in rows) == [(three_a.id, 8), (three_b.id, 16)]


def test_models_of_a_set_are_the_finished_ones_oldest_first(db_session, world):
    dataset, three_a, three_b, _ = world
    service = EmbeddingService(db_session)
    job_a = service.create_embedding_job(dataset, three_a)
    job_b = service.create_embedding_job(dataset, three_b)
    _finish(db_session, job_a)

    assert models_for_set(db_session, job_a.snippet_set_id) == [three_a.id]
    _finish(db_session, job_b)
    assert models_for_set(db_session, job_a.snippet_set_id) == [three_a.id, three_b.id]
    assert default_model_for_set(db_session, job_a.snippet_set_id) == three_a.id


def test_snippet_set_for_model_finds_a_non_default_set(db_session, world):
    dataset, three_a, _, five = world
    service = EmbeddingService(db_session)
    job_a = service.create_embedding_job(dataset, three_a)
    job_five = service.create_embedding_job(dataset, five)
    _finish(db_session, job_a)
    _finish(db_session, job_five)

    assert snippet_set_for_model(db_session, dataset, three_a.id) == job_a.snippet_set_id
    assert snippet_set_for_model(db_session, dataset, five.id) == job_five.snippet_set_id


def test_embedding_options_describe_the_switcher(db_session, world):
    dataset, three_a, three_b, five = world
    service = EmbeddingService(db_session)
    job_a = service.create_embedding_job(dataset, three_a)
    job_b = service.create_embedding_job(dataset, three_b)
    job_five = service.create_embedding_job(dataset, five)
    _finish(db_session, job_a)
    db_session.add(ALModelCheckpoint(
        dataset_id=dataset.id, model_family_name="fam_a", version="v0", checkpoint_path="c",
        label_config_path="l", model_type="pam_linear_multilabel_classifier",
        hyperparameters={"embedding_model_id": three_a.id},
    ))
    db_session.flush()
    ckpt = db_session.query(ALModelCheckpoint).one()
    ckpt_h.set_active_family_checkpoint(db_session, dataset.id, "fam_a", ckpt.id)
    db_session.commit()

    options = {o["embedding_model_id"]: o for o in embedding_options(db_session, dataset)}

    assert options[three_a.id]["is_default"] and not options[three_b.id]["is_default"]
    assert options[three_a.id]["shares_snippet_set"] and options[three_b.id]["shares_snippet_set"]
    assert not options[five.id]["shares_snippet_set"]
    assert options[three_a.id]["active_checkpoint_id"] == ckpt.id
    assert options[three_a.id]["model_family_name"] == "fam_a"
    assert options[three_b.id]["active_checkpoint_id"] is None
    assert options[three_b.id]["model_family_name"] == "three_b"
    assert options[three_b.id]["status"] == job_b.status.value
    assert options[five.id]["snippet_set_id"] == job_five.snippet_set_id


def test_classifier_family_is_locked_to_one_embedding_model(db_session, world):
    dataset, three_a, three_b, _ = world
    db_session.add(ALModelCheckpoint(
        dataset_id=dataset.id, model_family_name="fam", version="v0", checkpoint_path="c",
        label_config_path="l", model_type="pam_linear_multilabel_classifier",
        hyperparameters={"embedding_model_id": three_a.id},
    ))
    db_session.commit()

    ckpt_h.assert_family_uses_embedding_model(db_session, dataset.id, "fam", three_a.id)
    with pytest.raises(ValueError, match="separate family"):
        ckpt_h.assert_family_uses_embedding_model(db_session, dataset.id, "fam", three_b.id)


@pytest.fixture
def client(db_session):
    user = _user(db_session)
    app.dependency_overrides[get_db] = lambda: db_session
    app.dependency_overrides[get_current_active_user] = lambda: user
    yield TestClient(app)
    app.dependency_overrides.clear()


TAXON = {"taxon_id": "local:frog_a", "display_name": "Frog A"}


def _shared_snippet(db, world):
    dataset, three_a, three_b, _ = world
    service = EmbeddingService(db)
    job_a = service.create_embedding_job(dataset, three_a)
    job_b = service.create_embedding_job(dataset, three_b)
    _finish(db, job_a)
    _finish(db, job_b)
    return _snippet(db, dataset, job_a.snippet_set_id)


def test_annotation_records_the_selected_model(db_session, world, client):
    _, _, three_b, _ = world
    snippet = _shared_snippet(db_session, world)

    r = client.post("/api/annotations/", json={
        **TAXON, "snippet_id": snippet.id, "embedding_model_id": three_b.id,
    })

    assert r.status_code == 201, r.text
    assert r.json()["embedding_model_id"] == three_b.id
    assert db_session.query(Annotation).one().embedding_model_id == three_b.id


def test_annotation_on_a_shared_set_must_name_its_model(db_session, world, client):
    snippet = _shared_snippet(db_session, world)

    r = client.post("/api/annotations/", json={**TAXON, "snippet_id": snippet.id})

    assert r.status_code == 422
    assert db_session.query(Annotation).count() == 0


def test_annotation_on_a_single_model_set_fills_in_the_model(db_session, world, client):
    dataset, three_a, _, _ = world
    job = EmbeddingService(db_session).create_embedding_job(dataset, three_a)
    _finish(db_session, job)
    snippet = _snippet(db_session, dataset, job.snippet_set_id)

    r = client.post("/api/annotations/", json={**TAXON, "snippet_id": snippet.id})

    assert r.status_code == 201, r.text
    assert r.json()["embedding_model_id"] == three_a.id


def test_annotation_rejects_a_model_that_has_not_embedded_the_set(db_session, world, client):
    dataset, three_a, _, five = world
    job = EmbeddingService(db_session).create_embedding_job(dataset, three_a)
    _finish(db_session, job)
    snippet = _snippet(db_session, dataset, job.snippet_set_id)

    r = client.post("/api/annotations/", json={
        **TAXON, "snippet_id": snippet.id, "embedding_model_id": five.id,
    })

    assert r.status_code == 400


def test_label_made_under_one_model_is_visible_under_the_other(db_session, world, client):
    _, three_a, three_b, _ = world
    snippet = _shared_snippet(db_session, world)
    client.post("/api/annotations/", json={
        **TAXON, "snippet_id": snippet.id, "embedding_model_id": three_a.id,
    })

    # Annotations belong to the clip, so the same label cannot be added again under B.
    r = client.post("/api/annotations/", json={
        **TAXON, "snippet_id": snippet.id, "embedding_model_id": three_b.id,
    })
    assert r.status_code == 409
    listed = client.get("/api/annotations/", params={"snippet_id": snippet.id}).json()
    assert [(a["taxon_id"], a["embedding_model_id"]) for a in listed] == [("local:frog_a", three_a.id)]


def test_registry_runs_a_registered_embedder(monkeypatch):
    class Fake(AudioEmbedder):
        SAMPLE_RATE = 16000
        WINDOW_SECONDS = 3.0
        DIM = 8

        @classmethod
        def embed(cls, audio_path, start_time):
            return [0.0] * 8

        @classmethod
        def embed_batch_from_recording(cls, audio_path, snippet_windows):
            return [[0.0] * 8 for _ in snippet_windows]

    register_embedder("fake_8", Fake)
    try:
        assert get_embedder("fake_8") is Fake
        assert weights_available("fake_8")
    finally:
        unregister_embedder("fake_8")
    assert not weights_available("fake_8")
    with pytest.raises(ValueError):
        get_embedder("fake_8")


def _second_model_on_own_set(db, world):
    """`three_a` has a classifier; `five` has its own snippet set and no classifier."""
    dataset, three_a, _three_b, five = world
    dataset.retrain_after_threshold = 2
    service = EmbeddingService(db)
    three_job = service.create_embedding_job(dataset, three_a)
    five_job = service.create_embedding_job(dataset, five)
    _finish(db, three_job)
    _finish(db, five_job)
    db.add(ALModelCheckpoint(
        dataset_id=dataset.id, model_family_name="three_family", version="v0", checkpoint_path="",
        label_config_path="", model_type="pam_linear_multilabel", embedding_model_id=three_a.id,
    ))
    db.commit()
    return dataset, five, five_job.snippet_set_id, three_job.snippet_set_id


def _label(db, dataset, model, snippet_set_id, user):
    return PAMActiveLearningService(db).submit_feedback(ALFeedbackSubmit(
        dataset_id=dataset.id,
        model_family_name="five_family",
        snippet_id=_snippet(db, dataset, snippet_set_id).id,
        embedding_model_id=model.id,
        action="MODIFY",
        labels=["sp"],
        user_id=user.id,
    ))


def test_labels_train_the_selected_models_first_classifier_in_the_background(db_session, world):
    db = db_session
    dataset, five, five_set, three_set = _second_model_on_own_set(db, world)
    user = _user(db)

    # Labels on another model's snippets don't count toward this one.
    _label(db, dataset, world[1], three_set, user)
    first = _label(db, dataset, five, five_set, user)
    assert first["retrain_triggered"] is False
    assert first["feedback_count_since_retrain"] == 1

    second = _label(db, dataset, five, five_set, user)
    assert second["retrain_triggered"] is True
    assert second["active_checkpoint_id"] is None
    job = db.get(ALRetrainJob, second["auto_retrain_job_id"])
    assert job.status == ALRetrainStatus.PENDING
    checkpoint = db.get(ALModelCheckpoint, job.model_checkpoint_id)
    assert checkpoint.model_family_name == "five_family"
    assert checkpoint.hyperparameters["embedding_model_id"] == five.id
    assert checkpoint.hyperparameters["snippet_set_id"] == five_set
    assert checkpoint.hyperparameters["use_existing_labels"] is True

    # Already training: more labels don't start another run.
    third = _label(db, dataset, five, five_set, user)
    assert third["retrain_triggered"] is False
    assert third["auto_retrain_job_id"] is None


def test_a_failed_first_classifier_waits_for_new_labels(db_session, world):
    db = db_session
    dataset, five, five_set, _three_set = _second_model_on_own_set(db, world)
    user = _user(db)
    _label(db, dataset, five, five_set, user)
    job_id = _label(db, dataset, five, five_set, user)["auto_retrain_job_id"]

    job = db.get(ALRetrainJob, job_id)
    job.status = ALRetrainStatus.FAILED
    now = datetime.now(timezone.utc)
    job.completed_at = now - timedelta(minutes=30)
    db.get(ALModelCheckpoint, job.model_checkpoint_id).status = ALModelStatus.ERROR
    # Labelled before the run failed.
    db.query(ALFeedbackEvent).update({ALFeedbackEvent.created_at: now - timedelta(hours=1)})
    db.commit()

    assert _label(db, dataset, five, five_set, user)["retrain_triggered"] is False
    retry = _label(db, dataset, five, five_set, user)
    assert retry["retrain_triggered"] is True
    assert retry["auto_retrain_job_id"] != job_id
    # The failed run keeps its checkpoint row and version.
    assert db.get(ALModelCheckpoint, retry["auto_retrain_checkpoint_id"]).version == "v0_2"
