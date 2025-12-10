import pytest

from app.services.embedding_service import EmbeddingService
from app.models.dataset import Dataset
from app.models.embedding import (
    EmbeddingModel,
    SnippetSet,
    EmbeddingJob,
)


def test_embedding_model_get(db_session):
    model = EmbeddingModel(
        name="birdnet",
        version="2.4",
        window_size=3.0,
        step_size=3.0,
        overlap=0.0,
    )
    db_session.add(model)
    db_session.commit()

    service = EmbeddingService(db_session)
    retrieved = service.get_model(model.id)
    assert retrieved.id == model.id


def test_snippet_set_get_or_create_creates_new(db_session):
    ds = Dataset(name="D", source_uri="dummy")
    model = EmbeddingModel(
        name="yamnet",
        version="1.0",
        window_size=1.0,
        step_size=1.0,
        overlap=0.0,
    )
    db_session.add_all([ds, model])
    db_session.commit()

    svc = EmbeddingService(db_session)

    ss = svc.get_or_create_snippet_set(ds, model)
    assert ss.id is not None
    assert ss.window_size == model.window_size
    assert ss.embedding_model_id == model.id
    assert ss.dataset_id == ds.id


def test_snippet_set_get_or_create_reuses_existing(db_session):
    ds = Dataset(name="DS", source_uri="dummy")
    model = EmbeddingModel(
        name="birdnet",
        version="2.4",
        window_size=3.0,
        step_size=3.0,
        overlap=0.0,
    )
    db_session.add_all([ds, model])
    db_session.commit()

    svc = EmbeddingService(db_session)

    ss1 = svc.get_or_create_snippet_set(ds, model)
    ss2 = svc.get_or_create_snippet_set(ds, model)

    assert ss1.id == ss2.id  # reused


def test_create_embedding_job_creates_job_and_snippet_set(db_session):
    ds = Dataset(name="DS", source_uri="dummy")
    model = EmbeddingModel(
        name="bcresnet",
        version="1",
        window_size=2.0,
        step_size=1.0,
        overlap=0.0,
    )
    db_session.add_all([ds, model])
    db_session.commit()

    svc = EmbeddingService(db_session)
    job = svc.create_embedding_job(ds, model)

    assert isinstance(job.id, int)
    assert isinstance(job.snippet_set_id, int)

    ss = db_session.query(SnippetSet).get(job.snippet_set_id)
    assert ss is not None
    assert ss.dataset_id == ds.id
    assert ss.embedding_model_id == model.id
