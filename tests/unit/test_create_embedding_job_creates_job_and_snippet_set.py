import pytest

from app.services.embedding_service import EmbeddingService
from app.models.dataset import Dataset
from app.models.embedding import (
    EmbeddingModel,
    SnippetSet,
    EmbeddingJobStatus,
)


def test_create_embedding_job_creates_job_and_snippet_set(db_session):
    # -----------------------------------------------
    # Create dataset + embedding model
    # -----------------------------------------------
    dataset = Dataset(
        name="Test Dataset",
        source_uri="dummy"
    )
    db_session.add(dataset)

    model = EmbeddingModel(
        name="birdnet",
        version="2.4",
        window_size=3.0,
        step_size=1.0,
        overlap=0.0,
        requires_fixed_window=True,
        requires_fixed_step=True,
        requires_fixed_overlap=True,
    )
    db_session.add(model)

    db_session.commit()

    # -----------------------------------------------
    # Create job (which implicitly creates SnippetSet)
    # -----------------------------------------------
    svc = EmbeddingService(db_session)
    job = svc.create_embedding_job(dataset, model)

    # -----------------------------------------------
    # Assertions
    # -----------------------------------------------
    # Job was created
    assert job.id is not None
    assert job.dataset_id == dataset.id
    assert job.embedding_model_id == model.id
    assert job.status == EmbeddingJobStatus.PENDING

    # SnippetSet was created and linked
    assert isinstance(job.snippet_set_id, int)

    ss = db_session.query(SnippetSet).get(job.snippet_set_id)
    assert ss is not None
    assert ss.dataset_id == dataset.id
    assert ss.embedding_model_id == model.id

    # SnippetSet parameters should match model's fixed parameters
    assert ss.window_size == model.window_size
    assert ss.step_size == model.step_size
    assert ss.overlap == model.overlap

    # SnippetSet status should be PENDING initially
    assert ss.status.name.lower() == "pending"
