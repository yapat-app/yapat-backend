import pytest

from app.services.embedding_service import EmbeddingService
from app.models.dataset import Dataset
from app.models.embedding import EmbeddingModel, EmbeddingJobStatus


def test_create_embedding_job_creates_snippet_config_and_job(db_session):
    service = EmbeddingService(db_session)

    # Create dataset
    dataset = Dataset(name="TestDS", source_uri="dummy")
    db_session.add(dataset)
    db_session.commit()

    # Create embedding model
    model = EmbeddingModel(
        name="birdnet",
        version="1.0",
        default_window_size=3.0,
        default_step_size=1.0,
        default_overlap=2.0,
    )
    db_session.add(model)
    db_session.commit()

    # Run service
    job = service.create_embedding_job(dataset, model)

    assert job.id is not None
    assert job.dataset_id == dataset.id
    assert job.embedding_model_id == model.id
    assert job.status == EmbeddingJobStatus.PENDING

    # SnippetConfig 1:1 relationship
    assert job.snippet_config is not None
    assert job.snippet_config.window_size == 3.0
    assert job.snippet_config.step_size == 1.0
    assert job.snippet_config.overlap == 2.0

    # Ensure it is persisted
    fetched = service.get_job(job.id)
    assert fetched.snippet_config is not None
