from fastapi.testclient import TestClient

from app.services.embedding_service import EmbeddingService
from app.models.dataset import Dataset
from app.models.embedding import EmbeddingModel, EmbeddingJobStatus


def test_create_embedding_job_api(client: TestClient, db_session):
    # 1. Create dataset in DB
    dataset = Dataset(name="DS1", source_uri="uri")
    db_session.add(dataset)
    db_session.commit()

    # 2. Create embedding model in DB
    model = EmbeddingModel(
        name="birdnet",
        version="1.0",
        default_window_size=3.0,
        default_step_size=1.0,
        default_overlap=2.0,
    )
    db_session.add(model)
    db_session.commit()

    # 3. Call API
    payload = {
        "embedding_model_id": model.id
    }

    resp = client.post(f"/api/v1/datasets/{dataset.id}/embeddings", json=payload)
    assert resp.status_code == 200, resp.text

    data = resp.json()

    assert "embedding_job_id" in data
    assert "snippet_config_id" in data
    assert data["model_id"] == model.id
    assert data["status"] == "pending"

    # 4. Verify in DB
    service = EmbeddingService(db_session)
    job = service.get_job(data["embedding_job_id"])

    assert job.embedding_model_id == model.id
    assert job.snippet_config.id == data["snippet_config_id"]

    # Celery eager means task should have run immediately
    assert job.status in (EmbeddingJobStatus.COMPLETED, EmbeddingJobStatus.FAILED)
