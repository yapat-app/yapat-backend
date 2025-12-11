import pytest

from app.models.dataset import Dataset
from app.models.embedding import EmbeddingModel, EmbeddingJob, SnippetSet
from app.models.embedding import EmbeddingVector
from app.models.recording import Recording
from app.models.snippet import Snippet
from app.tasks.embedding_tasks import generate_embedding_for_snippet


@pytest.fixture
def fk_graph(db_session):
    """Build the minimal FK graph required by the embedding task."""
    # Dataset
    dataset = Dataset(name="ds", description=None, source_uri="test://")
    db_session.add(dataset)
    db_session.commit()

    # Recording
    rec = Recording(
        dataset_id=dataset.id,
        file_path="/tmp/a.wav",
        file_name="a.wav",
        duration=10.0,
        sample_rate=44100,
    )
    db_session.add(rec)
    db_session.commit()

    # EmbeddingModel
    model = EmbeddingModel(
        name="model",
        version="1",
        description=None,
        source_uri=None,
        window_size=1.0,
        step_size=0.5,
        overlap=0.0,
    )
    db_session.add(model)
    db_session.commit()

    # SnippetSet
    ss = SnippetSet(
        dataset_id=dataset.id,
        embedding_model_id=model.id,
        window_size=1.0,
        step_size=0.5,
        overlap=0.0,
        status="pending",
    )
    db_session.add(ss)
    db_session.commit()

    # Snippet
    snip = Snippet(
        recording_id=rec.id,
        snippet_set_id=ss.id,
        start_time=0.0,
        end_time=1.0,
        duration=1.0,
    )
    db_session.add(snip)
    db_session.commit()

    # Job
    job = EmbeddingJob(
        dataset_id=dataset.id,
        embedding_model_id=model.id,
        snippet_set_id=ss.id,
        status="running",
    )
    db_session.add(job)
    db_session.commit()

    return snip, job, model


def test_embedding_task_inserts_vector(db_session, monkeypatch, fk_graph):
    snip, job, model = fk_graph

    # Monkeypatch EmbeddingService.get_model to avoid failures
    class FakeEmbeddingService:
        def __init__(self, db): pass

        def get_model(self, model_id):
            return model  # return the existing model

    # Monkeypatch the module's EmbeddingService
    monkeypatch.setattr(
        "app.tasks.embedding_tasks.EmbeddingService",
        FakeEmbeddingService
    )

    # Monkeypatch the dummy inference
    monkeypatch.setattr(
        "app.tasks.embedding_tasks.dummy_vector",
        [0.9, 0.1, -0.5],  # optional, but illustrates control
        raising=False
    )

    # Invoke the task **directly**
    result = generate_embedding_for_snippet.run(
        snip.id,
        model.id,
    )

    assert result["status"] == "success"

    ev = db_session.query(EmbeddingVector).first()
    assert ev is not None
    assert ev.snippet_id == snip.id
    assert ev.embedding_model_id == model.id
    assert ev.embedding_job_id == job.id
    assert ev.dim == 3


def test_birdnet_task_inserts_vector(db_session, monkeypatch, fk_graph):
    snip, job, model = fk_graph

    # Mock BirdNETEmbedder.embed to return a known vector
    fake_vec = [0.9, -0.1, 0.3]

    monkeypatch.setattr(
        "app.tasks.embedding_tasks.BirdNETEmbedder.embed",
        lambda audio_path, start_time, end_time: fake_vec
    )

    # Monkeypatch EmbeddingService.get_model to avoid DB lookup issues
    class FakeEmbeddingService:
        def __init__(self, db): pass

        def get_model(self, model_id): return model

    monkeypatch.setattr(
        "app.tasks.embedding_tasks.EmbeddingService",
        FakeEmbeddingService
    )

    from app.tasks.embedding_tasks import generate_embedding_for_snippet

    result = generate_embedding_for_snippet(
        None, snip.id, model.id
    )

    assert result["status"] == "success"

    # Verify vector is stored
    from app.models.embedding import EmbeddingVector
    ev = db_session.query(EmbeddingVector).filter_by(snippet_id=snip.id).first()

    assert ev is not None
    assert ev.vector == fake_vec
    assert ev.dim == len(fake_vec)


def test_run_embedding_uses_birdnet(db_session, monkeypatch, fk_graph):
    snip, job, model = fk_graph

    # Make 1-second recording duration so segmentation produces exactly one snippet
    rec = snip.recording
    rec.duration = 1.0
    db_session.commit()

    # Mock BirdNET output
    monkeypatch.setattr(
        "app.tasks.embedding_tasks.BirdNETEmbedder.embed",
        lambda *args, **kwargs: [1.0, 2.0, 3.0]
    )

    # Patch EmbeddingService so we don't use real DB calls
    class FakeES:
        def __init__(self, db): pass

        def get_model(self, _): return model

        def update_job_status(self, *args, **kwargs): pass

    monkeypatch.setattr(
        "app.tasks.embedding_tasks.EmbeddingService",
        FakeES
    )

    from app.tasks.embedding_tasks import run_embedding

    result = run_embedding(None, job.id)

    assert result["failed"] == 0

    from app.models.embedding import EmbeddingVector
    ev = db_session.query(EmbeddingVector).first()

    assert ev.vector == [1.0, 2.0, 3.0]
