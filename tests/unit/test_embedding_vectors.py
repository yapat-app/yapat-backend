import math

from app.models.dataset import Dataset
from app.models.embedding import EmbeddingJob, EmbeddingModel, SnippetSet
from app.models.embedding import EmbeddingVector
from app.models.recording import Recording
from app.models.snippet import Snippet
from app.services.embedding_service import VectorStore


def test_embedding_vector_basic(db_session):
    # Dataset
    dataset = Dataset(
        name="ds1",
        description=None,
        source_uri="test://",
    )
    db_session.add(dataset)
    db_session.commit()

    # Recording (required for Snippet)
    rec = Recording(
        dataset_id=dataset.id,
        file_path="/tmp/test.wav",
        file_name="test.wav",
        duration=10.0,
        sample_rate=44100,
    )
    db_session.add(rec)
    db_session.commit()

    # Embedding model
    model = EmbeddingModel(
        name="m",
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

    # Embedding job
    job = EmbeddingJob(
        dataset_id=dataset.id,
        embedding_model_id=model.id,
        snippet_set_id=ss.id,
        status="pending",
    )
    db_session.add(job)
    db_session.commit()

    # Embedding vector
    ev = EmbeddingVector(
        snippet_id=snip.id,
        embedding_job_id=job.id,
        embedding_model_id=model.id,
        dim=3,
        vector=[0.1, 0.2, 0.3],
    )
    db_session.add(ev)
    db_session.commit()

    loaded = db_session.query(EmbeddingVector).first()
    assert loaded.dim == 3
    assert loaded.vector == [0.1, 0.2, 0.3]


def test_vector_store_insert(db_session):
    # --- Create FK parents (minimal valid set) ---

    # Dataset
    dataset = Dataset(
        name="ds1",
        description=None,
        source_uri="test://",
    )
    db_session.add(dataset)
    db_session.commit()

    # Recording
    rec = Recording(
        dataset_id=dataset.id,
        file_path="/tmp/test.wav",
        file_name="test.wav",
        duration=10.0,
        sample_rate=44100,
    )
    db_session.add(rec)
    db_session.commit()

    # Embedding model
    model = EmbeddingModel(
        name="m",
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

    # Embedding job
    job = EmbeddingJob(
        dataset_id=dataset.id,
        embedding_model_id=model.id,
        snippet_set_id=ss.id,
        status="pending",
    )
    db_session.add(job)
    db_session.commit()

    # --- Test VectorStore.insert ---
    store = VectorStore(db_session)

    vector = [0.1, 0.2, 0.3]
    store.insert(
        snippet_id=snip.id,
        job_id=job.id,
        model_id=model.id,
        vector=vector,
    )

    # --- Assertions ---
    ev = db_session.query(EmbeddingVector).first()
    assert ev is not None
    assert ev.snippet_id == snip.id
    assert ev.embedding_job_id == job.id
    assert ev.embedding_model_id == model.id
    assert ev.dim == 3
    assert ev.vector == vector


def create_fk_graph(db):
    """Helper to create the minimum valid object graph."""
    dataset = Dataset(name="ds1", description=None, source_uri="x")
    db.add(dataset)
    db.commit()

    rec = Recording(
        dataset_id=dataset.id,
        file_path="/tmp/a.wav",
        file_name="a.wav",
        duration=10.0,
        sample_rate=44100,
    )
    db.add(rec)
    db.commit()

    model = EmbeddingModel(
        name="m",
        version="1",
        description=None,
        source_uri=None,
        window_size=1.0,
        step_size=0.5,
        overlap=0.0,
    )
    db.add(model)
    db.commit()

    ss = SnippetSet(
        dataset_id=dataset.id,
        embedding_model_id=model.id,
        window_size=1.0,
        step_size=0.5,
        overlap=0.0,
        status="pending",
    )
    db.add(ss)
    db.commit()

    snip = Snippet(
        recording_id=rec.id,
        snippet_set_id=ss.id,
        start_time=0.0,
        end_time=1.0,
        duration=1.0,
    )
    db.add(snip)
    db.commit()

    job = EmbeddingJob(
        dataset_id=dataset.id,
        embedding_model_id=model.id,
        snippet_set_id=ss.id,
        status="pending",
    )
    db.add(job)
    db.commit()

    return snip, job, model


def test_vector_store_get(db_session):
    snip, job, model = create_fk_graph(db_session)

    store = VectorStore(db_session)
    vec = [0.1, 0.2, 0.3]

    store.insert(
        snippet_id=snip.id,
        job_id=job.id,
        model_id=model.id,
        vector=vec,
    )

    row = store.get(snippet_id=snip.id, model_id=model.id)

    assert row is not None
    assert row.vector == vec
    assert row.dim == 3
    assert row.snippet_id == snip.id
    assert row.embedding_model_id == model.id
    assert row.embedding_job_id == job.id


def make_graph(db):
    """Helper to create minimal FK graph once."""
    dataset = Dataset(name="ds1", description=None, source_uri="x")
    db.add(dataset)
    db.commit()

    rec = Recording(
        dataset_id=dataset.id,
        file_path="/tmp/a.wav",
        file_name="a.wav",
        duration=10.0,
        sample_rate=44100,
    )
    db.add(rec)
    db.commit()

    model = EmbeddingModel(
        name="m",
        version="1",
        source_uri=None,
        description=None,
        window_size=1.0,
        step_size=0.5,
        overlap=0.0,
    )
    db.add(model)
    db.commit()

    ss = SnippetSet(
        dataset_id=dataset.id,
        embedding_model_id=model.id,
        window_size=1.0,
        step_size=0.5,
        overlap=0.0,
        status="pending",
    )
    db.add(ss)
    db.commit()

    # Make a few snippets
    def add_snip(start, end):
        s = Snippet(
            recording_id=rec.id,
            snippet_set_id=ss.id,
            start_time=start,
            end_time=end,
            duration=end - start,
        )
        db.add(s)
        db.commit()
        return s

    snip1 = add_snip(0, 1)
    snip2 = add_snip(1, 2)
    snip3 = add_snip(2, 3)

    # Single embedding job
    job = EmbeddingJob(
        dataset_id=dataset.id,
        embedding_model_id=model.id,
        snippet_set_id=ss.id,
        status="pending",
    )
    db.add(job)
    db.commit()

    return (model, job, [snip1, snip2, snip3])


def test_vector_store_search(db_session):
    model, job, [s1, s2, s3] = make_graph(db_session)

    store = VectorStore(db_session)

    # Insert vectors
    store.insert(s1.id, job.id, model.id, [1.0, 0.0])  # aligned with query
    store.insert(s2.id, job.id, model.id, [0.0, 1.0])  # orthogonal
    store.insert(s3.id, job.id, model.id, [-1.0, 0.0])  # opposite

    # Query: aligned with s1
    results = store.search(model_id=model.id, query_vector=[1.0, 0.0], k=3)

    # Expect ordering: s1 (1.0), s2 (~0.0), s3 (-1.0)
    assert len(results) == 3

    ids = [r[0] for r in results]
    scores = [r[1] for r in results]

    assert ids[0] == s1.id
    assert math.isclose(scores[0], 1.0, rel_tol=1e-6)

    assert ids[1] == s2.id
    assert abs(scores[1]) < 1e-6  # approx 0

    assert ids[2] == s3.id
    assert math.isclose(scores[2], -1.0, rel_tol=1e-6)
