import pytest

from app.services.snippet_service import SnippetService
from app.models.dataset import Dataset
from app.models.recording import Recording
from app.models.snippet import Snippet
from app.models.annotation import Annotation
from app.models.embedding import EmbeddingModel, SnippetSet


def test_list_snippets_filters_correctly(db_session):
    # Dataset, model, snippet_set
    ds = Dataset(name="D", source_uri="dummy")
    model = EmbeddingModel(
        name="birdnet",
        version="2.4",
        window_size=3.0,
        step_size=3.0,
        overlap=0.0,
    )
    ss = SnippetSet(
        dataset_id=1,  # will be overwritten after commit
        embedding_model_id=1,
        window_size=3.0,
        step_size=3.0,
        overlap=0.0,
    )

    db_session.add_all([ds, model])
    db_session.commit()

    ss.dataset_id = ds.id
    ss.embedding_model_id = model.id
    db_session.add(ss)
    db_session.commit()

    # Recording
    rec = Recording(
        dataset_id=ds.id,
        file_path="dummy.wav",
        file_name="dummy.wav",
        duration=10.0,
        sample_rate=44100,
        extra_metadata=None,
        audio_sha256="x",
    )
    db_session.add(rec)
    db_session.commit()

    # Snippets
    s1 = Snippet(
        recording_id=rec.id,
        snippet_set_id=ss.id,
        start_time=0.0,
        duration=3.0,
    )
    s2 = Snippet(
        recording_id=rec.id,
        snippet_set_id=ss.id,
        start_time=3.0,
        duration=3.0,
    )
    db_session.add_all([s1, s2])
    db_session.commit()

    svc = SnippetService(db_session)
    results = svc.list_snippets(ds.id, ss.id)

    assert len(results) == 2
    assert results[0].start_time == 0.0


def test_annotation_count(db_session):
    # Setup
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

    ss = SnippetSet(
        dataset_id=ds.id,
        embedding_model_id=model.id,
        window_size=3.0,
        step_size=3.0,
        overlap=0.0,
    )
    db_session.add(ss)
    db_session.commit()

    rec = Recording(
        dataset_id=ds.id,
        file_path="xx.wav",
        file_name="xx.wav",
        duration=10.0,
        sample_rate=44100,
        extra_metadata=None,
        audio_sha256="y",
    )
    db_session.add(rec)
    db_session.commit()

    snip = Snippet(
        recording_id=rec.id,
        snippet_set_id=ss.id,
        start_time=1.0,
        duration=3.0,
    )
    db_session.add(snip)
    db_session.commit()

    ann1 = Annotation(snippet_id=snip.id, taxon_id="gbif:123")
    ann2 = Annotation(snippet_id=snip.id, taxon_id="gbif:456")
    db_session.add_all([ann1, ann2])
    db_session.commit()

    from app.services.snippet_service import SnippetService
    svc = SnippetService(db_session)

    assert svc.annotation_count(snip.id) == 2
