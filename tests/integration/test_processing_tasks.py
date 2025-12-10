from app.models.dataset import Dataset
from app.models.embedding import SnippetSet
from app.tasks.processing_tasks import scan_dataset, generate_snippets


def test_scan_dataset_task(db_session, temp_data_root, tiny_wav_file):
    folder = temp_data_root / "dset"
    folder.mkdir()
    tiny_wav_file(folder / "a.wav")

    ds = Dataset(team_id=None, name="A", description=None, source_uri="dset")
    db_session.add(ds)
    db_session.commit()

    result = scan_dataset.apply(args=[ds.id]).get()

    assert result["status"] == "ok"
    assert result["recordings_created"] == 1


def test_snippet_generation_placeholder(db_session):
    ds = Dataset(team_id=None, name="C", description=None, source_uri="unused")
    db_session.add(ds)
    db_session.commit()

    # Provide dummy embedding_model_id=1 (a placeholder)
    ss = SnippetSet(
        dataset_id=ds.id,
        embedding_model_id=1,
        window_size=3.0,
        step_size=1.5,
        overlap=0.5,
    )

    db_session.add(ss)
    db_session.commit()

    result = generate_snippets.apply(args=[ds.id, ss.id]).get()
    assert result["status"] == "not_implemented"

