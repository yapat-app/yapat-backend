from app.models.dataset import Dataset
from app.tasks.processing_tasks import scan_dataset


def test_scan_dataset_task_creates_recordings(db_session, temp_data_root, tiny_wav_file):
    folder = temp_data_root / "dset1"
    folder.mkdir()
    tiny_wav_file(folder / "a.wav")

    ds = Dataset(team_id=None, name="D", description=None, source_uri="dset1")
    db_session.add(ds)
    db_session.commit()

    result = scan_dataset.apply(args=[ds.id]).get()

    assert result["status"] == "ok"
    assert result["recordings_created"] == 1
    assert result["dataset_id"] == ds.id
