import pytest
from app.models.dataset import Dataset
from app.models.recording import Recording
from app.services.dataset_service import DatasetService


def test_scan_recordings_creates_entries(db_session, temp_data_root, tiny_wav_file):
    svc = DatasetService(db_session)

    folder = temp_data_root / "myds"
    folder.mkdir()

    wav1 = tiny_wav_file(folder / "a.wav")
    wav2 = tiny_wav_file(folder / "b.wav")

    ds = Dataset(team_id=None, name="DS", description="d", source_uri="myds")
    db_session.add(ds)
    db_session.commit()

    new = svc.scan_recordings(ds)

    assert len(new) == 2
    paths = {rec.file_path for rec in new}
    assert str(wav1) in paths
    assert str(wav2) in paths


def test_scan_recordings_idempotent(db_session, temp_data_root, tiny_wav_file):
    svc = DatasetService(db_session)

    folder = temp_data_root / "ds2"
    folder.mkdir()
    tiny_wav_file(folder / "x.wav")

    ds = Dataset(team_id=None, name="DS2", description="d", source_uri="ds2")
    db_session.add(ds)
    db_session.commit()

    first = svc.scan_recordings(ds)
    second = svc.scan_recordings(ds)

    assert len(first) == 1
    assert len(second) == 0

    assert db_session.query(Recording).filter_by(dataset_id=ds.id).count() == 1


def test_scan_recordings_ignores_non_audio(db_session, temp_data_root, tiny_wav_file):
    svc = DatasetService(db_session)

    folder = temp_data_root / "ds3"
    folder.mkdir()
    tiny_wav_file(folder / "real.wav")
    (folder / "ignore.txt").write_text("hi")

    ds = Dataset(team_id=None, name="DS3", description="d", source_uri="ds3")
    db_session.add(ds)
    db_session.commit()

    recs = svc.scan_recordings(ds)
    assert len(recs) == 1
    assert recs[0].file_name == "real.wav"


def test_scan_invalid_path_raises(db_session):
    svc = DatasetService(db_session)

    ds = Dataset(team_id=None, name="Bad", description=None, source_uri="not_there")
    db_session.add(ds)
    db_session.commit()

    with pytest.raises(ValueError):
        svc.scan_recordings(ds)


def test_delete_dataset_cascades_recordings(db_session, temp_data_root, tiny_wav_file):
    svc = DatasetService(db_session)

    folder = temp_data_root / "dsX"
    folder.mkdir()
    tiny_wav_file(folder / "a.wav")

    ds = Dataset(team_id=None, name="X", description=None, source_uri="dsX")
    db_session.add(ds)
    db_session.commit()

    svc.scan_recordings(ds)

    svc.delete_dataset(ds)

    assert db_session.query(Dataset).count() == 0
    assert db_session.query(Recording).count() == 0
