from app.models.dataset import Dataset
from app.services.dataset_service import DatasetService


def test_checksum_is_generated(db_session, temp_data_root, tiny_wav_file):
    """
    Ensure DatasetService computes and stores a valid SHA-256 checksum
    when scanning recordings.
    """
    svc = DatasetService(db_session)

    # Create a folder inside our test data root
    folder = temp_data_root / "ds"
    folder.mkdir()

    # Create a tiny WAV file
    tiny_wav_file(folder / "x.wav")

    # Create dataset pointing to that folder
    ds = Dataset(
        team_id=None,
        name="C",
        description=None,
        source_uri="ds",       # matches folder name
    )
    db_session.add(ds)
    db_session.commit()

    # Scan for recordings
    recs = svc.scan_recordings(ds)

    # Assertions
    assert len(recs) == 1
    assert recs[0].audio_sha256 is not None
    assert len(recs[0].audio_sha256) == 64  # SHA-256 hex string
