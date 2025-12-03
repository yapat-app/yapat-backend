import shutil
import tempfile
from pathlib import Path

import pytest
import soundfile as sf
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models.dataset import Dataset
from app.models.recording import Recording
from app.models.team import Team
from app.models.user import User, UserRole
from app.schemas.dataset import DatasetCreate
from app.services.dataset_service import DatasetService


# ------------------------------------------------------
# Test fixtures
# ------------------------------------------------------

@pytest.fixture(scope="session")
def engine():
    # SQLite in-memory DB for service-layer tests
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    return engine


@pytest.fixture(scope="function")
def db_session(engine):
    """Fresh DB session per test."""
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    yield session
    session.close()


@pytest.fixture
def temp_data_root(monkeypatch):
    """Create a temporary directory to act as DATA_ROOT."""
    tmp = tempfile.mkdtemp(prefix="yapat_data_")
    monkeypatch.setenv("INTERNAL_DATA_ROOT", tmp)
    yield Path(tmp)
    shutil.rmtree(tmp)


@pytest.fixture
def tiny_wav_file():
    """Return a helper that creates a minimal WAV file."""

    def _create(path: Path, duration_sec: float = 0.1, sr: int = 16000):
        samples = int(duration_sec * sr)
        data = [0.0] * samples
        sf.write(str(path), data, sr)
        return path

    return _create


# ------------------------------------------------------
# Tests
# ------------------------------------------------------

def test_create_dataset(db_session):
    svc = DatasetService(db_session)

    user = User(
        username="test@example.com",
        hashed_password="x",
        role=UserRole.ADMIN
    )
    team = Team(name="Team A")
    db_session.add_all([user, team])
    db_session.commit()

    ds_in = DatasetCreate(
        team_id=team.id,
        name="MyDataset",
        description="Test dataset",
        source_uri="ds1"
    )

    dataset = svc.create_dataset(ds_in, user)

    assert dataset.id is not None
    assert dataset.name == "MyDataset"
    assert dataset.source_uri == "ds1"


def test_scan_recordings_creates_entries(db_session, temp_data_root, tiny_wav_file):
    svc = DatasetService(db_session)

    # Prepare dataset and directory
    (temp_data_root / "myds").mkdir()
    wav1 = tiny_wav_file(temp_data_root / "myds" / "a.wav")
    wav2 = tiny_wav_file(temp_data_root / "myds" / "b.wav")

    dataset = Dataset(
        team_id=None,
        name="DS",
        description="d",
        source_uri="myds"
    )
    db_session.add(dataset)
    db_session.commit()
    db_session.refresh(dataset)

    new_recs = svc.scan_recordings(dataset)

    assert len(new_recs) == 2
    paths = {rec.file_path for rec in new_recs}
    assert str(wav1) in paths
    assert str(wav2) in paths


def test_scan_recordings_idempotency(db_session, temp_data_root, tiny_wav_file):
    svc = DatasetService(db_session)

    p = temp_data_root / "ds2"
    p.mkdir()
    wav = tiny_wav_file(p / "x.wav")

    dataset = Dataset(
        team_id=None,
        name="DS2",
        description="d",
        source_uri="ds2"
    )
    db_session.add(dataset)
    db_session.commit()
    db_session.refresh(dataset)

    # First scan → 1 new recording
    recs1 = svc.scan_recordings(dataset)
    assert len(recs1) == 1

    # Second scan → 0 new recordings
    recs2 = svc.scan_recordings(dataset)
    assert len(recs2) == 0

    total_recs = db_session.query(Recording).filter_by(dataset_id=dataset.id).count()
    assert total_recs == 1


def test_scan_recordings_ignores_non_audio(db_session, temp_data_root, tiny_wav_file):
    svc = DatasetService(db_session)

    d = temp_data_root / "ds3"
    d.mkdir()
    tiny_wav_file(d / "real.wav")
    (d / "not_audio.txt").write_text("hello world")

    dataset = Dataset(
        team_id=None,
        name="DS3",
        description="d",
        source_uri="ds3"
    )
    db_session.add(dataset)
    db_session.commit()
    db_session.refresh(dataset)

    recs = svc.scan_recordings(dataset)
    assert len(recs) == 1
    assert recs[0].file_name == "real.wav"


def test_scan_invalid_path_raises(db_session, temp_data_root):
    svc = DatasetService(db_session)

    dataset = Dataset(
        team_id=None,
        name="Bad",
        description="d",
        source_uri="does_not_exist"
    )
    db_session.add(dataset)
    db_session.commit()

    with pytest.raises(ValueError):
        svc.scan_recordings(dataset)


def test_create_dataset_duplicate_raises(db_session):
    svc = DatasetService(db_session)

    user = User(username="u", hashed_password="x", role=UserRole.ADMIN)
    team = Team(name="T")
    db_session.add(user)
    db_session.commit()

    ds_in = DatasetCreate(
        team_id=team.id,
        name="D1",
        description=None,
        source_uri="sameuri"
    )

    ds1 = svc.create_dataset(ds_in, user)
    assert ds1 is not None

    # Creating the same dataset a second time → duplicate
    with pytest.raises(ValueError) as exc:
        svc.create_dataset(ds_in, user)

    assert str(exc.value) == "duplicate_dataset"


def test_nonadmin_requires_team_id(db_session):
    svc = DatasetService(db_session)

    user = User(username="u", hashed_password="x", role=UserRole.USER)
    db_session.add(user)
    db_session.commit()

    ds_in = DatasetCreate(
        team_id=None,
        name="D1",
        description=None,
        source_uri="x"
    )

    with pytest.raises(ValueError):  # service raises cleanly
        svc.create_dataset(ds_in, user)


def test_claim_admin_dataset(db_session):
    svc = DatasetService(db_session)

    # Admin-created dataset → team_id = None
    admin = User(username="adm", hashed_password="x", role=UserRole.ADMIN)
    user = User(username="u", hashed_password="x", role=UserRole.USER, team_id=10)

    db_session.add_all([admin, user])
    db_session.commit()

    ds_in = DatasetCreate(
        team_id=None,
        name="D",
        description=None,
        source_uri="claimtest"
    )
    ds = svc.create_dataset(ds_in, admin)
    assert ds.team_id is None

    claimed = svc.claim_dataset(ds, user)
    assert claimed.team_id == user.team_id


def test_claim_dataset_already_owned_fails(db_session):
    svc = DatasetService(db_session)

    admin = User(username="adm", hashed_password="x", role=UserRole.ADMIN)
    owner = User(username="own", hashed_password="x", role=UserRole.USER, team_id=1)
    other = User(username="o", hashed_password="x", role=UserRole.USER, team_id=2)

    db_session.add_all([admin, owner, other])
    db_session.commit()

    ds_in = DatasetCreate(
        team_id=owner.team_id,
        name="D",
        description=None,
        source_uri="owned"
    )
    ds = svc.create_dataset(ds_in, admin)
    assert ds.team_id == owner.team_id

    with pytest.raises(ValueError):
        svc.claim_dataset(ds, other)


def test_delete_dataset_cascades_recordings(db_session, temp_data_root, tiny_wav_file):
    svc = DatasetService(db_session)

    # Create dataset + recordings
    p = temp_data_root / "dsX"
    p.mkdir()
    tiny_wav_file(p / "a.wav")

    ds = Dataset(team_id=None, name="X", description=None, source_uri="dsX")
    db_session.add(ds)
    db_session.commit()

    recs = svc.scan_recordings(ds)
    assert len(recs) == 1

    # Delete dataset
    svc.delete_dataset(ds)

    # All gone
    assert db_session.query(Dataset).count() == 0
    assert db_session.query(Recording).count() == 0


