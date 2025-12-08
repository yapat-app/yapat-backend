import shutil
import tempfile
from pathlib import Path
import pytest
import numpy as np
import soundfile as sf

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from app.database import Base


# ------------------------------------------------------
# Engine (shared across tests)
# ------------------------------------------------------

@pytest.fixture(scope="session")
def engine():
    """
    A single in-memory SQLite engine shared across the entire test session.
    Celery tasks open new sessions, so the engine must persist for the full run.
    """
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )

    # Enforce SQLite foreign key constraints
    @event.listens_for(engine, "connect")
    def enforce_fk(dbapi_conn, conn_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    # Create all tables on this shared engine
    Base.metadata.create_all(bind=engine)

    return engine


# ------------------------------------------------------
# Session (per test)
# ------------------------------------------------------

@pytest.fixture(scope="function")
def db_session(engine):
    """
    Each test gets a fresh session.
    The engine itself persists across tests so Celery tasks can connect.
    """
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    try:
        yield session
    finally:
        session.close()


# ------------------------------------------------------
# Temporary test data directory
# ------------------------------------------------------

@pytest.fixture
def temp_data_root(monkeypatch):
    """
    Creates a temp folder and sets INTERNAL_DATA_ROOT to it.
    Cleaned after test completion.
    """
    tmp = tempfile.mkdtemp(prefix="yapat_data_")
    monkeypatch.setenv("INTERNAL_DATA_ROOT", tmp)

    yield Path(tmp)

    shutil.rmtree(tmp)


# ------------------------------------------------------
# Utility: create small WAV files
# ------------------------------------------------------

@pytest.fixture
def tiny_wav_file():
    """
    Writes a minimal WAV file using float32 NumPy array (safe for soundfile).
    """

    def _make(path: Path, duration_sec=0.1, sr=16000):
        samples = int(duration_sec * sr)
        data = np.zeros(samples, dtype=np.float32)
        sf.write(str(path), data, sr)
        return path

    return _make
