import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.main import app



@pytest.fixture
def client():
    """
    FastAPI test client for API integration tests.
    """
    return TestClient(app)


@pytest.fixture
def auth_headers(db_session):
    """
    Provide authentication headers for API tests.
    Creates a test user and returns headers with a valid JWT token.
    """
    from app.models.user import User, UserRole
    from app.core.security import create_access_token
    
    # Create test user if not exists
    test_user = db_session.query(User).filter_by(email="test@example.com").first()
    if not test_user:
        test_user = User(
            email="test@example.com",
            hashed_password="dummy_hash",
            role=UserRole.USER,
        )
        db_session.add(test_user)
        db_session.commit()
    
    # Generate access token
    access_token = create_access_token(data={"sub": test_user.email})
    
    return {"Authorization": f"Bearer {access_token}"}


@pytest.fixture(autouse=True, scope="session")
def celery_eager():
    """
    Run Celery tasks synchronously in tests.
    """
    from app.celery_app import celery_app

    celery_app.conf.task_always_eager = True
    celery_app.conf.task_eager_propagates = True

    yield

    # restore defaults (optional)
    celery_app.conf.task_always_eager = False
    celery_app.conf.task_eager_propagates = False


# ------------------------------------------------------
# Engine (shared across whole session)
# ------------------------------------------------------

@pytest.fixture(scope="session")
def engine():
    """
    A single in-memory SQLite engine for the entire test session.
    Celery eager tasks will open new DB sessions, so the engine must persist.
    """
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(engine, "connect")
    def enforce_fk(dbapi_conn, conn_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(bind=engine)

    return engine


# ------------------------------------------------------
# Patch SessionLocal so Celery tasks use test engine
# ------------------------------------------------------

@pytest.fixture(autouse=True)
def patch_sessionlocal(monkeypatch, engine):
    """
    Ensure Celery tasks use the SAME database engine as tests.
    Prevents creation of a second engine inside app.database.
    """
    SessionLocal = sessionmaker(bind=engine)

    monkeypatch.setattr("app.database.SessionLocal", SessionLocal)
    monkeypatch.setattr("app.tasks.processing_tasks.SessionLocal", SessionLocal)

    # If embedding_tasks or others touch the DB, patch them too:
    monkeypatch.setattr("app.tasks.embedding_tasks.SessionLocal", SessionLocal, raising=False)

    yield


# ------------------------------------------------------
# Session factory (fresh per test)
# ------------------------------------------------------

@pytest.fixture(scope="function")
def db_session(engine):
    """
    Each test gets a fresh session, but all share the same engine.
    """
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    try:
        yield session
    finally:
        session.close()


# ------------------------------------------------------
# Temporary dataset root
# ------------------------------------------------------

@pytest.fixture
def temp_data_root(monkeypatch):
    """
    Creates a temp directory and sets INTERNAL_DATA_ROOT to it.
    """
    tmp = tempfile.mkdtemp(prefix="yapat_data_")
    monkeypatch.setenv("INTERNAL_DATA_ROOT", tmp)

    yield Path(tmp)

    shutil.rmtree(tmp)


# ------------------------------------------------------
# Tiny WAV generator
# ------------------------------------------------------

@pytest.fixture
def tiny_wav_file():
    """
    Writes a minimal WAV file with float32 samples.
    """

    def _create(path: Path, duration_sec=0.1, sr=16000):
        samples = int(duration_sec * sr)
        data = np.zeros(samples, dtype=np.float32)
        sf.write(str(path), data, sr)
        return path

    return _create
