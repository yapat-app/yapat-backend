import shutil
import tempfile
from pathlib import Path
import pytest
import soundfile as sf

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from app.database import Base


from app.database import Base

@pytest.fixture(autouse=True)
def reset_db(engine):
    # datasets <-> snippet_sets is a foreign-key cycle, so SQLAlchemy cannot
    # topologically sort the DROPs and emits them in an order that trips the
    # FK enforcement the engine fixture switches on. Disable enforcement for
    # the teardown only -- leaving it on would fail every test at setup.
    with engine.connect() as conn:
        conn.exec_driver_sql("PRAGMA foreign_keys=OFF")
        Base.metadata.drop_all(bind=conn)
        conn.exec_driver_sql("PRAGMA foreign_keys=ON")
        conn.commit()
    Base.metadata.create_all(bind=engine)
    yield


# ------------------------------------------------------
# Shared test database (SQLite in-memory)
# ------------------------------------------------------

@pytest.fixture(scope="session")
def engine():
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


@pytest.fixture(scope="function")
def db_session(engine):
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    yield session
    session.close()


# ------------------------------------------------------
# Temporary data directory
# ------------------------------------------------------

@pytest.fixture
def temp_data_root(monkeypatch):
    tmp = tempfile.mkdtemp(prefix="yapat_data_")
    monkeypatch.setenv("INTERNAL_DATA_ROOT", tmp)
    yield Path(tmp)
    shutil.rmtree(tmp)


# ------------------------------------------------------
# Tiny WAV generator
# ------------------------------------------------------

@pytest.fixture
def tiny_wav_file():
    def _make(path: Path, duration_sec=0.1, sr=16000):
        samples = int(duration_sec * sr)
        sf.write(str(path), [0.0] * samples, sr)
        return path
    return _make
