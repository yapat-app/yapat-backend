import importlib.util
from pathlib import Path

import pytest
from sqlalchemy import select

from app.core.security import verify_password
from app.models.user import User, UserRole

_MIGRATION = Path(__file__).parents[2] / "alembic" / "versions" / "2026_10_06_seed_default_admin.py"
_spec = importlib.util.spec_from_file_location("seed_default_admin_migration", _MIGRATION)
migration = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(migration)


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    monkeypatch.delenv("DEFAULT_ADMIN_USERNAME", raising=False)
    monkeypatch.delenv("DEFAULT_ADMIN_PASSWORD", raising=False)


def _seed(engine):
    with engine.begin() as conn:
        migration.seed_default_admin(conn)


def test_creates_default_admin_on_empty_db(engine, db_session):
    _seed(engine)

    user = db_session.execute(select(User)).scalar_one()
    assert user.username == "admin"
    assert user.role == UserRole.ADMIN
    assert user.is_active is True
    assert verify_password("admin12345", user.hashed_password)


def test_does_not_duplicate_when_admin_exists(engine, db_session):
    db_session.add(User(username="existing", hashed_password="x", role=UserRole.ADMIN))
    db_session.commit()

    _seed(engine)

    assert [u.username for u in db_session.execute(select(User)).scalars()] == ["existing"]


def test_is_idempotent(engine, db_session):
    _seed(engine)
    _seed(engine)

    assert len(db_session.execute(select(User)).scalars().all()) == 1


def test_uses_env_overrides(engine, db_session, monkeypatch):
    monkeypatch.setenv("DEFAULT_ADMIN_USERNAME", "root")
    monkeypatch.setenv("DEFAULT_ADMIN_PASSWORD", "s3cret-pass-1")

    _seed(engine)

    user = db_session.execute(select(User)).scalar_one()
    assert user.username == "root"
    assert verify_password("s3cret-pass-1", user.hashed_password)


def test_skips_when_username_taken_by_non_admin(engine, db_session):
    db_session.add(User(username="admin", hashed_password="x", role=UserRole.USER))
    db_session.commit()

    _seed(engine)

    user = db_session.execute(select(User)).scalar_one()
    assert user.role == UserRole.USER
