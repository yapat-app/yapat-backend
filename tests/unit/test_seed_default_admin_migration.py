import importlib.util
from pathlib import Path

import pytest

from app.config import Settings
from app.core.security import verify_password
from app.models.user import User, UserRole

MIGRATION_PATH = (
    Path(__file__).resolve().parents[2] / "alembic" / "versions" / "2026_10_07_seed_default_admin.py"
)

DEFAULT_USER = "admin"
DEFAULT_PASSWORD = "admin12345"


@pytest.fixture(scope="module")
def migration():
    # The filename starts with a digit, so it cannot be imported normally.
    spec = importlib.util.spec_from_file_location("seed_default_admin_migration", MIGRATION_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _seed(engine, migration, username=DEFAULT_USER, password=DEFAULT_PASSWORD):
    with engine.begin() as conn:
        return migration.seed_default_admin(conn, username, password)


def test_revision_chain(migration):
    assert migration.revision == "2026_10_07_seed_default_admin"
    assert migration.down_revision == "2026_09_22_al_pred_composite_idx"


def test_defaults_are_admin_admin12345():
    s = Settings(_env_file=None)
    assert (s.DEFAULT_ADMIN_USERNAME, s.DEFAULT_ADMIN_PASSWORD) == (DEFAULT_USER, DEFAULT_PASSWORD)


def test_fresh_database_creates_admin(engine, db_session, migration, capsys):
    assert _seed(engine, migration) == "created"

    user = db_session.query(User).filter_by(username="admin").one()
    assert user.role == UserRole.ADMIN
    assert user.is_active is True
    assert verify_password(DEFAULT_PASSWORD, user.hashed_password)  # can log in right away

    out = capsys.readouterr().out
    assert "Default administrator account created successfully" in out
    assert "Username: admin" in out
    assert "Password: admin12345" in out
    assert "Role:     ADMIN" in out
    assert "change the password after your first login" in out


def test_password_is_hashed_not_plaintext(engine, db_session, migration):
    _seed(engine, migration)

    user = db_session.query(User).filter_by(username="admin").one()
    assert user.hashed_password != DEFAULT_PASSWORD
    assert DEFAULT_PASSWORD not in user.hashed_password
    assert user.hashed_password.startswith("$2")  # bcrypt


def test_existing_admin_is_unchanged(engine, db_session, migration, capsys):
    db_session.add(User(username="root", hashed_password="root-hash", role=UserRole.ADMIN))
    db_session.commit()

    assert _seed(engine, migration) == "exists"

    db_session.expire_all()
    assert db_session.query(User).count() == 1
    root = db_session.query(User).one()
    assert (root.username, root.role, root.hashed_password) == ("root", UserRole.ADMIN, "root-hash")

    out = capsys.readouterr().out
    assert "Existing administrator account found" in out
    assert "No changes were made to existing accounts" in out
    assert "Password:" not in out


def test_existing_admin_named_admin_keeps_its_password(engine, db_session, migration):
    db_session.add(User(username="admin", hashed_password="their-hash", role=UserRole.ADMIN))
    db_session.commit()

    assert _seed(engine, migration) == "exists"

    db_session.expire_all()
    admin = db_session.query(User).one()
    assert admin.hashed_password == "their-hash"  # not reset to admin12345


def test_existing_users_without_admin_creates_admin(engine, db_session, migration, capsys):
    db_session.add_all([
        User(username="alice", hashed_password="a-hash", role=UserRole.USER),
        User(username="bob", hashed_password="b-hash", role=UserRole.TEAM_OWNER),
    ])
    db_session.commit()

    assert _seed(engine, migration) == "created"

    db_session.expire_all()
    assert db_session.query(User).count() == 3
    assert db_session.query(User).filter_by(username="admin").one().role == UserRole.ADMIN
    # Pre-existing accounts are untouched.
    alice = db_session.query(User).filter_by(username="alice").one()
    assert (alice.role, alice.hashed_password) == (UserRole.USER, "a-hash")
    assert "Default administrator account created successfully" in capsys.readouterr().out


def test_non_admin_named_admin_is_untouched_and_warned(engine, db_session, migration, capsys):
    db_session.add(User(username="admin", hashed_password="orig-hash", role=UserRole.USER))
    db_session.commit()

    assert _seed(engine, migration) == "username_taken"

    db_session.expire_all()
    assert db_session.query(User).count() == 1
    user = db_session.query(User).one()
    assert user.role == UserRole.USER
    assert user.hashed_password == "orig-hash"

    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "'admin' is already in use" in out
    assert "was NOT modified" in out
    assert "Password:" not in out


def test_repeated_runs_do_not_duplicate(engine, db_session, migration):
    assert _seed(engine, migration) == "created"
    assert _seed(engine, migration) == "exists"
    assert _seed(engine, migration) == "exists"

    assert db_session.query(User).filter(User.role == UserRole.ADMIN).count() == 1
    assert db_session.query(User).count() == 1


def test_configured_credentials_are_respected(engine, db_session, migration, capsys):
    assert _seed(engine, migration, "boss", "my-own-pass-1") == "created"

    assert db_session.query(User).filter_by(username="admin").count() == 0
    boss = db_session.query(User).filter_by(username="boss").one()
    assert boss.role == UserRole.ADMIN
    assert verify_password("my-own-pass-1", boss.hashed_password)
    assert not verify_password(DEFAULT_PASSWORD, boss.hashed_password)

    out = capsys.readouterr().out
    assert "Username: boss" in out
    # A custom password is a secret: never written to the logs.
    assert "my-own-pass-1" not in out
    assert "Password: configured via DEFAULT_ADMIN_PASSWORD" in out


def test_upgrade_reads_credentials_from_settings(engine, db_session, migration, monkeypatch):
    # upgrade() must pass the (env-driven) settings values through.
    monkeypatch.setattr(migration.settings, "DEFAULT_ADMIN_USERNAME", "envadmin")
    monkeypatch.setattr(migration.settings, "DEFAULT_ADMIN_PASSWORD", "from-env-pass-1")
    monkeypatch.setattr(migration.op, "get_bind", lambda: conn)

    with engine.begin() as conn:
        migration.upgrade()

    user = db_session.query(User).filter_by(username="envadmin").one()
    assert user.role == UserRole.ADMIN
    assert verify_password("from-env-pass-1", user.hashed_password)


def test_downgrade_keeps_seeded_admin(engine, db_session, migration):
    _seed(engine, migration)
    migration.downgrade()
    assert db_session.query(User).filter_by(username="admin").count() == 1


@pytest.mark.parametrize("bad_password", ["short", "x" * 73, "é" * 37, ""])
def test_invalid_configured_password_fails(engine, db_session, migration, bad_password):
    with pytest.raises(ValueError):
        _seed(engine, migration, "admin", bad_password)
    assert db_session.query(User).count() == 0


def test_invalid_password_ignored_when_nothing_to_create(engine, db_session, migration):
    db_session.add(User(username="root", hashed_password="h", role=UserRole.ADMIN))
    db_session.commit()
    assert _seed(engine, migration, "admin", "short") == "exists"


def test_empty_username_fails(engine, migration):
    with pytest.raises(ValueError):
        _seed(engine, migration, "", DEFAULT_PASSWORD)
