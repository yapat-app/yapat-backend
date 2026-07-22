"""Unit tests for admin-registration bootstrap gating (app/api/auth.py).

Self-service admin registration (POST /api/auth/register with role="admin")
must only work to create the very first admin on a fresh instance -- once
one exists, further attempts must be rejected server-side (not just hidden
in the frontend), and GET /api/auth/admin-exists must reflect that for the
frontend to hide the option proactively.

Once that bootstrap admin exists, further admin accounts are created via
POST /api/auth/admin/create-user, which requires an authenticated admin
caller (app/api/deps.get_current_admin_user) and is exempt from the
single-admin gate above -- an authenticated admin is the trusted party the
gate exists to require in the first place.

GET /api/auth/admin/users (also admin-only) backs the "manage users" screen
that lists existing accounts before creating a new one -- there's otherwise
no way to enumerate users at all.

Uses an isolated in-memory SQLite DB (StaticPool -- a single shared
connection, since sqlite:///:memory: otherwise gives each new connection its
own empty DB) with only the users table created, since User has no foreign
keys of its own. Overrides conftest.py's autouse reset_db (which does an
unfiltered Base.metadata.create_all() against the full app schema) the same
way tests/unit/test_visualisation_service_fpv.py does, so this file doesn't
depend on the full-metadata FK-drop-ordering issue being fixed.
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base
from app.models.user import User


@pytest.fixture(autouse=True)
def reset_db():
    yield


@pytest.fixture()
def client():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine, tables=[User.__table__])
    SessionLocal = sessionmaker(bind=engine)

    # Deliberately not importing app.main: the full app registers routers
    # (pam_active_learning, embeddings, ...) that transitively import torch,
    # which isn't installed in this sandbox. app/api/auth.py itself has no
    # torch dependency, so mounting just its router avoids that chain
    # entirely -- this test is about the auth endpoints, not the full app.
    from app.api import auth
    from app.api.deps import get_db

    app = FastAPI()
    app.include_router(auth.router, prefix="/api/auth")

    def override_get_db():
        session = SessionLocal()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_db, None)


def _register(client, username, role="user"):
    return client.post(
        "/api/auth/register",
        json={"username": username, "password": "a-password-123", "role": role},
    )


def _login(client, username):
    resp = client.post(
        "/api/auth/login",
        json={"username": username, "password": "a-password-123"},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


def _auth_headers(token):
    return {"Authorization": f"Bearer {token}"}


def _admin_create_user(client, token, username, role="admin"):
    return client.post(
        "/api/auth/admin/create-user",
        json={"username": username, "password": "a-password-123", "role": role},
        headers=_auth_headers(token),
    )


class TestAdminExistsEndpoint:
    def test_false_on_fresh_instance(self, client):
        resp = client.get("/api/auth/admin-exists")
        assert resp.status_code == 200
        assert resp.json() == {"admin_exists": False}

    def test_true_once_an_admin_registers(self, client):
        _register(client, "first-admin", role="admin")
        resp = client.get("/api/auth/admin-exists")
        assert resp.json() == {"admin_exists": True}


class TestAdminRegistrationGating:
    def test_first_admin_registration_succeeds(self, client):
        resp = _register(client, "first-admin", role="admin")
        assert resp.status_code == 201
        assert resp.json()["role"] == "admin"

    def test_second_admin_registration_is_rejected(self, client):
        first = _register(client, "first-admin", role="admin")
        assert first.status_code == 201

        second = _register(client, "second-admin", role="admin")
        assert second.status_code == 403
        assert "already exists" in second.json()["detail"]

    def test_plain_user_registration_unaffected_by_existing_admin(self, client):
        _register(client, "first-admin", role="admin")
        resp = _register(client, "just-a-user", role="user")
        assert resp.status_code == 201
        assert resp.json()["role"] == "user"

    def test_rejected_admin_registration_does_not_create_a_user_row(self, client):
        _register(client, "first-admin", role="admin")
        _register(client, "second-admin", role="admin")

        # The rejected attempt must not have left a "second-admin" user
        # behind at some other (e.g. non-admin) role -- it should be a
        # clean no-op, not a silent downgrade.
        login = client.post(
            "/api/auth/login",
            json={"username": "second-admin", "password": "a-password-123"},
        )
        assert login.status_code == 401


class TestAdminCreateUserEndpoint:
    def test_admin_can_create_a_second_admin(self, client):
        _register(client, "first-admin", role="admin")
        token = _login(client, "first-admin")

        resp = _admin_create_user(client, token, "second-admin", role="admin")
        assert resp.status_code == 201
        assert resp.json()["role"] == "admin"

        # And the new admin can actually log in.
        login = client.post(
            "/api/auth/login",
            json={"username": "second-admin", "password": "a-password-123"},
        )
        assert login.status_code == 200

    def test_admin_can_create_users_of_any_role(self, client):
        _register(client, "first-admin", role="admin")
        token = _login(client, "first-admin")

        resp = _admin_create_user(client, token, "new-team-owner", role="team_owner")
        assert resp.status_code == 201
        assert resp.json()["role"] == "team_owner"

    def test_non_admin_cannot_create_a_user(self, client):
        _register(client, "plain-user", role="user")
        token = _login(client, "plain-user")

        resp = _admin_create_user(client, token, "sneaky-admin", role="admin")
        assert resp.status_code == 403

    def test_unauthenticated_request_is_rejected(self, client):
        resp = client.post(
            "/api/auth/admin/create-user",
            json={"username": "sneaky-admin", "password": "a-password-123", "role": "admin"},
        )
        assert resp.status_code in (401, 403)

    def test_duplicate_username_is_rejected(self, client):
        _register(client, "first-admin", role="admin")
        token = _login(client, "first-admin")

        resp = _admin_create_user(client, token, "first-admin", role="admin")
        assert resp.status_code == 400


class TestAdminListUsersEndpoint:
    def test_admin_sees_all_users(self, client):
        _register(client, "first-admin", role="admin")
        token = _login(client, "first-admin")
        _admin_create_user(client, token, "second-admin", role="admin")
        _register(client, "plain-user", role="user")

        resp = client.get("/api/auth/admin/users", headers=_auth_headers(token))
        assert resp.status_code == 200
        usernames = {u["username"] for u in resp.json()}
        assert usernames == {"first-admin", "second-admin", "plain-user"}

    def test_non_admin_cannot_list_users(self, client):
        _register(client, "plain-user", role="user")
        token = _login(client, "plain-user")

        resp = client.get("/api/auth/admin/users", headers=_auth_headers(token))
        assert resp.status_code == 403

    def test_unauthenticated_request_is_rejected(self, client):
        resp = client.get("/api/auth/admin/users")
        assert resp.status_code in (401, 403)
