import uuid
from unittest.mock import patch

from app.api.deps import get_current_active_user
from app.main import app
from app.models.team import Team
from app.models.user import User, UserRole


def test_create_dataset_triggers_process_task(db_session, client):
    # Create admin + team
    admin = User(username="adm", hashed_password="x", role=UserRole.ADMIN)
    team = Team(name="T")
    db_session.add_all([admin, team])
    db_session.commit()

    # Override FastAPI authentication
    app.dependency_overrides[get_current_active_user] = lambda: admin

    called = {}

    # Fake Celery delay() method
    def fake_delay(dataset_id):
        called["dataset_id"] = dataset_id

        class Result:
            id = "fake-task-id"

        return Result()

    # Patch the exact function that the endpoint calls
    with patch("app.api.datasets.process_dataset.delay", side_effect=fake_delay):
        res = client.post(
            "/api/datasets/",
            json={
                "team_id": team.id,
                "name": "MyDS",
                "description": None,
                "source_uri": f"abc_{uuid.uuid4()}",
            },
        )

    assert res.status_code == 201
    body = res.json()

    assert "dataset" in body
    ds = body["dataset"]

    assert ds["name"] == "MyDS"
    assert ds["team_id"] == team.id
    assert "id" in ds

    # Verify Celery was called correctly
    assert called["dataset_id"] == ds["id"]

    # Verify returned task id
    assert body["process_task_id"] == "fake-task-id"

    # Cleanup
    app.dependency_overrides = {}
