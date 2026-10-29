"""
POST/GET /api/datasets/{id}/add-recordings (yapat-backend#70, step 2).

Covers request validation, permissions, the one-run-per-dataset lock and the
active_ingest_job field. The Celery pipeline itself is dispatched through a stub
here; its stages are tested separately.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_current_active_user, get_db
from app.config import settings
from app.main import app
from app.models.dataset import Dataset, DatasetType
from app.models.dataset_ingest_job import DatasetIngestJob, IngestJobStatus, IngestStage
from app.models.embedding import (
    EmbeddingJob,
    EmbeddingJobStatus,
    EmbeddingModel,
    SnippetSet,
    SnippetSetStatus,
)
from app.models.team import Team, TeamMembership, TeamRole
from app.models.user import User, UserRole


@pytest.fixture
def data_root(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "data"
    (root / "site" / "current").mkdir(parents=True)
    monkeypatch.setattr(settings, "DATA_ROOT", str(root))
    return root


@pytest.fixture
def dispatched(monkeypatch):
    """Capture task dispatches instead of running the pipeline."""
    calls = []

    class _Result:
        id = "celery-task-1"

    def _delay(job_id):
        calls.append(job_id)
        return _Result()

    monkeypatch.setattr("app.api.dataset_ingest.run_dataset_ingest.delay", _delay)
    return calls


@pytest.fixture
def people(db_session):
    team = Team(name="ingest-team")
    db_session.add(team)
    db_session.flush()
    admin = User(username="ingest_admin", hashed_password="x", role=UserRole.ADMIN)
    owner = User(username="ingest_owner", hashed_password="x", role=UserRole.TEAM_OWNER)
    member = User(username="ingest_member", hashed_password="x", role=UserRole.USER)
    db_session.add_all([admin, owner, member])
    db_session.flush()
    db_session.add_all([
        TeamMembership(team_id=team.id, user_id=owner.id, role=TeamRole.OWNER),
        TeamMembership(team_id=team.id, user_id=member.id, role=TeamRole.USER),
    ])
    db_session.commit()
    return {"team": team, "admin": admin, "owner": owner, "member": member}


def _make_dataset(db_session, team_id, source_uri="site/current", snippet_status=SnippetSetStatus.READY):
    ds = Dataset(
        name=f"ds-{source_uri}",
        source_uri=source_uri,
        dataset_type=DatasetType.PAM,
        team_id=team_id,
    )
    db_session.add(ds)
    db_session.flush()
    model = db_session.query(EmbeddingModel).first()
    if model is None:
        model = EmbeddingModel(name="BirdNET", window_size=3.0, step_size=3.0, overlap=0.0)
        db_session.add(model)
        db_session.flush()
    ss = SnippetSet(
        dataset_id=ds.id,
        embedding_model_id=model.id,
        window_size=3.0,
        step_size=3.0,
        overlap=0.0,
        status=snippet_status,
    )
    db_session.add(ss)
    db_session.flush()
    db_session.add(EmbeddingJob(
        dataset_id=ds.id,
        embedding_model_id=model.id,
        snippet_set_id=ss.id,
        status=EmbeddingJobStatus.COMPLETED,
    ))
    ds.default_snippet_set_id = ss.id
    db_session.commit()
    return ds


@pytest.fixture
def dataset(db_session, people, data_root):
    return _make_dataset(db_session, people["team"].id)


@pytest.fixture
def api(db_session):
    def _as(user: User) -> TestClient:
        app.dependency_overrides[get_db] = lambda: db_session
        app.dependency_overrides[get_current_active_user] = lambda: user
        return TestClient(app)

    yield _as
    app.dependency_overrides.clear()


def _start(client, dataset_id, body=None):
    return client.post(f"/api/datasets/{dataset_id}/add-recordings", json=body if body is not None else {})


# ----------------------------------------------------------------------
# Start
# ----------------------------------------------------------------------

def test_team_owner_starts_a_rescan(api, people, dataset, dispatched, db_session):
    resp = _start(api(people["owner"]), dataset.id)

    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "pending"
    assert body["stage"] == "pending"
    assert body["summary"]["recordings_new"] == 0
    assert dispatched == [body["job_id"]]

    job = db_session.get(DatasetIngestJob, body["job_id"])
    assert job.snippet_set_id == dataset.default_snippet_set_id
    assert job.source_uri is None
    assert job.celery_task_id == "celery-task-1"


def test_no_body_is_accepted(api, people, dataset, dispatched):
    client = api(people["admin"])
    resp = client.post(f"/api/datasets/{dataset.id}/add-recordings")
    assert resp.status_code == 202


def test_team_member_is_refused(api, people, dataset, dispatched):
    resp = _start(api(people["member"]), dataset.id)
    assert resp.status_code == 403
    assert dispatched == []


def test_unknown_dataset_is_404(api, people, data_root, dispatched):
    assert _start(api(people["admin"]), 999_999).status_code == 404


def test_second_start_returns_running_job(api, people, dataset, dispatched):
    client = api(people["admin"])
    first = _start(client, dataset.id).json()

    resp = _start(client, dataset.id)

    assert resp.status_code == 409
    assert resp.json() == {"detail": "ingest_job_running", "job_id": first["job_id"]}
    assert len(dispatched) == 1


def test_running_embedding_job_blocks(api, people, dataset, dispatched, db_session):
    running = EmbeddingJob(
        dataset_id=dataset.id,
        embedding_model_id=db_session.query(EmbeddingModel).first().id,
        snippet_set_id=dataset.default_snippet_set_id,
        status=EmbeddingJobStatus.RUNNING,
    )
    db_session.add(running)
    db_session.commit()

    resp = _start(api(people["admin"]), dataset.id)

    assert resp.status_code == 409
    assert resp.json() == {"detail": "embedding_job_running", "job_id": running.id}


def test_never_embedded_dataset_is_refused(api, people, data_root, dispatched, db_session):
    ds = Dataset(name="raw", source_uri="site/current", dataset_type=DatasetType.PAM, team_id=people["team"].id)
    db_session.add(ds)
    db_session.commit()

    resp = _start(api(people["admin"]), ds.id)

    assert resp.status_code == 400
    assert resp.json()["detail"] == "no_embeddings_yet"


def test_failed_embeddings_are_refused(api, people, data_root, dispatched, db_session):
    ds = _make_dataset(db_session, people["team"].id, snippet_status=SnippetSetStatus.FAILED)

    resp = _start(api(people["admin"]), ds.id)

    assert resp.status_code == 400
    assert resp.json()["detail"] == "embeddings_not_ready"


def test_dataset_responses_count_recordings_with_metadata(api, people, dataset, db_session):
    from app.models.recording import Recording

    full = {"location": "CIPAP07", "recorded_date": "2024-11-11", "recorded_time": 85800}
    db_session.add_all([
        Recording(dataset_id=dataset.id, file_path="a.wav", file_name="a.wav", extra_metadata=full),
        Recording(dataset_id=dataset.id, file_path="b.wav", file_name="b.wav",
                  extra_metadata={**full, "recorded_time": 0}),          # midnight still counts
        Recording(dataset_id=dataset.id, file_path="c.wav", file_name="c.wav",
                  extra_metadata={"location": "CIPAP07"}),               # no date/time
        Recording(dataset_id=dataset.id, file_path="d.wav", file_name="d.wav",
                  extra_metadata={**full, "location": ""}),              # empty location
        Recording(dataset_id=dataset.id, file_path="e.wav", file_name="e.wav", extra_metadata=None),
    ])
    db_session.commit()
    client = api(people["admin"])

    detail = client.get(f"/api/datasets/{dataset.id}").json()
    listed = {d["id"]: d for d in client.get("/api/datasets/").json()}[dataset.id]

    for body in (detail, listed):
        assert body["recording_count"] == 5
        assert body["recordings_with_metadata"] == 2


def _age(db_session, job_id, minutes):
    from datetime import datetime, timedelta, timezone

    job = db_session.get(DatasetIngestJob, job_id)
    job.created_at = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    db_session.commit()


def test_pending_job_no_worker_picked_up_is_replaced(api, people, dataset, dispatched, db_session):
    client = api(people["admin"])
    stuck = _start(client, dataset.id).json()["job_id"]
    _age(db_session, stuck, minutes=31)

    resp = _start(client, dataset.id)

    assert resp.status_code == 202
    assert resp.json()["job_id"] != stuck
    db_session.expire_all()
    old = db_session.get(DatasetIngestJob, stuck)
    assert old.status == IngestJobStatus.FAILED
    assert "Never started" in old.error_message


def test_reading_a_dead_job_releases_the_dataset(api, people, dataset, dispatched, db_session):
    """The UI disables the start button while a job is active, so reads must clear it too."""
    client = api(people["admin"])
    stuck = _start(client, dataset.id).json()["job_id"]
    _age(db_session, stuck, minutes=31)

    assert client.get(f"/api/datasets/{dataset.id}").json()["active_ingest_job"] is None
    listed = {d["id"]: d for d in client.get("/api/datasets/").json()}
    assert listed[dataset.id]["active_ingest_job"] is None
    polled = client.get(f"/api/datasets/{dataset.id}/add-recordings/{stuck}").json()
    assert polled["status"] == "failed"
    assert "Never started" in polled["error"]


def test_recent_pending_job_still_blocks(api, people, dataset, dispatched, db_session):
    client = api(people["admin"])
    queued = _start(client, dataset.id).json()["job_id"]
    _age(db_session, queued, minutes=20)

    resp = _start(client, dataset.id)

    assert resp.status_code == 409
    assert resp.json()["job_id"] == queued


def test_old_pending_job_reports_stalled(api, people, dataset, dispatched, db_session):
    client = api(people["admin"])
    job_id = _start(client, dataset.id).json()["job_id"]
    assert client.get(f"/api/datasets/{dataset.id}/add-recordings/{job_id}").json()["stalled"] is False

    _age(db_session, job_id, minutes=11)

    assert client.get(f"/api/datasets/{dataset.id}/add-recordings/{job_id}").json()["stalled"] is True


# ----------------------------------------------------------------------
# source_uri
# ----------------------------------------------------------------------

def test_parent_folder_becomes_the_new_source_uri(api, people, dataset, dispatched, db_session):
    resp = _start(api(people["admin"]), dataset.id, {"source_uri": "site/"})

    assert resp.status_code == 202
    db_session.refresh(dataset)
    assert dataset.source_uri == "site"
    assert db_session.get(DatasetIngestJob, resp.json()["job_id"]).source_uri == "site"


def test_same_folder_counts_as_rescan(api, people, dataset, dispatched, db_session):
    resp = _start(api(people["admin"]), dataset.id, {"source_uri": "site/current"})

    assert resp.status_code == 202
    assert db_session.get(DatasetIngestJob, resp.json()["job_id"]).source_uri is None


@pytest.mark.parametrize("source_uri", ["elsewhere", "site/current/deeper", "../outside", "missing"])
def test_folder_that_does_not_contain_current_is_refused(api, people, dataset, data_root, dispatched, source_uri):
    (data_root / "elsewhere").mkdir()
    (data_root / "site" / "current" / "deeper").mkdir()

    resp = _start(api(people["admin"]), dataset.id, {"source_uri": source_uri})

    assert resp.status_code == 400
    assert resp.json()["detail"] == "invalid_source_uri"


def test_folder_holding_another_dataset_is_refused(api, people, dataset, data_root, dispatched, db_session):
    (data_root / "site" / "neighbour").mkdir()
    db_session.add(Dataset(
        name="neighbour", source_uri="site/neighbour", dataset_type=DatasetType.PAM, team_id=None,
    ))
    db_session.commit()

    resp = _start(api(people["admin"]), dataset.id, {"source_uri": "site"})

    assert resp.status_code == 400
    assert resp.json()["detail"] == "source_uri_overlaps_other_dataset"
    db_session.refresh(dataset)
    assert dataset.source_uri == "site/current"


def test_reference_dataset_needs_its_metadata_csv(api, people, dataset, data_root, dispatched, db_session):
    dataset.is_reference = True
    db_session.commit()
    client = api(people["admin"])

    missing = _start(client, dataset.id)
    assert missing.status_code == 400
    assert missing.json()["detail"] == "reference_metadata_missing"

    (data_root / "site" / "current" / "pam_metadata.csv").write_text("filename\n")
    assert _start(client, dataset.id).status_code == 202


# ----------------------------------------------------------------------
# Poll + active_ingest_job
# ----------------------------------------------------------------------

def test_poll_returns_job_and_member_can_read_it(api, people, dataset, dispatched):
    job_id = _start(api(people["owner"]), dataset.id).json()["job_id"]

    resp = api(people["member"]).get(f"/api/datasets/{dataset.id}/add-recordings/{job_id}")

    assert resp.status_code == 200
    assert resp.json()["job_id"] == job_id
    assert resp.json()["percent"] is None


def test_poll_completed_job_reports_100_percent(api, people, dataset, db_session):
    job = DatasetIngestJob(
        dataset_id=dataset.id,
        snippet_set_id=dataset.default_snippet_set_id,
        embedding_job_id=db_session.query(EmbeddingJob.id).first()[0],
        status=IngestJobStatus.COMPLETED,
        stage=IngestStage.COMPLETED,
        summary={"recordings_new": 3},
    )
    db_session.add(job)
    db_session.commit()

    body = api(people["admin"]).get(f"/api/datasets/{dataset.id}/add-recordings/{job.id}").json()

    assert body["percent"] == 100.0
    assert body["summary"]["recordings_new"] == 3
    assert body["summary"]["snippets_new"] == 0  # missing keys are filled in


def test_poll_job_of_another_dataset_is_404(api, people, dataset, data_root, dispatched, db_session):
    (data_root / "other").mkdir()
    other = _make_dataset(db_session, people["team"].id, source_uri="other")
    job_id = _start(api(people["admin"]), dataset.id).json()["job_id"]

    resp = api(people["admin"]).get(f"/api/datasets/{other.id}/add-recordings/{job_id}")

    assert resp.status_code == 404


def test_dataset_responses_carry_active_ingest_job(api, people, dataset, dispatched):
    client = api(people["admin"])
    assert client.get(f"/api/datasets/{dataset.id}").json()["active_ingest_job"] is None

    job_id = _start(client, dataset.id).json()["job_id"]

    detail = client.get(f"/api/datasets/{dataset.id}").json()["active_ingest_job"]
    assert detail["id"] == job_id
    assert detail["stage"] == "pending"
    listed = {d["id"]: d for d in client.get("/api/datasets/").json()}
    assert listed[dataset.id]["active_ingest_job"]["id"] == job_id


# ----------------------------------------------------------------------
# Embedding failures: count, details, dismiss, retry_failed_only
# ----------------------------------------------------------------------

def _failure(db_session, dataset, file_name, missing, reason, dismissed=False):
    from datetime import datetime, timezone

    from app.models.recording import Recording
    from app.models.recording_embedding_failure import RecordingEmbeddingFailure

    rec = Recording(dataset_id=dataset.id, file_path=f"site/current/{file_name}", file_name=file_name)
    db_session.add(rec)
    db_session.flush()
    db_session.add(RecordingEmbeddingFailure(
        dataset_id=dataset.id, snippet_set_id=dataset.default_snippet_set_id, recording_id=rec.id,
        snippets_missing=missing, reason=reason,
        dismissed_at=datetime.now(timezone.utc) if dismissed else None,
    ))
    db_session.commit()
    return rec


def test_failures_are_counted_listed_and_dismissed(api, people, dataset, db_session):
    _failure(db_session, dataset, "b.wav", 4, "RuntimeError: Error opening audio file")
    _failure(db_session, dataset, "a.wav", 1, "BirdNET returned no embedding for 1 of 20 snippets")
    _failure(db_session, dataset, "old.wav", 9, "gone", dismissed=True)
    # api() sets the current user globally, so switch user before each block.
    owner = api(people["owner"])
    assert owner.get(f"/api/datasets/{dataset.id}").json()["embedding_failures"] == {"recordings": 2, "snippets": 5}
    listed = {d["id"]: d for d in owner.get("/api/datasets/").json()}[dataset.id]
    assert listed["embedding_failures"] == {"recordings": 2, "snippets": 5}

    member = api(people["member"])
    body = member.get(f"/api/datasets/{dataset.id}/embedding-failures").json()
    assert (body["recordings"], body["snippets"], body["dismissed_recordings"]) == (2, 5, 1)
    assert [i["file_name"] for i in body["items"]] == ["a.wav", "b.wav"]
    assert body["items"][1]["reason"] == "RuntimeError: Error opening audio file"
    assert body["items"][1]["snippets_missing"] == 4
    with_dismissed = member.get(
        f"/api/datasets/{dataset.id}/embedding-failures?include_dismissed=true").json()
    assert [i["file_name"] for i in with_dismissed["items"]] == ["a.wav", "b.wav", "old.wav"]
    assert with_dismissed["items"][2]["dismissed"] is True

    assert member.post(f"/api/datasets/{dataset.id}/embedding-failures/dismiss").status_code == 403
    owner = api(people["owner"])
    assert owner.post(f"/api/datasets/{dataset.id}/embedding-failures/dismiss").json() == {"dismissed": 2}
    assert owner.get(f"/api/datasets/{dataset.id}").json()["embedding_failures"] == {"recordings": 0, "snippets": 0}


def test_retry_failed_only_creates_a_retry_job(api, people, dataset, dispatched, db_session):
    _failure(db_session, dataset, "b.wav", 4, "boom", dismissed=True)
    client = api(people["admin"])

    resp = _start(client, dataset.id, {"retry_failed_only": True})

    assert resp.status_code == 202
    assert resp.json()["mode"] == "retry_failed"
    assert client.get(f"/api/datasets/{dataset.id}").json()["active_ingest_job"]["mode"] == "retry_failed"


def test_retry_failed_only_needs_failures_and_no_source_uri(api, people, dataset, dispatched, db_session):
    client = api(people["admin"])
    resp = _start(client, dataset.id, {"retry_failed_only": True})
    assert resp.status_code == 400 and resp.json()["detail"] == "no_failed_recordings"

    _failure(db_session, dataset, "b.wav", 4, "boom")
    resp = _start(client, dataset.id, {"retry_failed_only": True, "source_uri": "site"})
    assert resp.status_code == 400 and resp.json()["detail"] == "source_uri_not_allowed_with_retry"
    assert dispatched == []
