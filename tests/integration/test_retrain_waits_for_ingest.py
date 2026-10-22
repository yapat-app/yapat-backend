"""
Retrains and cold starts wait for a running "add recordings" run on the same dataset.

Covers app/services/retrain_deferral.py (no stored link: a held retrain is one
that is PENDING, never started and created after the run started) and its
release points: run completion, run failure, a dead run failed on read, the
cleanup beat, plus the atomic claim that makes a duplicate dispatch harmless.
Celery dispatch is captured instead of running a real retrain.
"""

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from app.models.dataset import Dataset, DatasetType
from app.models.dataset_ingest_job import DatasetIngestJob, IngestJobStatus, IngestStage, empty_summary
from app.models.embedding import EmbeddingJob, EmbeddingJobStatus, EmbeddingModel, SnippetSet, SnippetSetStatus
from app.models.pam_active_learning import ALModelCheckpoint, ALModelStatus, ALRetrainJob, ALRetrainStatus
from app.models.user import User, UserRole
from app.services import retrain_deferral
from app.services.dataset_ingest_service import DatasetIngestService


def _now():
    return datetime.now(timezone.utc)


@pytest.fixture
def dispatched(monkeypatch):
    from app.tasks import pam_al_tasks

    calls = []
    monkeypatch.setattr(pam_al_tasks.pam_al_auto_retrain, "delay",
                        lambda **kw: calls.append(("auto", kw["job_id"])))
    monkeypatch.setattr(pam_al_tasks.pam_al_manual_retrain, "delay",
                        lambda **kw: calls.append(("manual", kw["job_id"])))
    monkeypatch.setattr(pam_al_tasks.pam_al_train_from_scratch, "delay",
                        lambda **kw: calls.append(("cold", kw["job_id"])))
    return calls


@pytest.fixture
def training_files(tmp_path, monkeypatch):
    """The metadata CSV and label config setup_train_from_scratch requires to exist."""
    from app.services.pam_al import service as service_module

    site = tmp_path / "site"
    site.mkdir()
    (site / "pam_metadata.csv").write_text("filename,species\n")
    (site / "pam_label_config.json").write_text('["a"]')
    monkeypatch.setattr(service_module, "DATA_ROOT", str(tmp_path))


@pytest.fixture
def env(db_session, engine, monkeypatch):
    from app.tasks import ingest_tasks, pam_al_tasks

    factory = sessionmaker(bind=engine)
    monkeypatch.setattr(ingest_tasks, "SessionLocal", factory)
    monkeypatch.setattr(pam_al_tasks, "SessionLocal", factory)

    em = EmbeddingModel(name="BirdNET", window_size=3.0, step_size=3.0, overlap=0.0)
    ds = Dataset(name="retrain-waits", source_uri="site", dataset_type=DatasetType.PAM)
    db_session.add_all([em, ds])
    db_session.flush()
    ss = SnippetSet(dataset_id=ds.id, embedding_model_id=em.id, window_size=3.0, step_size=3.0,
                    overlap=0.0, status=SnippetSetStatus.READY)
    db_session.add(ss)
    db_session.flush()
    ejob = EmbeddingJob(dataset_id=ds.id, embedding_model_id=em.id, snippet_set_id=ss.id,
                        status=EmbeddingJobStatus.COMPLETED)
    ckpt = ALModelCheckpoint(dataset_id=ds.id, model_family_name="pam", version="v2",
                             checkpoint_path="x.pt", label_config_path="", status=ALModelStatus.LOADING)
    db_session.add_all([ejob, ckpt])
    db_session.commit()

    def ingest(status=IngestJobStatus.RUNNING, minutes_ago=0):
        job = DatasetIngestJob(
            dataset_id=ds.id, snippet_set_id=ss.id, embedding_job_id=ejob.id,
            status=status, stage=IngestStage.EMBEDDING, summary=empty_summary(),
            created_at=_now() - timedelta(minutes=minutes_ago),
            started_at=_now() - timedelta(minutes=minutes_ago),
        )
        db_session.add(job)
        db_session.commit()
        return job

    def retrain(trigger="auto_feedback", minutes_ago=0, status=ALRetrainStatus.PENDING):
        job = ALRetrainJob(model_checkpoint_id=ckpt.id, dataset_id=ds.id, trigger=trigger,
                           feedback_count=10, status=status,
                           created_at=_now() - timedelta(minutes=minutes_ago))
        db_session.add(job)
        db_session.commit()
        return job

    return type("Env", (), {"dataset": ds, "ingest": staticmethod(ingest), "retrain": staticmethod(retrain)})


def _fresh(db_session, obj):
    db_session.expire_all()
    return db_session.get(type(obj), obj.id)


# ----------------------------------------------------------------------
# Hold and release
# ----------------------------------------------------------------------

def test_no_ingest_running_dispatches_right_away(db_session, env, dispatched):
    job = env.retrain()

    assert retrain_deferral.dispatch_or_wait(db_session, job) is None

    assert dispatched == [("auto", job.id)]
    assert retrain_deferral.waiting_ingest_job_id(db_session, job) is None


def test_running_ingest_holds_the_retrain(db_session, env, dispatched):
    run = env.ingest()
    job = env.retrain(trigger="manual")

    assert retrain_deferral.dispatch_or_wait(db_session, job) == run.id

    assert dispatched == []
    held = _fresh(db_session, job)
    assert held.status == ALRetrainStatus.PENDING and held.started_at is None
    assert retrain_deferral.waiting_ingest_job_id(db_session, held) == run.id


def test_run_completion_dispatches_the_held_retrain(db_session, env, dispatched):
    from app.tasks.ingest_tasks import complete_job

    run = env.ingest()
    job = env.retrain()
    retrain_deferral.dispatch_or_wait(db_session, job)

    complete_job(run.id, snippets_scored=5)

    assert dispatched == [("auto", job.id)]
    assert retrain_deferral.waiting_ingest_job_id(db_session, _fresh(db_session, job)) is None


def test_run_failure_dispatches_the_held_retrain(db_session, env, dispatched):
    from app.tasks.ingest_tasks import _mark_failed

    run = env.ingest()
    job = env.retrain(trigger="manual")
    retrain_deferral.dispatch_or_wait(db_session, job)

    _mark_failed(run.id, "scan failed")

    assert dispatched == [("manual", job.id)]


def test_dead_run_failed_on_read_dispatches_the_held_retrain(db_session, env, dispatched, monkeypatch):
    from app.services import ingest_progress

    monkeypatch.setattr(ingest_progress, "read", lambda job_id: (None, None, None))
    run = env.ingest(minutes_ago=61)   # running, no progress for over an hour
    job = env.retrain()
    retrain_deferral.dispatch_or_wait(db_session, job)
    assert dispatched == []

    DatasetIngestService(db_session).active_jobs_by_dataset([env.dataset.id])

    assert _fresh(db_session, run).status == IngestJobStatus.FAILED
    assert dispatched == [("auto", job.id)]


@pytest.mark.parametrize("trigger", ["auto_feedback", "manual"])
def test_retrains_created_by_the_service_are_held_and_released(db_session, env, dispatched, trigger):
    """Jobs built by setup_auto_retrain / setup_manual_retrain, not by hand,
    so a setup that stamps started_at early (and hides the job from release)
    is caught."""
    from app.models.pam_active_learning import ALModelFamilyState
    from app.schemas.pam_active_learning import ALRetrainRequest
    from app.services.pam_al.service import PAMActiveLearningService
    from app.tasks.ingest_tasks import complete_job

    parent = ALModelCheckpoint(
        dataset_id=env.dataset.id, model_family_name="default", version="v0",
        checkpoint_path="p.pt", label_config_path="", status=ALModelStatus.AVAILABLE,
        hyperparameters={"embedding_model_id": 1, "resolved_snippet_set_id": 1, "label_order": ["a"]},
    )
    db_session.add(parent)
    db_session.flush()
    db_session.add(ALModelFamilyState(dataset_id=env.dataset.id, model_family_name="default",
                                      active_model_checkpoint_id=parent.id))
    db_session.commit()

    # A minute ago: the job's created_at comes from SQLite's CURRENT_TIMESTAMP,
    # which has whole seconds only and could read as older than a run started
    # in the same second.
    run = env.ingest(minutes_ago=1)
    svc = PAMActiveLearningService(db_session)
    if trigger == "manual":
        _, job = svc.setup_manual_retrain(
            ALRetrainRequest(dataset_id=env.dataset.id, model_family_name="default"))
    else:
        _, job = svc.setup_auto_retrain(parent.id)

    assert job.started_at is None
    assert retrain_deferral.dispatch_or_wait(db_session, job) == run.id
    assert dispatched == []

    complete_job(run.id)

    kind = "manual" if trigger == "manual" else "auto"
    assert dispatched == [(kind, job.id)]


def test_cold_start_created_by_the_service_is_held_and_released(
    db_session, env, dispatched, training_files
):
    """Same as above for setup_train_from_scratch, released to the cold-start task."""
    from app.schemas.pam_active_learning import ALTrainFromScratchRequest
    from app.services.pam_al.service import PAMActiveLearningService
    from app.tasks.ingest_tasks import complete_job

    snippet_set = db_session.query(SnippetSet).filter_by(dataset_id=env.dataset.id).one()
    run = env.ingest(minutes_ago=1)   # see the SQLite note in the test above
    _, job = PAMActiveLearningService(db_session).setup_train_from_scratch(
        ALTrainFromScratchRequest(dataset_id=env.dataset.id, snippet_set_id=snippet_set.id,
                                  embedding_model_id=snippet_set.embedding_model_id))

    assert job.trigger == "cold_start" and job.started_at is None
    assert retrain_deferral.dispatch_or_wait(db_session, job) == run.id
    assert dispatched == []
    assert retrain_deferral.waiting_ingest_job_id(db_session, _fresh(db_session, job)) == run.id

    complete_job(run.id)

    assert dispatched == [("cold", job.id)]


def test_release_only_touches_retrains_the_run_held(db_session, env, dispatched):
    from app.tasks.ingest_tasks import complete_job

    before = env.retrain(minutes_ago=30)                     # existed before the run
    run = env.ingest(minutes_ago=20)
    started = env.retrain(minutes_ago=5)
    started.started_at = _now()                              # already running
    db_session.commit()
    cold = env.retrain(trigger="cold_start", minutes_ago=5)  # held like a retrain
    inference = env.retrain(trigger="inference", minutes_ago=5)  # never waits
    held = env.retrain(minutes_ago=1)

    complete_job(run.id)

    assert dispatched == [("cold", cold.id), ("auto", held.id)]
    assert {before.id, started.id, inference.id}.isdisjoint({j for _, j in dispatched})


# ----------------------------------------------------------------------
# Duplicate dispatch is harmless
# ----------------------------------------------------------------------

def test_claim_succeeds_once(db_session, env):
    job = env.retrain()

    assert retrain_deferral.claim_retrain_job(db_session, job.id) is True
    assert retrain_deferral.claim_retrain_job(db_session, job.id) is False

    claimed = _fresh(db_session, job)
    assert claimed.status == ALRetrainStatus.RUNNING
    assert claimed.started_at is not None


@pytest.mark.parametrize(
    "task_name", ["pam_al_auto_retrain", "pam_al_manual_retrain", "pam_al_train_from_scratch"])
def test_second_copy_of_a_retrain_task_skips(db_session, env, monkeypatch, task_name):
    from app.tasks import pam_al_tasks

    job = env.retrain(status=ALRetrainStatus.RUNNING)   # another copy already claimed it
    monkeypatch.setattr(pam_al_tasks, "_make_service", lambda db: pytest.fail("trained twice"))

    result = getattr(pam_al_tasks, task_name).apply(
        kwargs={"checkpoint_id": job.model_checkpoint_id, "job_id": job.id}).get()

    assert result["status"] == "skipped"


# ----------------------------------------------------------------------
# Cleanup beat
# ----------------------------------------------------------------------

def test_cleanup_spares_held_retrains_and_resends_lost_ones(db_session, env, dispatched):
    from app.tasks.pam_al_tasks import pam_al_cleanup_stale_jobs

    # Dataset A: a long run holds a retrain created 7 h ago (past the 6 h cutoff).
    run = env.ingest(minutes_ago=8 * 60)
    held = env.retrain(minutes_ago=7 * 60)

    pam_al_cleanup_stale_jobs.apply()

    assert _fresh(db_session, held).status == ALRetrainStatus.PENDING
    assert dispatched == []

    # The run ends but its release pass is lost: the beat re-sends the retrain.
    run.status = IngestJobStatus.COMPLETED
    db_session.commit()

    pam_al_cleanup_stale_jobs.apply()

    assert dispatched == [("auto", held.id)]


def test_cleanup_leaves_fresh_undispatched_retrains_alone(db_session, env, dispatched):
    from app.tasks.pam_al_tasks import pam_al_cleanup_stale_jobs

    env.retrain(minutes_ago=2)   # probably just queued in Celery

    pam_al_cleanup_stale_jobs.apply()

    assert dispatched == []


# ----------------------------------------------------------------------
# API: the poll endpoint reports the computed field
# ----------------------------------------------------------------------

def test_poll_endpoint_reports_waiting(db_session, env, dispatched):
    from app.api.deps import get_current_active_user, get_db
    from app.main import app

    admin = User(username="retrain_admin", hashed_password="x", role=UserRole.ADMIN)
    db_session.add(admin)
    db_session.commit()
    run = env.ingest()
    job = env.retrain()
    app.dependency_overrides[get_db] = lambda: db_session
    app.dependency_overrides[get_current_active_user] = lambda: admin
    try:
        client = TestClient(app)
        polled = client.get(f"/api/pam-al/retrain/jobs/{job.id}").json()
        listed = client.get(f"/api/pam-al/retrain/jobs?dataset_id={env.dataset.id}").json()
    finally:
        app.dependency_overrides.clear()

    assert polled["status"] == "PENDING"
    assert polled["waiting_for_ingest_job_id"] == run.id
    assert [j["waiting_for_ingest_job_id"] for j in listed if j["id"] == job.id] == [run.id]


def test_train_from_scratch_endpoint_waits_for_a_running_ingest(
    db_session, env, dispatched, training_files
):
    from app.api.deps import get_current_active_user, get_db
    from app.main import app

    admin = User(username="cold_start_admin", hashed_password="x", role=UserRole.ADMIN)
    db_session.add(admin)
    db_session.commit()
    snippet_set = db_session.query(SnippetSet).filter_by(dataset_id=env.dataset.id).one()
    run = env.ingest(minutes_ago=1)
    app.dependency_overrides[get_db] = lambda: db_session
    app.dependency_overrides[get_current_active_user] = lambda: admin
    try:
        response = TestClient(app).post("/api/pam-al/train-from-scratch", json={
            "dataset_id": env.dataset.id,
            "snippet_set_id": snippet_set.id,
            "embedding_model_id": snippet_set.embedding_model_id,
        })
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 202, response.text
    body = response.json()
    assert body["waiting_for_ingest_job_id"] == run.id
    assert dispatched == []
