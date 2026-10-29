"""
Add-recordings pipeline, steps scan -> segment -> embed (yapat-backend#70).

Celery runs eagerly (tests/integration/conftest.py), so one apply() of
run_dataset_ingest walks the whole chain including the chord callback. BirdNET
is replaced by a fake module so TensorFlow is never loaded.
"""

import sys
import types
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.models.dataset import Dataset, DatasetType
from app.models.dataset_ingest_job import (
    DatasetIngestJob,
    IngestJobStatus,
    IngestStage,
    empty_summary,
)
from app.models.embedding import (
    EmbeddingJob,
    EmbeddingJobStatus,
    EmbeddingModel,
    EmbeddingVector,
    SnippetSet,
    SnippetSetStatus,
)
from app.models.recording import Recording
from app.models.snippet import Snippet
from app.services.embedding_service import VectorStore
from app.tasks import ingest_tasks

SR = 16000


def _wav(path: Path, seconds: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.zeros(int(seconds * SR), dtype=np.float32), SR)


@pytest.fixture
def data_root(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "data"
    (root / "site").mkdir(parents=True)
    monkeypatch.setattr(settings, "DATA_ROOT", str(root))
    return root


@pytest.fixture
def birdnet(monkeypatch):
    """Fake BirdNET: one vector per window; records which files were embedded."""
    calls = []
    fail_windows = set()
    broken_files = set()

    class FakeEmbedder:
        @classmethod
        def embed_batch_from_recording(cls, audio_path, snippet_windows):
            calls.append(Path(audio_path).name)
            if Path(audio_path).name in broken_files:
                raise RuntimeError("Error opening audio file: corrupt header")
            return [
                None if (Path(audio_path).name, start) in fail_windows else [0.1] * 1024
                for start, _end in snippet_windows
            ]

    module = types.ModuleType("app.services.birdnet_model")
    module.BirdNetEmbedder = FakeEmbedder
    monkeypatch.setitem(sys.modules, "app.services.birdnet_model", module)
    monkeypatch.setenv("DATA_ROOT", settings.DATA_ROOT or "/data")
    return types.SimpleNamespace(calls=calls, fail_windows=fail_windows, broken_files=broken_files)


@pytest.fixture
def progress(monkeypatch):
    """Record progress calls instead of talking to Redis."""
    events = []
    monkeypatch.setattr(ingest_tasks.ingest_progress, "start_stage", lambda jid, total=0: events.append(("start", total)))
    monkeypatch.setattr(ingest_tasks.ingest_progress, "incr", lambda jid, n: events.append(("incr", n)))
    return events


@pytest.fixture(autouse=True)
def task_sessions(engine, monkeypatch):
    from app.tasks import pam_al_tasks

    factory = sessionmaker(bind=engine)
    monkeypatch.setattr(ingest_tasks, "SessionLocal", factory)
    monkeypatch.setattr(pam_al_tasks, "SessionLocal", factory)


@pytest.fixture(autouse=True)
def refreshes(monkeypatch):
    """Record the finalize step's cache refreshes instead of touching disk/Redis/FPV."""
    from app.services.explore import hooks
    from app.tasks import embedding_tasks

    calls = []
    monkeypatch.setattr(hooks, "invalidate_dataset_base", lambda db, dataset_id: calls.append(("base", dataset_id)))
    monkeypatch.setattr(ingest_tasks, "_invalidate_embedding_cache", lambda ss, em: calls.append(("embedding_cache", ss, em)))
    monkeypatch.setattr(
        embedding_tasks.generate_fpv_for_dataset, "delay",
        lambda **kw: calls.append(("fpv", kw["dataset_id"], kw["embedding_model_id"])),
    )
    return calls


@pytest.fixture
def embedded_dataset(db_session, data_root, birdnet):
    """A dataset already embedded once: site/old.wav, 6 s -> 2 snippets with vectors."""
    _wav(data_root / "site" / "old.wav", 6.0)
    model = EmbeddingModel(name="BirdNET", window_size=3.0, step_size=3.0, overlap=0.0)
    ds = Dataset(name="pipeline", source_uri="site", dataset_type=DatasetType.PAM)
    db_session.add_all([model, ds])
    db_session.flush()
    ss = SnippetSet(
        dataset_id=ds.id, embedding_model_id=model.id,
        window_size=3.0, step_size=3.0, overlap=0.0, status=SnippetSetStatus.READY,
    )
    db_session.add(ss)
    db_session.flush()
    ejob = EmbeddingJob(
        dataset_id=ds.id, embedding_model_id=model.id, snippet_set_id=ss.id,
        status=EmbeddingJobStatus.COMPLETED,
    )
    rec = Recording(dataset_id=ds.id, file_path="site/old.wav", file_name="old.wav", duration=6.0, sample_rate=SR)
    db_session.add_all([ejob, rec])
    db_session.flush()
    for start in (0.0, 3.0):
        sn = Snippet(recording_id=rec.id, snippet_set_id=ss.id, start_time=start, end_time=start + 3.0, duration=3.0)
        db_session.add(sn)
        db_session.flush()
        db_session.add(EmbeddingVector(
            snippet_id=sn.id, embedding_job_id=ejob.id, embedding_model_id=model.id, dim=1024, vector=[0.0] * 1024,
        ))
    ds.default_snippet_set_id = ss.id
    db_session.commit()
    return types.SimpleNamespace(dataset=ds, snippet_set=ss, embedding_job=ejob, model=model)


def _job(db_session, env) -> DatasetIngestJob:
    job = DatasetIngestJob(
        dataset_id=env.dataset.id,
        snippet_set_id=env.snippet_set.id,
        embedding_job_id=env.embedding_job.id,
        status=IngestJobStatus.PENDING,
        stage=IngestStage.PENDING,
        summary=empty_summary(),
    )
    db_session.add(job)
    db_session.commit()
    return job


def _run(db_session, job) -> DatasetIngestJob:
    ingest_tasks.run_dataset_ingest.apply(args=[job.id])
    db_session.expire_all()
    return db_session.get(DatasetIngestJob, job.id)


def test_new_file_is_scanned_segmented_and_embedded(db_session, embedded_dataset, data_root, birdnet, progress):
    _wav(data_root / "site" / "new" / "fresh.wav", 9.5)  # 3 full windows
    vectors_before = {v.id: v.snippet_id for v in db_session.query(EmbeddingVector).all()}

    job = _run(db_session, _job(db_session, embedded_dataset))

    assert job.status == IngestJobStatus.COMPLETED, job.error_message
    assert job.stage == IngestStage.COMPLETED
    assert job.summary == {
        **empty_summary(),
        "files_found": 2,
        "files_skipped": 1,
        "recordings_new": 1,
        # "fresh.wav" carries no site/date/time in its name.
        "recordings_with_metadata": 0,
        "recordings_missing_metadata": 1,
        "snippets_new": 3,
        "snippets_embedded": 3,
        # The scoring stage ran; this dataset has no model to score with.
        "scoring_skipped_reason": "no_active_checkpoint",
    }

    fresh = db_session.query(Recording).filter_by(file_name="fresh.wav").one()
    assert fresh.file_path == "site/new/fresh.wav"
    fresh_snippets = db_session.query(Snippet).filter_by(recording_id=fresh.id).all()
    assert sorted(s.start_time for s in fresh_snippets) == [0.0, 3.0, 6.0]
    assert {s.snippet_set_id for s in fresh_snippets} == {embedded_dataset.snippet_set.id}

    # Only the new recording went through BirdNET; old vectors are untouched.
    assert birdnet.calls == ["fresh.wav"]
    vectors_after = {v.id: v.snippet_id for v in db_session.query(EmbeddingVector).all()}
    assert {k: vectors_after[k] for k in vectors_before} == vectors_before
    new_vectors = db_session.query(EmbeddingVector).filter(
        EmbeddingVector.snippet_id.in_([s.id for s in fresh_snippets])
    ).all()
    assert len(new_vectors) == 3
    assert {v.embedding_job_id for v in new_vectors} == {embedded_dataset.embedding_job.id}

    # Snippet set never left READY; no extra EmbeddingJob was created.
    db_session.refresh(embedded_dataset.snippet_set)
    assert embedded_dataset.snippet_set.status == SnippetSetStatus.READY
    assert db_session.query(EmbeddingJob).count() == 1

    # Progress: scanning (1 file), segmenting (1 recording), embedding (3 snippets).
    assert ("start", 1) in progress and ("start", 3) in progress
    assert sum(n for kind, n in progress if kind == "incr") == 1 + 1 + 3


def test_finalize_refreshes_layers_cache_and_projection(db_session, embedded_dataset, data_root, birdnet, progress, refreshes):
    _wav(data_root / "site" / "fresh.wav", 6.0)

    job = _run(db_session, _job(db_session, embedded_dataset))

    assert job.status == IngestJobStatus.COMPLETED, job.error_message
    ds, ss, em = embedded_dataset.dataset.id, embedded_dataset.snippet_set.id, embedded_dataset.model.id
    assert sorted(refreshes) == sorted([("base", ds), ("embedding_cache", ss, em), ("fpv", ds, em)])
    assert ("start", 0) in progress  # finalizing stage reported


def test_nothing_new_refreshes_nothing(db_session, embedded_dataset, birdnet, progress, refreshes):
    job = _run(db_session, _job(db_session, embedded_dataset))

    assert job.status == IngestJobStatus.COMPLETED, job.error_message
    assert refreshes == []


def test_retried_vectors_refresh_base_cache_and_projection(db_session, embedded_dataset, data_root, birdnet, progress, refreshes):
    # No new snippets, but the retried ones now have vectors, so they join the
    # hub's base layer (which only lists embedded snippets).
    _wav(data_root / "site" / "fresh.wav", 6.0)
    birdnet.fail_windows.add(("fresh.wav", 3.0))
    _run(db_session, _job(db_session, embedded_dataset))
    birdnet.fail_windows.clear()
    refreshes.clear()

    _run(db_session, _job(db_session, embedded_dataset))

    kinds = sorted(call[0] for call in refreshes)
    assert kinds == ["base", "embedding_cache", "fpv"]


def test_refresh_failure_does_not_fail_the_run(db_session, embedded_dataset, data_root, birdnet, progress, monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("disk gone")

    monkeypatch.setattr(ingest_tasks, "_invalidate_embedding_cache", boom)
    _wav(data_root / "site" / "fresh.wav", 6.0)

    job = _run(db_session, _job(db_session, embedded_dataset))

    assert job.status == IngestJobStatus.COMPLETED, job.error_message


def test_metadata_from_file_names_is_counted(db_session, embedded_dataset, data_root, birdnet, progress):
    _wav(data_root / "site" / "CIPAP07_20241111_235000.wav", 3.0)  # site + date + time
    _wav(data_root / "site" / "unnamed_take.wav", 3.0)              # nothing parseable

    job = _run(db_session, _job(db_session, embedded_dataset))

    assert job.status == IngestJobStatus.COMPLETED, job.error_message
    assert job.summary["recordings_new"] == 2
    assert job.summary["recordings_with_metadata"] == 1
    assert job.summary["recordings_missing_metadata"] == 1
    parsed = db_session.query(Recording).filter_by(file_name="CIPAP07_20241111_235000.wav").one()
    assert parsed.extra_metadata["location"] == "CIPAP07"
    assert parsed.extra_metadata["recorded_date"] == "2024-11-11"


def test_nothing_new_completes_with_zeros(db_session, embedded_dataset, birdnet, progress):
    job = _run(db_session, _job(db_session, embedded_dataset))

    assert job.status == IngestJobStatus.COMPLETED, job.error_message
    assert job.summary["files_found"] == 1
    assert job.summary["files_skipped"] == 1
    assert job.summary["recordings_new"] == 0
    assert job.summary["snippets_new"] == 0
    assert job.summary["snippets_embedded"] == 0
    assert birdnet.calls == []


def test_unreadable_file_is_counted_and_skipped(db_session, embedded_dataset, data_root, birdnet, progress):
    (data_root / "site" / "broken.wav").write_bytes(b"not audio")

    job = _run(db_session, _job(db_session, embedded_dataset))

    assert job.status == IngestJobStatus.COMPLETED, job.error_message
    assert job.summary["files_unreadable"] == 1
    assert job.summary["recordings_new"] == 0


def test_birdnet_gap_is_reported_as_failed_snippet(db_session, embedded_dataset, data_root, birdnet, progress):
    _wav(data_root / "site" / "fresh.wav", 6.0)
    birdnet.fail_windows.add(("fresh.wav", 3.0))

    job = _run(db_session, _job(db_session, embedded_dataset))

    assert job.status == IngestJobStatus.COMPLETED, job.error_message
    assert job.summary["snippets_new"] == 2
    assert job.summary["snippets_embedded"] == 1
    assert job.summary["snippets_failed"] == 1


def test_snippet_left_without_vector_is_retried_next_run(db_session, embedded_dataset, data_root, birdnet, progress):
    _wav(data_root / "site" / "fresh.wav", 6.0)
    birdnet.fail_windows.add(("fresh.wav", 3.0))
    _run(db_session, _job(db_session, embedded_dataset))
    birdnet.fail_windows.clear()
    birdnet.calls.clear()

    job = _run(db_session, _job(db_session, embedded_dataset))

    assert job.summary["snippets_new"] == 0
    assert job.summary["snippets_embedded"] == 1
    assert birdnet.calls == ["fresh.wav"]


def test_missing_folder_fails_the_job(db_session, embedded_dataset, data_root, birdnet, progress):
    embedded_dataset.dataset.source_uri = "gone"
    db_session.commit()

    job = _run(db_session, _job(db_session, embedded_dataset))

    assert job.status == IngestJobStatus.FAILED
    assert job.stage == IngestStage.FAILED
    assert "Invalid dataset path" in job.error_message
    assert job.completed_at is not None


def test_bulk_insert_skips_snippets_that_already_have_a_vector(db_session, embedded_dataset):
    existing = db_session.query(EmbeddingVector).first()
    row = {
        "snippet_id": existing.snippet_id,
        "job_id": embedded_dataset.embedding_job.id,
        "model_id": embedded_dataset.model.id,
        "vector": [0.5] * 1024,
    }

    inserted = VectorStore(db_session).bulk_insert([row])

    assert inserted == 0
    assert db_session.query(EmbeddingVector).filter_by(snippet_id=existing.snippet_id).count() == 1


# ----------------------------------------------------------------------
# Embedding failures: recorded, retried, dismissed
# ----------------------------------------------------------------------

def _failures(db_session):
    from app.models.recording_embedding_failure import RecordingEmbeddingFailure

    db_session.expire_all()
    return {
        db_session.get(Recording, f.recording_id).file_name: f
        for f in db_session.query(RecordingEmbeddingFailure).all()
    }


def _job_mode(db_session, env, mode):
    from app.models.dataset_ingest_job import IngestMode

    job = _job(db_session, env)
    job.mode = IngestMode(mode)
    db_session.commit()
    return job


def test_failed_recordings_are_recorded_with_reason(db_session, embedded_dataset, data_root, birdnet, progress):
    _wav(data_root / "site" / "broken.wav", 6.0)   # whole recording errors: 2 snippets
    _wav(data_root / "site" / "gappy.wav", 9.0)    # BirdNET misses 1 of 3
    _wav(data_root / "site" / "fine.wav", 3.0)
    birdnet.broken_files.add("broken.wav")
    birdnet.fail_windows.add(("gappy.wav", 3.0))

    job = _run(db_session, _job(db_session, embedded_dataset))

    assert job.status == IngestJobStatus.COMPLETED, job.error_message
    assert job.summary["embedding_failures"] == {"recordings": 2, "snippets": 3}
    failures = _failures(db_session)
    assert set(failures) == {"broken.wav", "gappy.wav"}
    assert failures["broken.wav"].snippets_missing == 2
    assert "corrupt header" in failures["broken.wav"].reason
    assert failures["gappy.wav"].snippets_missing == 1
    assert failures["gappy.wav"].reason == "BirdNET returned no embedding for 1 of 3 snippets"
    assert failures["broken.wav"].last_job_id == job.id


def test_retry_failed_only_skips_the_scan_and_clears_fixed_recordings(db_session, embedded_dataset, data_root, birdnet, progress):
    _wav(data_root / "site" / "broken.wav", 6.0)
    birdnet.broken_files.add("broken.wav")
    _run(db_session, _job(db_session, embedded_dataset))
    # The user fixes the file, and a new file appears that the retry must ignore.
    birdnet.broken_files.clear()
    birdnet.calls.clear()
    _wav(data_root / "site" / "later.wav", 3.0)

    job = _run(db_session, _job_mode(db_session, embedded_dataset, "retry_failed"))

    assert job.status == IngestJobStatus.COMPLETED, job.error_message
    assert birdnet.calls == ["broken.wav"]
    assert job.summary["files_found"] == 0          # no scan
    assert job.summary["snippets_embedded"] == 2
    assert job.summary["embedding_failures"] == {"recordings": 0, "snippets": 0}
    assert _failures(db_session) == {}
    assert db_session.query(Recording).filter_by(file_name="later.wav").count() == 0


def test_dismissed_failures_are_not_retried_by_normal_runs(db_session, embedded_dataset, data_root, birdnet, progress):
    from sqlalchemy.sql import func as sa_func

    _wav(data_root / "site" / "broken.wav", 6.0)
    birdnet.broken_files.add("broken.wav")
    _run(db_session, _job(db_session, embedded_dataset))
    failure = _failures(db_session)["broken.wav"]
    failure.dismissed_at = sa_func.now()
    db_session.commit()
    birdnet.calls.clear()

    job = _run(db_session, _job(db_session, embedded_dataset))

    assert birdnet.calls == []
    assert job.summary["embedding_failures"] == {"recordings": 0, "snippets": 0}
    assert _failures(db_session)["broken.wav"].dismissed_at is not None


def test_explicit_retry_includes_dismissed_and_reactivates_on_new_failure(db_session, embedded_dataset, data_root, birdnet, progress):
    from sqlalchemy.sql import func as sa_func

    _wav(data_root / "site" / "broken.wav", 6.0)
    birdnet.broken_files.add("broken.wav")
    _run(db_session, _job(db_session, embedded_dataset))
    _failures(db_session)["broken.wav"].dismissed_at = sa_func.now()
    db_session.commit()
    birdnet.calls.clear()

    job = _run(db_session, _job_mode(db_session, embedded_dataset, "retry_failed"))

    assert birdnet.calls == ["broken.wav"]         # retried although dismissed
    assert job.summary["embedding_failures"] == {"recordings": 1, "snippets": 2}
    assert _failures(db_session)["broken.wav"].dismissed_at is None
