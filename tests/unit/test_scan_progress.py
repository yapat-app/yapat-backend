"""Tests for live dataset-scan progress (services/scan_progress + scan_recordings callback)."""

import pytest

from app.config import settings
from app.models.dataset import Dataset
from app.services import scan_progress
from app.services.dataset_service import DatasetService


class _FakePipeline:
    def __init__(self, client):
        self._client = client
        self._ops = []

    def __getattr__(self, name):
        def queue(*args, **kwargs):
            self._ops.append((name, args, kwargs))
            return self
        return queue

    def execute(self):
        return [getattr(self._client, name)(*a, **kw) for name, a, kw in self._ops]


class FakeRedis:
    """Just enough of redis-py's hash API for scan_progress."""

    def __init__(self):
        self.hashes = {}

    def pipeline(self):
        return _FakePipeline(self)

    def hset(self, key, mapping):
        self.hashes.setdefault(key, {}).update({k: str(v).encode() for k, v in mapping.items()})

    def hgetall(self, key):
        return {k.encode(): v for k, v in self.hashes.get(key, {}).items()}

    def delete(self, key):
        self.hashes.pop(key, None)

    def expire(self, key, seconds):
        return True


@pytest.fixture
def fake_redis(monkeypatch):
    client = FakeRedis()
    monkeypatch.setattr(scan_progress, "_redis", lambda: client)
    return client


# ---------------------------------------------------------------- build_progress

def test_unknown_scan_is_idle():
    p = scan_progress.build_progress(7, {})
    assert p["stage"] == "idle"
    assert p["percent"] is None
    assert p["stalled"] is False


def test_percent_spans_reading_then_saving():
    reading = scan_progress.build_progress(1, {"stage": "reading", "done": "50", "total": "100", "ts": "0"}, now=0)
    saving_end = scan_progress.build_progress(1, {"stage": "saving", "done": "100", "total": "100", "ts": "0"}, now=0)
    completed = scan_progress.build_progress(1, {"stage": "completed", "created": "100"}, now=0)
    assert reading["percent"] == 47.5
    assert saving_end["percent"] == 100.0
    assert completed["percent"] == 100.0
    assert completed["recordings_created"] == 100


def test_stalled_only_in_working_stages():
    old = str(0)
    late = scan_progress.STALL_AFTER_SECONDS + 1
    working = scan_progress.build_progress(1, {"stage": "reading", "done": "1", "total": "9", "ts": old}, now=late)
    queued = scan_progress.build_progress(1, {"stage": "queued", "ts": old}, now=late)
    assert working["stalled"] is True
    # Waiting behind a long embedding job for a free worker is normal, not a stall.
    assert queued["stalled"] is False


# ---------------------------------------------------------------- writes / reads

def test_mark_queued_clears_previous_failure(fake_redis):
    scan_progress.mark_failed(3, "disk gone")
    scan_progress.mark_queued(3)
    p = scan_progress.get_progress(3)
    assert p["stage"] == "queued"
    assert p["error_message"] is None


def test_active_scans_skip_completed_and_unknown(fake_redis):
    scan_progress.mark_queued(1)
    scan_progress.update(2, "reading", done=5, total=10)
    scan_progress.mark_failed(3, "boom")
    scan_progress.mark_completed(4, 12)
    active = scan_progress.get_active_scans([1, 2, 3, 4, 5])
    assert set(active) == {1, 2, 3}
    assert active[2]["done"] == 5
    assert active[3]["error_message"] == "boom"


def test_redis_unavailable_fails_soft(monkeypatch):
    monkeypatch.setattr(scan_progress, "_redis", lambda: None)
    scan_progress.mark_queued(1)
    scan_progress.update(1, "reading", done=1, total=2)
    assert scan_progress.get_progress(1)["stage"] == "idle"
    assert scan_progress.get_active_scans([1]) == {}


def test_reporter_throttles_but_keeps_stage_changes_and_stage_end(fake_redis, monkeypatch):
    writes = []
    monkeypatch.setattr(scan_progress, "update", lambda ds, stage, done=None, total=None: writes.append((stage, done)))
    report = scan_progress.make_reporter(1, min_interval=1000, min_step=25, clock=lambda: 0.0)

    report("listing", 0, 0)
    for i in range(0, 101):
        report("reading", i, 100)
    report("saving", 0, 100)

    assert writes[0] == ("listing", 0)
    reading = [d for s, d in writes if s == "reading"]
    assert reading == [0, 25, 50, 75, 100]
    assert writes[-1] == ("saving", 0)


# ---------------------------------------------------------------- scan_recordings

@pytest.fixture
def scan_dataset_dir(db_session, temp_data_root, tiny_wav_file, monkeypatch):
    monkeypatch.setattr(settings, "DATA_ROOT", str(temp_data_root))
    folder = temp_data_root / "progress_ds"
    folder.mkdir()
    for name in ("a.wav", "b.wav", "c.wav"):
        tiny_wav_file(folder / name)
    ds = Dataset(team_id=None, name="P", description="d", source_uri="progress_ds")
    db_session.add(ds)
    db_session.commit()
    return ds


def test_scan_recordings_reports_each_stage(db_session, scan_dataset_dir):
    events = []
    new = DatasetService(db_session).scan_recordings(
        scan_dataset_dir, progress=lambda stage, done, total: events.append((stage, done, total))
    )

    assert len(new) == 3
    stages = [stage for stage, _, _ in events]
    assert stages[0] == "listing"
    assert stages.index("reading") < stages.index("saving")
    assert ("reading", 3, 3) in events
    assert events[-1] == ("saving", 3, 3)


def test_scan_recordings_survives_failing_callback(db_session, scan_dataset_dir):
    def broken(stage, done, total):
        raise RuntimeError("redis exploded")

    new = DatasetService(db_session).scan_recordings(scan_dataset_dir, progress=broken)
    assert len(new) == 3
