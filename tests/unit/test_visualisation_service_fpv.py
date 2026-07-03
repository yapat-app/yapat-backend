"""Unit tests for VISService.generate_fpv_for_dataset_embeddings's snippet-set
resolution branching (app/services/visualisation_service.py).

DR is keyed by dataset_id and can in principle span multiple SnippetSets for
one embedding_model_id, while AL's embedding cache (load_embeddings_cached)
is keyed by a single snippet_set_id. The service only reuses the cache when
exactly one matching SnippetSet is found; otherwise it falls back to the
original direct query. These tests isolate that branching decision -- not
the full DR compute (_compute_visualizations) or persistence path, which are
covered elsewhere / are straightforward passthroughs once X and snippet_ids
are resolved correctly.

Uses an in-memory SQLite DB with StaticPool (a single shared connection --
without it, sqlite:///:memory: gives each new connection its own empty DB,
which breaks under FastAPI/SQLAlchemy's default connection handling) and an
explicit table subset (Base.metadata.create_all(tables=[...])) to avoid an
unrelated JSONB-on-sqlite compile error from custom_taxonomy models pulled
in transitively via the full metadata.
"""
import sys
import types

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base
from app.models.embedding import EmbeddingVector, SnippetSet
from app.models.snippet import Snippet
from app.schemas.visualisation import FPVDatasetRequest


@pytest.fixture(autouse=True)
def reset_db():
    """Override conftest.py's autouse reset_db(engine) for this module.

    That fixture does Base.metadata.create_all(bind=engine) with no table
    filter, against the shared session-scoped sqlite engine. Importing
    app.schemas.visualisation here pulls in custom_taxonomy models
    transitively, which register a JSONB column on Base.metadata -- sqlite
    can't compile JSONB, so the unfiltered create_all crashes. This module
    uses its own isolated, table-filtered engine (db_session below) instead,
    so the shared fixture's DB setup is not needed for these tests.
    """
    yield


@pytest.fixture()
def db_session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine, tables=[
        SnippetSet.__table__, Snippet.__table__, EmbeddingVector.__table__,
    ])
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def _make_snippet_set(db, dataset_id, embedding_model_id):
    ss = SnippetSet(
        dataset_id=dataset_id, embedding_model_id=embedding_model_id,
        window_size=1.0, step_size=1.0, overlap=0.0,
    )
    db.add(ss)
    db.flush()
    return ss.id


def _make_snippet_with_embedding(db, snippet_set_id, recording_id, embedding_model_id, vector):
    snippet = Snippet(recording_id=recording_id, snippet_set_id=snippet_set_id,
                       start_time=0.0, end_time=1.0, duration=1.0)
    db.add(snippet)
    db.flush()
    ev = EmbeddingVector(
        snippet_id=snippet.id, embedding_job_id=1,
        embedding_model_id=embedding_model_id, dim=len(vector), vector=vector,
    )
    db.add(ev)
    db.flush()
    return snippet.id


class TestSnippetSetResolution:
    def test_single_snippet_set_uses_cache(self, db_session, monkeypatch):
        """Exactly one matching SnippetSet -> route through load_embeddings_cached."""
        from app.services.visualisation_service import VISService

        ss_id = _make_snippet_set(db_session, dataset_id=1, embedding_model_id=1)
        expected_snippet_ids = [
            _make_snippet_with_embedding(db_session, ss_id, recording_id=10,
                                          embedding_model_id=1, vector=[0.1, 0.2])
            for _ in range(3)
        ]
        db_session.commit()

        captured = {}

        def fake_load_embeddings_cached(db, snippet_set_id, embedding_model_id):
            captured["snippet_set_id"] = snippet_set_id
            captured["embedding_model_id"] = embedding_model_id
            X = np.array([[0.1, 0.2]] * 3, dtype=np.float32)
            rows = [{"snippet_id": sid} for sid in expected_snippet_ids]
            return X, rows

        # generate_fpv_for_dataset_embeddings does a *local* import of
        # load_embeddings_cached (`from app.services.pam_al._embedding_cache
        # import ...`). The real app.services.pam_al package (via its
        # __init__.py -> service.py -> active_learning) requires torch,
        # which isn't installed in this sandbox -- but that's an environment
        # gap, not something this test should depend on. Stub both the
        # parent package and the submodule in sys.modules so the local
        # import resolves to our fake without executing the real package's
        # __init__.py / torch-dependent chain at all.
        fake_pkg = types.ModuleType("app.services.pam_al")
        fake_submodule = types.ModuleType("app.services.pam_al._embedding_cache")
        fake_submodule.load_embeddings_cached = fake_load_embeddings_cached
        monkeypatch.setitem(sys.modules, "app.services.pam_al", fake_pkg)
        monkeypatch.setitem(sys.modules, "app.services.pam_al._embedding_cache", fake_submodule)
        monkeypatch.setattr(
            "app.services.visualisation_service.VISService._compute_visualizations",
            lambda self, X, run_3d: {},
        )
        monkeypatch.setattr(
            "app.services.visualisation_service.persist_fpv_vis_dataset_rows",
            lambda **kwargs: captured.setdefault("persisted", kwargs),
        )
        monkeypatch.setattr(
            "app.services.visualisation_service.VISService.get_fpv_for_dataset_embeddings",
            lambda self, body: "SENTINEL_RESPONSE",
        )

        svc = VISService(db_session)
        body = FPVDatasetRequest(dataset_id=1, embedding_model_id=1, run_3d=False)
        result = svc.generate_fpv_for_dataset_embeddings(body)

        assert result == "SENTINEL_RESPONSE"
        assert captured["snippet_set_id"] == ss_id
        assert captured["embedding_model_id"] == 1
        assert captured["persisted"]["snippet_ids"] == expected_snippet_ids

    def test_zero_snippet_sets_falls_back_to_direct_query(self, db_session, monkeypatch):
        """No SnippetSet row matches dataset_id/embedding_model_id at all ->
        direct query path, which correctly raises since there's no data."""
        from app.services.visualisation_service import VISService

        # No stubbing of load_embeddings_cached here: with zero matching
        # SnippetSets, the code never takes the cache branch, so it never
        # even imports app.services.pam_al -- proven by this test passing
        # without the torch-avoidance sys.modules stub used in the
        # single-snippet-set test above.
        svc = VISService(db_session)
        body = FPVDatasetRequest(dataset_id=999, embedding_model_id=1, run_3d=False)
        with pytest.raises(ValueError, match="No embeddings found"):
            svc.generate_fpv_for_dataset_embeddings(body)

    def test_multiple_snippet_sets_falls_back_to_direct_query(self, db_session, monkeypatch):
        """Two SnippetSet rows for the same (dataset_id, embedding_model_id) --
        e.g. a re-segmentation left an old one behind -- must not silently
        drop data by picking one; falls back to the direct query, which
        aggregates across all of them (existing, pre-cache-reuse behavior)."""
        from app.services.visualisation_service import VISService

        ss_a = _make_snippet_set(db_session, dataset_id=2, embedding_model_id=1)
        ss_b = _make_snippet_set(db_session, dataset_id=2, embedding_model_id=1)
        ids_a = [_make_snippet_with_embedding(db_session, ss_a, recording_id=20,
                                               embedding_model_id=1, vector=[1.0, 0.0])
                 for _ in range(2)]
        ids_b = [_make_snippet_with_embedding(db_session, ss_b, recording_id=21,
                                               embedding_model_id=1, vector=[0.0, 1.0])
                 for _ in range(2)]
        db_session.commit()

        # Same reasoning as the zero-snippet-set test: two matching
        # SnippetSets also skips the cache branch entirely, so
        # app.services.pam_al is never imported here either.
        captured = {}
        def _fake_compute(self, X, run_3d):
            captured["X"] = X
            return {}

        monkeypatch.setattr(
            "app.services.visualisation_service.VISService._compute_visualizations",
            _fake_compute,
        )
        monkeypatch.setattr(
            "app.services.visualisation_service.persist_fpv_vis_dataset_rows",
            lambda **kwargs: captured.setdefault("persisted", kwargs),
        )
        monkeypatch.setattr(
            "app.services.visualisation_service.VISService.get_fpv_for_dataset_embeddings",
            lambda self, body: "SENTINEL_RESPONSE",
        )

        svc = VISService(db_session)
        body = FPVDatasetRequest(dataset_id=2, embedding_model_id=1, run_3d=False)
        result = svc.generate_fpv_for_dataset_embeddings(body)

        assert result == "SENTINEL_RESPONSE"
        # both snippet sets' snippets present -- nothing silently dropped
        assert set(captured["persisted"]["snippet_ids"]) == set(ids_a + ids_b)
