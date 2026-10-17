"""
Scoring stage of "add recordings" (yapat-backend#70, step 5).

The parity test is the important one: full inference scores every snippet, then
the predictions of a few snippets are deleted and re-created by the incremental
scorer. Values that depend only on the snippet must match exactly; the ones that
depend on the population must match closely.
"""

import types

import numpy as np
import pytest
import torch
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
from app.models.pam_active_learning import (
    ALAnnotationSource,
    ALModelCheckpoint,
    ALModelFamilyState,
    ALModelStatus,
    ALPrediction,
    ALSnippetAnnotation,
)
from app.models.recording import Recording
from app.models.snippet import Snippet
from app.services.explore import storage
from app.services.explore.builders import model_key
from app.services.explore.hooks import append_to_model_layer
from app.services.pam_al import _ann_index_cache, _embedding_cache
from app.services.pam_al import _incremental_scoring as inc

DIM = 16
LABELS = ["a", "b", "c"]


class FakeLinearHead(torch.nn.Module):
    """Linear multilabel head whose features are its inputs (like the PAM linear model)."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.fc = torch.nn.Linear(DIM, len(LABELS))
        self.label_order = LABELS

    def predict_with_features(self, x, threshold=0.3):
        probs = torch.sigmoid(self.fc(x))
        return x, probs, (probs >= threshold).float()


@pytest.fixture(autouse=True)
def isolated_caches(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "PAM_CHECKPOINTS_DIR", str(tmp_path / "pam" / "checkpoints"))
    monkeypatch.setattr(settings, "EXPLORE_CACHE_DIR", str(tmp_path / "explore"))
    _embedding_cache._MEMORY_CACHE.clear()
    _ann_index_cache._MEMORY_CACHE.clear()
    yield
    _embedding_cache._MEMORY_CACHE.clear()
    _ann_index_cache._MEMORY_CACHE.clear()


@pytest.fixture
def model(monkeypatch):
    head = FakeLinearHead().eval()
    from app.services.pam_al import _checkpoint_helpers as ckpt_h

    monkeypatch.setattr(ckpt_h, "load_model_from_checkpoint", lambda ckpt, device: head)
    return head


@pytest.fixture
def scored_set(db_session):
    """300 snippets with random vectors, 15 labelled, one active checkpoint."""
    rng = np.random.default_rng(42)
    em = EmbeddingModel(name="BirdNET", window_size=3.0, step_size=3.0, overlap=0.0)
    ds = Dataset(name="scoring", source_uri="site", dataset_type=DatasetType.PAM)
    db_session.add_all([em, ds])
    db_session.flush()
    ss = SnippetSet(dataset_id=ds.id, embedding_model_id=em.id, window_size=3.0, step_size=3.0,
                    overlap=0.0, status=SnippetSetStatus.READY)
    db_session.add(ss)
    db_session.flush()
    ejob = EmbeddingJob(dataset_id=ds.id, embedding_model_id=em.id, snippet_set_id=ss.id,
                        status=EmbeddingJobStatus.COMPLETED)
    rec = Recording(dataset_id=ds.id, file_path="site/r.wav", file_name="r.wav", duration=900.0, sample_rate=16000)
    db_session.add_all([ejob, rec])
    db_session.flush()
    # Three clusters so density/diversity have structure.
    centres = rng.normal(size=(3, DIM)) * 4
    snippet_ids = []
    for i in range(300):
        sn = Snippet(recording_id=rec.id, snippet_set_id=ss.id, start_time=i * 3.0,
                     end_time=i * 3.0 + 3.0, duration=3.0)
        db_session.add(sn)
        db_session.flush()
        vec = (centres[i % 3] + rng.normal(size=DIM)).astype(np.float32)
        db_session.add(EmbeddingVector(snippet_id=sn.id, embedding_job_id=ejob.id,
                                       embedding_model_id=em.id, dim=DIM, vector=vec.tolist()))
        snippet_ids.append(sn.id)
    for sid in snippet_ids[:15]:
        db_session.add(ALSnippetAnnotation(dataset_id=ds.id, snippet_id=sid, label="a",
                                           source=ALAnnotationSource.GROUND_TRUTH))
    ckpt = ALModelCheckpoint(
        dataset_id=ds.id, model_family_name="pam", version="v1", checkpoint_path="unused.pt",
        label_config_path="unused.json", status=ALModelStatus.AVAILABLE,
        hyperparameters={"embedding_model_id": em.id, "label_order": LABELS,
                         "resolved_snippet_set_id": ss.id},
    )
    db_session.add(ckpt)
    db_session.flush()
    db_session.add(ALModelFamilyState(dataset_id=ds.id, model_family_name="pam",
                                      active_model_checkpoint_id=ckpt.id))
    ds.default_snippet_set_id = ss.id
    db_session.commit()
    return types.SimpleNamespace(dataset=ds, snippet_set=ss, embedding_job=ejob, model=em,
                                 checkpoint=ckpt, snippet_ids=snippet_ids)


def _full_inference(db_session, env, model):
    from app.services.pam_al import _annotation_helpers as ann_h
    from app.services.pam_al import _data_helpers as data_h
    from app.services.pam_al import _inference_helpers as inf_h

    X, rows = data_h.load_embeddings(db_session, env.snippet_set.id, env.model.id)
    labeled = ann_h.get_labeled_snippet_ids_for_dataset(db_session, env.dataset.id)
    inf_h.run_and_store_inference(db_session, env.dataset.id, env.checkpoint, model, X, rows, LABELS, labeled)


def _predictions(db_session, ckpt_id):
    db_session.expire_all()
    return {p.snippet_id: p for p in db_session.query(ALPrediction).filter_by(model_checkpoint_id=ckpt_id)}


def test_incremental_matches_full_inference(db_session, scored_set, model, monkeypatch):
    # Turn off diversity's greedy redundancy correction, the one step the
    # incremental scorer does not reproduce: with 100 centres x 200 neighbours it
    # rewrites every row of this 300-row set, while on a real set it touches
    # ~20k rows of millions. See the next test for that difference.
    from app.services.pam_al import _inference_helpers as inf_h

    monkeypatch.setattr(inf_h, "DIVERSITY_NUM_CENTERS", 0)
    _full_inference(db_session, scored_set, model)
    full = {sid: (p.predicted_probabilities, p.predicted_labels, p.uncertainty, p.diversity, p.density, p.composite_score)
            for sid, p in _predictions(db_session, scored_set.checkpoint.id).items()}
    rescored = scored_set.snippet_ids[-20:]
    db_session.query(ALPrediction).filter(ALPrediction.snippet_id.in_(rescored)).delete(synchronize_session=False)
    db_session.commit()

    n = inc.score_unscored_snippets(db_session, scored_set.checkpoint, scored_set.snippet_set.id)

    assert n == 20
    after = _predictions(db_session, scored_set.checkpoint.id)
    assert len(after) == 300
    for sid in rescored:
        probs, labels, unc, div, dens, comp = full[sid]
        p = after[sid]
        # Depend only on the snippet: identical.
        assert p.predicted_labels == labels
        assert p.predicted_probabilities == pytest.approx(probs, abs=1e-6)
        assert p.uncertainty == pytest.approx(unc, abs=1e-6)
        # Distance to the nearest labelled point, as in full inference.
        assert p.diversity == pytest.approx(div, rel=1e-4)
        # Same neighbours over the same index.
        assert p.density == pytest.approx(dens, rel=1e-3)
        # z-scored against the 265 remaining unlabelled rows instead of all 285.
        assert p.composite_score == pytest.approx(comp, abs=0.1)


def test_redundancy_correction_is_the_only_diversity_gap(db_session, scored_set, model):
    """With the correction on, full diversity can only be lower, never higher."""
    _full_inference(db_session, scored_set, model)
    full = {sid: p.diversity for sid, p in _predictions(db_session, scored_set.checkpoint.id).items()}
    rescored = scored_set.snippet_ids[-20:]
    db_session.query(ALPrediction).filter(ALPrediction.snippet_id.in_(rescored)).delete(synchronize_session=False)
    db_session.commit()

    inc.score_unscored_snippets(db_session, scored_set.checkpoint, scored_set.snippet_set.id)

    after = _predictions(db_session, scored_set.checkpoint.id)
    for sid in rescored:
        assert after[sid].diversity >= full[sid] - 1e-5


def test_labelled_unscored_snippet_gets_probabilities_but_no_scores(db_session, scored_set, model):
    _full_inference(db_session, scored_set, model)
    labelled = scored_set.snippet_ids[0]
    db_session.query(ALPrediction).filter_by(snippet_id=labelled).delete()
    db_session.commit()

    assert inc.score_unscored_snippets(db_session, scored_set.checkpoint, scored_set.snippet_set.id) == 1

    p = _predictions(db_session, scored_set.checkpoint.id)[labelled]
    assert set(p.predicted_probabilities) == set(LABELS)
    assert p.composite_score is None and p.density is None


def test_nothing_to_score_does_not_load_anything(db_session, scored_set, model, monkeypatch):
    _full_inference(db_session, scored_set, model)
    from app.services.pam_al import _data_helpers as data_h

    monkeypatch.setattr(data_h, "load_embeddings", lambda *a: pytest.fail("loaded embeddings"))

    assert inc.score_unscored_snippets(db_session, scored_set.checkpoint, scored_set.snippet_set.id) == 0


def test_model_layer_gets_new_rows_appended(db_session, scored_set, model):
    _full_inference(db_session, scored_set, model)  # publishes the layer
    rescored = scored_set.snippet_ids[-20:]
    db_session.query(ALPrediction).filter(ALPrediction.snippet_id.in_(rescored)).delete(synchronize_session=False)
    db_session.commit()
    key = model_key(scored_set.checkpoint.id, scored_set.snippet_set.id)
    before = storage.read_pointer("model", key).version

    inc.score_unscored_snippets(db_session, scored_set.checkpoint, scored_set.snippet_set.id)

    pointer = storage.read_pointer("model", key)
    assert pointer.version != before
    layer = storage.load_layer("model", key, pointer.version)
    ids = np.asarray(layer["snippet_ids"])
    assert ids.tolist() == sorted(scored_set.snippet_ids)
    assert layer["scores"].shape == (300, 4)
    assert layer.meta["source"] == "inference+incremental"


def test_append_with_other_label_order_invalidates(scored_set, db_session, model):
    _full_inference(db_session, scored_set, model)
    key = model_key(scored_set.checkpoint.id, scored_set.snippet_set.id)

    append_to_model_layer(
        scored_set.checkpoint.id, scored_set.snippet_set.id, [999_999],
        np.zeros((1, 2)), np.zeros((1, 2)), np.zeros((1, 4), dtype=np.float32), ["x", "y"],
    )

    assert storage.read_pointer("model", key) is None


def test_population_stats_match_numpy(db_session, scored_set, model):
    _full_inference(db_session, scored_set, model)
    stored = [p for p in _predictions(db_session, scored_set.checkpoint.id).values() if p.composite_score is not None]

    stats = inc.population_stats(db_session, scored_set.checkpoint.id)

    for name in inc.SCORE_FIELDS:
        values = np.asarray([getattr(p, name) for p in stored], dtype=np.float64)
        n, mean, std = stats[name]
        assert n == values.size
        assert mean == pytest.approx(values.mean(), rel=1e-6)
        assert std == pytest.approx(values.std(ddof=1), rel=1e-6)


# ----------------------------------------------------------------------
# Celery task: completes the ingest job
# ----------------------------------------------------------------------

def _ingest_job(db_session, env):
    job = DatasetIngestJob(
        dataset_id=env.dataset.id, snippet_set_id=env.snippet_set.id, embedding_job_id=env.embedding_job.id,
        status=IngestJobStatus.RUNNING, stage=IngestStage.SCORING, summary=empty_summary(),
    )
    db_session.add(job)
    db_session.commit()
    return job


@pytest.fixture
def task_sessions(engine, monkeypatch):
    from app.tasks import ingest_tasks, pam_al_tasks

    factory = sessionmaker(bind=engine)
    monkeypatch.setattr(ingest_tasks, "SessionLocal", factory)
    monkeypatch.setattr(pam_al_tasks, "SessionLocal", factory)
    monkeypatch.setattr(ingest_tasks.ingest_progress, "start_stage", lambda *a, **k: None)
    monkeypatch.setattr(ingest_tasks.ingest_progress, "incr", lambda *a, **k: None)


def _run_task(db_session, job_id):
    from app.tasks.pam_al_tasks import pam_al_score_ingested_snippets

    pam_al_score_ingested_snippets.apply(args=[job_id])
    db_session.expire_all()
    return db_session.get(DatasetIngestJob, job_id)


def test_task_scores_and_completes_job(db_session, scored_set, model, task_sessions):
    _full_inference(db_session, scored_set, model)
    db_session.query(ALPrediction).filter(ALPrediction.snippet_id.in_(scored_set.snippet_ids[-5:])).delete(
        synchronize_session=False)
    db_session.commit()
    job = _ingest_job(db_session, scored_set)

    job = _run_task(db_session, job.id)

    assert job.status == IngestJobStatus.COMPLETED
    assert job.stage == IngestStage.COMPLETED
    assert job.summary["snippets_scored"] == 5
    assert job.summary["scoring_skipped_reason"] is None


def test_task_without_active_checkpoint_skips_scoring(db_session, scored_set, task_sessions):
    db_session.query(ALModelFamilyState).delete()
    db_session.commit()
    job = _ingest_job(db_session, scored_set)

    job = _run_task(db_session, job.id)

    assert job.status == IngestJobStatus.COMPLETED
    assert job.summary["scoring_skipped_reason"] == "no_active_checkpoint"


def test_task_failure_still_completes_the_run(db_session, scored_set, task_sessions, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("model file missing")

    monkeypatch.setattr(inc, "score_unscored_snippets", boom)
    job = _ingest_job(db_session, scored_set)

    job = _run_task(db_session, job.id)

    assert job.status == IngestJobStatus.COMPLETED
    assert job.summary["scoring_skipped_reason"] == "scoring_failed"
    assert "model file missing" in job.error_message


def test_memory_embedding_cache_notices_appended_vectors(db_session, scored_set):
    from app.services.pam_al import _data_helpers as data_h

    X1, _ = data_h.load_embeddings(db_session, scored_set.snippet_set.id, scored_set.model.id)
    sn = Snippet(recording_id=db_session.query(Recording.id).first()[0], snippet_set_id=scored_set.snippet_set.id,
                 start_time=5000.0, end_time=5003.0, duration=3.0)
    db_session.add(sn)
    db_session.flush()
    db_session.add(EmbeddingVector(snippet_id=sn.id, embedding_job_id=scored_set.embedding_job.id,
                                   embedding_model_id=scored_set.model.id, dim=DIM, vector=[0.0] * DIM))
    db_session.commit()

    X2, rows2 = data_h.load_embeddings(db_session, scored_set.snippet_set.id, scored_set.model.id)

    assert X2.shape[0] == X1.shape[0] + 1
    assert rows2[-1]["snippet_id"] == sn.id
