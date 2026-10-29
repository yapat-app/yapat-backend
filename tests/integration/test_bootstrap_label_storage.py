"""
Bootstrap training (first model from users' labels) keeps both label stores right.

Bootstrap = POST /pam-al/feedback while the dataset has no active checkpoint
(PAMActiveLearningService._submit_bootstrap_feedback). Once the threshold is
reached the same request trains the first model (sync train_from_scratch).

Pinned here:
* every bootstrap label also lands in the canonical `annotations` table, as
  feedback on a trained model does (dataset stats and exports read only that);
* the bootstrap training does not copy users' labels in as GROUND_TRUTH, so a
  later REJECT removes the label from training data.
"""

import numpy as np
import pytest

from app.config import settings
from app.core import taxonomy
from app.models.annotation import Annotation
from app.models.dataset import Dataset, DatasetType
from app.models.embedding import (
    EmbeddingJob,
    EmbeddingJobStatus,
    EmbeddingModel,
    EmbeddingVector,
    SnippetSet,
    SnippetSetStatus,
)
from app.models.pam_active_learning import ALAnnotationSource, ALSnippetAnnotation
from app.models.recording import Recording
from app.models.snippet import Snippet
from app.models.user import User, UserRole
from app.schemas.pam_active_learning import ALFeedbackSubmit
from app.services.pam_al import _annotation_helpers as ann_h
from app.services.pam_al import _ann_index_cache, _embedding_cache
from app.services.pam_al import _checkpoint_helpers as ckpt_h
from app.services.pam_al.service import PAMActiveLearningService

DIM = 16
FAMILY = "default"


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "PAM_CHECKPOINTS_DIR", str(tmp_path / "pam" / "checkpoints"))
    monkeypatch.setattr(settings, "EXPLORE_CACHE_DIR", str(tmp_path / "explore"))
    # Label codes like "a" must not go to GBIF: they fall through to a custom taxon.
    monkeypatch.setattr(taxonomy, "match_species_name", lambda name: None)
    _embedding_cache._MEMORY_CACHE.clear()
    _ann_index_cache._MEMORY_CACHE.clear()
    yield
    _embedding_cache._MEMORY_CACHE.clear()
    _ann_index_cache._MEMORY_CACHE.clear()


@pytest.fixture
def env(db_session):
    """40 embedded snippets in two clusters, no model yet, bootstrap threshold 3."""
    rng = np.random.default_rng(7)
    em = EmbeddingModel(name="BirdNET", window_size=3.0, step_size=3.0, overlap=0.0)
    ds = Dataset(name="bootstrap", source_uri="site", dataset_type=DatasetType.PAM,
                 retrain_after_threshold=3)
    user = User(username="bootstrap_user", hashed_password="x", role=UserRole.ADMIN)
    db_session.add_all([em, ds, user])
    db_session.flush()
    ss = SnippetSet(dataset_id=ds.id, embedding_model_id=em.id, window_size=3.0, step_size=3.0,
                    overlap=0.0, status=SnippetSetStatus.READY)
    db_session.add(ss)
    db_session.flush()
    ds.default_snippet_set_id = ss.id
    rec = Recording(dataset_id=ds.id, file_path="site/r.wav", file_name="r.wav",
                    duration=200.0, sample_rate=16000)
    ejob = EmbeddingJob(dataset_id=ds.id, embedding_model_id=em.id, snippet_set_id=ss.id,
                        status=EmbeddingJobStatus.COMPLETED)
    db_session.add_all([ejob, rec])
    db_session.flush()
    centres = rng.normal(size=(2, DIM)) * 4
    snippet_ids = []
    for i in range(40):
        sn = Snippet(recording_id=rec.id, snippet_set_id=ss.id, start_time=i * 3.0,
                     end_time=i * 3.0 + 3.0, duration=3.0)
        db_session.add(sn)
        db_session.flush()
        vec = centres[i % 2] + rng.normal(size=DIM)
        db_session.add(EmbeddingVector(snippet_id=sn.id, embedding_job_id=ejob.id,
                                       embedding_model_id=em.id, dim=DIM,
                                       vector=vec.astype(np.float32).tolist()))
        snippet_ids.append(sn.id)
    db_session.commit()
    return type("Env", (), {"dataset": ds, "user": user, "em": em, "snippets": snippet_ids})


def _feedback(env, snippet_id, labels, action="MODIFY"):
    return ALFeedbackSubmit(
        dataset_id=env.dataset.id, model_family_name=FAMILY, snippet_id=snippet_id,
        embedding_model_id=env.em.id, action=action, labels=labels, user_id=env.user.id,
    )


def _al_rows(db_session, env, source):
    return (
        db_session.query(ALSnippetAnnotation)
        .filter(ALSnippetAnnotation.dataset_id == env.dataset.id, ALSnippetAnnotation.source == source)
        .all()
    )


def test_bootstrap_label_is_written_to_both_stores(db_session, env):
    svc = PAMActiveLearningService(db_session)

    result = svc.submit_feedback(_feedback(env, env.snippets[0], ["a", "b"]))

    assert result["retrain_triggered"] is False
    assert {r.label for r in _al_rows(db_session, env, ALAnnotationSource.USER)} == {"a", "b"}
    canonical = db_session.query(Annotation).filter(Annotation.snippet_id == env.snippets[0]).all()
    assert sorted(a.resolved_name_snapshot for a in canonical) == ["a", "b"]
    assert all(a.taxon_id.startswith("custom:") for a in canonical)


def test_persist_annotations_false_skips_the_canonical_write(db_session, env):
    body = _feedback(env, env.snippets[0], ["a"])
    body.persist_annotations = False

    PAMActiveLearningService(db_session).submit_feedback(body)

    assert db_session.query(Annotation).count() == 0
    assert len(_al_rows(db_session, env, ALAnnotationSource.USER)) == 1


def test_bootstrap_training_keeps_labels_as_user_only_and_reject_removes_them(db_session, env):
    svc = PAMActiveLearningService(db_session)
    s1, s2, s3 = env.snippets[0], env.snippets[1], env.snippets[2]
    svc.submit_feedback(_feedback(env, s1, ["a"]))
    svc.submit_feedback(_feedback(env, s2, ["b"]))

    result = svc.submit_feedback(_feedback(env, s3, ["a"]))   # threshold 3: trains

    assert result["retrain_triggered"] is True
    ckpt = ckpt_h.get_active_checkpoint_for_model_family(db_session, env.dataset.id, FAMILY)
    assert ckpt is not None and ckpt.id == result["active_checkpoint_id"]
    assert _al_rows(db_session, env, ALAnnotationSource.GROUND_TRUTH) == []
    assert db_session.query(Annotation.snippet_id).distinct().count() == 3

    # Rejecting a bootstrap label takes it out of training data and the canonical table.
    svc.submit_feedback(_feedback(env, s1, None, action="REJECT"))

    trusted = ann_h.get_trusted_annotations(db_session, env.dataset.id)
    assert s1 not in trusted
    assert set(trusted) == {s2, s3}
    assert db_session.query(Annotation).filter(Annotation.snippet_id == s1).count() == 0
