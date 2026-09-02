"""
Retrain-time feedback sync must not overwrite newer annotator decisions.

`sync_feedback_events_to_annotations` replays every feedback event since the
last completed retrain. On 2026-09-01 that replayed 2.5 months of backlog over
labels annotators had since corrected in the classic hub: 1867 rows rewritten,
35 of them contradicting the canonical table (e.g. snippet 84223 was relabelled
Colibri serrirostris on 07-13, and the 07-02 feedback event put Augastes
scutatus back).
"""

from datetime import datetime, timedelta, timezone

import pytest

from app.models.annotation import Annotation
from app.models.dataset import Dataset
from app.models.embedding import EmbeddingModel, SnippetSet
from app.models.pam_active_learning import (
    ALAnnotationSource,
    ALFeedbackAction,
    ALFeedbackEvent,
    ALModelCheckpoint,
    ALModelType,
    ALSnippetAnnotation,
)
from app.models.recording import Recording
from app.models.snippet import Snippet
from app.models.user import User
from app.services.pam_al._feedback_helpers import sync_feedback_events_to_annotations

T0 = datetime(2026, 7, 2, 14, 50, tzinfo=timezone.utc)


@pytest.fixture
def env(db_session):
    db = db_session
    user = User(username="sara", hashed_password="x")
    ds = Dataset(name="D", source_uri="dummy")
    model = EmbeddingModel(
        name="birdnet", version="2.4", window_size=3.0, step_size=3.0, overlap=0.0
    )
    db.add_all([user, ds, model])
    db.commit()

    ss = SnippetSet(
        dataset_id=ds.id, embedding_model_id=model.id,
        window_size=3.0, step_size=3.0, overlap=0.0,
    )
    db.add(ss)
    db.commit()

    rec = Recording(dataset_id=ds.id, file_path="a.wav", file_name="a.wav", duration=10.0)
    db.add(rec)
    db.commit()

    snippet = Snippet(
        recording_id=rec.id, snippet_set_id=ss.id,
        start_time=0.0, end_time=3.0, duration=3.0,
    )
    ckpt = ALModelCheckpoint(
        dataset_id=ds.id, model_family_name="fam", version="v1",
        checkpoint_path="", label_config_path="",
        model_type=ALModelType.PAM_MLP_MULTILABEL,
    )
    db.add_all([snippet, ckpt])
    db.commit()

    return {"db": db, "ds": ds.id, "snippet": snippet.id, "user": user.id, "ckpt": ckpt.id}


def _event(env, labels, when, action=ALFeedbackAction.MODIFY):
    return ALFeedbackEvent(
        dataset_id=env["ds"], model_checkpoint_id=env["ckpt"],
        snippet_id=env["snippet"], user_id=env["user"],
        action=action, final_labels=labels, created_at=when,
    )


def _al_labels(env):
    return {
        row.label
        for row in env["db"].query(ALSnippetAnnotation).filter(
            ALSnippetAnnotation.snippet_id == env["snippet"],
            ALSnippetAnnotation.source == ALAnnotationSource.USER,
        )
    }


def test_cleared_labels_are_not_resurrected(env):
    """
    A REJECT (empty final_labels) is the annotator clearing the snippet. It must
    supersede the earlier labelled event, not be skipped over.
    """
    db = env["db"]
    db.add_all([
        _event(env, ["wind"], T0),
        _event(env, [], T0 + timedelta(minutes=5), action=ALFeedbackAction.REJECT),
    ])
    db.commit()

    sync_feedback_events_to_annotations(db, env["ckpt"])
    db.commit()

    assert _al_labels(env) == set()


def test_newer_canonical_annotation_is_not_overwritten(env):
    """
    The hub relabelled this snippet 11 days after the feedback event. Replaying
    the older event must not reinstate the superseded label nor delete the
    newer one.
    """
    db = env["db"]
    db.add(_event(env, ["Augastes scutatus"], T0))
    db.add(Annotation(
        snippet_id=env["snippet"], user_id=env["user"],
        taxon_id="local:colibri_serrirostris",
        resolved_name_snapshot="Colibri serrirostris",
        created_at=T0 + timedelta(days=11),
    ))
    db.add(ALSnippetAnnotation(
        dataset_id=env["ds"], snippet_id=env["snippet"],
        label="Colibri serrirostris", source=ALAnnotationSource.USER,
        user_id=env["user"], created_at=T0 + timedelta(days=11),
    ))
    db.commit()

    sync_feedback_events_to_annotations(db, env["ckpt"])
    db.commit()

    assert _al_labels(env) == {"Colibri serrirostris"}


def test_event_newer_than_canonical_still_applies(env):
    """The guard must not freeze the sync: newer feedback still wins."""
    db = env["db"]
    db.add(Annotation(
        snippet_id=env["snippet"], user_id=env["user"],
        taxon_id="local:aves", resolved_name_snapshot="Aves",
        created_at=T0 - timedelta(days=1),
    ))
    db.add(_event(env, ["Augastes scutatus"], T0))
    db.commit()

    sync_feedback_events_to_annotations(db, env["ckpt"])
    db.commit()

    assert _al_labels(env) == {"Augastes scutatus"}
