"""
The export dialog's label picker offers whatever `_collect_annotation_labels`
returns, so every value it lists must be one the export's scope filter matches
exactly -- otherwise a user picks a label from the dropdown and gets an empty
CSV back.
"""

import pytest

from app.api.datasets import _collect_annotation_labels, _resolve_scope_snippet_ids
from app.models.annotation import Annotation
from app.models.dataset import Dataset
from app.models.embedding import EmbeddingModel, SnippetSet
from app.models.pam_active_learning import ALAnnotationSource, ALSnippetAnnotation
from app.models.recording import Recording
from app.models.snippet import Snippet
from app.models.user import User


@pytest.fixture
def env(db_session):
    db = db_session
    user = User(username="sara", hashed_password="x")
    ds = Dataset(name="D", source_uri="dummy")
    other_ds = Dataset(name="Other", source_uri="dummy")
    model = EmbeddingModel(
        name="birdnet", version="2.4", window_size=3.0, step_size=3.0, overlap=0.0
    )
    db.add_all([user, ds, other_ds, model])
    db.commit()

    def _snippet(dataset, file_name):
        ss = SnippetSet(
            dataset_id=dataset.id, embedding_model_id=model.id,
            window_size=3.0, step_size=3.0, overlap=0.0,
        )
        db.add(ss)
        db.commit()
        rec = Recording(
            dataset_id=dataset.id, file_path=file_name,
            file_name=file_name, duration=10.0,
        )
        db.add(rec)
        db.commit()
        snippet = Snippet(
            recording_id=rec.id, snippet_set_id=ss.id,
            start_time=0.0, end_time=3.0, duration=3.0,
        )
        db.add(snippet)
        db.commit()
        return snippet

    return {
        "db": db,
        "ds": ds.id,
        "user": user.id,
        "snippet": _snippet(ds, "a.wav"),
        "second_snippet": _snippet(ds, "b.wav"),
        "foreign_snippet": _snippet(other_ds, "c.wav"),
    }


def _al(env, snippet, label, source=ALAnnotationSource.USER):
    return ALSnippetAnnotation(
        dataset_id=snippet.recording.dataset_id, snippet_id=snippet.id,
        label=label, source=source, user_id=env["user"],
    )


def _canonical(env, snippet, taxon_id, name):
    return Annotation(
        snippet_id=snippet.id, user_id=env["user"],
        taxon_id=taxon_id, resolved_name_snapshot=name,
    )


def test_unions_both_label_stores(env):
    """
    Neither store is a superset of the other, so a picker fed by one alone
    hides labels the export can still scope by.
    """
    db = env["db"]
    db.add_all([
        _al(env, env["snippet"], "rain"),
        _canonical(env, env["snippet"], "gbif:2427091", "Boana cipoensis"),
    ])
    db.commit()

    assert _collect_annotation_labels(db, env["ds"]) == ["Boana cipoensis", "rain"]


def test_offers_display_names_not_taxon_ids(env):
    """
    `gbif:2427091` scopes an export just as well, but no annotator recognises
    it -- the picker shows the resolved name instead.
    """
    db = env["db"]
    db.add(_canonical(env, env["snippet"], "gbif:2427091", "Boana cipoensis"))
    db.commit()

    assert _collect_annotation_labels(db, env["ds"]) == ["Boana cipoensis"]


def test_excludes_ground_truth_and_other_datasets(env):
    """
    The export excludes ground-truth imports and other datasets, so offering
    either yields a label that scopes to nothing.
    """
    db = env["db"]
    db.add_all([
        _al(env, env["snippet"], "rain"),
        _al(env, env["snippet"], "imported-truth", source=ALAnnotationSource.GROUND_TRUTH),
        _al(env, env["foreign_snippet"], "other-dataset-label"),
    ])
    db.commit()

    assert _collect_annotation_labels(db, env["ds"]) == ["rain"]


def test_keeps_case_variants_apart(env):
    """
    The scope match is exact, so collapsing "Rain" into "rain" would drop the
    other spelling's snippets from the export.
    """
    db = env["db"]
    db.add_all([
        _al(env, env["snippet"], "rain"),
        _al(env, env["second_snippet"], "Rain"),
    ])
    db.commit()

    assert _collect_annotation_labels(db, env["ds"]) == ["Rain", "rain"]


def test_every_offered_label_scopes_to_at_least_one_snippet(env):
    """
    The contract the picker relies on: nothing in the list produces an empty
    export.
    """
    db = env["db"]
    db.add_all([
        _al(env, env["snippet"], "rain"),
        _al(env, env["second_snippet"], "wind"),
        _canonical(env, env["snippet"], "gbif:2427091", "Boana cipoensis"),
    ])
    db.commit()

    for label in _collect_annotation_labels(db, env["ds"]):
        assert _resolve_scope_snippet_ids(db, env["ds"], [label]), label


def test_dataset_without_annotations_is_empty(env):
    assert _collect_annotation_labels(env["db"], env["ds"]) == []
