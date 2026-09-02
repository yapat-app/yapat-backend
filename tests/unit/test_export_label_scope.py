"""
"Only selected labels" must mean what it says.

The first cut of this filter resolved at snippet level: exporting SCIALT
returned every annotation on every snippet carrying SCIALT, with an `in_scope`
column marking the ones that matched. On a dataset where snippets carry several
species (DENMIN, LEPFUS, SCIALT on one snippet) that buried the requested rows
among rows nobody asked for. Row filtering is now the default and the
snippet-level view is opt-in.
"""

import pytest

from app.api.datasets import _collect_export_rows
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
    snippets = []
    for start in (0.0, 3.0):
        sn = Snippet(
            recording_id=rec.id, snippet_set_id=ss.id,
            start_time=start, end_time=start + 3.0, duration=3.0,
        )
        db.add(sn)
        db.commit()
        snippets.append(sn)

    # Snippet 0 mirrors the reported case: three species on one snippet.
    db.add_all([
        Annotation(snippet_id=snippets[0].id, user_id=user.id,
                   taxon_id="custom:a", resolved_name_snapshot="SCIALT"),
        Annotation(snippet_id=snippets[0].id, user_id=user.id,
                   taxon_id="custom:b", resolved_name_snapshot="DENMIN"),
        Annotation(snippet_id=snippets[0].id, user_id=user.id,
                   taxon_id="custom:c", resolved_name_snapshot="LEPFUS"),
        # Second snippet has no SCIALT at all.
        Annotation(snippet_id=snippets[1].id, user_id=user.id,
                   taxon_id="custom:d", resolved_name_snapshot="DENMIN"),
        ALSnippetAnnotation(dataset_id=ds.id, snippet_id=snippets[1].id,
                            label="rain", source=ALAnnotationSource.USER,
                            user_id=user.id),
    ])
    db.commit()
    return {"db": db, "ds": ds.id}


def _names(rows):
    return sorted(row["label"] for row in rows)


def test_label_filter_returns_only_that_label(env):
    rows = _collect_export_rows(env["db"], env["ds"], labels=["SCIALT"])

    assert _names(rows) == ["SCIALT"]


def test_export_names_the_label_column_label(env):
    """`resolved_name_snapshot` is how the row was stored, not what it is."""
    rows = _collect_export_rows(env["db"], env["ds"], labels=["SCIALT"])

    assert "label" in rows[0]
    assert "resolved_name_snapshot" not in rows[0]


def test_export_omits_taxon_id(env):
    """
    A `custom:<uuid>` identifies nothing outside this database, and the column
    name invites a group_by that returns one group per row.
    """
    rows = _collect_export_rows(env["db"], env["ds"])

    assert all("taxon_id" not in row for row in rows)


def test_co_occurring_mode_brings_the_whole_snippet(env):
    rows = _collect_export_rows(
        env["db"], env["ds"], labels=["SCIALT"], include_co_occurring=True
    )

    assert _names(rows) == ["DENMIN", "LEPFUS", "SCIALT"]


def test_co_occurring_mode_does_not_reach_unmatched_snippets(env):
    """The second snippet carries DENMIN but no SCIALT — it stays out."""
    rows = _collect_export_rows(
        env["db"], env["ds"], labels=["SCIALT"], include_co_occurring=True
    )

    assert len(rows) == 3


def test_no_export_carries_an_in_scope_column(env):
    """
    Dropped: which rows matched is already legible from `label`, so the column
    only ever restated the filter.
    """
    for rows in (
        _collect_export_rows(env["db"], env["ds"]),
        _collect_export_rows(env["db"], env["ds"], labels=["SCIALT"]),
        _collect_export_rows(
            env["db"], env["ds"], labels=["SCIALT"], include_co_occurring=True
        ),
    ):
        assert all("in_scope" not in row for row in rows)


def test_filter_matches_taxon_id_and_al_label_too(env):
    """
    All three spellings still scope identically even though taxon_id is no
    longer a column: dropping it from the output did not drop it as a filter.
    """
    by_taxon = _collect_export_rows(env["db"], env["ds"], labels=["custom:a"])
    al_label = _collect_export_rows(env["db"], env["ds"], labels=["rain"])

    assert _names(by_taxon) == ["SCIALT"]
    assert _names(al_label) == ["rain"]


def test_taxon_id_alias_filters_by_row(env):
    rows = _collect_export_rows(env["db"], env["ds"], taxon_id="SCIALT")

    assert _names(rows) == ["SCIALT"]
