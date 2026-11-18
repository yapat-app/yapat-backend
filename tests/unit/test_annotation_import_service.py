import json
from types import SimpleNamespace

import pytest

from app.core import taxonomy
from app.models.annotation import Annotation
from app.models.dataset import Dataset
from app.models.embedding import EmbeddingModel, SnippetSet
from app.models.pam_active_learning import ALSnippetAnnotation
from app.models.recording import Recording
from app.models.snippet import Snippet
from app.models.user import User, UserRole
from app.services.annotation_import_service import (
    IMPORT_SOURCE,
    AnnotationImportError,
    AnnotationImportService,
)

HEADER = (
    "recording_file_name,snippet_start_time,snippet_end_time,"
    "taxon_id,resolved_name_snapshot,confidence\n"
)


def _csv(*rows: str) -> bytes:
    return (HEADER + "\n".join(rows) + "\n").encode()


@pytest.fixture
def world(db_session, monkeypatch):
    """A dataset with one recording (a.wav) and two snippets: 0-3 s and 3-6 s."""

    def fake_resolve(taxon_id, db_session=None):
        if taxon_id.startswith("gbif:"):
            return {
                "taxon_id": taxon_id,
                "canonical_name": f"Species {taxon_id[5:]}",
                "taxonomy_type": "gbif",
            }
        return None

    monkeypatch.setattr(taxonomy, "resolve_taxon_id", fake_resolve)

    user = User(username="admin", hashed_password="x", role=UserRole.ADMIN)
    ds = Dataset(name="D", source_uri="d")
    model = EmbeddingModel(
        name="birdnet", version="2.4", window_size=3.0, step_size=3.0, overlap=0.0
    )
    db_session.add_all([user, ds, model])
    db_session.commit()

    ss = SnippetSet(
        dataset_id=ds.id,
        embedding_model_id=model.id,
        window_size=3.0,
        step_size=3.0,
        overlap=0.0,
    )
    db_session.add(ss)
    db_session.commit()

    rec = Recording(
        dataset_id=ds.id,
        file_path="a.wav",
        file_name="a.wav",
        duration=10.0,
        sample_rate=44100,
    )
    db_session.add(rec)
    db_session.commit()

    s1 = Snippet(
        recording_id=rec.id, snippet_set_id=ss.id,
        start_time=0.0, end_time=3.0, duration=3.0,
    )
    s2 = Snippet(
        recording_id=rec.id, snippet_set_id=ss.id,
        start_time=3.0, end_time=6.0, duration=3.0,
    )
    db_session.add_all([s1, s2])
    db_session.commit()

    return SimpleNamespace(
        svc=AnnotationImportService(db_session),
        db=db_session,
        user=user,
        ds=ds,
        s1=s1,
        s2=s2,
    )


def test_preview_counts_each_kind_of_row_and_writes_nothing(world):
    world.db.add(
        Annotation(
            snippet_id=world.s2.id, user_id=world.user.id,
            taxon_id="gbif:2", resolved_name_snapshot="Species 2",
        )
    )
    world.db.commit()

    file = _csv(
        "a.wav,0,3,gbif:1,Species 1,0.9",        # matches -> will import
        "a.wav,3,6,gbif:2,Species 2,0.9",        # snippet already has gbif:2
        "b.wav,0,3,gbif:1,Species 1,0.9",        # recording not in dataset
        "a.wav,1,2,gbif:1,Species 1,0.9",        # no snippet at that time
        "a.wav,0,3,nope:1,Nope,0.9",             # species cannot be resolved
        "a.wav,abc,3,gbif:1,Species 1,0.9",      # times are not numbers
        "a.wav,0,3,gbif:1,Species 1,0.9",        # same row again in the file
    )
    result = world.svc.preview(world.ds.id, world.user, file, "x.csv")

    assert result["total_rows"] == 7
    assert result["will_import"] == 1
    assert result["already_exist"] == 2
    assert result["no_match"] == 2
    assert result["invalid"] == 2
    by_status = {}
    for row in result["rows"]:
        by_status.setdefault(row["status"], []).append(row["row"])
    # CSV rows are numbered by file line: the header is line 1.
    assert by_status == {
        "will_import": [2],
        "already_exist": [3, 8],
        "no_match": [4, 5],
        "invalid": [6, 7],
    }
    reasons = {r["row"]: r["reason"] for r in result["rows"]}
    assert "already has this species" in reasons[3]
    assert "earlier row in this file" in reasons[8]
    first = result["rows"][0]
    assert (first["recording"], first["start"], first["end"], first["taxon_id"]) == (
        "a.wav", "0", "3", "gbif:1",
    )
    assert all(r["reason"] for r in result["rows"] if r["status"] != "will_import")
    assert world.db.query(Annotation).count() == 1  # only the pre-existing one

    labels = {l["taxon_id"]: l for l in result["labels"]}
    assert labels["gbif:1"]["name"] == "Species 1"
    assert [labels["gbif:1"][k] for k in ("will_import", "already_exist", "no_match", "invalid")] == [1, 1, 2, 1]
    assert [labels["gbif:2"][k] for k in ("will_import", "already_exist", "no_match", "invalid")] == [0, 1, 0, 0]
    assert [labels["nope:1"][k] for k in ("will_import", "already_exist", "no_match", "invalid")] == [0, 0, 0, 1]
    assert result["labels"][0]["taxon_id"] == "gbif:1"  # most rows first


def test_rows_show_the_files_own_text_and_line_numbers(world):
    file = (
        HEADER
        + "a.wav,0.0,3.0,gbif:1,Species 1,0.9\n"
        + "\n"  # a blank line still counts as a line of the file
        + "a.wav,3.000,6.0000,gbif:3,,\n"
    ).encode()
    rows = world.svc.preview(world.ds.id, world.user, file, "x.csv")["rows"]

    assert [(r["row"], r["start"], r["end"]) for r in rows] == [
        (2, "0.0", "3.0"),
        (4, "3.000", "6.0000"),  # not re-formatted to 3 and 6
    ]


def test_json_rows_are_numbered_from_one(world):
    items = [
        {"recording_file_name": "a.wav", "snippet_start_time": 0.0,
         "snippet_end_time": 3.0, "taxon_id": "gbif:1"},
        {"recording_file_name": "b.wav", "snippet_start_time": 0.0,
         "snippet_end_time": 3.0, "taxon_id": "gbif:1"},
    ]
    rows = world.svc.preview(
        world.ds.id, world.user, json.dumps(items).encode(), "x.json"
    )["rows"]
    assert [(r["row"], r["status"]) for r in rows] == [(1, "will_import"), (2, "no_match")]


def test_preview_lists_a_limited_number_of_rows_per_status(world, monkeypatch):
    from app.services import annotation_import_service as module

    monkeypatch.setattr(module, "_MAX_ROWS_RETURNED_PER_STATUS", 2)
    file = _csv(*["b.wav,0,3,gbif:1,,"] * 5)  # five rows, none match
    result = world.svc.preview(world.ds.id, world.user, file, "x.csv")

    assert result["no_match"] == 5  # counts cover the whole file
    assert len(result["rows"]) == 2  # the list is capped


def test_import_adds_next_to_existing_and_marks_source(world):
    world.db.add(
        Annotation(
            snippet_id=world.s2.id, user_id=world.user.id,
            taxon_id="gbif:2", resolved_name_snapshot="Species 2",
        )
    )
    world.db.commit()

    file = _csv(
        "a.wav,0,3,gbif:1,My name for 1,0.7",
        "a.wav,3,6,gbif:3,,",
    )
    result = world.svc.import_annotations(world.ds.id, world.user, file, "x.csv")

    assert result["imported"] == 2
    assert result["no_match"] == result["invalid"] == result["already_exist"] == 0

    on_s2 = (
        world.db.query(Annotation).filter(Annotation.snippet_id == world.s2.id).all()
    )
    assert {a.taxon_id for a in on_s2} == {"gbif:2", "gbif:3"}  # added, not replaced

    first = (
        world.db.query(Annotation)
        .filter(Annotation.snippet_id == world.s1.id)
        .one()
    )
    assert first.source == IMPORT_SOURCE
    assert first.user_id == world.user.id
    assert first.resolved_name_snapshot == "My name for 1"
    assert first.confidence == 0.7

    second = (
        world.db.query(Annotation)
        .filter(Annotation.snippet_id == world.s2.id, Annotation.taxon_id == "gbif:3")
        .one()
    )
    assert second.resolved_name_snapshot == "Species 3"  # no name in file -> resolved name

    untouched = (
        world.db.query(Annotation)
        .filter(Annotation.snippet_id == world.s2.id, Annotation.taxon_id == "gbif:2")
        .one()
    )
    assert untouched.source is None  # existing annotation left as it was

    labels = world.db.query(ALSnippetAnnotation).all()
    assert {(l.snippet_id, l.label) for l in labels} == {
        (world.s1.id, "My name for 1"),
        (world.s2.id, "Species 3"),
    }


def test_import_twice_adds_nothing_the_second_time(world):
    file = _csv("a.wav,0,3,gbif:1,Species 1,0.9")
    first = world.svc.import_annotations(world.ds.id, world.user, file, "x.csv")
    second = world.svc.import_annotations(world.ds.id, world.user, file, "x.csv")

    assert first["imported"] == 1
    assert second["imported"] == 0 and second["already_exist"] == 1
    assert world.db.query(Annotation).count() == 1


def test_times_match_within_a_small_tolerance(world):
    file = _csv("a.wav,0.0004,2.9996,gbif:1,,")
    result = world.svc.preview(world.ds.id, world.user, file, "x.csv")
    assert result["will_import"] == 1


def test_accepts_the_json_export_format(world):
    rows = [
        {
            "annotation_id": 1,
            "dataset_id": world.ds.id,
            "snippet_id": 999,  # ids in the file are ignored
            "taxon_id": "gbif:1",
            "resolved_name_snapshot": "Species 1",
            "confidence": 0.8,
            "created_at": "2026-01-01T00:00:00",
            "created_by": 5,
            "recording_file_name": "a.wav",
            "recording_file_path": "a.wav",
            "snippet_start_time": 0.0,
            "snippet_end_time": 3.0,
            "snippet_duration": 3.0,
        }
    ]
    result = world.svc.import_annotations(
        world.ds.id, world.user, json.dumps(rows).encode(), "x.json"
    )
    assert result["imported"] == 1
    annotation = world.db.query(Annotation).one()
    assert annotation.snippet_id == world.s1.id
    assert annotation.user_id == world.user.id  # credited to whoever imported


@pytest.mark.parametrize(
    "content, filename, message",
    [
        (b"", "x.csv", "empty"),
        (b"recording_file_name,taxon_id\na.wav,gbif:1\n", "x.csv", "missing required"),
        (HEADER.encode(), "x.csv", "no rows"),
        (b"not json", "x.json", "valid JSON"),
        (b'{"a": 1}', "x.json", "list of annotation objects"),
    ],
)
def test_unusable_files_are_rejected(world, content, filename, message):
    with pytest.raises(AnnotationImportError, match=message):
        world.svc.preview(world.ds.id, world.user, content, filename)
