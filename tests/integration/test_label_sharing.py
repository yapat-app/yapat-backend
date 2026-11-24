"""Labels shared between snippet sets of one dataset (issue #93, part 2).

One 15 s recording, cut into 3 s clips (0, 3, 6, 9, 12) and 5 s clips (0, 5, 10).
"""

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_current_active_user, get_db
from app.config import settings
from app.main import app
from app.models.annotation import Annotation
from app.models.dataset import Dataset, DatasetType
from app.models.embedding import EmbeddingJobStatus, EmbeddingModel, SnippetSetStatus
from app.models.label_share import LabelShareCandidate
from app.models.pam_active_learning import ALAnnotationSource, ALSnippetAnnotation
from app.models.recording import Recording
from app.models.snippet import Snippet
from app.models.user import User, UserRole
from app.schemas.pam_active_learning import ALFeedbackSubmit
from app.services import label_sharing
from app.services.embedding_service import EmbeddingService
from app.services.pam_al import _annotation_helpers as ann_h
from app.services.pam_al.service import PAMActiveLearningService

FROG = {"taxon_id": "local:frog_a", "display_name": "Frog A"}
TOAD = {"taxon_id": "local:toad_b", "display_name": "Toad B"}
SILENT = {"taxon_id": "local:no_biophony", "display_name": "No biophony"}


@pytest.fixture
def world(db_session):
    db = db_session
    dataset = Dataset(name="share", source_uri="site", dataset_type=DatasetType.PAM)
    db.add(dataset)
    db.flush()
    three = EmbeddingModel(name="three", window_size=3.0, step_size=3.0, overlap=0.0, dim=8)
    five = EmbeddingModel(name="five", window_size=5.0, step_size=5.0, overlap=0.0, dim=8)
    db.add_all([three, five])
    db.flush()
    service = EmbeddingService(db)
    jobs = [service.create_embedding_job(dataset, three), service.create_embedding_job(dataset, five)]
    for job in jobs:
        job.status = EmbeddingJobStatus.COMPLETED
        job.snippet_set.status = SnippetSetStatus.READY
    dataset.default_snippet_set_id = jobs[0].snippet_set_id
    recording = Recording(dataset_id=dataset.id, file_path="site/a.wav", file_name="a.wav",
                          duration=15.0, sample_rate=48000)
    db.add(recording)
    db.flush()
    clips = {}
    for job, window in ((jobs[0], 3.0), (jobs[1], 5.0)):
        start = 0.0
        while start + window <= 15.0:
            snippet = Snippet(recording_id=recording.id, snippet_set_id=job.snippet_set_id,
                              start_time=start, end_time=start + window, duration=window)
            db.add(snippet)
            clips[(int(window), int(start))] = snippet
            start += window
    user = User(username="ann", hashed_password="x", role=UserRole.ADMIN)
    db.add(user)
    db.commit()
    return {"dataset": dataset, "three": three, "five": five, "clips": clips, "user": user}


@pytest.fixture
def client(db_session, world):
    app.dependency_overrides[get_db] = lambda: db_session
    app.dependency_overrides[get_current_active_user] = lambda: world["user"]
    yield TestClient(app)
    app.dependency_overrides.clear()


def _label(client, world, clip, taxon):
    window = clip[0]
    model = world["three"] if window == 3 else world["five"]
    r = client.post("/api/annotations/", json={
        **taxon, "snippet_id": world["clips"][clip].id, "embedding_model_id": model.id,
    })
    assert r.status_code == 201, r.text
    return r.json()


def _labels_on(db, world, clip):
    return {
        (a.taxon_id, a.origin)
        for a in db.query(Annotation).filter(Annotation.snippet_id == world["clips"][clip].id)
    }


def _queue(db):
    return {
        (c.target_snippet_id, c.taxon_id, c.reason, c.status)
        for c in db.query(LabelShareCandidate)
    }


def test_species_label_is_copied_to_the_longer_clip_that_contains_it(db_session, world, client):
    source = _label(client, world, (3, 0), FROG)

    copy = db_session.query(Annotation).filter_by(snippet_id=world["clips"][(5, 0)].id).one()
    assert copy.origin == "auto_shared"
    assert copy.derived_from_id == source["id"]
    assert copy.taxon_id == FROG["taxon_id"]
    mirror = db_session.query(ALSnippetAnnotation).filter_by(snippet_id=copy.snippet_id).one()
    assert (mirror.label, mirror.source) == ("Frog A", ALAnnotationSource.DERIVED)
    assert _queue(db_session) == set()


def test_partial_overlaps_go_to_the_queue(db_session, world, client):
    _label(client, world, (3, 3), FROG)  # 3-6 s: 2 s in clip 0-5, 1 s in clip 5-10

    clips = world["clips"]
    assert _labels_on(db_session, world, (5, 0)) == set()
    assert _queue(db_session) == {
        (clips[(5, 0)].id, "local:frog_a", "partial_overlap", "pending"),
        (clips[(5, 5)].id, "local:frog_a", "partial_overlap", "pending"),
    }


def test_species_label_on_a_longer_clip_is_only_suggested_for_shorter_ones(db_session, world, client):
    _label(client, world, (5, 0), FROG)

    clips = world["clips"]
    assert _labels_on(db_session, world, (3, 0)) == set()
    assert _queue(db_session) == {
        (clips[(3, 0)].id, "local:frog_a", "target_shorter", "pending"),
        (clips[(3, 3)].id, "local:frog_a", "partial_overlap", "pending"),
    }


def test_no_event_label_goes_from_the_longer_clip_to_the_shorter_one(db_session, world, client):
    _label(client, world, (5, 0), SILENT)

    assert _labels_on(db_session, world, (3, 0)) == {("local:no_biophony", "auto_shared")}
    assert ("local:no_biophony", "auto_shared") not in _labels_on(db_session, world, (3, 3))

    db_session.query(Annotation).delete()
    db_session.query(ALSnippetAnnotation).delete()
    db_session.query(LabelShareCandidate).delete()
    db_session.commit()
    _label(client, world, (3, 0), SILENT)
    assert _labels_on(db_session, world, (5, 0)) == set()
    assert _queue(db_session) == {
        (world["clips"][(5, 0)].id, "local:no_biophony", "target_longer", "pending"),
    }


def test_a_contradicting_label_on_the_target_is_kept_and_the_copy_queued(db_session, world, client):
    _label(client, world, (5, 0), SILENT)
    db_session.query(LabelShareCandidate).delete()
    db_session.query(Annotation).filter(Annotation.origin == "auto_shared").delete()
    db_session.commit()

    _label(client, world, (3, 0), FROG)

    assert _labels_on(db_session, world, (5, 0)) == {("local:no_biophony", "manual")}
    assert _queue(db_session) == {
        (world["clips"][(5, 0)].id, "local:frog_a", "conflict", "pending"),
    }


def test_a_label_the_target_has_already_is_left_alone(db_session, world, client):
    target = _label(client, world, (5, 0), FROG)
    _label(client, world, (3, 0), FROG)

    rows = db_session.query(Annotation).filter_by(snippet_id=world["clips"][(5, 0)].id).all()
    assert [(r.id, r.origin) for r in rows] == [(target["id"], "manual")]


def test_a_label_with_the_same_name_under_another_taxon_id_counts_as_there(db_session, world, client):
    # Hub labels without a known taxon become custom taxa, and the same name
    # can end up under two ids. Clip 0-5 s shows "Frog A" already.
    _label(client, world, (5, 0), FROG)
    _label(client, world, (3, 3), {"taxon_id": "local:frog_a_2", "display_name": "Frog A"})

    targets = {c.target_snippet_id for c in db_session.query(LabelShareCandidate)}
    assert world["clips"][(5, 0)].id not in targets
    assert world["clips"][(5, 5)].id in targets


def test_a_copy_settles_a_suggestion_queued_before_it(db_session, world, client):
    # 3-6 s overlaps 0-5 s only partly, so it is queued; then 0-3 s, which lies
    # inside 0-5 s, copies the same label there. The suggestion is moot.
    _label(client, world, (3, 3), FROG)
    assert world["clips"][(5, 0)].id in {c.target_snippet_id for c in db_session.query(LabelShareCandidate)}

    _label(client, world, (3, 0), FROG)

    pending = {
        c.target_snippet_id
        for c in db_session.query(LabelShareCandidate).filter_by(status="pending")
    }
    assert world["clips"][(5, 0)].id not in pending
    assert world["clips"][(5, 5)].id in pending


def test_deleting_the_source_removes_its_copy(db_session, world, client):
    source = _label(client, world, (3, 0), FROG)
    assert client.delete(f"/api/annotations/{source['id']}").status_code == 204

    assert db_session.query(Annotation).count() == 0
    assert db_session.query(ALSnippetAnnotation).count() == 0


def test_a_deleted_copy_does_not_come_back(db_session, world, client):
    _label(client, world, (3, 0), FROG)
    copy = db_session.query(Annotation).filter_by(origin="auto_shared").one()

    assert client.delete(f"/api/annotations/{copy.id}").status_code == 204
    label_sharing.share_dataset(db_session, world["dataset"].id)

    assert _labels_on(db_session, world, (5, 0)) == set()
    assert _queue(db_session) == {
        (world["clips"][(5, 0)].id, "local:frog_a", "target_shorter", "rejected"),
    }


def test_labelling_a_clip_with_its_copy_makes_the_copy_manual(db_session, world, client):
    _label(client, world, (3, 0), FROG)
    copy = db_session.query(Annotation).filter_by(origin="auto_shared").one()

    confirmed = _label(client, world, (5, 0), FROG)

    assert confirmed["id"] == copy.id
    assert confirmed["origin"] == "manual"
    sources = {r.source for r in db_session.query(ALSnippetAnnotation).filter_by(snippet_id=copy.snippet_id)}
    assert sources == {ALAnnotationSource.USER}


def _pending(db, clip_id):
    return db.query(LabelShareCandidate).filter_by(target_snippet_id=clip_id, status="pending").one()


def test_snippet_breakdown_counts_each_snippet_once_per_set(db_session, world, client):
    _label(client, world, (3, 0), FROG)  # copied onto 0-5 s
    _label(client, world, (3, 3), TOAD)  # suggested for 0-5 s and 5-10 s
    url = f"/api/label-sharing/datasets/{world['dataset'].id}/snippet-breakdown"
    three_id = world["clips"][(3, 0)].snippet_set_id
    five_id = world["clips"][(5, 0)].snippet_set_id

    def rows():
        response = client.get(url)
        assert response.status_code == 200, response.text
        return {row["snippet_set_id"]: row for row in response.json()["snippet_sets"]}

    before = rows()
    pending = db_session.query(LabelShareCandidate).filter_by(
        target_snippet_set_id=five_id, status="pending",
    ).count()
    assert pending > 0
    assert before[three_id] == {
        "snippet_set_id": three_id, "window_size": 3.0, "status": "ready", "total_snippets": 5,
        "recording_count": 1, "labelled": 2, "auto_copied": 0, "to_review": 0,
    }
    assert (before[five_id]["total_snippets"], before[five_id]["labelled"],
            before[five_id]["auto_copied"], before[five_id]["to_review"]) == (3, 0, 1, pending)

    candidate = _pending(db_session, world["clips"][(5, 0)].id)
    assert client.post(f"/api/label-sharing/candidates/{candidate.id}/accept", json={}).status_code == 200

    # 0-5 s now has a copy and an accepted label: one labelled snippet.
    after = rows()[five_id]
    assert (after["labelled"], after["auto_copied"], after["to_review"]) == (1, 0, pending - 1)


def test_accept_adds_the_label_under_the_reviewer(db_session, world, client):
    source = _label(client, world, (3, 3), FROG)
    candidate = _pending(db_session, world["clips"][(5, 0)].id)

    r = client.post(f"/api/label-sharing/candidates/{candidate.id}/accept",
                    json={"embedding_model_id": world["five"].id})

    assert r.status_code == 200, r.text
    assert r.json()["candidate"]["status"] == "accepted"
    added = db_session.get(Annotation, r.json()["annotation_id"])
    assert (added.origin, added.derived_from_id, added.embedding_model_id) == (
        "reviewed_shared", source["id"], world["five"].id,
    )
    mirror = db_session.query(ALSnippetAnnotation).filter_by(snippet_id=added.snippet_id).one()
    assert mirror.source == ALAnnotationSource.USER
    assert client.post(f"/api/label-sharing/candidates/{candidate.id}/reject").status_code == 409


def test_accepting_a_name_the_clip_has_under_another_taxon_id_adds_no_second_row(db_session, world, client):
    # Older data: clip 0-5 s already shows "Frog A" (copied under another
    # custom id), and two suggestions of "Frog A" wait under different ids.
    target_id = world["clips"][(5, 0)].id
    _label(client, world, (3, 3), FROG)
    first = _pending(db_session, target_id)
    other_source = Annotation(snippet_id=first.source_snippet_id, user_id=world["user"].id,
                              taxon_id="local:frog_a_3", resolved_name_snapshot="Frog A", origin="manual")
    db_session.add(other_source)
    db_session.flush()
    db_session.add(LabelShareCandidate(
        dataset_id=first.dataset_id, source_annotation_id=other_source.id,
        source_snippet_id=first.source_snippet_id, target_snippet_id=target_id,
        target_snippet_set_id=first.target_snippet_set_id, overlap_seconds=first.overlap_seconds,
        taxon_id="local:frog_a_3", resolved_name_snapshot="Frog A", al_label="Frog A",
        reason="partial_overlap", status="pending",
    ))
    old_copy = Annotation(snippet_id=target_id, user_id=world["user"].id, taxon_id="local:frog_a_2",
                          resolved_name_snapshot="Frog A", origin="auto_shared",
                          extra_metadata={"al_label": "Frog A"})
    db_session.add(old_copy)
    db_session.commit()

    r = client.post(f"/api/label-sharing/candidates/{first.id}/accept")

    assert r.status_code == 200, r.text
    assert r.json()["annotation_id"] == old_copy.id
    assert _labels_on(db_session, world, (5, 0)) == {("local:frog_a_2", "reviewed_shared")}
    assert db_session.query(LabelShareCandidate).filter_by(target_snippet_id=target_id, status="pending").count() == 0


@pytest.mark.parametrize("decision", ["accept", "relabel"])
def test_a_reviewed_suggestion_keeps_its_record_when_the_session_does_not_autoflush(
    db_session, world, client, monkeypatch, decision
):
    # The app's sessions run with autoflush off (app/database.py).
    monkeypatch.setattr(db_session, "autoflush", False)
    _label(client, world, (3, 3), FROG)
    candidate = _pending(db_session, world["clips"][(5, 0)].id)
    candidate_id = candidate.id

    if decision == "accept":
        r = client.post(f"/api/label-sharing/candidates/{candidate_id}/accept")
        expected = "accepted"
    else:
        r = client.post(
            f"/api/label-sharing/candidates/{candidate_id}/relabel",
            json={"taxon_id": "local:frog_a_2", "display_name": "Frog A"},
        )
        expected = "relabelled"

    assert r.status_code == 200, r.text
    assert r.json()["candidate"]["status"] == expected
    assert db_session.get(LabelShareCandidate, candidate_id).status == expected


def test_reviewed_label_stays_when_its_source_is_deleted(db_session, world, client):
    source = _label(client, world, (3, 3), FROG)
    candidate = _pending(db_session, world["clips"][(5, 0)].id)
    added_id = client.post(f"/api/label-sharing/candidates/{candidate.id}/accept").json()["annotation_id"]

    client.delete(f"/api/annotations/{source['id']}")

    kept = db_session.get(Annotation, added_id)
    assert kept is not None and kept.derived_from_id is None


def _recheck(db, clip_id):
    return db.query(LabelShareCandidate).filter_by(
        target_snippet_id=clip_id, status="pending", reason="source_removed",
    ).one_or_none()


def _accepted_then_source_deleted(db_session, world, client):
    """Frog on 3-6 s, accepted for 0-5 s, then deleted from 3-6 s."""
    source = _label(client, world, (3, 3), FROG)
    candidate = _pending(db_session, world["clips"][(5, 0)].id)
    added_id = client.post(f"/api/label-sharing/candidates/{candidate.id}/accept").json()["annotation_id"]
    assert client.delete(f"/api/annotations/{source['id']}").status_code == 204
    return added_id


def test_deleting_a_source_offers_to_recheck_its_reviewed_label(db_session, world, client):
    added_id = _accepted_then_source_deleted(db_session, world, client)

    recheck = _recheck(db_session, world["clips"][(5, 0)].id)
    assert recheck is not None
    assert (recheck.target_annotation_id, recheck.source_annotation_id, recheck.source_snippet_id) == (
        added_id, None, world["clips"][(3, 3)].id,
    )
    listed = client.get(
        f"/api/label-sharing/datasets/{world['dataset'].id}/candidates",
        params={"snippet_set_id": recheck.target_snippet_set_id},
    ).json()["items"]
    assert [(i["id"], i["reason"], i["label"]) for i in listed] == [(recheck.id, "source_removed", "Frog A")]
    assert client.post(f"/api/label-sharing/candidates/{recheck.id}/relabel", json=TOAD).status_code == 400

    kept = client.post(f"/api/label-sharing/candidates/{recheck.id}/accept")

    assert kept.status_code == 200, kept.text
    assert (kept.json()["candidate"]["status"], kept.json()["annotation_id"]) == ("accepted", added_id)
    assert _labels_on(db_session, world, (5, 0)) == {("local:frog_a", "reviewed_shared")}


def test_removing_a_rechecked_label_deletes_it_from_both_stores(db_session, world, client):
    db_session.autoflush = False  # as in the app (app/database.py)
    try:
        _accepted_then_source_deleted(db_session, world, client)
        recheck = _recheck(db_session, world["clips"][(5, 0)].id)

        removed = client.post(f"/api/label-sharing/candidates/{recheck.id}/reject")

        assert removed.status_code == 200, removed.text
        assert removed.json()["candidate"]["status"] == "rejected"
        assert _labels_on(db_session, world, (5, 0)) == set()
        assert db_session.query(ALSnippetAnnotation).filter_by(
            snippet_id=world["clips"][(5, 0)].id,
        ).count() == 0
    finally:
        db_session.autoflush = True


def test_a_recheck_closes_when_its_label_is_deleted(db_session, world, client):
    added_id = _accepted_then_source_deleted(db_session, world, client)

    assert client.delete(f"/api/annotations/{added_id}").status_code == 204

    assert _recheck(db_session, world["clips"][(5, 0)].id) is None


def test_no_recheck_while_another_overlapping_label_still_supports_it(db_session, world, client):
    source = _label(client, world, (3, 3), FROG)
    candidate = _pending(db_session, world["clips"][(5, 0)].id)
    client.post(f"/api/label-sharing/candidates/{candidate.id}/accept")
    _label(client, world, (3, 0), FROG)  # also inside 0-5 s, which has Frog already

    client.delete(f"/api/annotations/{source['id']}")

    assert _recheck(db_session, world["clips"][(5, 0)].id) is None
    assert _labels_on(db_session, world, (5, 0)) == {("local:frog_a", "reviewed_shared")}


def test_reject_and_relabel(db_session, world, client):
    _label(client, world, (3, 3), FROG)
    first = _pending(db_session, world["clips"][(5, 0)].id)
    second = _pending(db_session, world["clips"][(5, 5)].id)

    assert client.post(f"/api/label-sharing/candidates/{first.id}/reject").json()["candidate"]["status"] == "rejected"
    r = client.post(f"/api/label-sharing/candidates/{second.id}/relabel", json=TOAD)

    assert r.status_code == 200, r.text
    assert r.json()["candidate"]["status"] == "relabelled"
    assert _labels_on(db_session, world, (5, 5)) == {("local:toad_b", "reviewed_shared")}
    assert _labels_on(db_session, world, (5, 0)) == set()


def test_queue_list_and_summary(db_session, world, client):
    _label(client, world, (3, 0), FROG)
    _label(client, world, (3, 3), TOAD)
    dataset_id = world["dataset"].id

    listing = client.get(f"/api/label-sharing/datasets/{dataset_id}/candidates",
                         params={"embedding_model_id": world["five"].id}).json()
    summary = client.get(f"/api/label-sharing/datasets/{dataset_id}/summary").json()

    assert listing["total"] == 2
    first = listing["items"][0]
    assert (first["target"]["start_time"], first["label"], first["source_labeled_by"]) == (0.0, "Toad B", "ann")
    assert first["target_labels"] == ["Frog A"]
    assert (summary["auto_shared"], summary["pending"]) == (1, 2)
    five_set = str(world["clips"][(5, 0)].snippet_set_id)
    assert summary["pending_by_snippet_set_reason"] == {five_set: {"partial_overlap": 2}}
    empty = client.get(f"/api/label-sharing/datasets/{dataset_id}/candidates",
                       params={"embedding_model_id": world["three"].id}).json()
    assert empty["total"] == 0


def test_backfill_shares_labels_made_before_sharing(db_session, world, client, monkeypatch):
    monkeypatch.setattr(settings, "LABEL_SHARING_ENABLED", False)
    _label(client, world, (3, 0), FROG)
    _label(client, world, (3, 3), FROG)
    assert db_session.query(Annotation).count() == 2
    monkeypatch.setattr(settings, "LABEL_SHARING_ENABLED", True)

    first = label_sharing.share_dataset(db_session, world["dataset"].id)
    again = label_sharing.share_dataset(db_session, world["dataset"].id)

    # Clip 0-5 s already has the copy, so 3-6 s only queues for clip 5-10 s.
    assert (first["auto_shared"], first["queued"]) == (1, 1)
    assert (again["auto_shared"], again["queued"]) == (0, 0)


def test_sharing_onto_a_new_set_touches_only_that_set(db_session, world, client, monkeypatch):
    # As after a new model's embedding finishes: labels from the other sets go
    # onto the new set's clips only; labels on the new set are not shared out.
    monkeypatch.setattr(settings, "LABEL_SHARING_ENABLED", False)
    _label(client, world, (3, 0), FROG)
    _label(client, world, (5, 5), TOAD)
    monkeypatch.setattr(settings, "LABEL_SHARING_ENABLED", True)
    five_set = world["clips"][(5, 0)].snippet_set_id
    three_set = world["clips"][(3, 0)].snippet_set_id

    totals = label_sharing.share_dataset(
        db_session, world["dataset"].id, target_snippet_set_id=five_set
    )

    assert totals["source_clips"] == 1  # only the 3 s clip's label
    assert ("local:frog_a", "auto_shared") in _labels_on(db_session, world, (5, 0))
    three_clips = {s.id for (w, _), s in world["clips"].items() if w == 3}
    assert not db_session.query(LabelShareCandidate).filter(
        LabelShareCandidate.target_snippet_set_id == three_set
    ).count()
    assert not (
        db_session.query(Annotation)
        .filter(Annotation.snippet_id.in_(three_clips), Annotation.origin == "auto_shared")
        .count()
    )

    # The full backfill still shares the 5 s label onto the 3 s clips.
    label_sharing.share_dataset(db_session, world["dataset"].id)
    assert db_session.query(LabelShareCandidate).filter(
        LabelShareCandidate.target_snippet_set_id == three_set
    ).count() > 0


def test_a_transfer_goes_one_way_only(db_session, world, client, monkeypatch):
    monkeypatch.setattr(settings, "LABEL_SHARING_ENABLED", False)
    _label(client, world, (3, 0), FROG)
    _label(client, world, (5, 5), TOAD)
    monkeypatch.setattr(settings, "LABEL_SHARING_ENABLED", True)
    five_set = world["clips"][(5, 0)].snippet_set_id
    three_set = world["clips"][(3, 0)].snippet_set_id

    totals = label_sharing.share_dataset(
        db_session, world["dataset"].id,
        source_snippet_set_id=five_set, target_snippet_set_id=three_set,
    )

    # Only the 5 s label moved, onto 3 s clips; the 3 s label stayed put.
    assert totals["source_clips"] == 1
    assert {c.target_snippet_set_id for c in db_session.query(LabelShareCandidate)} == {three_set}
    assert _labels_on(db_session, world, (5, 0)) == set()


def test_transfer_endpoint_maps_models_to_their_clips(db_session, world, client, monkeypatch):
    from app.tasks import label_sharing_tasks

    calls = []
    monkeypatch.setattr(
        label_sharing_tasks.share_labels_for_dataset, "delay",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    dataset_id = world["dataset"].id
    url = f"/api/label-sharing/datasets/{dataset_id}/share"

    r = client.post(url, json={"from_embedding_model_id": world["three"].id,
                               "to_embedding_model_id": world["five"].id})

    assert r.status_code == 202, r.text
    assert calls == [((dataset_id,), {
        "target_snippet_set_id": world["clips"][(5, 0)].snippet_set_id,
        "source_snippet_set_id": world["clips"][(3, 0)].snippet_set_id,
    })]
    same = client.post(url, json={"from_embedding_model_id": world["three"].id,
                                  "to_embedding_model_id": world["three"].id})
    assert same.status_code == 400
    half = client.post(url, json={"from_embedding_model_id": world["three"].id})
    assert half.status_code == 400
    assert client.post(url).status_code == 202  # no body: every way, as before


def test_summary_counts_labelled_clips_per_set(db_session, world, client):
    _label(client, world, (3, 0), FROG)  # also copied onto 0-5 s, which is not counted
    _label(client, world, (3, 3), TOAD)

    summary = client.get(f"/api/label-sharing/datasets/{world['dataset'].id}/summary").json()

    assert summary["labelled_by_snippet_set"] == {str(world["clips"][(3, 0)].snippet_set_id): 2}


def test_hub_feedback_shares_and_unshares(db_session, world):
    dataset, three, user = world["dataset"], world["three"], world["user"]
    service = PAMActiveLearningService(db_session)

    def feedback(labels):
        service.submit_feedback(ALFeedbackSubmit(
            dataset_id=dataset.id, model_family_name="three_family",
            snippet_id=world["clips"][(3, 0)].id, embedding_model_id=three.id,
            action="MODIFY", labels=labels, user_id=user.id,
        ))

    feedback(["sp1", "sp2"])
    target = world["clips"][(5, 0)].id
    copies = db_session.query(Annotation).filter_by(snippet_id=target, origin="auto_shared").all()
    assert sorted(c.resolved_name_snapshot for c in copies) == ["sp1", "sp2"]
    assert sorted(ann_h.get_trusted_annotations(db_session, dataset.id)[target]) == ["sp1", "sp2"]

    feedback(["sp1"])
    copies = db_session.query(Annotation).filter_by(snippet_id=target, origin="auto_shared").all()
    assert [c.resolved_name_snapshot for c in copies] == ["sp1"]
    assert {r.label for r in db_session.query(ALSnippetAnnotation).filter_by(snippet_id=target)} == {"sp1"}


def test_shared_labels_can_be_left_out_of_training(db_session, world, client, monkeypatch):
    _label(client, world, (3, 0), FROG)
    target = world["clips"][(5, 0)].id
    monkeypatch.setattr(settings, "TRAIN_ON_SHARED_LABELS", False)

    assert target not in ann_h.get_trusted_annotations(db_session, world["dataset"].id)
    assert ann_h.get_labels_by_snippet(db_session, world["dataset"].id)[target] == ["Frog A"]


def test_hub_can_delete_a_shared_copy(db_session, world, client):
    _label(client, world, (3, 0), FROG)
    target = world["clips"][(5, 0)].id

    r = client.delete("/api/pam-al/snippet-labels", params={
        "dataset_id": world["dataset"].id, "snippet_id": target, "label": "Frog A", "source": "derived",
    })

    assert r.status_code == 204, r.text
    assert _labels_on(db_session, world, (5, 0)) == set()
    assert db_session.query(ALSnippetAnnotation).filter_by(snippet_id=target).count() == 0
    assert db_session.query(LabelShareCandidate).one().status == "rejected"
