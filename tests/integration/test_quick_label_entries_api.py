"""
Quick label entries API — the "GBIF pick becomes a quick label" flow.

Covers the two scoping rules that make this table work: personal rows are
invisible to other participants, and dataset-wide rows (the lane reserved for
Ontology Engineering) are visible to everyone but deletable by no one.
"""

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_db, get_current_active_user
from app.main import app
from app.models.dataset import Dataset, DatasetType
from app.models.quick_label_entry import QuickLabelEntry
from app.models.user import User, UserRole


@pytest.fixture
def dataset(db_session):
    ds = Dataset(
        name="quick-label-test",
        source_uri="/tmp/quick-label-test",
        dataset_type=DatasetType.PAM,
    )
    db_session.add(ds)
    db_session.commit()
    return ds


@pytest.fixture
def users(db_session):
    made = []
    for name in ("participant_a", "participant_b"):
        u = User(username=name, hashed_password="x", role=UserRole.USER)
        db_session.add(u)
        made.append(u)
    db_session.commit()
    return made


@pytest.fixture
def api(db_session, users):
    """TestClient factory bound to a chosen user, sharing the test session."""

    def _as(user: User) -> TestClient:
        app.dependency_overrides[get_db] = lambda: db_session
        app.dependency_overrides[get_current_active_user] = lambda: user
        return TestClient(app)

    yield _as
    app.dependency_overrides.clear()


def _post(client, dataset_id, **label):
    label.setdefault("display_name", "Turdus merula")
    label.setdefault("taxon_id", "gbif:2490719")
    return client.post(
        f"/api/datasets/{dataset_id}/quick-labels/mine", json={"labels": [label]}
    )


def test_gbif_pick_is_persisted_and_owned(api, dataset, users):
    client = api(users[0])

    resp = _post(client, dataset.id, taxon_id="gbif:2490719", rank="SPECIES")

    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0]["taxon_id"] == "gbif:2490719"
    assert body[0]["display_name"] == "Turdus merula"
    assert body[0]["rank"] == "SPECIES"
    assert body[0]["source"] == "gbif"
    assert body[0]["owned"] is True

    # And it survives to a fresh read.
    listed = client.get(f"/api/datasets/{dataset.id}/quick-labels/mine").json()
    assert [row["taxon_id"] for row in listed] == ["gbif:2490719"]


def test_repicking_the_same_species_does_not_duplicate(api, dataset, users, db_session):
    client = api(users[0])

    _post(client, dataset.id, taxon_id="gbif:2490719")
    first_id = client.get(f"/api/datasets/{dataset.id}/quick-labels/mine").json()[0]["id"]

    # Same taxon, different display name and source — the original row wins.
    resp = _post(
        client,
        dataset.id,
        taxon_id="gbif:2490719",
        display_name="Common Blackbird",
        source="manual",
    )

    body = resp.json()
    assert len(body) == 1
    assert body[0]["id"] == first_id
    assert body[0]["display_name"] == "Turdus merula"
    assert body[0]["source"] == "gbif"
    assert db_session.query(QuickLabelEntry).count() == 1


def test_a_batch_with_repeats_inserts_each_taxon_once(api, dataset, users):
    client = api(users[0])

    resp = client.post(
        f"/api/datasets/{dataset.id}/quick-labels/mine",
        json={
            "labels": [
                {"taxon_id": "gbif:1", "display_name": "One"},
                {"taxon_id": "gbif:2", "display_name": "Two"},
                {"taxon_id": "gbif:1", "display_name": "One again"},
            ]
        },
    )

    assert resp.status_code == 200
    assert sorted(row["taxon_id"] for row in resp.json()) == ["gbif:1", "gbif:2"]


def test_one_participants_picks_are_invisible_to_another(api, dataset, users):
    _post(api(users[0]), dataset.id, taxon_id="gbif:2490719")

    other = api(users[1]).get(f"/api/datasets/{dataset.id}/quick-labels/mine")

    assert other.status_code == 200
    assert other.json() == []


def test_dataset_wide_rows_are_visible_to_everyone_but_not_owned(
    api, dataset, users, db_session
):
    # The lane Ontology Engineering will write into: user_id NULL.
    db_session.add(
        QuickLabelEntry(
            dataset_id=dataset.id,
            user_id=None,
            taxon_id="envo:01001867",
            display_name="forest biome",
            source="oe",
        )
    )
    db_session.commit()

    for user in users:
        rows = api(user).get(f"/api/datasets/{dataset.id}/quick-labels/mine").json()
        assert [row["taxon_id"] for row in rows] == ["envo:01001867"]
        assert rows[0]["owned"] is False


def test_hand_picked_labels_sort_ahead_of_a_bulk_oe_import(
    api, dataset, users, db_session
):
    client = api(users[0])
    _post(client, dataset.id, taxon_id="gbif:2490719")

    # OE lands afterwards, so recency alone would push it to the front.
    db_session.add_all(
        QuickLabelEntry(
            dataset_id=dataset.id,
            user_id=None,
            taxon_id=f"envo:{i}",
            display_name=f"concept {i}",
            source="oe",
        )
        for i in range(3)
    )
    db_session.commit()

    rows = client.get(f"/api/datasets/{dataset.id}/quick-labels/mine").json()

    assert rows[0]["taxon_id"] == "gbif:2490719"
    assert all(row["source"] == "oe" for row in rows[1:])


def test_participant_can_delete_their_own_entry(api, dataset, users, db_session):
    client = api(users[0])
    _post(client, dataset.id, taxon_id="gbif:2490719")

    resp = client.delete(
        f"/api/datasets/{dataset.id}/quick-labels/mine",
        params={"taxon_id": "gbif:2490719"},
    )

    assert resp.status_code == 204
    assert client.get(f"/api/datasets/{dataset.id}/quick-labels/mine").json() == []
    assert db_session.query(QuickLabelEntry).count() == 0


def test_deleting_another_participants_entry_is_a_404(api, dataset, users, db_session):
    _post(api(users[0]), dataset.id, taxon_id="gbif:2490719")

    resp = api(users[1]).delete(
        f"/api/datasets/{dataset.id}/quick-labels/mine",
        params={"taxon_id": "gbif:2490719"},
    )

    assert resp.status_code == 404
    assert db_session.query(QuickLabelEntry).count() == 1


def test_dataset_wide_entries_cannot_be_deleted_by_a_participant(
    api, dataset, users, db_session
):
    db_session.add(
        QuickLabelEntry(
            dataset_id=dataset.id,
            user_id=None,
            taxon_id="envo:01001867",
            display_name="forest biome",
            source="oe",
        )
    )
    db_session.commit()

    resp = api(users[0]).delete(
        f"/api/datasets/{dataset.id}/quick-labels/mine",
        params={"taxon_id": "envo:01001867"},
    )

    assert resp.status_code == 404
    assert db_session.query(QuickLabelEntry).count() == 1


def test_unknown_dataset_is_a_404(api, users):
    resp = api(users[0]).get("/api/datasets/999999/quick-labels/mine")
    assert resp.status_code == 404


def test_entries_are_removed_with_their_dataset(api, dataset, users, db_session):
    _post(api(users[0]), dataset.id, taxon_id="gbif:2490719")

    db_session.delete(db_session.get(Dataset, dataset.id))
    db_session.commit()

    assert db_session.query(QuickLabelEntry).count() == 0
