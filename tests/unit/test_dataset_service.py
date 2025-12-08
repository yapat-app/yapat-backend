import pytest

from app.models.team import Team
from app.models.user import User, UserRole
from app.schemas.dataset import DatasetCreate
from app.services.dataset_service import DatasetService


# ------------------------------------------------------
# DatasetService tests
# ------------------------------------------------------

def test_create_dataset(db_session):
    svc = DatasetService(db_session)

    admin = User(username="admin@example.com", hashed_password="x", role=UserRole.ADMIN)
    team = Team(name="Team A")
    db_session.add_all([admin, team])
    db_session.commit()

    ds_in = DatasetCreate(
        team_id=team.id,
        name="MyDataset",
        description="Test",
        source_uri="ds1",
    )

    dataset = svc.create_dataset(ds_in, admin)

    assert dataset.id is not None
    assert dataset.name == "MyDataset"
    assert dataset.source_uri == "ds1"


def test_nonadmin_requires_team_id(db_session):
    svc = DatasetService(db_session)

    user = User(username="user", hashed_password="x", role=UserRole.USER)
    db_session.add(user)
    db_session.commit()

    ds_in = DatasetCreate(
        team_id=None,
        name="D1",
        description=None,
        source_uri="x",
    )

    # Non-admin must supply a team_id
    with pytest.raises(ValueError):
        svc.create_dataset(ds_in, user)


def test_create_dataset_duplicate_raises(db_session):
    svc = DatasetService(db_session)

    admin = User(username="adm", hashed_password="x", role=UserRole.ADMIN)
    team = Team(name="T")
    db_session.add_all([admin, team])
    db_session.commit()

    ds_in = DatasetCreate(
        team_id=team.id,
        name="D1",
        description=None,
        source_uri="duplicate",
    )

    svc.create_dataset(ds_in, admin)

    # Creating exact same dataset should fail
    with pytest.raises(ValueError):
        svc.create_dataset(ds_in, admin)


def test_claim_admin_dataset(db_session):
    svc = DatasetService(db_session)

    admin = User(username="adm", hashed_password="x", role=UserRole.ADMIN)
    user = User(username="user", hashed_password="x", role=UserRole.USER, team_id=99)

    db_session.add_all([admin, user])
    db_session.commit()

    ds_in = DatasetCreate(
        team_id=None,  # admin creates unowned dataset
        name="D",
        description=None,
        source_uri="claimable",
    )

    ds = svc.create_dataset(ds_in, admin)
    assert ds.team_id is None

    # User claims dataset
    claimed = svc.claim_dataset(ds, user)
    assert claimed.team_id == user.team_id


def test_claim_dataset_already_owned_fails(db_session):
    svc = DatasetService(db_session)

    admin = User(username="adm", hashed_password="x", role=UserRole.ADMIN)
    owner = User(username="owner", hashed_password="x", role=UserRole.USER, team_id=1)
    other = User(username="other", hashed_password="x", role=UserRole.USER, team_id=2)

    db_session.add_all([admin, owner, other])
    db_session.commit()

    ds_in = DatasetCreate(
        team_id=owner.team_id,  # dataset already has an owner
        name="D",
        description=None,
        source_uri="owned",
    )

    ds = svc.create_dataset(ds_in, admin)

    # Another user cannot claim an already-owned dataset
    with pytest.raises(ValueError):
        svc.claim_dataset(ds, other)
