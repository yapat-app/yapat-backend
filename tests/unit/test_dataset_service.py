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

