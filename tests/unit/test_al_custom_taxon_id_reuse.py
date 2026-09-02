"""
`Annotation.taxon_id` has to be an identifier, not a per-row uuid.

For a dataset with no team -- a supported mode, since `create_dataset` lets an
admin omit team_id -- AL used to mint a fresh `custom:<uuid>` on every call and
create no CustomTaxonomy row at all. On the local database that produced 768
distinct "taxon ids" for 9 labels on one dataset, none of them resolving to
anything, and made `group_by(taxon_id)` return one group per annotation.
"""

import pytest

from app.models.custom_taxonomy import CustomTaxonomy
from app.models.dataset import Dataset, DatasetType
from app.models.team import Team, TeamMembership
from app.models.user import User
from app.services.pam_al.service import PAMActiveLearningService


@pytest.fixture
def env(db_session):
    db = db_session
    admin = User(username="admin", hashed_password="x")
    member = User(username="mila", hashed_password="x")
    team = Team(name="Test Team")
    db.add_all([admin, member, team])
    db.commit()

    solo = Dataset(name="Admin dataset", source_uri="a", dataset_type=DatasetType.PAM)
    other_solo = Dataset(name="Other admin dataset", source_uri="b",
                         dataset_type=DatasetType.PAM)
    owned = Dataset(name="Team dataset", source_uri="c",
                    dataset_type=DatasetType.PAM, team_id=team.id)
    db.add_all([solo, other_solo, owned])
    db.commit()

    return {
        "svc": PAMActiveLearningService(db), "db": db,
        "admin": admin.id, "member": member.id, "team": team.id,
        "solo": solo.id, "other_solo": other_solo.id, "owned": owned.id,
    }


def _id_for(env, dataset_key, code, user_key="admin"):
    return env["svc"]._get_or_create_custom_taxon_id_for_code(
        dataset_id=env[dataset_key], code=code, user_id=env[user_key]
    )


def test_teamless_dataset_reuses_one_id_per_label(env):
    first = _id_for(env, "solo", "SCIALT")
    second = _id_for(env, "solo", "SCIALT")
    third = _id_for(env, "solo", "SCIALT")

    assert first == second == third


def test_teamless_id_resolves_to_a_taxonomy_row(env):
    """The old fallback returned an id backed by no row at all."""
    taxon_id = _id_for(env, "solo", "SCIALT")

    row = (
        env["db"].query(CustomTaxonomy)
        .filter(CustomTaxonomy.taxonomy_id == taxon_id)
        .one()
    )
    assert row.name == "SCIALT"
    assert row.team_id is None
    assert row.dataset_id == env["solo"]


def test_distinct_labels_still_get_distinct_ids(env):
    assert _id_for(env, "solo", "SCIALT") != _id_for(env, "solo", "DENMIN")


def test_teamless_datasets_do_not_share_label_ids(env):
    """
    Dataset-scoped means scoped: two admin datasets using the same code are not
    asserting they mean the same taxon.
    """
    assert _id_for(env, "solo", "SCIALT") != _id_for(env, "other_solo", "SCIALT")


def test_team_dataset_still_reuses_team_wide_row(env):
    """A team's label code keeps meaning the same across its datasets."""
    first = _id_for(env, "owned", "SCIALT", user_key="member")
    second = _id_for(env, "owned", "SCIALT", user_key="member")

    assert first == second
    row = (
        env["db"].query(CustomTaxonomy)
        .filter(CustomTaxonomy.taxonomy_id == first)
        .one()
    )
    assert row.team_id == env["team"]
    assert row.dataset_id is None


def test_team_and_dataset_scoped_rows_coexist(env):
    """
    The same code under a team and under an admin dataset are separate rows;
    the partial unique index must not collapse them.
    """
    team_id_value = _id_for(env, "owned", "SCIALT", user_key="member")
    solo_id_value = _id_for(env, "solo", "SCIALT")

    assert team_id_value != solo_id_value
    assert env["db"].query(CustomTaxonomy).filter(
        CustomTaxonomy.name == "SCIALT"
    ).count() == 2


def test_membership_still_attaches_a_teamless_dataset_to_the_users_team(env):
    """
    Pre-existing behaviour: an annotator who belongs to a team gets team scope
    even on a dataset that has none. Unchanged by the fix.
    """
    env["db"].add(TeamMembership(user_id=env["member"], team_id=env["team"]))
    env["db"].commit()

    taxon_id = _id_for(env, "solo", "LEPFUS", user_key="member")

    row = (
        env["db"].query(CustomTaxonomy)
        .filter(CustomTaxonomy.taxonomy_id == taxon_id)
        .one()
    )
    assert row.team_id == env["team"]
    assert row.dataset_id is None


def test_blank_code_yields_no_id(env):
    assert _id_for(env, "solo", "   ") is None
