from active_learning.config import RETRAIN_AFTER
from app.models.dataset import Dataset
from app.schemas.dataset import Dataset as DatasetSchema
from app.utils.dataset_response import dataset_to_dict


def _dataset(**kwargs) -> Dataset:
    return Dataset(id=1, name="D", source_uri="d", is_reference=False, **kwargs)


def test_response_includes_the_platform_default_threshold():
    data = dataset_to_dict(_dataset())

    assert data["default_retrain_threshold"] == RETRAIN_AFTER
    assert data["retrain_after_threshold"] is None


def test_default_is_sent_next_to_a_custom_threshold():
    data = dataset_to_dict(_dataset(retrain_after_threshold=3))

    assert data["retrain_after_threshold"] == 3
    assert data["default_retrain_threshold"] == RETRAIN_AFTER


def test_default_survives_the_response_schema():
    data = dataset_to_dict(_dataset())
    data["created_at"] = "2026-01-01T00:00:00"

    assert DatasetSchema(**data).default_retrain_threshold == RETRAIN_AFTER
