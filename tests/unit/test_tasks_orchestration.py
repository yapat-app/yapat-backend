import pytest
from unittest.mock import MagicMock, patch

from app.tasks.processing_tasks import process_dataset, scan_dataset


def test_process_dataset_submits_scan_task():
    """Ensure process_dataset schedules scan_dataset.delay properly."""
    with patch("app.tasks.processing_tasks.scan_dataset.delay") as mock_delay:
        mock_delay.return_value.id = "fake123"

        result = process_dataset.apply(args=[99]).get()

        mock_delay.assert_called_once_with(99)
        assert result["status"] == "submitted"
        assert result["dataset_id"] == 99
        assert result["scan_task_id"] == "fake123"


def test_scan_dataset_handles_missing_dataset(db_session):
    """When dataset does not exist, return error."""
    result = scan_dataset.apply(args=[99999]).get()
    assert result["status"] == "error"
    assert result["message"] == "dataset_not_found"
