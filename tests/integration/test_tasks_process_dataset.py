from unittest.mock import patch
from app.models.dataset import Dataset
from app.tasks.processing_tasks import process_dataset


def test_process_dataset_triggers_scan_task(db_session):
    ds = Dataset(team_id=None, name="D", description=None, source_uri="src")
    db_session.add(ds)
    db_session.commit()

    with patch("app.tasks.processing_tasks.scan_dataset.delay") as mock_delay:
        mock_delay.return_value.id = "tid123"

        res = process_dataset.apply(args=[ds.id]).get()

        assert res["status"] == "submitted"
        assert res["scan_task_id"] == "tid123"
        mock_delay.assert_called_once_with(ds.id)
