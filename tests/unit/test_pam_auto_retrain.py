from types import SimpleNamespace

import torch

from app.models.pam_active_learning import (
    ALModelCheckpoint,
    ALModelStatus,
    ALModelType,
    ALRetrainJob,
)
from app.services.pam_al import service as pam_service
from app.services.pam_al import _checkpoint_helpers as checkpoint_helpers


class _FakeSession:
    def __init__(self):
        self.added = []
        self._next_id = 100

    def add(self, obj):
        self.added.append(obj)

    def flush(self):
        for obj in self.added:
            if getattr(obj, "id", None) is None:
                obj.id = self._next_id
                self._next_id += 1

    def commit(self):
        pass

    def refresh(self, _obj):
        pass


def test_resolve_label_order_from_checkpoint_payload(tmp_path):
    checkpoint_path = tmp_path / "legacy.pt"
    torch.save({"label_order": ["gbif:123", "gbif:456"]}, checkpoint_path)
    checkpoint = ALModelCheckpoint(
        id=15,
        hyperparameters={},
        label_config_path="",
        checkpoint_path=str(checkpoint_path),
    )

    assert checkpoint_helpers.resolve_checkpoint_label_order(checkpoint) == [
        "gbif:123",
        "gbif:456",
    ]


def test_setup_auto_retrain_resolves_and_persists_parent_label_order(monkeypatch):
    parent = ALModelCheckpoint(
        id=15,
        dataset_id=7,
        model_family_name="birdnet",
        version="v1",
        checkpoint_path="/checkpoints/parent.pt",
        label_config_path="/checkpoints/parent-labels.json",
        model_type=ALModelType.PAM_LINEAR_MULTILABEL.value,
        hyperparameters={
            "resolved_snippet_set_id": 11,
            "embedding_model_id": 3,
        },
        is_base=1,
        status=ALModelStatus.AVAILABLE,
    )
    db = _FakeSession()

    monkeypatch.setattr(pam_service.ckpt_h, "get_checkpoint", lambda _db, _id: parent)
    monkeypatch.setattr(
        pam_service.ckpt_h,
        "get_pam_dataset",
        lambda _db, _id: SimpleNamespace(default_snippet_set_id=12),
    )
    monkeypatch.setattr(
        pam_service.fb_h,
        "feedback_count_since_retrain",
        lambda _db, _id: 5,
    )
    monkeypatch.setattr(
        pam_service.ckpt_h,
        "resolve_checkpoint_label_order",
        lambda checkpoint: ["gbif:123", "gbif:456"],
    )

    checkpoint, job = pam_service.PAMActiveLearningService(db).setup_auto_retrain(parent.id)

    assert isinstance(checkpoint, ALModelCheckpoint)
    assert isinstance(job, ALRetrainJob)
    assert checkpoint.parent_checkpoint_id == parent.id
    assert checkpoint.hyperparameters["resolved_snippet_set_id"] == 12
    assert checkpoint.hyperparameters["embedding_model_id"] == 3
    assert checkpoint.hyperparameters["label_order"] == ["gbif:123", "gbif:456"]
    assert checkpoint.hyperparameters["run_inference"] is True
    assert parent.hyperparameters["label_order"] == ["gbif:123", "gbif:456"]
    assert job.model_checkpoint_id == checkpoint.id
