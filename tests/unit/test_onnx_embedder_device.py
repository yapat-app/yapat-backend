"""ONNX embedding models use the device the image was built for."""

import sys
from types import SimpleNamespace

import pytest

from app.config import settings
from app.services.embedders import onnx_models
from app.services.embedders.onnx_models import BirdNetV3Embedder


@pytest.fixture
def load_session(monkeypatch, tmp_path):
    """Returns a loader whose fake session runs on `active`, and the providers it was asked for."""
    weights = tmp_path / "model.onnx"
    weights.touch()
    monkeypatch.setattr(onnx_models, "weights_path", lambda key: weights)
    monkeypatch.setattr(BirdNetV3Embedder, "_session", None)
    monkeypatch.setattr(BirdNetV3Embedder, "_input_name", None)

    def load(device, active):
        requested = []

        def inference_session(path, sess_options, providers):
            requested.extend(providers)
            return SimpleNamespace(
                get_providers=lambda: active,
                get_inputs=lambda: [SimpleNamespace(name="inputs")],
            )

        fake_ort = SimpleNamespace(
            SessionOptions=lambda: SimpleNamespace(),
            InferenceSession=inference_session,
        )
        monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
        monkeypatch.setattr(settings, "DEVICE", device)
        return BirdNetV3Embedder._load_session(), requested

    return load


def test_cpu_build_only_asks_for_cpu(load_session):
    session, requested = load_session("cpu", ["CPUExecutionProvider"])
    assert requested == ["CPUExecutionProvider"]
    assert BirdNetV3Embedder._session is session


def test_gpu_build_runs_on_cuda(load_session):
    session, requested = load_session("gpu", ["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert requested[0][0] == "CUDAExecutionProvider"
    assert requested[1] == "CPUExecutionProvider"
    assert BirdNetV3Embedder._session is session


def test_gpu_build_refuses_to_fall_back_to_cpu(load_session):
    with pytest.raises(RuntimeError, match="could not load on the GPU"):
        load_session("gpu", ["CPUExecutionProvider"])
    assert BirdNetV3Embedder._session is None
