import pytest
from unittest.mock import MagicMock, patch

from app.services.birdnet_model import BirdNetEmbedder


def test_birdnet_instance_is_singleton(monkeypatch):
    """Analyzer should load only once per worker."""
    mock_analyzer = MagicMock()
    monkeypatch.setattr("app.services.birdnet_model.Analyzer", lambda: mock_analyzer)

    inst1 = BirdNetEmbedder.instance()
    inst2 = BirdNetEmbedder.instance()

    assert inst1 is inst2
    assert inst1 is mock_analyzer


def test_birdnet_embed_returns_list(monkeypatch):
    """embed() should return a Python list, even if BirdNET returns numpy-like arrays."""

    # --- Fake BirdNET Recording ---
    fake_vector = MagicMock()
    fake_vector.tolist.return_value = [1.0, 2.0, 3.0]

    fake_recording = MagicMock()
    fake_recording.data = {"embedding": fake_vector}

    # Monkeypatch Analyzer instance
    monkeypatch.setattr(
        "app.services.birdnet_model.BirdNetEmbedder.instance",
        lambda: MagicMock()
    )

    # Monkeypatch BirdNET Recording class
    monkeypatch.setattr(
        "app.services.birdnet_model.BirdNetRecording",
        lambda path, analyzer, min_confidence, start_time, end_time: fake_recording
    )

    result = BirdNetEmbedder.embed("fake.wav", 0.0, 1.0)

    assert result == [1.0, 2.0, 3.0]
    fake_vector.tolist.assert_called_once()


def test_birdnet_embed_handles_missing_embedding(monkeypatch):
    """embed() should return None if BirdNET produced no embedding."""

    fake_recording = MagicMock()
    fake_recording.data = {}  # missing "embedding"

    monkeypatch.setattr(
        "app.services.birdnet_model.BirdNetEmbedder.instance",
        lambda: MagicMock()
    )

    monkeypatch.setattr(
        "app.services.birdnet_model.BirdNetRecording",
        lambda path, analyzer, min_confidence, start_time, end_time: fake_recording
    )

    result = BirdNetEmbedder.embed("fake.wav", 0.0, 1.0)

    assert result is None


def test_birdnet_embed_calls_analyze(monkeypatch):
    """Ensure embed() triggers BirdNET inference via rec.analyze()."""

    fake_recording = MagicMock()
    fake_recording.data = {"embedding": [0.1, 0.2, 0.3]}

    monkeypatch.setattr(
        "app.services.birdnet_model.BirdNetEmbedder.instance",
        lambda: MagicMock()
    )

    monkeypatch.setattr(
        "app.services.birdnet_model.BirdNetRecording",
        lambda path, analyzer, min_confidence, start_time, end_time: fake_recording
    )

    BirdNetEmbedder.embed("fake.wav", 0.0, 1.0)

    fake_recording.analyze.assert_called_once()
