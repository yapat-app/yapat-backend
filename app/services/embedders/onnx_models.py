"""Embedding models that run on ONNX Runtime."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import librosa
import numpy as np

from app.config import settings
from app.services.embedders import weights_path
from app.services.embedders.base import AudioEmbedder

logger = logging.getLogger(__name__)


class OnnxEmbedder(AudioEmbedder):
    IMPLEMENTATION_KEY: str
    EMBEDDING_OUTPUT: str
    INFERENCE_BATCH_SIZE = 32
    # Peak each window is scaled to before inference; None leaves the audio as is.
    TARGET_PEAK: Optional[float] = None

    _session = None
    _input_name: Optional[str] = None

    @classmethod
    def window_samples(cls) -> int:
        return int(round(cls.WINDOW_SECONDS * cls.SAMPLE_RATE))

    @classmethod
    def _providers(cls) -> list:
        if settings.DEVICE != "gpu":
            return ["CPUExecutionProvider"]
        # Steps with no GPU version (the spectrogram) need the CPU provider too. Each
        # worker process loads its own copy, so take GPU memory only as it is needed.
        return [
            ("CUDAExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}),
            "CPUExecutionProvider",
        ]

    @classmethod
    def _load_session(cls):
        """Load the model once per worker process."""
        if cls._session is None:
            import onnxruntime as ort

            path: Path = weights_path(cls.IMPLEMENTATION_KEY)
            if not path.exists():
                raise FileNotFoundError(
                    f"Weights for {cls.IMPLEMENTATION_KEY} not found at {path}. "
                    "Run scripts/download_embedding_weights.py."
                )
            options = ort.SessionOptions()
            # Celery runs several worker processes, and each would otherwise
            # start one thread per core.
            options.intra_op_num_threads = settings.EMBEDDING_ONNX_THREADS
            session = ort.InferenceSession(str(path), sess_options=options, providers=cls._providers())
            active = session.get_providers()
            if settings.DEVICE == "gpu" and "CUDAExecutionProvider" not in active:
                raise RuntimeError(
                    f"DEVICE=gpu, but {cls.IMPLEMENTATION_KEY} could not load on the GPU "
                    f"(running on {active}). Check the container's GPU access and CUDA libraries."
                )
            logger.info("%s running on %s", cls.IMPLEMENTATION_KEY, active[0])
            cls._session = session
            cls._input_name = session.get_inputs()[0].name
        return cls._session

    @classmethod
    def _prepare_window(cls, audio: np.ndarray) -> np.ndarray:
        """Pad or trim to the model's input length, then apply its normalisation."""
        length = cls.window_samples()
        if audio.shape[0] < length:
            audio = np.pad(audio, (0, length - audio.shape[0]))
        else:
            audio = audio[:length]
        audio = audio.astype(np.float32)
        if cls.TARGET_PEAK is not None:
            audio = audio - audio.mean()
            peak = np.abs(audio).max()
            if peak > 0:
                audio = audio / peak * cls.TARGET_PEAK
        return audio

    @classmethod
    def _infer(cls, windows: List[np.ndarray]) -> List[List[float]]:
        session = cls._load_session()
        embeddings: List[List[float]] = []
        for start in range(0, len(windows), cls.INFERENCE_BATCH_SIZE):
            batch = np.stack(windows[start:start + cls.INFERENCE_BATCH_SIZE])
            output = session.run([cls.EMBEDDING_OUTPUT], {cls._input_name: batch})[0]
            embeddings.extend(output.astype(np.float32).tolist())
        return embeddings

    @classmethod
    def embed(cls, audio_path: str, start_time: float) -> Optional[List[float]]:
        audio, _ = librosa.load(
            audio_path,
            sr=cls.SAMPLE_RATE,
            offset=start_time,
            duration=cls.WINDOW_SECONDS,
            mono=True,
        )
        if audio.size == 0:
            return None
        return cls._infer([cls._prepare_window(audio)])[0]

    @classmethod
    def embed_batch_from_recording(
        cls, audio_path: str, snippet_windows: List[Tuple[float, float]]
    ) -> List[Optional[List[float]]]:
        if not snippet_windows:
            return []
        full_audio, _ = librosa.load(audio_path, sr=cls.SAMPLE_RATE, mono=True)

        windows: List[np.ndarray] = []
        valid_indices: List[int] = []
        for index, (start_time, end_time) in enumerate(snippet_windows):
            audio = full_audio[int(start_time * cls.SAMPLE_RATE):int(end_time * cls.SAMPLE_RATE)]
            if audio.size == 0:
                continue
            windows.append(cls._prepare_window(audio))
            valid_indices.append(index)

        results: List[Optional[List[float]]] = [None] * len(snippet_windows)
        if windows:
            for index, embedding in zip(valid_indices, cls._infer(windows)):
                results[index] = embedding
        return results


class BirdNetV3Embedder(OnnxEmbedder):
    """BirdNET+ V3.0 developer preview 3.1. Takes variable-length input; run at its 3 s default."""

    IMPLEMENTATION_KEY = "birdnet_v3"
    SAMPLE_RATE = 32000
    WINDOW_SECONDS = 3.0
    DIM = 1280
    EMBEDDING_OUTPUT = "embeddings_out"


class PerchV2Embedder(OnnxEmbedder):
    """Perch 2.0. Expects 5 s windows at 32 kHz, peak-normalised as in perch-hoplite."""

    IMPLEMENTATION_KEY = "perch_v2"
    SAMPLE_RATE = 32000
    WINDOW_SECONDS = 5.0
    DIM = 1536
    EMBEDDING_OUTPUT = "embedding"
    TARGET_PEAK = 0.25
