from pathlib import Path

import librosa
import numpy as np
from tensorflow.keras.layers import TFSMLayer


class BirdNetEmbedder:
    """
    BirdNET V2.4 embedder (raw waveform → 1024-d embedding).

    - Uses SavedModel endpoint "embeddings".
    - Expects exactly 144000 float32 samples (3s @ 48 kHz).
    - Returns Python list[float].
    """

    _instance = None
    _model_path = (
            Path(__file__).resolve().parent
            / ".." / "assets" / "models" / "birdnet"
    )

    SAMPLE_RATE = 48000
    WINDOW_SAMPLES = 144000  # 3 seconds

    @classmethod
    def instance(cls):
        """Load BirdNET SavedModel endpoint once per worker."""
        if cls._instance is None:
            cls._instance = TFSMLayer(
                str(cls._model_path),
                call_endpoint="embeddings"  # Your endpoint name
            )
        return cls._instance

    @classmethod
    def embed(cls, audio_path: str, start_time: float):
        """
        Extract a BirdNET embedding for a 3-second snippet.

        Args:
            audio_path: path to audio file
            start_time: snippet start time in seconds

        Returns:
            list[float] (size 1024) or None
        """

        # 1. Load exactly 3 seconds from file
        audio, _ = librosa.load(
            audio_path,
            sr=cls.SAMPLE_RATE,
            offset=start_time,
            duration=3.0,
            mono=True
        )

        if audio.size == 0:
            return None

        # 2. Pad or trim to match model requirements
        if len(audio) < cls.WINDOW_SAMPLES:
            audio = np.pad(audio, (0, cls.WINDOW_SAMPLES - len(audio)))
        else:
            audio = audio[:cls.WINDOW_SAMPLES]

        # Shape: (1, 144000)
        batch = audio.astype(np.float32)[None, :]

        # 3. Run inference
        model = cls.instance()
        outputs = model(batch)

        # 4. Extract embedding
        emb = outputs["embedding"].numpy()[0]

        return emb.tolist()
