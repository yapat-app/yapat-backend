from birdnetlib.analyzer import Analyzer
from birdnetlib import Recording as BirdNetRecording


class BirdNetEmbedder:
    """
    BirdNET embedder for snippet-level embeddings.

    - Model loads once per worker (Analyzer is expensive).
    - Embeddings returned as Python list[float].
    """

    _instance = None

    @classmethod
    def instance(cls):
        """Load Analyzer once per process."""
        if cls._instance is None:
            cls._instance = Analyzer()   # loads model weights
        return cls._instance

    @classmethod
    def embed(cls, audio_path: str, start_time: float, end_time: float):
        """
        Compute a BirdNET embedding for the audio segment.

        Returns:
            list[float] | None
        """
        analyzer = cls.instance()

        rec = BirdNetRecording(
            path=audio_path,
            analyzer=analyzer,
            min_conf=0.0,
            start_time=start_time,
            end_time=end_time,
        )

        # This runs BirdNET inference for the snippet duration
        rec.analyze()

        # BirdNET stores embeddings as rec.data["embedding"]
        vector = rec.data.get("embedding")

        if vector is None:
            return None

        # birdnetlib returns numpy array → convert to python list
        try:
            return vector.tolist()
        except AttributeError:
            # Already a list – safe fallback
            return vector
