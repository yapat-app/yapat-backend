"""Interface every embedding model implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple


class AudioEmbedder(ABC):
    """Class-level API: an embedder loads its model once per worker process."""

    SAMPLE_RATE: int
    WINDOW_SECONDS: float
    DIM: int

    @classmethod
    @abstractmethod
    def embed(cls, audio_path: str, start_time: float) -> Optional[List[float]]:
        """Embedding of the window starting at start_time, or None if there is no audio."""

    @classmethod
    @abstractmethod
    def embed_batch_from_recording(
        cls, audio_path: str, snippet_windows: List[Tuple[float, float]]
    ) -> List[Optional[List[float]]]:
        """One embedding per (start, end) window, in order; None where a window has no audio."""
