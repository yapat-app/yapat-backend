"""
PAM Custom Classifier

Provides inference over audio snippet embeddings and produces
per-snippet predictions (label + confidence score).

When a valid checkpoint file is supplied the factory will attempt to
load weights via ``torch.load``; if torch is unavailable or the file
cannot be loaded it falls back to the stub so the pipeline never
crashes.
"""

from __future__ import annotations

import os
from typing import List, Tuple, Optional, Protocol
import numpy as np
import logging

logger = logging.getLogger(__name__)


# ── Classifier Protocol ────────────────────────────────────────────────

class PAMClassifierProtocol(Protocol):
    """
    Any object satisfying this protocol can be used as the PAM classifier.
    """

    def predict(
        self, embeddings: np.ndarray
    ) -> Tuple[List[str], np.ndarray]:
        """
        Run inference on a batch of embeddings.

        Args:
            embeddings: [N, D] float32 embedding matrix.

        Returns:
            labels:      length-N list of predicted label strings.
            confidences: [N] float array of confidence scores in [0, 1].
        """
        ...


# ── Placeholder Implementation ─────────────────────────────────────────

class PAMClassifierStub:
    """
    Stub classifier that returns a fixed label with random confidence.

    Replace this class with a real model once the checkpoint format and
    inference logic are finalised.
    """

    def __init__(
        self,
        default_label: str = "unknown",
        seed: int = 42,
        checkpoint_path: Optional[str] = None,
    ):
        self.default_label = default_label
        self.checkpoint_path = checkpoint_path
        self._rng = np.random.default_rng(seed)
        logger.info(
            "PAMClassifierStub initialised (placeholder)  weights=%s",
            checkpoint_path or "<none>",
        )

    def predict(
        self, embeddings: np.ndarray
    ) -> Tuple[List[str], np.ndarray]:
        """
        Return placeholder predictions.

        Confidences are drawn from U(0,1) so the combined scoring module
        has non-trivial uncertainty to rank on.
        """
        n = embeddings.shape[0]
        labels = [self.default_label] * n
        confidences = self._rng.uniform(0.0, 1.0, size=n).astype(np.float32)
        logger.info(f"PAMClassifierStub.predict: {n} samples → label='{self.default_label}'")
        return labels, confidences


# ── Factory ────────────────────────────────────────────────────────────

def _try_torch_load(checkpoint_path: str, device: str) -> Optional[object]:
    """
    Attempt to load a state_dict from *checkpoint_path* using torch.

    Returns the loaded object (usually ``OrderedDict``) on success,
    or ``None`` when torch is unavailable / loading fails.
    """
    try:
        import torch  # type: ignore[import-unresolved]
    except ImportError:
        logger.warning("torch not installed — cannot load checkpoint from '%s'", checkpoint_path)
        return None

    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        logger.info("Successfully loaded checkpoint from '%s'", checkpoint_path)
        return state
    except Exception as exc:
        logger.warning("Failed to load checkpoint '%s': %s", checkpoint_path, exc)
        return None


def load_pam_classifier(
    checkpoint_path: Optional[str] = None,
    model_type: str = "pam_classifier",
    device: str = "cpu",
) -> PAMClassifierProtocol:
    """
    Load (or create) a PAM classifier.

    Resolution logic:
      1. If *checkpoint_path* is ``None`` or the file doesn't exist →
         return stub.
      2. Try ``torch.load`` on the file.  If successful, wrap the loaded
         state into a real model class (TODO once architecture is
         finalised).  For now the state is loaded to verify the file is
         valid and the stub is returned with an acknowledgement.
      3. On any failure → return stub so the pipeline never crashes.

    Args:
        checkpoint_path: optional filesystem path to model weights.
        model_type:      identifier for model architecture dispatch.
        device:          'cpu' or 'cuda'.

    Returns:
        An object implementing :class:`PAMClassifierProtocol`.
    """
    if checkpoint_path is None:
        logger.warning("No checkpoint_path provided — using PAMClassifierStub")
        return PAMClassifierStub()

    if not os.path.isfile(checkpoint_path):
        logger.warning(
            "Checkpoint file does not exist at '%s' — using PAMClassifierStub",
            checkpoint_path,
        )
        return PAMClassifierStub()

    # Attempt to load the weights to validate the file
    state = _try_torch_load(checkpoint_path, device)
    if state is not None:
        # TODO: construct real model from state_dict once architecture is locked
        logger.info(
            "Checkpoint loaded successfully from '%s' (model_type='%s'). "
            "Real model wrapping not yet implemented — using stub with weights acknowledged.",
            checkpoint_path, model_type,
        )
        return PAMClassifierStub(checkpoint_path=checkpoint_path)

    # Fallback
    logger.warning(
        "Could not load checkpoint from '%s' — using PAMClassifierStub",
        checkpoint_path,
    )
    return PAMClassifierStub(checkpoint_path=checkpoint_path)
