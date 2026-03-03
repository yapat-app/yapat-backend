"""
PAM Retraining Orchestration

Handles:
  - Interaction counter: track feedback events since last retrain
  - Auto-retrain trigger: fire after N interactions (default N=5)
  - Manual retrain trigger
  - Placeholder training entrypoint
  - Versioned checkpoint file storage on disk
"""

from __future__ import annotations

import os
import shutil
from typing import Any, Dict, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────

AUTO_RETRAIN_THRESHOLD: int = 5  # retrain after this many feedbacks


# ── Interaction counter ────────────────────────────────────────────────

class InteractionCounter:
    """
    Keeps an in-process count of feedback events since the last retrain
    for each model checkpoint.  For persistence, the service layer writes
    to the DB; this class is a fast, non-DB helper.
    """

    def __init__(self) -> None:
        self._counts: Dict[int, int] = {}  # checkpoint_id → count

    def increment(self, checkpoint_id: int) -> int:
        """Increment and return the new count."""
        self._counts[checkpoint_id] = self._counts.get(checkpoint_id, 0) + 1
        return self._counts[checkpoint_id]

    def get(self, checkpoint_id: int) -> int:
        return self._counts.get(checkpoint_id, 0)

    def reset(self, checkpoint_id: int) -> None:
        self._counts[checkpoint_id] = 0

    def should_retrain(self, checkpoint_id: int, threshold: int = AUTO_RETRAIN_THRESHOLD) -> bool:
        return self.get(checkpoint_id) >= threshold


# Global singleton (reset on worker restart — for dev mode; prod should
# query the DB).
_counter = InteractionCounter()


def get_interaction_counter() -> InteractionCounter:
    return _counter


# ── Versioned checkpoint storage ──────────────────────────────────────

def _make_versioned_filename(
    name: str,
    new_version: str,
    timestamp: Optional[datetime] = None,
) -> str:
    """
    Build a filename for a new retrained checkpoint.

    Format: ``<name>_<version>_<YYYYMMDD_HHMMSS>.pt``
    """
    ts = timestamp or datetime.utcnow()
    ts_str = ts.strftime("%Y%m%d_%H%M%S")
    safe_name = name.lower().replace(" ", "_").replace("/", "_")
    return f"{safe_name}_{new_version}_{ts_str}.pt"


def save_retrained_checkpoint(
    checkpoints_dir: str,
    name: str,
    new_version: str,
    source_path: Optional[str] = None,
) -> str:
    """
    Persist a retrained model checkpoint to the versioned storage directory.

    In the real implementation this will receive the trained model's state
    dict and call ``torch.save``.  For now it either:
      • copies *source_path* (the parent weights) into the new versioned
        file, or
      • writes a minimal placeholder when no source is available.

    Args:
        checkpoints_dir: Root directory for versioned checkpoints
                         (``Settings.PAM_CHECKPOINTS_DIR``).
        name:            Model name used for the filename.
        new_version:     Version tag for the new checkpoint (e.g. "v1").
        source_path:     Path to the parent / base weights to copy as
                         starting point.

    Returns:
        Absolute path of the newly written checkpoint file.
    """
    os.makedirs(checkpoints_dir, exist_ok=True)

    filename = _make_versioned_filename(name, new_version)
    dest = os.path.join(checkpoints_dir, filename)

    if source_path and os.path.isfile(source_path):
        shutil.copy2(source_path, dest)
        logger.info(
            "Saved retrained checkpoint (copied from parent): %s → %s",
            source_path, dest,
        )
    else:
        # Write a minimal placeholder so the file physically exists.
        # Once real training is plugged in, this branch will be replaced
        # with torch.save(model.state_dict(), dest).
        try:
            import torch  # type: ignore[import-unresolved]
            import torch.nn as nn  # type: ignore[import-unresolved]
            dummy = nn.Linear(128, 10)
            torch.save(dummy.state_dict(), dest)
        except ImportError:
            import pickle
            import collections
            with open(dest, "wb") as f:
                pickle.dump(collections.OrderedDict(), f)
        logger.info("Saved retrained checkpoint (placeholder): %s", dest)

    return os.path.abspath(dest)


# ── Placeholder training entrypoint ───────────────────────────────────

def run_retrain(
    checkpoint_id: int,
    *,
    trigger: str = "auto",
    epochs: int = 5,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    feedback_count: int = 0,
    model_name: str = "pam_model",
    new_version: str = "v1",
    parent_checkpoint_path: Optional[str] = None,
    checkpoints_dir: str = "models_AL/pam/checkpoints",
) -> Dict[str, Any]:
    """
    Placeholder retraining function.

    In the real implementation this will:
      1. Collect all accepted/modified feedback labels for the checkpoint.
      2. Build a training DataLoader.
      3. Fine-tune the model.
      4. Save a new checkpoint version via :func:`save_retrained_checkpoint`.
      5. Return metrics.

    Currently it logs the call, saves a versioned checkpoint file to disk,
    and returns stub metrics including the new checkpoint path.

    Args:
        checkpoint_id:          PAMModelCheckpoint.id
        trigger:                "auto" or "manual"
        epochs:                 number of training epochs
        learning_rate:          optimizer LR
        device:                 "cpu" or "cuda"
        feedback_count:         how many feedback events triggered this retrain
        model_name:             human-readable name (used in filename)
        new_version:            version tag for the new checkpoint
        parent_checkpoint_path: path to the parent/current weights (base or previous version)
        checkpoints_dir:        directory where versioned checkpoints are stored

    Returns:
        dict with retrain result / metrics including ``new_checkpoint_path``.
    """
    logger.info(
        "PAM retrain triggered: checkpoint_id=%d  trigger=%s  "
        "epochs=%d  lr=%s  device=%s  feedback_count=%d  new_version=%s",
        checkpoint_id, trigger, epochs, learning_rate, device,
        feedback_count, new_version,
    )

    # TODO: plug in real training loop — e.g. call
    #   active_learning.active_learning.ActiveLearning.retrain(...)
    # or a custom training script.

    # Save the new versioned checkpoint to disk
    new_checkpoint_path = save_retrained_checkpoint(
        checkpoints_dir=checkpoints_dir,
        name=model_name,
        new_version=new_version,
        source_path=parent_checkpoint_path,
    )

    result = {
        "status": "completed",
        "checkpoint_id": checkpoint_id,
        "trigger": trigger,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "feedback_count": feedback_count,
        "new_version": new_version,
        "new_checkpoint_path": new_checkpoint_path,
        "loss": None,       # placeholder
        "accuracy": None,   # placeholder
        "started_at": datetime.utcnow().isoformat() + "Z",
        "completed_at": datetime.utcnow().isoformat() + "Z",
    }

    logger.info("PAM retrain completed (stub): %s", result)
    return result
