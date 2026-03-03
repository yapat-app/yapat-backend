"""
PAM Model Checkout

Provides a "checkout" interface: given a dataset and optional version,
return a *model handle* that downstream code can pass to the classifier.

Key behaviour:
  - If the checkpoint already has a ``checkpoint_path`` (retrained model),
    that path is used directly.
  - If ``checkpoint_path`` is ``None`` (first-time / base entry), the
    resolver falls back to the physical base model file on disk whose
    location is configured via ``PAM_BASE_MODEL_PATH`` in settings.

The real implementation will version-manage checkpoints on disk/object-
store; for now it wraps metadata from the DB record and resolves the
effective weights path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PAMModelHandle:
    """
    Lightweight value-object returned by :func:`checkout_model`.

    ``effective_path`` is the **resolved** path that should always be
    passed to the classifier loader — it may point to the base model
    when no retrained checkpoint exists.
    """
    checkpoint_id: int
    dataset_id: int
    name: str
    version: str
    checkpoint_path: Optional[str]       # path stored in DB (may be None)
    effective_path: Optional[str]         # resolved path (base or retrained)
    model_type: str
    is_base: bool = False
    parent_checkpoint_id: Optional[int] = None
    hyperparameters: Dict[str, Any] = field(default_factory=dict)


def _resolve_base_model_path(base_model_setting: Optional[str] = None) -> Optional[str]:
    """
    Return the absolute path to the base model file if it exists.

    Resolution order:
      1. ``base_model_setting`` (Settings.PAM_BASE_MODEL_PATH) as-is if absolute.
      2. Relative to the repository / working directory.
      3. ``None`` if the file cannot be found.
    """
    if base_model_setting is None:
        return None

    # Absolute path
    if os.path.isabs(base_model_setting) and os.path.isfile(base_model_setting):
        return base_model_setting

    # Relative to cwd (repo root when run via uvicorn / celery)
    candidate = os.path.join(os.getcwd(), base_model_setting)
    if os.path.isfile(candidate):
        return os.path.abspath(candidate)

    # Relative to this file's directory (active_learning/)
    candidate = os.path.join(os.path.dirname(__file__), "..", base_model_setting)
    if os.path.isfile(candidate):
        return os.path.abspath(candidate)

    logger.warning("Base model file not found at '%s'", base_model_setting)
    return None


def checkout_model(
    checkpoint_id: int,
    dataset_id: int,
    name: str,
    version: str = "v0",
    checkpoint_path: Optional[str] = None,
    model_type: str = "pam_classifier",
    hyperparameters: Optional[Dict[str, Any]] = None,
    is_base: bool = False,
    parent_checkpoint_id: Optional[int] = None,
    base_model_path_setting: Optional[str] = None,
) -> PAMModelHandle:
    """
    "Check out" a model version for use in the PAM active-learning pipeline.

    Path resolution:
      • If *checkpoint_path* is not ``None`` **and the file exists**, use it
        (this is a retrained checkpoint).
      • Otherwise fall back to the base model file located at
        *base_model_path_setting* (``Settings.PAM_BASE_MODEL_PATH``).

    Args:
        checkpoint_id:          DB row id of the PAMModelCheckpoint.
        dataset_id:             Associated dataset.
        name:                   Human-readable model name.
        version:                Version tag (e.g. "v0", "v3_retrained").
        checkpoint_path:        Filesystem path to weights stored in DB (may be None).
        model_type:             Architecture identifier.
        hyperparameters:        Training / model hyper-params dict.
        is_base:                Whether this checkpoint represents the base model entry.
        parent_checkpoint_id:   ID of the parent checkpoint (version lineage).
        base_model_path_setting: Value of ``Settings.PAM_BASE_MODEL_PATH``.

    Returns:
        PAMModelHandle with ``effective_path`` resolved to a real file.
    """
    # --- Resolve the effective weights path ---
    effective_path: Optional[str] = None

    if checkpoint_path and os.path.isfile(checkpoint_path):
        effective_path = os.path.abspath(checkpoint_path)
        logger.info(
            "Model checkout: using retrained checkpoint at '%s'", effective_path,
        )
    else:
        effective_path = _resolve_base_model_path(base_model_path_setting)
        if effective_path:
            logger.info(
                "Model checkout: no retrained checkpoint — falling back to base model at '%s'",
                effective_path,
            )
        else:
            logger.warning(
                "Model checkout: no checkpoint_path and base model not found; "
                "classifier will receive None."
            )

    handle = PAMModelHandle(
        checkpoint_id=checkpoint_id,
        dataset_id=dataset_id,
        name=name,
        version=version,
        checkpoint_path=checkpoint_path,
        effective_path=effective_path,
        model_type=model_type,
        is_base=is_base,
        parent_checkpoint_id=parent_checkpoint_id,
        hyperparameters=hyperparameters or {},
    )
    logger.info(
        "Model checked out: id=%d  name=%s  version=%s  effective_path=%s",
        checkpoint_id, name, version, effective_path,
    )
    return handle
