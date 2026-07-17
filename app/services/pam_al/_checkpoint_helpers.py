"""
Checkpoint and model-family state helpers.

All functions take ``db: Session`` as their first argument so they stay
decoupled from the service class and are independently testable.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

import torch
from sqlalchemy import and_
from sqlalchemy.orm import Session

from app.config import settings
from app.models.dataset import Dataset, DatasetType
from app.models.pam_active_learning import (
    ALModelCheckpoint,
    ALModelFamilyState,
    ALModelStatus,
)
from app.schemas.pam_active_learning import ALModelType
from active_learning.model_zoo.mlp_multilabel_classifier import MultiLabelMLPClassifier
from active_learning.model_zoo.linear_multilabel_classifier import MultiLabelLinearClassifier
from app.models.wssed_pytorch_models import SimpleLinearClassifier
from active_learning.config import RETRAIN_AFTER

logger = logging.getLogger(__name__)


_AL_ELIGIBLE_DATASET_TYPES = frozenset({DatasetType.PAM, DatasetType.FOCAL_RECORDINGS})


def get_pam_dataset(db: Session, dataset_id: int) -> Dataset:
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if ds is None:
        raise ValueError(f"Dataset {dataset_id} not found")
    if ds.dataset_type not in _AL_ELIGIBLE_DATASET_TYPES:
        raise ValueError(
            f"Dataset {dataset_id} is of type '{ds.dataset_type.value}', "
            f"expected one of {[t.value for t in _AL_ELIGIBLE_DATASET_TYPES]}"
        )
    return ds


def get_retrain_threshold(db: Session, dataset_id: int) -> int:
    """Effective feedback-count threshold that triggers auto-retrain for a dataset.

    Uses the dataset's ``retrain_after_threshold`` override when set, otherwise
    falls back to the global ``RETRAIN_AFTER`` default.
    """
    override = (
        db.query(Dataset.retrain_after_threshold)
        .filter(Dataset.id == dataset_id)
        .scalar()
    )
    return override if override is not None else RETRAIN_AFTER


def get_checkpoint(db: Session, checkpoint_id: int) -> Optional[ALModelCheckpoint]:
    return (
        db.query(ALModelCheckpoint)
        .filter(ALModelCheckpoint.id == checkpoint_id)
        .first()
    )

def list_active_family_checkpoints(
    db: Session,
    dataset_id: Optional[int] = None,
) -> List[ALModelCheckpoint]:
    q = (
        db.query(ALModelCheckpoint)
        .join(
            ALModelFamilyState,
            ALModelFamilyState.active_model_checkpoint_id == ALModelCheckpoint.id,
        )
    )

    if dataset_id is not None:
        q = q.filter(ALModelFamilyState.dataset_id == dataset_id)

    return q.order_by(ALModelFamilyState.created_at.desc()).all()

def list_checkpoints(
    db: Session, dataset_id: Optional[int] = None
) -> List[ALModelCheckpoint]:
    q = db.query(ALModelCheckpoint)
    if dataset_id is not None:
        q = q.filter(ALModelCheckpoint.dataset_id == dataset_id)
    return q.order_by(ALModelCheckpoint.created_at.desc()).all()


def register_checkpoint(
    db: Session,
    dataset_id: int,
    model_family_name: str,
    version: str = "v0",
    checkpoint_path: Optional[str] = None,
    label_config_path: Optional[str] = None,
    model_type: str = ALModelType.PAM_LINEAR_MULTILABEL.value,
    hyperparameters: Optional[Dict[str, Any]] = None,
    is_base: bool = False,
    parent_checkpoint_id: Optional[int] = None,
) -> ALModelCheckpoint:
    get_pam_dataset(db, dataset_id)

    existing = (
        db.query(ALModelCheckpoint)
        .filter(
            and_(
                ALModelCheckpoint.dataset_id == dataset_id,
                ALModelCheckpoint.model_family_name == model_family_name,
                ALModelCheckpoint.version == version,
            )
        )
        .first()
    )

    if existing:
        existing.checkpoint_path = checkpoint_path or ""
        existing.label_config_path = label_config_path or ""
        existing.model_type = model_type
        existing.hyperparameters = hyperparameters
        existing.is_base = int(is_base)
        existing.parent_checkpoint_id = parent_checkpoint_id
        existing.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(existing)
        logger.info("Updated PAM checkpoint id=%d", existing.id)
        return existing

    ckpt = ALModelCheckpoint(
        dataset_id=dataset_id,
        model_family_name=model_family_name,
        version=version,
        checkpoint_path=checkpoint_path or "",
        label_config_path=label_config_path or "",
        model_type=model_type,
        hyperparameters=hyperparameters,
        is_base=int(is_base),
        parent_checkpoint_id=parent_checkpoint_id,
        status=ALModelStatus.AVAILABLE,
    )
    db.add(ckpt)
    db.flush()
    _ensure_family_state(
        db=db,
        dataset_id=dataset_id,
        model_family_name=model_family_name,
        checkpoint_id=ckpt.id,
    )
    db.commit()
    db.refresh(ckpt)
    logger.info("Registered PAM checkpoint id=%d name=%s is_base=%s", ckpt.id, model_family_name, is_base)
    return ckpt

def _ensure_family_state(
    db: Session,
    dataset_id: int,
    model_family_name: str,
    checkpoint_id: int,
) -> None:
    family = (
        db.query(ALModelFamilyState)
        .filter(
            ALModelFamilyState.dataset_id == dataset_id,
            ALModelFamilyState.model_family_name == model_family_name,
        )
        .one_or_none()
    )

    if family is None:
        family = ALModelFamilyState(
            dataset_id=dataset_id,
            model_family_name=model_family_name,
            active_model_checkpoint_id=checkpoint_id,
        )
        db.add(family)
    else:
        family.active_model_checkpoint_id = checkpoint_id

    logger.info("Created new entry in ALModelFamilyState")



def get_active_checkpoint_for_model_family(
    db: Session,
    dataset_id: int,
    model_family_name: str,
) -> ALModelCheckpoint:
    family = (
        db.query(ALModelFamilyState)
        .filter(
            ALModelFamilyState.dataset_id == dataset_id,
            ALModelFamilyState.model_family_name == model_family_name,
        )
        .one_or_none()
    )
    if family is None or family.active_model_checkpoint_id is None:
        return None

    ckpt = get_checkpoint(db, family.active_model_checkpoint_id)
    if ckpt is None:
        raise ValueError(
            f"Active checkpoint {family.active_model_checkpoint_id} not found."
        )
    return ckpt


def set_active_family_checkpoint(
    db: Session,
    dataset_id: int,
    model_family_name: str,
    checkpoint_id: int,
) -> None:
    row = (
        db.query(ALModelFamilyState)
        .filter(
            ALModelFamilyState.dataset_id == dataset_id,
            ALModelFamilyState.model_family_name == model_family_name,
        )
        .one_or_none()
    )

    if row is None:
        row = ALModelFamilyState(
            dataset_id=dataset_id,
            model_family_name=model_family_name,
            active_model_checkpoint_id=checkpoint_id,
        )
        db.add(row)
    else:
        row.active_model_checkpoint_id = checkpoint_id


# ── Disk I/O ────────────────────────────────────────────────────────────

def ensure_dir(dir_path: str) -> str:
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def save_classifier_checkpoint(
    model,
    checkpoint_path: str,
    hidden_dim: int | None,
    dropout: float | None,
    label_order: list[str],
) -> None:
    if hasattr(model, "model") and model.model is None:
        raise ValueError("Cannot save checkpoint: classifier architecture has not been created.")

    checkpoint = {
        "model_type": getattr(model, "model_type", None),
        "n_dim": getattr(model, "n_dim", getattr(model, "input_dim", None)),
        "num_classes": model.num_classes,
        "state_dict": model.state_dict(),
        "label_order": label_order,
    }
    if hidden_dim is not None:
        checkpoint["hidden_dim"] = hidden_dim

    if dropout is not None:
        checkpoint["dropout"] = dropout

    torch.save(checkpoint, checkpoint_path)


def save_label_config(label_config_path: str, species_list: List[str]) -> None:
    payload = {"species_list": species_list}
    with open(label_config_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_species_from_label_config(label_config_path: str) -> List[str]:
    if not label_config_path:
        raise ValueError("label_config_path is required.")
    if not os.path.isfile(label_config_path):
        raise ValueError(f"Label config file not found: {label_config_path}")

    with open(label_config_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    # Backward compatibility:
    # Some deployments historically used {"labels": [...]} instead of
    # {"species_list": [...]}.
    species_list = payload.get("species_list")
    if species_list is None:
        species_list = payload.get("labels")

    if not isinstance(species_list, list) or len(species_list) == 0:
        raise ValueError(
            "Label config must contain a non-empty 'species_list' field (or legacy 'labels' field)."
        )

    return [str(s) for s in species_list]


def make_checkpoint_path(dataset_id: int, family_name: str, version: str, ckpt_id: int) -> str:
    checkpoint_dir = ensure_dir(
        os.path.join(settings.PAM_CHECKPOINTS_DIR, "pam_active_learning", str(dataset_id))
    )
    return os.path.join(checkpoint_dir, f"{family_name}_{version}_ckpt_{ckpt_id}.pt")


def make_label_config_path(dataset_id: int, family_name: str, version: str, ckpt_id: int) -> str:
    checkpoint_dir = ensure_dir(
        os.path.join(settings.PAM_CHECKPOINTS_DIR, "pam_active_learning", str(dataset_id))
    )
    return os.path.join(checkpoint_dir, f"{family_name}_{version}_labels_{ckpt_id}.json")

CheckpointLayout = Literal["linear", "mlp", "unknown"]

_LINEAR_MODEL_TYPES = frozenset(
    {
        "pam_multilabel_classifier",
        "pam_multi_label_classifier",
        ALModelType.PAM_LINEAR_MULTILABEL.value,
    }
)


def detect_checkpoint_layout(state_dict: Dict[str, Any]) -> CheckpointLayout:
    """Infer classifier architecture from saved ``state_dict`` key names."""
    if not state_dict:
        return "unknown"
    keys = state_dict.keys()
    if "model.weight" in keys:
        return "linear"
    # MLP Sequential: Linear(0) -> ReLU(1) -> Dropout(2) -> Linear(3)
    if "model.3.weight" in keys:
        return "mlp"
    if "model.0.weight" in keys:
        return "linear"
    weight_keys = [k for k in keys if k.endswith(".weight") and isinstance(state_dict[k], torch.Tensor)]
    if len(weight_keys) == 1:
        return "linear"
    if len(weight_keys) >= 2:
        return "mlp"
    return "unknown"


def remap_legacy_linear_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remap legacy single-layer Sequential checkpoints (``model.0.*``) to the
    current ``nn.Linear`` layout (``model.*``).

    MLP checkpoints (``model.0.*`` + ``model.3.*``) are returned unchanged;
    callers should load those with :class:`MultiLabelMLPClassifier`.
    """
    if not state_dict or "model.weight" in state_dict:
        return state_dict
    if "model.3.weight" in state_dict:
        return state_dict

    remapped: Dict[str, Any] = {}
    for key, value in state_dict.items():
        if key.startswith("model.0."):
            remapped["model." + key[len("model.0.") :]] = value
        else:
            remapped[key] = value
    return remapped


def prepare_linear_classifier_checkpoint(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    """Return a checkpoint dict whose ``state_dict`` is compatible with linear load."""
    payload = dict(checkpoint)
    state_dict = payload.get("state_dict")
    if isinstance(state_dict, dict):
        payload["state_dict"] = remap_legacy_linear_state_dict(state_dict)
    return payload


def _is_linear_model_type(model_type: ALModelType | str) -> bool:
    if model_type == ALModelType.PAM_LINEAR_MULTILABEL:
        return True
    return str(model_type) in _LINEAR_MODEL_TYPES


def make_model(model_type: ALModelType | str):
    # Backward compatibility:
    # older checkpoints used 'pam_multilabel_classifier' (and sometimes the typo
    # 'pam_multi_label_classifier') as the linear model identifier.
    if model_type in {"pam_multilabel_classifier", "pam_multi_label_classifier"}:
        return MultiLabelLinearClassifier()
    if model_type == ALModelType.PAM_LINEAR_MULTILABEL or model_type == ALModelType.PAM_LINEAR_MULTILABEL.value:
        return MultiLabelLinearClassifier()
    if model_type == ALModelType.PAM_MLP_MULTILABEL or model_type == ALModelType.PAM_MLP_MULTILABEL.value:
        return MultiLabelMLPClassifier()
    if model_type == ALModelType.WSSED_BIRDNET_SEGMENT or model_type == ALModelType.WSSED_BIRDNET_SEGMENT.value:
        return SimpleLinearClassifier()
    raise ValueError(f"Unsupported model_type '{model_type}'")

def load_model_from_checkpoint(model_ckpt, device: str):
    path = model_ckpt.checkpoint_path
    model_type = model_ckpt.model_type

    if model_type == ALModelType.WSSED_BIRDNET_SEGMENT or model_type == ALModelType.WSSED_BIRDNET_SEGMENT.value:
        return SimpleLinearClassifier.load_from_checkpoint(path, device=device)

    if _is_linear_model_type(model_type):
        checkpoint = torch.load(path, map_location=device)
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Checkpoint at {path} is not a dict payload.")
        state_dict = checkpoint.get("state_dict")
        # Some checkpoints are saved as a bare state_dict (no "state_dict" wrapper key).
        # Fall back to treating the checkpoint itself as the state_dict for layout detection.
        effective_state_dict = state_dict if isinstance(state_dict, dict) else checkpoint
        layout = detect_checkpoint_layout(effective_state_dict)
        if layout == "mlp":
            logger.warning(
                "Checkpoint id=%s is registered as linear but contains MLP "
                "Sequential weights (model.0 + model.3); loading as MLP.",
                model_ckpt.id,
            )
            return MultiLabelMLPClassifier.load_from_checkpoint(path, device=device)
        prepared = prepare_linear_classifier_checkpoint(checkpoint)
        return MultiLabelLinearClassifier.load_from_checkpoint_dict(prepared, device=device)

    return MultiLabelMLPClassifier.load_from_checkpoint(path, device=device)
