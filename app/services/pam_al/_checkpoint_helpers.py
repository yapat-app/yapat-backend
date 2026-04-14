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
from typing import Any, Dict, List, Optional

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

logger = logging.getLogger(__name__)


def get_pam_dataset(db: Session, dataset_id: int) -> Dataset:
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if ds is None:
        raise ValueError(f"Dataset {dataset_id} not found")
    if ds.dataset_type != DatasetType.PAM:
        raise ValueError(
            f"Dataset {dataset_id} is of type '{ds.dataset_type.value}', expected 'PAM'"
        )
    return ds


def get_checkpoint(db: Session, checkpoint_id: int) -> Optional[ALModelCheckpoint]:
    return (
        db.query(ALModelCheckpoint)
        .filter(ALModelCheckpoint.id == checkpoint_id)
        .first()
    )


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
    model_type: str = "pam_multilabel_classifier",
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
    db.commit()
    db.refresh(ckpt)
    logger.info("Registered PAM checkpoint id=%d name=%s is_base=%s", ckpt.id, model_family_name, is_base)
    return ckpt


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
    hidden_dim: int,
    dropout: float,
    label_order: list[str],
) -> None:
    if model.model is None:
        raise ValueError("Cannot save checkpoint: classifier architecture has not been created.")

    checkpoint = {
        "model_type": "pam_multilabel_classifier",
        "n_dim": model.n_dim,
        "num_classes": model.num_classes,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "state_dict": model.state_dict(),
        "label_order": label_order,
    }
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

    species_list = payload.get("species_list")
    if not isinstance(species_list, list) or len(species_list) == 0:
        raise ValueError("Label config must contain a non-empty 'species_list' field.")

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
