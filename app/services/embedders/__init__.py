"""
Embedders turn audio into vectors. Each one is registered here under the key
stored in `embedding_models.implementation_key`, so a new model needs one
adapter class, one entry below and one `embedding_models` row.

Embedder classes are imported on first use because they pull in TensorFlow or
ONNX Runtime, which the API process should not load just to list models.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Dict, Optional, Type, Union

from app.config import settings
from app.services.embedders.base import AudioEmbedder

_BUNDLED_MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "assets" / "models"

# implementation_key -> "module:Class"
_EMBEDDERS: Dict[str, Union[str, Type[AudioEmbedder]]] = {
    "birdnet_v2_4": "app.services.birdnet_model:BirdNetEmbedder",
    "birdnet_v3": "app.services.embedders.onnx_models:BirdNetV3Embedder",
    "perch_v2": "app.services.embedders.onnx_models:PerchV2Embedder",
}

# implementation_key -> weights path. BirdNET 2.4 ships in git; the others are
# fetched by scripts/download_embedding_weights.py.
_WEIGHTS: Dict[str, Path] = {
    "birdnet_v2_4": _BUNDLED_MODELS_DIR / "birdnet" / "saved_model.pb",
    "birdnet_v3": Path("birdnet_v3") / "BirdNET+_V3.0-preview3.1_Global_11K_FP16_pruned.onnx",
    "perch_v2": Path("perch_v2") / "perch_v2.onnx",
}


class UnknownEmbedderError(ValueError):
    pass


def register_embedder(key: str, embedder: Type[AudioEmbedder]) -> None:
    """Register an embedder class directly (used by tests)."""
    _EMBEDDERS[key] = embedder


def unregister_embedder(key: str) -> None:
    _EMBEDDERS.pop(key, None)


def weights_path(key: str) -> Optional[Path]:
    relative = _WEIGHTS.get(key)
    if relative is None:
        return None
    return Path(settings.EMBEDDING_WEIGHTS_DIR) / relative


def weights_available(key: Optional[str]) -> bool:
    """False when the model cannot run here. Does not import the embedder."""
    if key is None or key not in _EMBEDDERS:
        return False
    path = weights_path(key)
    return path is None or path.exists()


def get_embedder(key: Optional[str]) -> Type[AudioEmbedder]:
    entry = _EMBEDDERS.get(key) if key else None
    if entry is None:
        raise UnknownEmbedderError(f"No embedder registered for implementation_key={key!r}")
    if isinstance(entry, str):
        module_name, class_name = entry.split(":")
        return getattr(importlib.import_module(module_name), class_name)
    return entry
