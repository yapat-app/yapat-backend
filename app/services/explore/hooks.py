"""
Integration points that keep explore layers in sync with the rest of the app.

Every hook is best-effort: an explore failure must never fail inference, FPV
generation or a metadata import. When publishing fails the affected layer is
invalidated instead, so the next explore request rebuilds it from Postgres.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def publish_model_layer_from_inference(
    db: Session,
    checkpoint_id: int,
    snippet_ids: Sequence[int],
    probs: np.ndarray,
    preds: np.ndarray,
    score_matrix: np.ndarray,
    label_order: Sequence[str],
) -> None:
    from app.models.snippet import Snippet
    from app.services.explore import storage
    from app.services.explore.builders import model_arrays_from_matrices, model_key

    if len(snippet_ids) == 0:
        return
    snippet_set_id = None
    try:
        snippet_set_id = (
            db.query(Snippet.snippet_set_id).filter(Snippet.id == int(snippet_ids[0])).scalar()
        )
        if snippet_set_id is None:
            return
        arrays, meta = model_arrays_from_matrices(
            snippet_ids, probs, preds, score_matrix, label_order,
            checkpoint_id=checkpoint_id, snippet_set_id=int(snippet_set_id),
        )
        storage.write_layer("model", model_key(checkpoint_id, snippet_set_id), arrays, meta)
        prune_model_layers(int(snippet_set_id), keep_checkpoint_id=int(checkpoint_id))
    except Exception:
        logger.exception(
            "explore: publishing model layer for checkpoint %s failed; invalidating", checkpoint_id
        )
        if snippet_set_id is not None:
            invalidate_model(checkpoint_id, int(snippet_set_id))


def append_to_model_layer(
    checkpoint_id: int,
    snippet_set_id: int,
    snippet_ids: Sequence[int],
    probs: np.ndarray,
    preds: np.ndarray,
    score_matrix: np.ndarray,
    label_order: Sequence[str],
) -> None:
    """Merge newly scored rows into the published model layer (add-recordings).

    The layer is republished as a new version holding the old rows plus these
    (re-scored ids replace their old row), so the hub never has to rebuild a
    multi-million-row layer from al_predictions JSON. When no layer is published
    yet there is nothing to merge into: the next reader builds it from Postgres,
    which already holds these rows. A label-order mismatch or any error falls
    back to invalidating, which is always correct, only slower.
    """
    from app.services.explore import storage
    from app.services.explore.builders import model_arrays_from_matrices, model_key

    if len(snippet_ids) == 0:
        return
    key = model_key(checkpoint_id, snippet_set_id)
    try:
        pointer = storage.read_pointer("model", key)
        if pointer is None:
            return
        current = storage.load_layer("model", key, pointer.version)
        if list(current.meta.get("label_order") or []) != [str(x) for x in label_order]:
            invalidate_model(checkpoint_id, snippet_set_id)
            return

        new_arrays, meta = model_arrays_from_matrices(
            snippet_ids, probs, preds, score_matrix, label_order,
            checkpoint_id=checkpoint_id, snippet_set_id=snippet_set_id,
        )
        old_ids = np.asarray(current["snippet_ids"])
        keep = ~np.isin(old_ids, new_arrays["snippet_ids"])
        merged_ids = np.concatenate([old_ids[keep], new_arrays["snippet_ids"]])
        order = np.argsort(merged_ids, kind="stable")
        merged = {"snippet_ids": merged_ids[order]}
        for name in ("scores", "probs", "pred_bits", "default_confidence"):
            merged[name] = np.concatenate([np.asarray(current[name])[keep], new_arrays[name]])[order]

        meta = {**current.meta, "n": int(merged_ids.size), "source": "inference+incremental"}
        for reserved in ("kind", "key", "version", "arrays", "created_at"):
            meta.pop(reserved, None)
        storage.write_layer("model", key, merged, meta)
    except Exception:
        logger.exception(
            "explore: merging %s rows into model layer %s failed; invalidating", len(snippet_ids), key
        )
        invalidate_model(checkpoint_id, snippet_set_id)


MODEL_LAYERS_KEPT_PER_SNIPPET_SET = 3


def prune_model_layers(
    snippet_set_id: int,
    keep_checkpoint_id: int,
    keep: int = MODEL_LAYERS_KEPT_PER_SNIPPET_SET,
) -> None:
    """Delete model layers of superseded checkpoints for a snippet set.

    Every retrain publishes a layer as large as the snippet set, so only the
    newest few checkpoints are kept on disk. A client still pointing at a
    pruned checkpoint gets it rebuilt from Postgres on demand.
    """
    import re
    import shutil

    from app.services.explore import storage

    root = storage.cache_root() / "model"
    if not root.is_dir():
        return
    pattern = re.compile(rf"^ckpt(\d+)_ss{int(snippet_set_id)}$")
    found: list[tuple[int, str]] = []
    for child in root.iterdir():
        match = pattern.match(child.name)
        if match and child.is_dir():
            found.append((int(match.group(1)), child.name))
    found.sort(reverse=True)
    kept = {name for _, name in found[:keep]}
    for checkpoint_id, name in found:
        if name in kept or checkpoint_id == keep_checkpoint_id:
            continue
        storage.invalidate_layer("model", name)
        shutil.rmtree(root / name, ignore_errors=True)
        logger.info("explore: pruned model layer %s", name)


def invalidate_model(checkpoint_id: int, snippet_set_id: int) -> None:
    try:
        from app.services.explore import registry
        from app.services.explore.builders import model_key

        registry.invalidate("model", model_key(checkpoint_id, snippet_set_id))
    except Exception:
        logger.exception("explore: invalidating model layer failed")


def invalidate_projection(dataset_id: int, embedding_model_id: int) -> None:
    try:
        from app.services.explore import registry
        from app.services.explore.builders import projection_key

        registry.invalidate("projection", projection_key(dataset_id, embedding_model_id))
    except Exception:
        logger.exception("explore: invalidating projection layer failed")


def invalidate_dataset_base(db: Session, dataset_id: int) -> None:
    """Recording metadata changed: rebuild base layers of every snippet set."""
    try:
        from app.models.embedding import SnippetSet
        from app.services.explore import registry
        from app.services.explore.builders import base_key

        ids = [row[0] for row in db.query(SnippetSet.id).filter(SnippetSet.dataset_id == dataset_id).all()]
        for snippet_set_id in ids:
            registry.invalidate("base", base_key(snippet_set_id))
    except Exception:
        logger.exception("explore: invalidating base layers for dataset %s failed", dataset_id)
