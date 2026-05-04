"""
Inference helpers: scoring, prediction CRUD, and suggestion ranking.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
from sqlalchemy.orm import Session

from app.models.snippet import Snippet
from app.models.pam_active_learning import ALPrediction
from app.schemas.pam_active_learning import ALInferenceRow

from active_learning.samplers import uncertainty, density, diversity, composite
from active_learning.config import (
    DEFAULT_INFERENCE_THRESHOLD,
    DEFAULT_DENSITY_K,
    DEFAULT_COMPOSITE_WU,
    DEFAULT_COMPOSITE_WD,
    DEFAULT_COMPOSITE_WR,
)


def resolve_inference_params(
    threshold: float | None,
    density_k: int | None,
    wu: float | None,
    wd: float | None,
    wr: float | None,
) -> tuple[float, int, float, float, float]:
    return (
        threshold if threshold is not None else DEFAULT_INFERENCE_THRESHOLD,
        density_k if density_k is not None else DEFAULT_DENSITY_K,
        wu if wu is not None else DEFAULT_COMPOSITE_WU,
        wd if wd is not None else DEFAULT_COMPOSITE_WD,
        wr if wr is not None else DEFAULT_COMPOSITE_WR,
    )


def build_inference_rows(
    probs: torch.Tensor,
    preds: torch.Tensor,
    embeddings: torch.Tensor,
    snippet_ids: Sequence[int],
    labeled_snippet_ids: set[int],
    label_order: List[str],
    density_k: int,
    wu: float,
    wd: float,
    wr: float,
) -> list[ALInferenceRow]:
    """
    Compute prediction rows for all snippets and attach acquisition scores
    for unlabeled snippets.
    """

    unlabeled_indices = [i for i, sid in enumerate(snippet_ids) if sid not in labeled_snippet_ids]
    labeled_indices = [i for i, sid in enumerate(snippet_ids) if sid in labeled_snippet_ids]

    z_u = embeddings[unlabeled_indices] if unlabeled_indices else torch.empty(
        (0, embeddings.shape[1]), device=embeddings.device
    )
    z_l = embeddings[labeled_indices] if labeled_indices else torch.empty(
        (0, embeddings.shape[1]), device=embeddings.device
    )

    uncertainty_scores_u = uncertainty(probs[unlabeled_indices]) if unlabeled_indices else torch.empty(
        0, device=embeddings.device
    )

    diversity_scores_u = diversity(z_u, z_l)
    density_scores_u = density(z_u, k=density_k)
    composite_scores_u = composite(
        uncertainty_scores=uncertainty_scores_u,
        diversity_scores=diversity_scores_u,
        density_scores=density_scores_u,
        wu=wu,
        wd=wd,
        wr=wr,
    )

    uncertainty_full = [None] * len(snippet_ids)
    diversity_full = [None] * len(snippet_ids)
    density_full = [None] * len(snippet_ids)
    composite_full = [None] * len(snippet_ids)

    for pos, idx in enumerate(unlabeled_indices):
        uncertainty_full[idx] = float(uncertainty_scores_u[pos].item())
        diversity_full[idx] = float(diversity_scores_u[pos].item())
        density_full[idx] = float(density_scores_u[pos].item())
        composite_full[idx] = float(composite_scores_u[pos].item())

    rows: list[ALInferenceRow] = []

    for i, snippet_id in enumerate(snippet_ids):
        pred_indices = torch.where(preds[i] > 0)[0].tolist()
        pred_labels = [label_order[j] for j in pred_indices]
        prob_dict = {
            label_order[j]: float(probs[i, j].item())
            for j in range(len(label_order))
        }

        rows.append(
            ALInferenceRow(
                snippet_id=snippet_id,
                embedding=embeddings[i].detach().cpu().tolist(),
                predicted_labels=pred_labels,
                predicted_probabilities=prob_dict,
                uncertainty=uncertainty_full[i],
                diversity=diversity_full[i],
                density=density_full[i],
                composite_score=composite_full[i],
            )
        )

    return rows


def save_prediction_rows(
    db: Session,
    model_checkpoint_id: int,
    rows,
) -> None:
    for row in rows:
        existing = (
            db.query(ALPrediction)
            .filter(
                ALPrediction.model_checkpoint_id == model_checkpoint_id,
                ALPrediction.snippet_id == row.snippet_id,
            )
            .one_or_none()
        )

        if existing is None:
            existing = ALPrediction(
                model_checkpoint_id=model_checkpoint_id,
                snippet_id=row.snippet_id,
            )
            db.add(existing)

        existing.embedding = row.embedding
        existing.predicted_labels = row.predicted_labels
        existing.predicted_probabilities = row.predicted_probabilities
        existing.uncertainty = row.uncertainty
        existing.diversity = row.diversity
        existing.density = row.density
        existing.composite_score = row.composite_score


def run_and_store_inference(
    db: Session,
    dataset_id: int,
    model_ckpt,
    model,
    X,
    snippet_rows,
    label_order: list[str],
    labeled_snippet_ids: set[int],
    threshold: float | None = None,
    density_k: int | None = None,
    wu: float | None = None,
    wd: float | None = None,
    wr: float | None = None,
) -> dict:
    threshold, density_k, wu, wd, wr = resolve_inference_params(
        threshold=threshold, density_k=density_k, wu=wu, wd=wd, wr=wr,
    )

    device = next(model.parameters()).device
    x_tensor = torch.tensor(X, dtype=torch.float32, device=device)

    features = model.extract_features(x_tensor)
    probs, preds = model.predict(x_tensor, threshold=threshold)
    snippet_ids = [row["snippet_id"] for row in snippet_rows]

    rows = build_inference_rows(
        probs=probs,
        preds=preds,
        embeddings=features,
        snippet_ids=snippet_ids,
        labeled_snippet_ids=labeled_snippet_ids,
        label_order=label_order,
        density_k=density_k,
        wu=wu,
        wd=wd,
        wr=wr,
    )

    save_prediction_rows(db=db, model_checkpoint_id=model_ckpt.id, rows=rows)

    return {
        "num_predictions": len(rows),
        "num_labeled_snippets": len(labeled_snippet_ids),
        "threshold": threshold,
        "density_k": density_k,
        "composite_wu": wu,
        "composite_wd": wd,
        "composite_wr": wr,
    }


def get_predictions_for_checkpoint_and_snippet_set(
    db: Session,
    model_checkpoint_id: int,
    snippet_set_id: int,
) -> list[ALPrediction]:
    return (
        db.query(ALPrediction)
        .join(Snippet, Snippet.id == ALPrediction.snippet_id)
        .filter(
            ALPrediction.model_checkpoint_id == model_checkpoint_id,
            Snippet.snippet_set_id == snippet_set_id,
        )
        .order_by(ALPrediction.composite_score.desc().nullslast(), ALPrediction.id.asc())
        .all()
    )


def rank_prediction_suggestions(
    db: Session,
    dataset_id: int,
    snippet_set_id: int,
    predictions: list[ALPrediction],
    strategy: str,
    annotated_ids: set[int],
) -> list[ALPrediction]:
    candidates = [p for p in predictions if p.snippet_id not in annotated_ids]

    if strategy == "random":
        import random
        candidates = candidates[:]
        random.shuffle(candidates)
        return candidates

    key_map = {
        "uncertainty": lambda p: p.uncertainty if p.uncertainty is not None else float("-inf"),
        "diversity": lambda p: p.diversity if p.diversity is not None else float("-inf"),
        "density": lambda p: p.density if p.density is not None else float("-inf"),
        "composite": lambda p: p.composite_score if p.composite_score is not None else float("-inf"),
    }

    if strategy not in key_map:
        raise ValueError(f"Unsupported suggestion strategy '{strategy}'.")

    return sorted(candidates, key=key_map[strategy], reverse=True)
