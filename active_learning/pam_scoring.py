"""
PAM Combined Scoring / Ranking Module

Produces a composite ranking score for each unlabeled snippet using a 
custom combined strategy that can have multiple factors:
  - Currently: Uncertainty score (from the classifier)
  - Future: Additional factors can be added to the combined score

The final score determines which snippets are shown to the annotator first.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional

from .samplers import entropy

import torch


# ── Combined Score Computation ────────────────────────────────────────

def _compute_uncertainty(probs: np.ndarray) -> np.ndarray:
    """
    Compute uncertainty score via binary entropy.
    
    This is the primary component of the combined score. In the future,
    additional factors can be integrated into the scoring strategy.

    Args:
        probs: [N] predicted probabilities in [0, 1].

    Returns:
        [N] uncertainty scores (higher = more uncertain).
    """
    P = np.stack([1.0 - probs, probs], axis=1)  # [N, 2]
    P_t = torch.from_numpy(P.astype(np.float32))
    ent = entropy(P_t).numpy()
    return ent.astype(np.float64)


def _compute_combined_strategy(
    probs: np.ndarray,
    embeddings: Optional[np.ndarray] = None,
    labeled_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute the combined sampling strategy score.
    
    Currently based on uncertainty (entropy). Future implementations can
    incorporate additional factors like diversity, novelty, etc.
    
    Args:
        probs:          [N] classifier output probabilities.
        embeddings:     [N, D] optional embedding vectors (for future use).
        labeled_mask:   [N] bool mask – True for already-labeled samples (for future use).
    
    Returns:
        [N] combined strategy scores (higher = more informative).
    """
    # Current implementation: Use uncertainty as the primary score
    # Future: Can combine multiple factors here
    uncertainty_scores = _compute_uncertainty(probs)
    
    # Placeholder for future factor integration
    # e.g., diversity_scores = _compute_diversity(embeddings, labeled_mask)
    # e.g., combined = 0.7 * uncertainty + 0.3 * diversity
    
    return uncertainty_scores


# ── Public API ─────────────────────────────────────────────────────────

def combined_score(
    probs: np.ndarray,
    *,
    embeddings: Optional[np.ndarray] = None,
    labeled_mask: Optional[np.ndarray] = None,
    factor_weights: Optional[Dict[str, float]] = None,  # Kept for backward compatibility
) -> np.ndarray:
    """
    Compute a combined ranking score for every sample using a custom strategy.
    
    Currently uses uncertainty-based scoring. Future implementations can
    incorporate additional factors into the combined strategy.

    Args:
        probs:          [N] classifier output probabilities.
        embeddings:     [N, D] optional embedding vectors (reserved for future use).
        labeled_mask:   [N] bool mask – True for already-labeled samples (reserved for future use).
        factor_weights: Deprecated - kept for backward compatibility, not used.

    Returns:
        scores: [N] float64 — higher means *more informative* / should be
                shown to annotator sooner.
    """
    return _compute_combined_strategy(probs, embeddings, labeled_mask)


def select_top_k(
    scores: np.ndarray,
    k: int,
    exclude_mask: Optional[np.ndarray] = None,
) -> List[int]:
    """
    Select the top-k indices by descending score, optionally excluding
    already-labeled samples.

    Args:
        scores:       [N] ranking scores.
        k:            number of samples to select.
        exclude_mask: [N] bool — True means *skip* this index.

    Returns:
        List of selected indices (length ≤ k).
    """
    if exclude_mask is not None:
        scores = scores.copy()
        scores[exclude_mask] = -np.inf

    k_eff = min(k, int((scores > -np.inf).sum()))
    if k_eff == 0:
        return []

    top_idx = np.argsort(scores)[::-1][:k_eff]
    return top_idx.tolist()
