"""
Parity between the rewritten scoring path and the original ALQueryScorer.

Option C moved neighbour lookups onto a shared index over *all* snippets and
dropped the `embeddings[unlabeled_indices]` copy. None of that is supposed to
change a score. This test pins that by computing both ways on the same input.

ALQueryScorer is left in place as the reference implementation; if these
diverge, one of the two changed.
"""

import numpy as np
import pytest
import torch

from active_learning.samplers import ALQueryScorer, composite, zscore
from app.services.pam_al import _inference_helpers as inf_h


def _inputs(n=400, dim=16, n_labels=3, n_labeled=50, seed=1):
    rng = np.random.default_rng(seed)
    embeddings = torch.tensor(rng.normal(size=(n, dim)).astype(np.float32))
    probs = torch.tensor(rng.uniform(0.01, 0.99, size=(n, n_labels)).astype(np.float32))
    preds = (probs > 0.5).to(torch.float32)
    snippet_ids = list(range(1000, 1000 + n))
    labeled = set(rng.choice(snippet_ids, size=n_labeled, replace=False).tolist())
    return embeddings, probs, preds, snippet_ids, labeled


def _reference(embeddings, probs, snippet_ids, labeled, density_k, wu, wd, wr):
    """The pre-change computation, verbatim in shape."""
    unlabeled_idx = [i for i, sid in enumerate(snippet_ids) if sid not in labeled]
    labeled_idx = [i for i, sid in enumerate(snippet_ids) if sid in labeled]

    z_u = embeddings[unlabeled_idx]
    z_l = embeddings[labeled_idx]
    scorer = ALQueryScorer(z_u, z_l)

    unc = scorer.uncertainty(probs[unlabeled_idx])
    div = scorer.diversity()
    den = scorer.density(k=density_k)
    comp = composite(zscore(unc), zscore(div), zscore(den), wu=wu, wd=wd, wr=wr)
    return unlabeled_idx, unc.numpy(), div.numpy(), den.numpy(), comp.numpy()


@pytest.fixture
def wide_update_k(monkeypatch):
    """Every candidate visited, so the greedy selection is deterministic in both paths."""
    monkeypatch.setattr(inf_h, "DIVERSITY_UPDATE_K", 10_000)
    return 10_000


def test_scores_match_the_reference_implementation(wide_update_k):
    density_k, wu, wd, wr = 10, 0.5, 0.25, 0.25
    embeddings, probs, preds, snippet_ids, labeled = _inputs()

    rows = inf_h.build_inference_rows(
        probs=probs,
        preds=preds,
        embeddings=embeddings,
        snippet_ids=snippet_ids,
        labeled_snippet_ids=labeled,
        label_order=["a", "b", "c"],
        density_k=density_k,
        wu=wu,
        wd=wd,
        wr=wr,
    )

    unlabeled_idx, unc, div, den, comp = _reference(
        embeddings, probs, snippet_ids, labeled, density_k, wu, wd, wr
    )

    got_unc = np.array([rows[i].uncertainty for i in unlabeled_idx])
    got_div = np.array([rows[i].diversity for i in unlabeled_idx])
    got_den = np.array([rows[i].density for i in unlabeled_idx])
    got_comp = np.array([rows[i].composite_score for i in unlabeled_idx])

    assert np.allclose(got_unc, unc, rtol=1e-5, atol=1e-6)
    assert np.allclose(got_div, div, rtol=1e-3, atol=1e-4)
    assert np.allclose(got_den, den, rtol=1e-3, atol=1e-4)
    assert np.allclose(got_comp, comp, rtol=1e-3, atol=1e-3)


def test_labeled_rows_carry_no_acquisition_scores():
    embeddings, probs, preds, snippet_ids, labeled = _inputs(n=200, n_labeled=30, seed=4)

    rows = inf_h.build_inference_rows(
        probs=probs, preds=preds, embeddings=embeddings, snippet_ids=snippet_ids,
        labeled_snippet_ids=labeled, label_order=["a", "b", "c"],
        density_k=5, wu=0.5, wd=0.25, wr=0.25,
    )

    for row in rows:
        if row.snippet_id in labeled:
            assert row.uncertainty is None
            assert row.diversity is None
            assert row.density is None
            assert row.composite_score is None
        else:
            assert row.uncertainty is not None
            assert row.composite_score is not None


def test_predictions_are_unaffected():
    """Scoring changed; the label/probability payload must not have."""
    embeddings, probs, preds, snippet_ids, labeled = _inputs(n=120, n_labeled=20, seed=9)
    label_order = ["a", "b", "c"]

    rows = inf_h.build_inference_rows(
        probs=probs, preds=preds, embeddings=embeddings, snippet_ids=snippet_ids,
        labeled_snippet_ids=labeled, label_order=label_order,
        density_k=5, wu=0.5, wd=0.25, wr=0.25,
    )

    preds_np = preds.numpy()
    probs_np = probs.numpy()
    for i, row in enumerate(rows):
        expected = [label_order[j] for j in np.flatnonzero(preds_np[i] > 0)]
        assert row.predicted_labels == expected
        assert row.predicted_probabilities == pytest.approx(
            dict(zip(label_order, map(float, probs_np[i])))
        )


def test_no_labels_yet_still_scores():
    """Cold start: nothing labelled, so diversity falls back to a constant."""
    embeddings, probs, preds, snippet_ids, _ = _inputs(n=150, seed=12)

    rows = inf_h.build_inference_rows(
        probs=probs, preds=preds, embeddings=embeddings, snippet_ids=snippet_ids,
        labeled_snippet_ids=set(), label_order=["a", "b", "c"],
        density_k=5, wu=0.5, wd=0.25, wr=0.25,
    )

    assert all(row.composite_score is not None for row in rows)
    assert len({round(row.diversity, 6) for row in rows}) == 1
