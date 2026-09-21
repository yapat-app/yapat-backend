"""
Parity tests for the explore query engine.

Each case pins a behaviour of the former client-side pipeline
(PredictionFeed / scoreVisibility / HistogramSlider / alSlice) so the server
returns exactly what the browser used to compute.
"""

import math

import numpy as np
import pytest

from app.services.explore.labels import labels_from_pairs
from app.services.explore.query import (
    MISSING_DAY,
    BaseData,
    Filters,
    ModelData,
    apply_filters,
    compute_default_confidence,
    compute_native_order,
    date_time_counts,
    histogram,
    order_rows,
    packbits_rows,
    resolve_anchor,
    score_histograms,
    unpack_labels,
)

NAN = math.nan
DAY = 19000  # 2022-01-08


def make_base() -> BaseData:
    return BaseData(
        version="b1",
        snippet_ids=np.array([10, 11, 12, 13, 14, 15], dtype=np.int64),
        recording_idx=np.array([0, 0, 0, 1, 1, 1], dtype=np.int32),
        duration=np.full(6, 3.0, dtype=np.float32),
        rec_ids=np.array([100, 101], dtype=np.int64),
        rec_location=np.array([0, -1], dtype=np.int32),
        rec_epoch_day=np.array([DAY, MISSING_DAY], dtype=np.int32),
        rec_month=np.array([1, 0], dtype=np.int8),
        rec_time=np.array([3600.0, NAN], dtype=np.float32),
        locations=["L1"],
    )


def make_model(base: BaseData) -> ModelData:
    label_order = ["a", "b", "c"]
    probs = np.array(
        [
            [0.9, 0.1, NAN],  # 10
            [0.2, 0.7, 0.1],  # 11
            [0.0, 0.0, 0.0],  # 12
            [0.4, 0.6, NAN],  # 13
            [0.5, NAN, NAN],  # 14
        ],
        dtype=np.float16,
    )
    preds = np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 0], [1, 1, 0], [0, 0, 0]], dtype=bool
    )
    scores_model = np.array(
        [
            [0.2, 0.5, 0.1, 1.0],
            [0.8, 0.4, 0.2, -0.5],
            [NAN, NAN, NAN, NAN],
            [0.55, 0.3, 0.3, 0.3],
            [NAN, 0.9, 0.9, 1.0],
        ],
        dtype=np.float32,
    )
    n = base.n
    has = np.array([True, True, True, True, True, False])
    model_row = np.array([0, 1, 2, 3, 4, -1], dtype=np.int64)
    scores = np.full((n, 4), NAN, dtype=np.float32)
    scores[:5] = scores_model
    default_conf = np.full(n, NAN, dtype=np.float32)
    default_conf[:5] = compute_default_confidence(probs)
    return ModelData(
        version="m1",
        has_prediction=has,
        model_row=model_row,
        scores=scores,
        probs=probs,
        pred_bits=packbits_rows(preds),
        label_order=label_order,
        default_confidence=default_conf,
        native_order=compute_native_order(has, scores[:, 3]),
    )


def make_labels(base: BaseData):
    return labels_from_pairs(
        base,
        np.array([12, 13, 13, 999], dtype=np.int64),
        ["x", "y", "x", "ignored-not-in-set"],
        version="l1",
    )


@pytest.fixture
def frame():
    base = make_base()
    return base, make_model(base), make_labels(base)


def visible_ids(base, result) -> list[int]:
    return base.snippet_ids[result.visible].tolist()


def ordered_ids(base, order) -> list[int]:
    return base.snippet_ids[order].tolist()


def test_native_order_is_composite_desc_nulls_last_then_id(frame):
    base, model, _ = frame
    assert ordered_ids(base, model.native_order) == [10, 14, 13, 11, 12]


def test_default_confidence_is_max_probability_or_undefined(frame):
    base, model, _ = frame
    conf = model.default_confidence
    assert conf[0] == pytest.approx(0.9, abs=1e-3)
    assert conf[1] == pytest.approx(0.7, abs=1e-3)
    assert math.isnan(conf[2])  # all probabilities 0 → undefined
    assert conf[4] == pytest.approx(0.5, abs=1e-3)


def test_population_excludes_rows_without_prediction(frame):
    base, model, labels = frame
    result = apply_filters(base, model, labels, Filters())
    assert visible_ids(base, result) == [10, 11, 12, 13, 14]


def test_no_model_population_is_every_snippet(frame):
    base, _, labels = frame
    result = apply_filters(base, None, labels, Filters())
    assert visible_ids(base, result) == [10, 11, 12, 13, 14, 15]
    assert ordered_ids(base, order_rows(base, None, result, [])) == [10, 11, 12, 13, 14, 15]


def test_predicted_species_narrows_and_rescopes_confidence(frame):
    base, model, labels = frame
    result = apply_filters(base, model, labels, Filters(predicted_species=("b", "c")))
    assert visible_ids(base, result) == [11, 13]
    conf = result.population.confidence
    assert conf[1] == pytest.approx(1 - (1 - 0.7) * (1 - 0.1), abs=1e-3)
    # c is absent (NaN) for 13, so only b counts.
    assert conf[3] == pytest.approx(0.6, abs=1e-3)


def test_unknown_predicted_species_matches_nothing(frame):
    base, model, labels = frame
    result = apply_filters(base, model, labels, Filters(predicted_species=("zzz",)))
    assert visible_ids(base, result) == []


def test_annotation_status_and_sticky_ids(frame):
    base, model, labels = frame
    annotated = apply_filters(base, model, labels, Filters(annotation_status="annotated"))
    assert visible_ids(base, annotated) == [12, 13]
    unannotated = apply_filters(base, model, labels, Filters(annotation_status="unannotated"))
    assert visible_ids(base, unannotated) == [10, 11, 14]
    sticky = apply_filters(
        base, model, labels, Filters(annotation_status="unannotated", sticky_ids=(12,))
    )
    assert visible_ids(base, sticky) == [10, 11, 12, 14]


def test_labels_are_sorted_and_first_label_is_alphabetical(frame):
    base, _, labels = frame
    assert labels.labels_for(3) == ["x", "y"]
    assert labels.vocab[labels.first_label[3]] == "x"
    assert labels.first_label[0] == -1
    # Labels on snippets outside the set never reach the species options.
    assert labels.vocab == ["x", "y"]


def test_annotated_species_and_label_scope(frame):
    base, model, labels = frame
    result = apply_filters(base, model, labels, Filters(annotated_species=("y",)))
    assert visible_ids(base, result) == [13]
    scoped = apply_filters(base, model, labels, Filters(label_scope=("a",)))
    assert visible_ids(base, scoped) == [10, 13]


def test_location_and_date_time_filters_require_metadata(frame):
    base, model, labels = frame
    loc = apply_filters(base, model, labels, Filters(locations=("L1",)))
    assert visible_ids(base, loc) == [10, 11, 12]
    date = apply_filters(base, model, labels, Filters(date_range=(DAY - 1, DAY + 0.5)))
    assert visible_ids(base, date) == [10, 11, 12]
    months = apply_filters(base, model, labels, Filters(months=(1,)))
    assert visible_ids(base, months) == [10, 11, 12]
    time_ = apply_filters(base, model, labels, Filters(time_range=(0, 100)))
    assert visible_ids(base, time_) == []


def test_score_ranges_use_population_domain_and_missing_scores_pass(frame):
    base, model, labels = frame
    # uncertainty domain over the population is [0.2, 0.8]; lo=0.5 → value ≥ 0.5.
    result = apply_filters(
        base, model, labels, Filters(score_ranges=(("uncertainty", 0.5, 1.0),))
    )
    assert visible_ids(base, result) == [11, 12, 13, 14]


def test_confidence_range_uses_declared_zero_one_domain(frame):
    base, model, labels = frame
    result = apply_filters(
        base, model, labels, Filters(score_ranges=(("confidence", 0.65, 1.0),))
    )
    # 12 has undefined confidence → passes (missing score).
    assert visible_ids(base, result) == [10, 11, 12]


def test_full_ranges_do_not_filter(frame):
    base, model, labels = frame
    result = apply_filters(
        base, model, labels, Filters(score_ranges=(("uncertainty", 0.0, 1.0),))
    )
    assert visible_ids(base, result) == [10, 11, 12, 13, 14]


def test_histogram_matches_compute_bins():
    assert histogram(np.array([0.0, 0.5, 1.0, 2.0, NAN]), 4, 0.0, 1.0) == [1, 0, 1, 2]
    # Degenerate domain: span falls back to 1.
    assert histogram(np.array([0.3, 0.3]), 4, 0.3, 0.3) == [2, 0, 0, 0]


def test_score_histograms_total_is_non_score_and_visible_is_filtered(frame):
    base, model, labels = frame
    result = apply_filters(
        base, model, labels, Filters(score_ranges=(("uncertainty", 0.5, 1.0),))
    )
    hist = score_histograms(model, result, 4, base.n)
    assert sum(hist["uncertainty"]["total"]) == 3  # 10, 11, 13 finite
    assert sum(hist["uncertainty"]["visible"]) == 2  # 11, 13


def test_sort_confidence_desc_treats_undefined_as_zero(frame):
    base, model, labels = frame
    result = apply_filters(base, model, labels, Filters())
    order = order_rows(base, model, result, [("confidence", "desc")])
    assert ordered_ids(base, order) == [10, 11, 13, 14, 12]


def test_sort_date_asc_puts_missing_first_and_keeps_native_ties(frame):
    base, model, labels = frame
    result = apply_filters(base, model, labels, Filters())
    order = order_rows(base, model, result, [("date", "asc")])
    assert ordered_ids(base, order) == [14, 13, 10, 11, 12]


def test_multi_key_sort(frame):
    base, model, labels = frame
    result = apply_filters(base, model, labels, Filters())
    order = order_rows(base, model, result, [("date", "desc"), ("uncertainty", "asc")])
    # date desc: dated rows (10, 11, 12) first; within, uncertainty asc with
    # missing (-inf) first: 12, 10(.2), 11(.8); then undated 14(nan), 13(.5).
    assert ordered_ids(base, order) == [12, 10, 11, 14, 13]


def test_resolve_anchor_prefers_next_unlabeled(frame):
    base, model, labels = frame
    result = apply_filters(base, model, labels, Filters())
    order = order_rows(base, model, result, [])
    anchor_row = int(base.index_of([13])[0])
    assert resolve_anchor(order, anchor_row, labels, prefer_unlabeled=True) == 3  # snippet 11
    assert resolve_anchor(order, anchor_row, labels, prefer_unlabeled=False) == 2
    assert resolve_anchor(order, -1, labels, prefer_unlabeled=False) is None


def test_pred_bits_roundtrip():
    labels = [f"s{i}" for i in range(11)]
    pred = np.zeros((1, 11), dtype=bool)
    pred[0, [0, 8, 10]] = True
    assert unpack_labels(packbits_rows(pred)[0], labels) == ["s0", "s8", "s10"]


def test_date_time_counts_are_recording_level(frame):
    base, _, _ = frame
    counts = date_time_counts(base)
    assert counts["has_date_time"] is True
    assert counts["date_domain"] == [DAY, DAY]
    assert counts["date_counts"] == [[DAY, 1]]
    assert counts["time_counts"] == [[60, 1]]
