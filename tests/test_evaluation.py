"""Tests for batchtextreview.evaluation."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from batchtextreview.evaluation import (
    INVALID,
    ClassMetrics,
    EvalReport,
    dataset_fingerprint,
    evaluate,
    slice_report,
)

ABC = ["a", "b", "c"]
SENT = ["positive", "negative", "neutral"]


def approx(value: float) -> object:
    return pytest.approx(value, abs=1e-12)


# --------------------------------------------------------------------------
# A hand-computed labels-mode example
# --------------------------------------------------------------------------
#
#   row  gold  pred     include   outcome
#   0    a     a                  correct
#   1    a     a                  correct
#   2    a     b                  wrong
#   3    b     b                  correct
#   4    b     "??"               wrong, INVALID
#   5    c     c                  correct
#   6    c     c                  correct
#   7    c     a                  wrong
#   8    NaN   a                  gold unscorable
#   9    b     ""       False     excluded
#   10   zzz   b                  gold unscorable (not in the label set)
#
# Scored: 8 rows, 5 correct.
#   a: support 3, predicted 3, tp 2 -> P 2/3, R 2/3, F1 2/3
#   b: support 2, predicted 2, tp 1 -> P 1/2, R 1/2, F1 1/2
#   c: support 3, predicted 2, tp 2 -> P 1,   R 2/3, F1 4/5
#   macro F1    = (2/3 + 1/2 + 4/5) / 3 = 59/90
#   weighted F1 = (3*2/3 + 2*1/2 + 3*4/5) / 8 = 5.4/8 = 0.675
#   kappa: po = 5/8; gold marginals a3 b2 c3 inv0, pred marginals a3 b2 c2 inv1
#          pe = (9 + 4 + 6 + 0) / 64 = 19/64
#          kappa = (5/8 - 19/64) / (1 - 19/64) = 21/45 = 7/15

HAND_PREDS = ["a", "a", "b", "b", "??", "c", "c", "a", "a", "", "b"]
HAND_GOLDS = ["a", "a", "a", "b", "b", "c", "c", "c", float("nan"), "b", "zzz"]
HAND_INCLUDE = [True] * 9 + [False, True]


@pytest.fixture
def hand_report() -> EvalReport:
    return evaluate(HAND_PREDS, HAND_GOLDS, allowed=ABC, include=HAND_INCLUDE)


def test_hand_example_counts(hand_report):
    r = hand_report
    assert r.mode == "labels"
    assert r.labels == ABC
    assert r.n_input == 11
    assert r.n_excluded == 1
    assert r.n_gold_unscorable == 2
    assert r.n_scored == 8
    assert r.n_correct == 5
    assert r.n_pred_invalid == 1
    assert r.n_scored + r.n_excluded + r.n_gold_unscorable == r.n_input


def test_hand_example_metrics(hand_report):
    r = hand_report
    assert r.accuracy == approx(5 / 8)
    assert r.macro_f1 == approx(59 / 90)
    assert r.weighted_f1 == approx(0.675)
    assert r.kappa == approx(7 / 15)


def test_hand_example_per_class(hand_report):
    by_label = {m.label: m for m in hand_report.per_class}
    assert [m.label for m in hand_report.per_class] == ABC
    a, b, c = by_label["a"], by_label["b"], by_label["c"]
    assert (a.precision, a.recall, a.f1) == (approx(2 / 3), approx(2 / 3), approx(2 / 3))
    assert (a.support, a.predicted) == (3, 3)
    assert (b.precision, b.recall, b.f1) == (approx(0.5), approx(0.5), approx(0.5))
    assert (b.support, b.predicted) == (2, 2)
    assert (c.precision, c.recall, c.f1) == (approx(1.0), approx(2 / 3), approx(0.8))
    assert (c.support, c.predicted) == (3, 2)


def test_hand_example_rows(hand_report):
    rows = hand_report.rows
    assert list(rows.columns) == ["row", "gold", "predicted", "correct"]
    assert rows["row"].tolist() == [0, 1, 2, 3, 4, 5, 6, 7]
    assert rows["gold"].tolist() == ["a", "a", "a", "b", "b", "c", "c", "c"]
    assert rows["predicted"].tolist() == ["a", "a", "b", "b", INVALID, "c", "c", "a"]
    assert rows["correct"].tolist() == [True, True, False, True, False, True, True, False]


def test_hand_example_confusion(hand_report):
    cm = hand_report.confusion()
    assert cm.index.tolist() == ["a", "b", "c"]
    assert cm.columns.tolist() == ["a", "b", "c", INVALID]
    assert cm.index.name == "gold"
    assert cm.columns.name == "predicted"
    assert cm.to_numpy().tolist() == [
        [2, 1, 0, 0],
        [0, 1, 0, 1],
        [1, 0, 2, 0],
    ]
    assert int(cm.to_numpy().sum()) == hand_report.n_scored


def test_hand_example_per_class_frame(hand_report):
    frame = hand_report.per_class_frame()
    assert list(frame.columns) == ["label", "precision", "recall", "f1", "support", "predicted"]
    assert frame["label"].tolist() == ABC
    assert frame["support"].tolist() == [3, 2, 3]
    assert frame["predicted"].tolist() == [3, 2, 2]
    assert frame["f1"].tolist() == [approx(2 / 3), approx(0.5), approx(0.8)]


def test_summary_is_flat_and_json_ready(hand_report):
    summary = hand_report.summary()
    assert summary == {
        "mode": "labels",
        "n_input": 11,
        "n_excluded": 1,
        "n_gold_unscorable": 2,
        "n_scored": 8,
        "n_correct": 5,
        "n_pred_invalid": 1,
        "accuracy": approx(5 / 8),
        "macro_f1": approx(59 / 90),
        "weighted_f1": approx(0.675),
        "kappa": approx(7 / 15),
    }
    json.dumps(summary)


# --------------------------------------------------------------------------
# Labels mode details
# --------------------------------------------------------------------------


def test_confusion_follows_allowed_order_and_keeps_zero_rows():
    allowed = ["positive", "negative", "neutral", "mixed"]
    r = evaluate(["Negative", "positive"], ["negative", "positive"], allowed=allowed)
    cm = r.confusion()
    assert cm.index.tolist() == allowed
    assert cm.columns.tolist() == allowed  # no INVALID column without invalid preds
    assert cm.loc["mixed"].sum() == 0
    assert cm.loc["negative", "negative"] == 1


def test_unused_and_never_gold_labels():
    # "c" never appears at all -> undefined everywhere, left out of macro F1.
    # "b" is predicted but never gold -> precision 0, recall undefined, F1 0.
    r = evaluate(["a", "b"], ["a", "a"], allowed=ABC)
    by_label = {m.label: m for m in r.per_class}
    assert by_label["c"] == ClassMetrics("c", None, None, None, 0, 0)
    assert by_label["b"] == ClassMetrics("b", 0.0, None, 0.0, 0, 1)
    a = by_label["a"]
    assert (a.precision, a.recall, a.f1) == (approx(1.0), approx(0.5), approx(2 / 3))
    assert r.macro_f1 == approx((2 / 3 + 0.0) / 2)
    assert r.weighted_f1 == approx(2 / 3)  # only "a" has support
    frame = r.per_class_frame()
    assert frame["precision"].isna().tolist() == [False, False, True]


def test_gold_never_predicted_has_zero_f1():
    r = evaluate(["a", "a"], ["a", "b"], allowed=["a", "b"])
    b = next(m for m in r.per_class if m.label == "b")
    assert (b.precision, b.recall, b.f1) == (None, 0.0, 0.0)


def test_invalid_predictions_count_as_wrong_and_reduce_recall():
    r = evaluate(["no idea", "positive or negative", "positive"], ["positive"] * 3, SENT)
    assert r.n_pred_invalid == 2
    assert r.n_correct == 1
    assert r.accuracy == approx(1 / 3)
    pos = next(m for m in r.per_class if m.label == "positive")
    assert (pos.precision, pos.recall) == (approx(1.0), approx(1 / 3))
    assert r.confusion().loc["positive", INVALID] == 2


def test_both_sides_are_normalised():
    preds = [
        "**Positive**",
        "Sentiment: negative",
        "<think>hm</think>Neutral",
        "The text is positive.",
    ]
    golds = ["Positive", "NEGATIVE ", "neutral.", "positive"]
    r = evaluate(preds, golds, allowed=["Positive", "Negative", "Neutral"])
    assert r.labels == SENT
    assert r.n_correct == 4
    assert r.accuracy == 1.0


def test_numeric_golds_read_as_floats():
    golds = [3.0, 1.0, np.float64(5.0), "2", float("nan")]
    preds = ["3", "1/5", "5 out of 5", "2.0", "4"]
    r = evaluate(preds, golds, allowed=["1", "2", "3", "4", "5"])
    assert r.n_scored == 4
    assert r.n_gold_unscorable == 1
    assert r.n_correct == 4


@pytest.mark.parametrize("missing", [None, float("nan"), pd.NA, "", "   "])
def test_missing_gold_is_unscorable_in_both_modes(missing):
    for allowed in (None, ["x", "y"]):
        r = evaluate(["x"], [missing], allowed=allowed)
        assert r.n_gold_unscorable == 1
        assert r.n_scored == 0
        assert r.accuracy is None
        assert r.kappa is None


def test_include_mask_excludes_before_anything_else():
    # Excluded rows are neither wrong answers nor unscorable golds.
    preds = ["", None, "a", "a"]
    golds = ["a", float("nan"), "a", "b"]
    r = evaluate(preds, golds, allowed=["a", "b"], include=[False, False, True, True])
    assert r.n_excluded == 2
    assert r.n_gold_unscorable == 0
    assert r.n_scored == 2
    assert r.n_pred_invalid == 0
    assert r.rows["row"].tolist() == [2, 3]


def test_include_accepts_series_and_arrays():
    preds = pd.Series(["a", "b", "a"], index=[7, 8, 9])
    golds = pd.Series(["a", "b", "b"], index=[1, 2, 3])
    for include in (pd.Series([True, False, True], index=[4, 5, 6]), np.array([1, 0, 1])):
        r = evaluate(preds, golds, allowed=["a", "b"], include=include)
        assert r.n_excluded == 1
        assert r.rows["row"].tolist() == [0, 2]
        assert r.n_correct == 1


def test_pandas_string_columns_with_missing_values():
    golds = pd.Series(["a", None, "b"], dtype="string")
    preds = pd.Series(["a", "b", None], dtype="string")
    r = evaluate(preds, golds, allowed=["a", "b"])
    assert r.n_gold_unscorable == 1
    assert r.n_scored == 2
    assert r.n_pred_invalid == 1
    assert r.n_correct == 1


def test_kappa_perfect_agreement():
    r = evaluate(["a", "b", "a", "b"], ["a", "b", "a", "b"], allowed=["a", "b"])
    assert r.kappa == approx(1.0)


def test_kappa_undefined_when_chance_agreement_is_total():
    r = evaluate(["a", "a", "a"], ["a", "a", "a"], allowed=["a", "b"])
    assert r.accuracy == 1.0
    assert r.kappa is None


def test_kappa_single_gold_class_with_varied_predictions():
    # po = 3/4, pe = 1 * 3/4 -> kappa 0, defined.
    r = evaluate(["a", "a", "a", "b"], ["a"] * 4, allowed=["a", "b"])
    assert r.kappa == approx(0.0)


def test_kappa_worse_than_chance_is_negative():
    r = evaluate(["b", "a"], ["a", "b"], allowed=["a", "b"])
    assert r.kappa == approx(-1.0)


def test_empty_label_set_falls_back_to_exact_mode():
    for allowed in ([], ["", "  "]):
        r = evaluate(["x"], ["X"], allowed=allowed)
        assert r.mode == "exact"
        assert r.accuracy == 1.0


# --------------------------------------------------------------------------
# Exact mode
# --------------------------------------------------------------------------


def test_exact_mode():
    preds = ["Paris", "london", None, "Rome.", "**Berlin**", "madrid"]
    golds = ["paris", "London", "Berlin", "rome", "berlin", float("nan")]
    r = evaluate(preds, golds)
    assert r.mode == "exact"
    assert r.n_scored == 5
    assert r.n_gold_unscorable == 1
    assert r.n_correct == 4
    assert r.accuracy == approx(0.8)
    assert r.macro_f1 is None
    assert r.weighted_f1 is None
    assert r.per_class == []
    assert r.per_class_frame().empty
    assert list(r.per_class_frame().columns) == [
        "label",
        "precision",
        "recall",
        "f1",
        "support",
        "predicted",
    ]
    # Most frequent gold first, ties in first-seen order.
    assert r.labels == ["berlin", "paris", "london", "rome"]
    # A missing prediction is simply wrong (shown as INVALID), but not
    # "invalid": there is no label set to fall outside.
    missing_pred = r.rows.set_index("row").loc[2]
    assert missing_pred["predicted"] == INVALID
    assert not missing_pred["correct"]
    assert r.n_pred_invalid == 0
    assert r.kappa is not None


def test_exact_mode_empty_outputs_are_wrong_not_invalid():
    r = evaluate(["", "<think>hmm", "English", "French"], ["English"] * 3 + ["French"])
    summary = r.summary()
    assert summary["mode"] == "exact"
    assert summary["n_pred_invalid"] == 0
    assert (summary["n_scored"], summary["n_correct"]) == (4, 2)
    assert r.rows["predicted"].tolist() == [INVALID, INVALID, "english", "french"]
    # Labels mode still counts them.
    assert evaluate(["", "English"], ["English"] * 2, allowed=["English"]).n_pred_invalid == 1


def test_exact_mode_confusion_buckets_beyond_top_15():
    golds: list[str] = []
    for i in range(20):
        golds += [f"v{i:02d}"] * (20 - i)  # v00 most frequent ... v19 least
    preds = list(golds)
    preds[0] = "something else"
    r = evaluate(preds, golds)
    cm = r.confusion()
    top = [f"v{i:02d}" for i in range(15)]
    assert cm.index.tolist() == [*top, "(other)"]
    assert cm.columns.tolist() == [*top, "(other)"]
    assert cm.loc["(other)", "(other)"] == sum(20 - i for i in range(15, 20))
    assert cm.loc["v00", "(other)"] == 1
    assert cm.loc["v00", "v00"] == 19
    assert int(cm.to_numpy().sum()) == len(golds)


def test_exact_mode_confusion_without_other():
    r = evaluate(["a", "b"], ["a", "a"])
    cm = r.confusion()
    assert cm.index.tolist() == ["a"]
    assert cm.columns.tolist() == ["a", "(other)"]


# --------------------------------------------------------------------------
# Input validation / empty inputs
# --------------------------------------------------------------------------


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="length"):
        evaluate(["a", "b"], ["a"])
    with pytest.raises(ValueError, match="include"):
        evaluate(["a", "b"], ["a", "b"], include=[True])


@pytest.mark.parametrize("allowed", [None, ABC])
def test_empty_inputs(allowed):
    r = evaluate([], [], allowed=allowed)
    assert r.n_input == r.n_scored == r.n_correct == 0
    assert r.accuracy is None
    assert r.macro_f1 is None
    assert r.weighted_f1 is None
    assert r.kappa is None
    assert list(r.rows.columns) == ["row", "gold", "predicted", "correct"]
    assert r.rows.empty
    cm = r.confusion()
    if allowed is None:
        assert cm.shape == (0, 0)
    else:
        assert cm.index.tolist() == ABC
        assert cm.columns.tolist() == ABC
        assert int(cm.to_numpy().sum()) == 0
    assert slice_report(r, []).empty


def test_everything_excluded():
    r = evaluate(["a", "b"], ["a", "b"], allowed=["a", "b"], include=[False, False])
    assert r.n_excluded == 2
    assert r.n_scored == 0
    assert r.accuracy is None
    assert r.macro_f1 is None
    assert all(m.f1 is None for m in r.per_class)


# --------------------------------------------------------------------------
# slice_report
# --------------------------------------------------------------------------


def test_slice_report_numeric_aware_order_and_missing():
    preds = ["a", "a", "b", "a", "a", "b", "a", "a"]
    golds = ["a", "a", "a", "a", "b", "b", "a", "a"]
    groups = ["2", "10", "10", None, "1", "2", float("nan"), "x"]
    include = [True] * 7 + [False]  # the "x" row isn't scored
    r = evaluate(preds, golds, allowed=["a", "b"], include=include)
    sliced = slice_report(r, groups)
    assert list(sliced.columns) == ["group", "n", "correct", "accuracy"]
    assert sliced["group"].tolist() == ["1", "2", "10", "(missing)"]
    assert sliced["n"].tolist() == [1, 2, 2, 2]
    assert sliced["correct"].tolist() == [0, 2, 1, 2]
    assert sliced["accuracy"].tolist() == [approx(0.0), approx(1.0), approx(0.5), approx(1.0)]


def test_slice_report_natural_text_order_and_float_groups():
    preds = ["a"] * 6
    golds = ["a"] * 6
    groups = ["tier 10", "Tier 2", "tier 1", 3.0, 1.5, "beta"]
    sliced = slice_report(evaluate(preds, golds, allowed=["a"]), groups)
    # numbers first (numerically), then text in natural, case-insensitive order
    assert sliced["group"].tolist() == ["1.5", "3", "beta", "tier 1", "Tier 2", "tier 10"]


def test_slice_report_keeps_large_integer_groups_exact():
    # Through float, these three ids merged into one "1234567890123456768".
    ids = pd.Series([1234567890123456789, 1234567890123456790, 1234567890123456791])
    r = evaluate(["a", "a", "b"], ["a", "b", "b"], allowed=["a", "b"])
    sliced = slice_report(r, ids)
    assert sliced["group"].tolist() == [
        "1234567890123456789",
        "1234567890123456790",
        "1234567890123456791",
    ]
    assert sliced["correct"].tolist() == [1, 0, 1]


def test_slice_report_large_and_negative_integer_groups():
    groups = [np.uint64(2**64 - 1), 10**400, -1234567890123456789, -1234567890123456790, 7.0]
    sliced = slice_report(evaluate(["a"] * 5, ["a"] * 5, allowed=["a"]), groups)
    # exact text (no OverflowError for 10**400) in exact numeric order
    assert sliced["group"].tolist() == [
        "-1234567890123456790",
        "-1234567890123456789",
        "7",
        "18446744073709551615",
        str(10**400),
    ]


def test_slice_report_uses_positions_not_index():
    groups = pd.Series(["g1", "g2"], index=[100, 200])
    r = evaluate(["a", "b"], ["a", "a"], allowed=["a", "b"])
    sliced = slice_report(r, groups)
    assert sliced.set_index("group")["correct"].to_dict() == {"g1": 1, "g2": 0}


def test_slice_report_length_mismatch():
    r = evaluate(["a"], ["a"], allowed=["a"])
    with pytest.raises(ValueError):
        slice_report(r, ["g1", "g2"])


# --------------------------------------------------------------------------
# dataset_fingerprint
# --------------------------------------------------------------------------


def test_fingerprint_shape_and_stability():
    texts = ["hello", "world", None]
    golds = ["a", "b", "c"]
    fp = dataset_fingerprint(texts, golds)
    assert len(fp) == 16
    assert all(ch in "0123456789abcdef" for ch in fp)
    assert dataset_fingerprint(texts, golds) == fp
    # same values in other containers -> same fingerprint
    assert dataset_fingerprint(tuple(texts), pd.Series(golds, index=[5, 6, 7])) == fp
    assert dataset_fingerprint(pd.Series(texts, dtype="string"), np.array(golds)) == fp


def test_fingerprint_is_pinned():
    # Stored in evaluation history: changing the scheme orphans old rows.
    assert dataset_fingerprint(["hello", "world", None], ["a", "b", "c"]) == (
        dataset_fingerprint(["hello", "world", float("nan")], ["a", "b", "c"])
    )
    assert dataset_fingerprint(["x"], ["y"]) == "060d512181ab49bc"


def test_fingerprint_missing_is_not_the_string_nan():
    assert dataset_fingerprint([float("nan")]) != dataset_fingerprint(["nan"])
    assert dataset_fingerprint([None]) != dataset_fingerprint(["None"])
    assert dataset_fingerprint([pd.NA]) != dataset_fingerprint(["<NA>"])
    # all spellings of missing are the same
    assert len({dataset_fingerprint([m]) for m in (None, float("nan"), pd.NA, np.nan)}) == 1


def test_fingerprint_is_sensitive_to_content_order_and_columns():
    base = dataset_fingerprint(["a", "b"], ["x", "y"])
    assert dataset_fingerprint(["b", "a"], ["y", "x"]) != base  # row order
    assert dataset_fingerprint(["x", "y"], ["a", "b"]) != base  # column order
    assert dataset_fingerprint(["a", "b"]) != base  # column count
    assert dataset_fingerprint(["a", "b"], ["x", "z"]) != base  # one value
    # value boundaries can't be faked by concatenation
    assert dataset_fingerprint(["ab", "c"]) != dataset_fingerprint(["a", "bc"])
    assert dataset_fingerprint(["a"], ["b"]) != dataset_fingerprint(["a", "b"])


def test_fingerprint_length_mismatch_and_no_columns():
    with pytest.raises(ValueError):
        dataset_fingerprint(["a", "b"], ["x"])
    assert len(dataset_fingerprint()) == 16
    assert dataset_fingerprint([]) != dataset_fingerprint()
