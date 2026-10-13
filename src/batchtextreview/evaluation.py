"""Scoring outputs against a gold-label column.

Pure Python + pandas (no scikit-learn dependency).
"""

from __future__ import annotations

import hashlib
import numbers
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Literal

import pandas as pd

from batchtextreview.labels import INVALID, canonical_labels, is_missing, normalise_label

__all__ = [
    "INVALID",
    "ClassMetrics",
    "EvalReport",
    "dataset_fingerprint",
    "evaluate",
    "slice_report",
]

CONFUSION_TOP_N = 15
"""Exact mode: gold values shown individually in :meth:`EvalReport.confusion`."""
OTHER = "(other)"
"""Exact-mode confusion bucket for everything outside the top values."""
MISSING_GROUP = "(missing)"
"""Group name :func:`slice_report` uses for missing grouping values."""


@dataclass(frozen=True)
class ClassMetrics:
    label: str
    precision: float | None
    """None when the label was never predicted."""
    recall: float | None
    """None when the label never appears in gold."""
    f1: float | None
    support: int
    """Gold count."""
    predicted: int
    """Prediction count."""


@dataclass
class EvalReport:
    mode: Literal["labels", "exact"]
    """"labels" when scored against a closed label set, "exact" for
    exact-match on normalised strings."""
    labels: list[str]
    """Canonical labels, in the allowed-set order (labels mode) or sorted by
    gold frequency (exact mode)."""
    n_input: int
    """Rows offered (length of preds/golds)."""
    n_excluded: int
    """Rows skipped because ``include`` was False (blank text, generation
    error, not run)."""
    n_gold_unscorable: int
    """Gold value missing, or (labels mode) not resolvable to the label set."""
    n_scored: int
    n_correct: int
    n_pred_invalid: int
    """Scored rows whose prediction didn't resolve (labels mode only; 0 in
    exact mode, where a missing prediction is simply wrong)."""
    accuracy: float | None
    macro_f1: float | None
    """Mean F1 over labels with support > 0 or predicted > 0 (labels mode);
    None in exact mode."""
    weighted_f1: float | None
    kappa: float | None
    """Cohen's kappa between gold and predicted (INVALID counts as its own
    category); None if undefined (e.g. expected agreement == 1)."""
    per_class: list[ClassMetrics] = field(default_factory=list)
    rows: pd.DataFrame = field(default_factory=pd.DataFrame)
    """One row per *scored* input row: columns ``row`` (position in the
    inputs), ``gold``, ``predicted`` (INVALID when unresolved), ``correct``."""

    def confusion(self) -> pd.DataFrame:
        """Counts, gold labels as the index, predicted labels as columns.

        Labels mode: all ``labels`` on both axes in order (zeros included),
        plus an INVALID column when any prediction was invalid. Exact mode:
        the top 15 gold values, everything else bucketed as "(other)".
        """
        golds = [str(v) for v in self.rows["gold"]] if len(self.rows) else []
        preds = [str(v) for v in self.rows["predicted"]] if len(self.rows) else []
        if self.mode == "labels":
            index = list(self.labels)
            columns = list(self.labels) + ([INVALID] if INVALID in preds else [])
        else:
            top = list(self.labels[:CONFUSION_TOP_N])
            keep = set(top)
            golds = [g if g in keep else OTHER for g in golds]
            preds = [p if p in keep else OTHER for p in preds]
            index = top + ([OTHER] if OTHER in golds else [])
            columns = top + ([OTHER] if OTHER in preds else [])
        position = {label: i for i, label in enumerate(columns)}
        counts = [[0] * len(columns) for _ in index]
        row_of = {label: i for i, label in enumerate(index)}
        for gold, pred in zip(golds, preds, strict=True):
            counts[row_of[gold]][position[pred]] += 1
        frame = pd.DataFrame(
            counts,
            index=pd.Index(index, name="gold", dtype=object),
            columns=pd.Index(columns, name="predicted", dtype=object),
            dtype="int64",
        )
        return frame

    def per_class_frame(self) -> pd.DataFrame:
        """``per_class`` as a DataFrame (label, precision, recall, f1,
        support, predicted)."""
        columns = ["label", "precision", "recall", "f1", "support", "predicted"]
        if not self.per_class:
            return pd.DataFrame(
                {
                    "label": pd.Series([], dtype=object),
                    "precision": pd.Series([], dtype="float64"),
                    "recall": pd.Series([], dtype="float64"),
                    "f1": pd.Series([], dtype="float64"),
                    "support": pd.Series([], dtype="int64"),
                    "predicted": pd.Series([], dtype="int64"),
                }
            )
        records = [
            {
                "label": m.label,
                "precision": m.precision,
                "recall": m.recall,
                "f1": m.f1,
                "support": m.support,
                "predicted": m.predicted,
            }
            for m in self.per_class
        ]
        frame = pd.DataFrame.from_records(records, columns=columns)
        for name in ("precision", "recall", "f1"):
            frame[name] = frame[name].astype("float64")
        return frame

    def summary(self) -> dict[str, Any]:
        """Flat dict of the scalar fields (for history logging / JSON)."""
        return {
            "mode": self.mode,
            "n_input": self.n_input,
            "n_excluded": self.n_excluded,
            "n_gold_unscorable": self.n_gold_unscorable,
            "n_scored": self.n_scored,
            "n_correct": self.n_correct,
            "n_pred_invalid": self.n_pred_invalid,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "kappa": self.kappa,
        }


def evaluate(
    preds: Sequence[Any],
    golds: Sequence[Any],
    allowed: Sequence[str] | None = None,
    include: Sequence[bool] | None = None,
) -> EvalReport:
    """Score predictions against gold values, row by row.

    ``preds`` and ``golds`` must have equal length (ValueError otherwise);
    ``include`` (same length) marks rows eligible for scoring - pass False for
    blank texts, generation errors and rows that never ran, so they are
    counted in ``n_excluded`` instead of as wrong answers.

    Both sides go through :func:`~batchtextreview.labels.normalise_label`
    with ``allowed`` (canonicalised). A gold that is missing or doesn't
    resolve is unscorable; a prediction that doesn't resolve counts as wrong
    and is recorded as INVALID. Without ``allowed`` both sides use
    ``label_key`` and are compared exactly (missing gold -> unscorable,
    missing prediction -> wrong).
    """
    preds = list(preds)
    golds = list(golds)
    if len(preds) != len(golds):
        raise ValueError(f"preds and golds differ in length ({len(preds)} vs {len(golds)}).")
    if include is None:
        mask = [True] * len(preds)
    else:
        mask = [bool(flag) for flag in include]
        if len(mask) != len(preds):
            raise ValueError(f"include has {len(mask)} entries for {len(preds)} rows.")

    allowed = None if allowed is None else list(allowed)
    labels = canonical_labels(allowed)
    mode: Literal["labels", "exact"] = "labels" if labels else "exact"
    # The raw list, not ``labels``: normalise_label canonicalises it the same
    # way, so every resolved value is guaranteed to be one of ``labels``.
    resolve_with = allowed if mode == "labels" else None

    n_excluded = n_gold_unscorable = n_pred_invalid = 0
    rows: list[int] = []
    gold_out: list[str] = []
    pred_out: list[str] = []
    for i, (pred, gold, eligible) in enumerate(zip(preds, golds, mask, strict=True)):
        if not eligible:
            n_excluded += 1
            continue
        gold_key = normalise_label(gold, resolve_with)
        if gold_key is None:
            n_gold_unscorable += 1
            continue
        pred_key = normalise_label(pred, resolve_with)
        if pred_key is None:
            # Exact mode has no label set to fall outside: an empty output
            # is just a wrong answer, not an invalid one.
            if mode == "labels":
                n_pred_invalid += 1
            pred_key = INVALID
        rows.append(i)
        gold_out.append(gold_key)
        pred_out.append(pred_key)

    correct = [g == p for g, p in zip(gold_out, pred_out, strict=True)]
    n_scored = len(rows)
    n_correct = sum(correct)
    if mode == "exact":
        # Most frequent first; Counter keeps first-seen order for ties.
        labels = [label for label, _ in Counter(gold_out).most_common()]

    per_class: list[ClassMetrics] = []
    macro_f1 = weighted_f1 = None
    if mode == "labels":
        per_class = _per_class(labels, gold_out, pred_out)
        active = [m.f1 for m in per_class if m.f1 is not None]
        macro_f1 = sum(active) / len(active) if active else None
        total_support = sum(m.support for m in per_class)
        if total_support:
            weighted_f1 = sum((m.f1 or 0.0) * m.support for m in per_class) / total_support

    return EvalReport(
        mode=mode,
        labels=labels,
        n_input=len(preds),
        n_excluded=n_excluded,
        n_gold_unscorable=n_gold_unscorable,
        n_scored=n_scored,
        n_correct=n_correct,
        n_pred_invalid=n_pred_invalid,
        accuracy=n_correct / n_scored if n_scored else None,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        kappa=_cohen_kappa(gold_out, pred_out),
        per_class=per_class,
        rows=pd.DataFrame(
            {
                "row": pd.Series(rows, dtype="int64"),
                "gold": pd.Series(gold_out, dtype=object),
                "predicted": pd.Series(pred_out, dtype=object),
                "correct": pd.Series(correct, dtype=bool),
            }
        ),
    )


def slice_report(report: EvalReport, groups: Sequence[Any]) -> pd.DataFrame:
    """Accuracy per value of a grouping column.

    ``groups`` is aligned with the original inputs (same length as preds);
    only scored rows (``report.rows.row``) are used. Missing group values
    become "(missing)". Columns: ``group`` (str), ``n``, ``correct``,
    ``accuracy``; sorted by group (numeric-aware: "2" before "10").
    """
    groups = list(groups)
    if len(groups) != report.n_input:
        raise ValueError(f"groups has {len(groups)} entries for {report.n_input} rows.")
    tally: dict[str, list[int]] = {}
    for row, correct in zip(report.rows["row"], report.rows["correct"], strict=True):
        counts = tally.setdefault(_group_name(groups[int(row)]), [0, 0])
        counts[0] += 1
        counts[1] += int(bool(correct))
    names = sorted(tally, key=_group_sort_key)
    return pd.DataFrame(
        {
            "group": pd.Series(names, dtype=object),
            "n": pd.Series([tally[g][0] for g in names], dtype="int64"),
            "correct": pd.Series([tally[g][1] for g in names], dtype="int64"),
            "accuracy": pd.Series([tally[g][1] / tally[g][0] for g in names], dtype="float64"),
        }
    )


def dataset_fingerprint(*columns: Sequence[Any]) -> str:
    """Short (16 hex) sha256 over the given aligned columns' string values,
    used to recognise "the same evaluation data" in history. Missing values
    hash as a sentinel distinct from the string "nan"."""
    materialised = [list(column) for column in columns]
    lengths = {len(column) for column in materialised}
    if len(lengths) > 1:
        raise ValueError(f"columns differ in length: {sorted(lengths)}")
    digest = hashlib.sha256()
    digest.update(
        f"btr-fingerprint-v1|{len(materialised)}|{lengths.pop() if lengths else 0}\n".encode()
    )
    for column in materialised:
        digest.update(b"\x1d")  # column separator
        for value in column:
            if is_missing(value):
                digest.update(b"\x00")
            else:
                # Length-prefixed, so no value can imitate a separator.
                data = str(value).encode("utf-8", "surrogatepass")
                digest.update(b"s%d:" % len(data))
                digest.update(data)
    return digest.hexdigest()[:16]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

_NUMERIC = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
_DIGITS = re.compile(r"(\d+)")


def _per_class(
    labels: Sequence[str], golds: Sequence[str], preds: Sequence[str]
) -> list[ClassMetrics]:
    support = Counter(golds)
    predicted = Counter(preds)
    hits = Counter(g for g, p in zip(golds, preds, strict=True) if g == p)
    metrics: list[ClassMetrics] = []
    for label in labels:
        tp, n_gold, n_pred = hits[label], support[label], predicted[label]
        precision = tp / n_pred if n_pred else None
        recall = tp / n_gold if n_gold else None
        # F1 = 2TP / (2TP + FP + FN) = 2TP / (gold + predicted); 0 rather
        # than undefined when only one side is empty, like scikit-learn.
        f1 = 2 * tp / (n_gold + n_pred) if (n_gold + n_pred) else None
        metrics.append(ClassMetrics(label, precision, recall, f1, n_gold, n_pred))
    return metrics


def _cohen_kappa(golds: Sequence[str], preds: Sequence[str]) -> float | None:
    """Cohen's kappa over whatever categories occur (INVALID included).

    Worked in integers - kappa = (n*agree - S) / (n^2 - S) with
    S = sum_k gold_k * pred_k - so "undefined" (S == n^2) is exact.
    """
    n = len(golds)
    if n == 0:
        return None
    agree = sum(g == p for g, p in zip(golds, preds, strict=True))
    gold_counts = Counter(golds)
    pred_counts = Counter(preds)
    chance = sum(count * pred_counts[label] for label, count in gold_counts.items())
    if chance == n * n:
        return None
    return (n * agree - chance) / (n * n - chance)


def _group_name(value: Any) -> str:
    if is_missing(value):
        return MISSING_GROUP
    if isinstance(value, bool):
        return str(value)
    # Integers stay exact: going through float would merge ids above 2**53.
    if isinstance(value, numbers.Integral):
        return str(int(value))
    # A float column (1.0, 2.0, NaN) reads better as "1", "2".
    if isinstance(value, numbers.Real):
        as_float = float(value)
        if as_float.is_integer():
            return str(int(as_float))
    return str(value).strip()


def _group_sort_key(name: str) -> tuple[Any, ...]:
    if name == MISSING_GROUP:
        return (2,)
    if _NUMERIC.fullmatch(name):
        # Decimal, not float, so large integer ids keep their exact order.
        return (0, Decimal(name), name)
    chunks = _DIGITS.split(name.casefold())
    natural = tuple(int(c) if i % 2 else c for i, c in enumerate(chunks))
    return (1, natural, name)
