"""Tests for the pure UI formatting helpers (batchtextreview.ui.format).

Edge cases (rounding boundaries, NaN-like values, NaT, accented names,
extension-less dataset names) are covered explicitly.
"""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from batchtextreview.ui import format as fmt

WHEN = datetime(2026, 10, 12, 14, 3, 27)


# ---- plural ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "expected"),
    [(0, "0 rows"), (1, "1 row"), (2, "2 rows"), (2000, "2,000 rows"), (1234567, "1,234,567 rows")],
)
def test_plural_counts_and_thousands_separator(n: int, expected: str) -> None:
    assert fmt.plural(n, "row") == expected


def test_plural_irregular_form() -> None:
    assert fmt.plural(1, "query", "queries") == "1 query"
    assert fmt.plural(3, "query", "queries") == "3 queries"


# ---- duration --------------------------------------------------------------


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0, "0 s"),
        (0.4, "under 1 s"),
        (1.2, "1 s"),
        (2.1, "2 s"),
        (9.94, "10 s"),
        (10, "10 s"),
        (42.4, "42 s"),
        (60, "1 min 00 s"),
        (75, "1 min 15 s"),
        (252, "4 min 12 s"),
        (3599, "59 min 59 s"),
        (3600, "1 h 00 min"),
        (3900, "1 h 05 min"),
        (90061, "25 h 01 min"),
    ],
)
def test_duration_formats(seconds: float, expected: str) -> None:
    assert fmt.duration(seconds) == expected


@pytest.mark.parametrize("missing", [None, float("nan"), np.float64("nan")])
def test_duration_missing_is_dash(missing: object) -> None:
    assert fmt.duration(missing) == "–"  # type: ignore[arg-type]


def test_duration_negative_clamps_to_zero() -> None:
    assert fmt.duration(-5) == "0 s"


def test_duration_rolls_minutes_into_hours() -> None:
    # 3599.6 rounds to 3600 s: must read as an hour, not "60 min 00 s".
    assert fmt.duration(3599.6) == "1 h 00 min"


def test_duration_rounding_at_minute_boundary() -> None:
    assert fmt.duration(59.6) == "1 min 00 s"


def test_duration_rounding_at_ten_seconds() -> None:
    assert fmt.duration(9.96) == "10 s"


def test_duration_infinity_does_not_raise() -> None:
    assert fmt.duration(float("inf")) == "–"


def test_duration_pd_na_is_dash() -> None:
    assert fmt.duration(pd.NA) == "–"  # type: ignore[arg-type]


def test_duration_numpy_float32_nan_is_dash() -> None:
    assert fmt.duration(np.float32("nan")) == "–"  # type: ignore[arg-type]


# ---- percent / number / rate ------------------------------------------------


def test_percent() -> None:
    assert fmt.percent(0.1234) == "12.3%"
    assert fmt.percent(1) == "100.0%"
    assert fmt.percent(0) == "0.0%"
    assert fmt.percent(0.5, digits=0) == "50%"
    assert fmt.percent(None) == "–"
    assert fmt.percent(float("nan")) == "–"


def test_percent_other_missing_values() -> None:
    assert fmt.percent(np.float32("nan")) == "–"  # type: ignore[arg-type]
    assert fmt.percent(pd.NA) == "–"  # type: ignore[arg-type]


def test_number() -> None:
    assert fmt.number(0.12345) == "0.12"
    assert fmt.number(0.5, digits=3) == "0.500"
    assert fmt.number(-0.25) == "-0.25"
    assert fmt.number(None) == "–"
    assert fmt.number(float("nan")) == "–"


def test_rate() -> None:
    assert fmt.rate(4.13) == "4.1 rows/s"
    assert fmt.rate(1) == "1.0 rows/s"
    assert fmt.rate(0.5) == "30.0 rows/min"
    assert fmt.rate(None) == "–"
    assert fmt.rate(0) == "–"


def test_rate_nan_is_dash() -> None:
    assert fmt.rate(float("nan")) == "–"


# ---- timestamp -------------------------------------------------------------


def test_timestamp_datetime_and_iso_string() -> None:
    # The tables' date format ("D MMM YYYY, HH:mm"): no zero-padded day.
    assert fmt.timestamp(WHEN) == "12 Oct 2026, 14:03"
    assert fmt.timestamp("2026-10-12T14:03:27") == "12 Oct 2026, 14:03"
    assert fmt.timestamp("2026-10-12 14:03:27") == "12 Oct 2026, 14:03"
    assert fmt.timestamp(pd.Timestamp("2026-01-02 09:05")) == "2 Jan 2026, 09:05"


def test_timestamp_none_and_unparsable() -> None:
    assert fmt.timestamp(None) == ""
    assert fmt.timestamp("yesterday") == "yesterday"
    assert fmt.timestamp("") == ""


def test_timestamp_nat_is_blank() -> None:
    assert fmt.timestamp(pd.NaT) == ""  # type: ignore[arg-type]


# ---- slug / export_filename ------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Qwen2.5-3B · Sentiment", "qwen2.5-3b-sentiment"),
        ("Yes/No flag (set {var})", "yes-no-flag-set-var"),
        ("  --Hello__World..  ", "hello-world"),
        ("", "results"),
        ("···", "results"),
        ("..", "results"),
        ("C:\\Users\\me\\data.csv", "c-users-me-data.csv"),
    ],
)
def test_slug(value: str, expected: str) -> None:
    assert fmt.slug(value) == expected


def test_slug_is_filesystem_safe_and_bounded() -> None:
    s = fmt.slug('a<b>c:d"e/f\\g|h?i*j' * 10, max_len=25)
    assert len(s) <= 25
    assert all(ch.isalnum() or ch in "-." for ch in s)
    assert not s.endswith(("-", "."))


def test_slug_keeps_accented_letters_readable() -> None:
    assert fmt.slug("Über Café") in {"uber-cafe", "über-café"}


def test_export_filename() -> None:
    name = fmt.export_filename("feedback.csv", "Sentiment", "Qwen2.5-3B-Instruct", "csv", WHEN)
    assert name == "feedback_sentiment_qwen2.5-3b-instruct_20261012-1403.csv"


def test_export_filename_ext_with_dot_and_multi_dot_dataset() -> None:
    name = fmt.export_filename(
        "my.data.v2.xlsx", "Yes/No question", "Demo (fake outputs)", ".xlsx", WHEN
    )
    assert name == "my.data.v2_yes-no-question_demo-fake-outputs_20261012-1403.xlsx"


def test_export_filename_parts_are_truncated() -> None:
    name = fmt.export_filename("x" * 100 + ".csv", "t" * 100, "m" * 100, "csv", WHEN)
    dataset, task, model, stamp = name.removesuffix(".csv").split("_")
    assert (len(dataset), len(task), len(model)) == (30, 24, 30)
    assert stamp == "20261012-1403"


def test_export_filename_defaults_to_now() -> None:
    name = fmt.export_filename("a.csv", "b", "c", "csv")
    stamp = name.removesuffix(".csv").rsplit("_", 1)[-1]
    parsed = datetime.strptime(stamp, "%Y%m%d-%H%M")
    assert abs((datetime.now() - parsed).total_seconds()) < 120


def test_export_filename_dataset_without_extension() -> None:
    name = fmt.export_filename("Qwen2.5 run", "t", "m", "csv", WHEN)
    assert name.startswith("qwen2.5-run_")


# ---- truncate --------------------------------------------------------------


def test_truncate_collapses_whitespace() -> None:
    assert fmt.truncate("a  b\n\tc ") == "a b c"


def test_truncate_long_text() -> None:
    out = fmt.truncate("word " * 50, limit=20)
    assert len(out) <= 20
    assert out.endswith("…")
    assert out == "word word word word…"


def test_truncate_exact_limit_untouched() -> None:
    assert fmt.truncate("x" * 80) == "x" * 80
    assert fmt.truncate("x" * 81).endswith("…")


def test_truncate_non_string() -> None:
    assert fmt.truncate(12345, limit=3) == "12…"  # type: ignore[arg-type]
    assert fmt.truncate(math.pi, limit=80) == str(math.pi)  # type: ignore[arg-type]


def test_truncate_zero_limit() -> None:
    assert len(fmt.truncate("abcdef", 0)) <= 1


# ---- shared copy helpers ------------------------------------------------------


def test_duration_under_a_second() -> None:
    assert fmt.duration(0.01) == "under 1 s"
    assert fmt.duration(0) == "0 s"
    assert fmt.duration(0.99) == "under 1 s"


@pytest.mark.parametrize(
    ("n_bytes", "expected"),
    [
        (0, "0 B"),
        (900, "900 B"),
        (69_632, "68.0 KB"),
        (5 * 1024**2, "5.0 MB"),
        (3 * 1024**3, "3.0 GB"),
        (2048 * 1024**3, "2048.0 GB"),
        (None, "–"),
        (float("nan"), "–"),
        (-5, "0 B"),
    ],
)
def test_file_size(n_bytes: float | None, expected: str) -> None:
    assert fmt.file_size(n_bytes) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("4bit", "4-bit"),
        ("fp16", "fp16"),
        ("bf16", "bf16"),
        ("fp32", "fp32"),
        ("auto", "Auto"),
        ("none", "n/a"),  # the demo model
        (" fp16 ", "fp16"),
        ("int8", "–"),
        (None, "–"),
        (float("nan"), "–"),
    ],
)
def test_precision(value: object, expected: str) -> None:
    assert fmt.precision(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("app", "App"),
        ("cli", "CLI"),
        ("legacy", "v1"),
        ("api", "Api"),
        (None, "App"),
        ("", "App"),
        (float("nan"), "App"),
        (pd.NA, "App"),
    ],
)
def test_source_word(value: object, expected: str) -> None:
    assert fmt.source_word(value) == expected


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("Qwen/Qwen2.5-3B-Instruct", "Qwen2.5-3B-Instruct"),
        ("C:\\models\\my-model\\", "my-model"),
        ("/opt/models/llama/", "llama"),
        ("plain", "plain"),
    ],
)
def test_model_name(model_id: str, expected: str) -> None:
    assert fmt.model_name(model_id) == expected


def test_md_escapes_markdown_and_collapses_whitespace() -> None:
    assert fmt.md("a_b *c*\n[x](y)") == "a\\_b \\*c\\* \\[x\\]\\(y\\)"
    assert fmt.md("1. item") == "1\\. item"  # not a list
    assert fmt.md(":material/x:") == "\\:material/x\\:"  # not an icon
    assert fmt.md(42) == "42"


def test_module_has_no_streamlit_import() -> None:
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(fmt))
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import | ast.ImportFrom)
        for alias in (
            node.names if isinstance(node, ast.Import) else [ast.alias(node.module or "")]
        )
    }
    assert "streamlit" not in imported


def test_estimate_never_says_about_under():
    assert fmt.estimate(0.4) == "under 1 s"
    assert fmt.estimate(0) == "under 1 s"
    assert fmt.estimate(170) == "about 2 min 50 s"
    assert fmt.estimate(None) == fmt.DASH


def test_distinct_model_names():
    names = ["tinymodel"] * 4 + ["Qwen2.5-3B-Instruct"]
    keys = [
        "/x/a/tinymodel",
        "/x/other/tinymodel",
        "/y/third/tinymodel",
        "/x/a/tinymodel",
        "Qwen/Qwen2.5-3B-Instruct",
    ]
    assert fmt.distinct_model_names(names, keys) == [
        "a/tinymodel",
        "other/tinymodel",
        "third/tinymodel",
        "a/tinymodel",
        "Qwen2.5-3B-Instruct",  # one model per name: unchanged
    ]
    # Windows paths, and parents that only differ further up.
    assert fmt.distinct_model_names(["m", "m"], ["C:\\a\\m", "D:\\b\\m"]) == ["a/m", "b/m"]
    assert fmt.distinct_model_names(["m", "m"], ["/a/x/m", "/b/x/m"]) == ["a/x/m", "b/x/m"]
    # A name that isn't the folder's own keeps it in front.
    assert fmt.distinct_model_names(["X", "X"], ["/a/foo", "/b/foo"]) == ["X (a/foo)", "X (b/foo)"]
    assert fmt.distinct_model_names([], []) == []
