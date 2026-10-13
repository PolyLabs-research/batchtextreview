"""Tests for batchtextreview.ui.analysis (pure; no Streamlit)."""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from batchtextreview.engine import RunConfig, RunResult, RunStats
from batchtextreview.labels import INVALID
from batchtextreview.ui import analysis as an
from batchtextreview.ui import state

SENTIMENT = ["positive", "negative", "mixed", "neutral"]


def make_result(outputs, statuses, errors=None, template="Classify: {text}", **stats) -> RunResult:
    config = RunConfig(
        model_id="demo://fake",
        model_label="Demo",
        precision="none",
        template=template,
        max_new_tokens=8,
    )
    n = len(outputs)
    counts = {
        "total": n,
        "generated": sum(s == "generated" for s in statuses),
        "cached": sum(s == "cached" for s in statuses),
        "blank": sum(s == "blank" for s in statuses),
        "errors": sum(s == "error" for s in statuses),
        "pending": sum(s == "pending" for s in statuses),
    }
    counts.update(stats)
    return RunResult(
        run_id=config.run_id,
        config=config,
        outputs=list(outputs),
        statuses=list(statuses),
        errors=list(errors) if errors is not None else [None] * n,
        stats=RunStats(**counts),
    )


def make_context(frame: pd.DataFrame, **overrides: Any) -> state.RunContext:
    values: dict[str, Any] = {
        "name": "Demo · Sentiment · 10:00",
        "dataset_name": "feedback.csv",
        "dataset_fingerprint": "fp-1",
        "frame": frame,
        "text_column": "TEXT",
        "preset_key": "Sentiment",
        "preset_title": "Sentiment",
        "template_raw": "Classify: {text}",
        "template_edited": False,
        "var": None,
        "labels": list(SENTIMENT),
        "output_kind": "label",
        "selection": "All 6 rows",
        "id_column": "ID",
        "gold_column": "GOLD",
        "created_at": datetime(2026, 10, 12, 10, 0, 0),
    }
    values.update(overrides)
    return state.RunContext(**values)


def make_job(result: RunResult | None, ctx: state.RunContext | None, job_id: str = "job-1"):
    context: dict[str, Any] = {} if ctx is None else {state.CONTEXT_KEY: ctx}
    return SimpleNamespace(id=job_id, result=result, request=SimpleNamespace(context=context))


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ID": ["1", "2", "3", "4", "5", "6"],
            "TEXT": ["great", "awful", "", "meh", "fine I guess", "boom"],
            "GOLD": ["positive", "negative", "neutral", "mixed", "neutral", "negative"],
            "TIER": ["1", "1", "2", "2", "3", "3"],
            "CONTESTED": ["0", "0", "0", "1", "1", "0"],
        }
    )


@pytest.fixture
def label_job(frame):
    result = make_result(
        ["Positive", "negative", "", "I am not sure", "neutral", ""],
        ["generated", "cached", "blank", "generated", "generated", "error"],
        errors=[None, None, None, None, None, "RuntimeError: boom"],
    )
    return make_job(result, make_context(frame))


class FakeStore:
    def __init__(self, fail: bool = False) -> None:
        self.records: list[dict[str, Any]] = []
        self.fail = fail

    def log_evaluation(self, record):
        if self.fail:
            raise OSError("disk full")
        self.records.append(dict(record))
        return len(self.records)


# ---- analyse ---------------------------------------------------------------


def test_analyse_needs_result_and_context(frame, label_job):
    assert an.analyse(make_job(None, make_context(frame))) is None
    assert an.analyse(make_job(label_job.result, None)) is None
    job = SimpleNamespace(
        id="x", result=label_job.result, request=SimpleNamespace(context={"run": "nope"})
    )
    assert an.analyse(job) is None


def test_analyse_label_task(label_job):
    built = an.analyse(label_job)
    assert built is not None
    assert built.job_id == "job-1"
    assert built.scorable == [True, True, False, True, True, False]
    cols = built.columns
    assert (cols.output, cols.label, cols.status, cols.correct) == (
        "OUTPUT",
        "LABEL",
        "STATUS",
        "CORRECT",
    )
    assert cols.error == "ERROR"
    table = built.table
    assert list(table["LABEL"]) == ["positive", "negative", "", INVALID, "neutral", ""]
    assert list(table["CORRECT"]) == [True, True, None, False, True, None]
    report = built.report
    assert report is not None and report.mode == "labels"
    assert (report.n_scored, report.n_correct, report.n_excluded, report.n_pred_invalid) == (
        4,
        3,
        2,
        1,
    )
    assert built.result.stats.errors
    # Original columns are kept, in order, ahead of the result columns.
    assert list(table.columns[:5]) == ["ID", "TEXT", "GOLD", "TIER", "CONTESTED"]


def test_label_counts_order_and_invalid(label_job):
    counts = an.analyse(label_job).label_counts()
    assert list(counts["label"]) == [*SENTIMENT, INVALID]
    assert list(counts["count"]) == [1, 1, 0, 1, 1]


def test_label_counts_without_invalid(frame):
    result = make_result(["positive"] * 6, ["generated"] * 6)
    counts = an.analyse(make_job(result, make_context(frame))).label_counts()
    assert list(counts["label"]) == SENTIMENT
    assert list(counts["count"]) == [6, 0, 0, 0]


def test_open_ended_not_scored_unless_exact_match(frame):
    result = make_result(["Positive", "Negative", "", "Mixed", "neutral", "x"], ["generated"] * 6)
    ctx = make_context(frame, labels=None, output_kind="text", preset_key="Language")
    built = an.analyse(make_job(result, ctx))
    assert built.report is None
    assert built.columns.correct is None and built.columns.label is None
    assert "CORRECT" not in built.table.columns
    assert built.label_counts().empty

    ctx_exact = make_context(frame, labels=None, output_kind="text", extra={"exact_match": True})
    exact = an.analyse(make_job(result, ctx_exact))
    assert exact.report is not None and exact.report.mode == "exact"
    assert list(exact.table["CORRECT"])[:2] == [True, True]


def test_no_gold_column_means_no_report(frame):
    result = make_result(["positive"] * 6, ["generated"] * 6)
    built = an.analyse(make_job(result, make_context(frame, gold_column=None)))
    assert built.report is None
    assert an.log_evaluation(FakeStore(), built) is None


def test_output_diagnostics_json(frame):
    outputs = ['{"people": ["Ann"]}', '```json\n{"people": []}\n```', "", "not json", "[1, 2]", ""]
    statuses = ["generated", "generated", "generated", "cached", "generated", "blank"]
    ctx = make_context(frame, labels=None, output_kind="json", gold_column=None)
    built = an.analyse(make_job(make_result(outputs, statuses), ctx))
    diag = built.output_diagnostics()
    assert diag["n_outputs"] == 5
    assert diag["empty_rate"] == pytest.approx(1 / 5)
    assert diag["json_valid_rate"] == pytest.approx(3 / 5)
    lengths = built.output_lengths()
    assert len(lengths) == 4
    assert diag["median_chars"] == pytest.approx(pd.Series(lengths).median())
    assert built.parsed_json(0) == {"people": ["Ann"]}
    assert built.parsed_json(3) is None


def test_output_diagnostics_text_task(frame):
    ctx = make_context(frame, labels=None, output_kind="text", gold_column=None)
    result = make_result(
        ["abc", "abcdef", "", "", "", ""],
        ["generated", "generated", "blank", "pending", "pending", "error"],
    )
    diag = an.analyse(make_job(result, ctx)).output_diagnostics()
    assert diag == {
        "n_outputs": 2,
        "empty_rate": 0.0,
        "mean_chars": 4.5,
        "median_chars": 4.5,
        "json_valid_rate": None,
    }
    nothing = an.analyse(make_job(make_result([""] * 6, ["pending"] * 6), ctx)).output_diagnostics()
    assert (
        nothing["n_outputs"] == 0
        and nothing["empty_rate"] is None
        and nothing["mean_chars"] is None
    )


def test_result_column_collisions(frame):
    # Re-running on an exported file: the result columns already exist.
    exported = frame.assign(OUTPUT="old", LABEL="old", STATUS="old", CORRECT="old")
    result = make_result(["positive"] * 6, ["generated"] * 6)
    built = an.analyse(make_job(result, make_context(exported)))
    cols = built.columns
    assert cols.output != "OUTPUT" and cols.output.startswith("OUTPUT")
    assert cols.label != "LABEL" and cols.label.startswith("LABEL")
    assert cols.correct != "CORRECT" and cols.correct.startswith("CORRECT")
    assert list(built.table["OUTPUT"]) == ["old"] * 6
    assert list(built.table[cols.output]) == ["positive"] * 6
    assert built.label_counts()["count"].sum() == 6


def test_get_analysis_is_memoised(label_job):
    first = an.get_analysis(label_job)
    assert an.get_analysis(label_job) is first
    assert an.get_analysis(make_job(None, first.context)) is None


# ---- history ---------------------------------------------------------------


def test_log_evaluation_record(label_job):
    store = FakeStore()
    built = an.analyse(label_job)
    assert an.log_evaluation(store, built) == 1
    record = store.records[0]
    assert record["preset"] == "Sentiment"
    assert record["dataset_fingerprint"] == "fp-1"
    assert record["gold_column"] == "GOLD"
    assert record["n_scored"] == 4
    assert record["selection"] == "All 6 rows"
    assert record["template_edited"] is False
    assert record["var"] is None


def test_log_evaluation_records_var_only_when_used(frame):
    result = make_result(["yes"] * 6, ["generated"] * 6, template="Q: is it bad? {text}")
    ctx = make_context(
        frame.assign(GOLD=["yes", "no", "yes", "no", "yes", "no"]),
        labels=["yes", "no"],
        var="is it bad?",
        template_raw="Q: {var} {text}",
    )
    store = FakeStore()
    an.log_evaluation(store, an.analyse(make_job(result, ctx)))
    assert store.records[0]["var"] == "is it bad?"

    unused = make_context(frame, var="leftover", template_raw="Classify: {text}")
    store = FakeStore()
    an.log_evaluation(
        store, an.analyse(make_job(make_result(["positive"] * 6, ["generated"] * 6), unused))
    )
    assert store.records[0]["var"] is None


def test_log_evaluation_nothing_scored(frame):
    result = make_result([""] * 6, ["error"] * 6)
    built = an.analyse(make_job(result, make_context(frame)))
    assert built.report is not None and built.report.n_scored == 0
    assert an.log_evaluation(FakeStore(), built) is None


def test_finish_hook_logs_and_caches(label_job):
    store = FakeStore()
    an.finish_hook(store)(label_job)
    assert len(store.records) == 1
    assert isinstance(label_job.request.context.get("_analysis"), an.JobAnalysis)
    assert an.FINISH_ERROR_KEY not in label_job.request.context


def test_finish_hook_never_raises(label_job, frame):
    an.finish_hook(FakeStore(fail=True))(label_job)
    assert "OSError: disk full" in label_job.request.context[an.FINISH_ERROR_KEY]
    # Jobs without a context / result are ignored quietly.
    an.finish_hook(FakeStore())(make_job(None, None))
    broken = SimpleNamespace(
        id="b",
        result=object(),
        request=SimpleNamespace(context={state.CONTEXT_KEY: make_context(frame)}),
    )
    an.finish_hook(FakeStore())(broken)
    assert an.FINISH_ERROR_KEY in broken.request.context


def test_log_evaluation_real_store_is_idempotent(label_job, tmp_path):
    from batchtextreview.store import Store

    store = Store(tmp_path / "btr.db")
    built = an.analyse(label_job)
    first = an.log_evaluation(store, built)
    second = an.log_evaluation(store, built)
    assert first == second
    history = store.list_evaluations()
    assert len(history) == 1
    assert history.iloc[0]["n_scored"] == 4


def _history_row(**values: Any) -> dict[str, Any]:
    row = {
        "id": 1,
        "run_id": "other-run",
        "dataset_fingerprint": "fp-1",
        "gold_column": "GOLD",
        "preset": "Sentiment",
        "labels": list(SENTIMENT),
        "accuracy": 0.5,
        "macro_f1": 0.4,
        "kappa": 0.3,
        "n_scored": 4,
        "n_invalid": 2,
        "n_rows": 6,
        "created_at": "2026-10-12T09:00:00",
    }
    row.update(values)
    return row


def test_previous_evaluation(label_job):
    built = an.analyse(label_job)
    history = pd.DataFrame(
        [
            _history_row(id=5, run_id=built.result.run_id, created_at="2026-10-12T10:00:00"),
            _history_row(id=4, created_at="2026-10-12T11:00:00", accuracy=0.9),  # later
            _history_row(id=3, dataset_fingerprint="fp-2"),
            _history_row(id=2, labels=["yes", "no"]),
            _history_row(id=1, accuracy=0.25, created_at="2026-10-12T08:00:00"),
            _history_row(id=0, accuracy=0.5, created_at="2026-10-12T09:30:00"),
        ]
    )
    previous = an.previous_evaluation(history, built)
    assert previous is not None and previous["id"] == 0
    assert an.previous_evaluation(history.iloc[:4], built) is None
    assert an.previous_evaluation(pd.DataFrame(), built) is None


def test_kpis(label_job):
    built = an.analyse(label_job)
    k = an.kpis(built.report, built.context.n_rows)
    assert k.accuracy == pytest.approx(3 / 4)
    assert k.valid_rate == pytest.approx(3 / 4)
    assert k.coverage == pytest.approx(4 / 6)
    h = an.history_kpis(_history_row())
    assert h.valid_rate == pytest.approx(0.5)
    assert h.coverage == pytest.approx(4 / 6)
    assert an.history_kpis({"accuracy": float("nan"), "labels": None}).accuracy is None


def test_exclusion_note(label_job):
    report = an.analyse(label_job).report
    statuses = label_job.result.statuses
    # Only the reasons that apply, counted.
    assert an.exclusion_note(report, statuses) == "left out: 1 blank text, 1 failed"
    assert an.exclusion_note(report) == "2 rows left out (no output to score)"
    pending = ["generated", "pending", "pending", "error", "generated", "generated"]
    assert an.exclusion_note(report, pending) == "left out: 1 failed, 2 not generated"
    report.n_gold_unscorable = 1
    assert an.exclusion_note(report).endswith(
        "1 row has a missing gold label or one outside the label set"
    )
    report.n_excluded = 0
    report.n_gold_unscorable = 0
    assert an.exclusion_note(report) == ""


# ---- table views -----------------------------------------------------------


def test_row_filters(label_job):
    built = an.analyse(label_job)
    filters = an.row_filters(built)
    assert list(filters) == ["All", "Wrong", "Correct", "Invalid", "Errors", "Blank"]
    counts = {k: int(v.sum()) for k, v in filters.items()}
    assert counts == {"All": 6, "Wrong": 1, "Correct": 3, "Invalid": 1, "Errors": 1, "Blank": 1}
    assert list(filters["Wrong"]) == [False, False, False, True, False, False]


def test_row_filters_pending_and_json(frame):
    ctx = make_context(frame, labels=None, output_kind="json", gold_column=None)
    result = make_result(
        ['{"a": 1}', "nope", "", "", "", ""],
        ["generated", "generated", "blank", "pending", "pending", "pending"],
    )
    filters = an.row_filters(an.analyse(make_job(result, ctx)))
    assert list(filters) == ["All", "Invalid JSON", "Blank", "Not generated"]
    assert int(filters["Not generated"].sum()) == 3
    assert list(filters["Invalid JSON"]) == [False, True, False, False, False, False]


@pytest.mark.parametrize("status", ["error", "pending", "blank"])
def test_row_filters_json_run_with_no_output(frame, status):
    """A JSON run where nothing could be checked (every row failed, the run
    was cancelled before its first batch, or every text is blank) - the
    Invalid JSON mask used to raise on pandas 3."""
    ctx = make_context(frame, labels=None, output_kind="json", gold_column=None)
    result = make_result([""] * 6, [status] * 6)
    built = an.analyse(make_job(result, ctx))
    filters = an.row_filters(built)
    assert "Invalid JSON" not in filters
    assert int(filters["All"].sum()) == 6
    assert built.output_diagnostics()["n_outputs"] == 0


def test_output_matches_label(frame, label_job):
    built = an.analyse(label_job)
    # "I am not sure" resolved to INVALID: the raw output says more than the label.
    assert not built.output_matches_label()
    clean = make_result(
        ["Positive.", "negative", "", "**mixed**", "neutral", ""],
        ["generated", "cached", "blank", "generated", "generated", "error"],
    )
    assert an.analyse(make_job(clean, make_context(frame))).output_matches_label()
    open_ended = make_context(frame, labels=None, output_kind="text")
    assert not an.analyse(make_job(clean, open_ended)).output_matches_label()


def test_gold_counts_and_majority_baseline(label_job):
    built = an.analyse(label_job)
    counts = built.gold_counts()
    assert counts == {"positive": 1, "negative": 1, "mixed": 1, "neutral": 1}
    label, share = an.majority_baseline(built.report)
    assert label in counts and share == pytest.approx(0.25)


def test_kappa_band():
    assert an.kappa_band(None) is None
    assert an.kappa_band(-0.1) == "worse than chance"
    assert an.kappa_band(0.1) == "slight agreement"
    assert an.kappa_band(0.33) == "fair agreement"
    assert an.kappa_band(0.5) == "moderate agreement"
    assert an.kappa_band(0.7) == "substantial agreement"
    assert an.kappa_band(0.95) == "almost perfect agreement"


def test_search_mask(label_job):
    table = an.analyse(label_job).table
    hits = an.search_mask(table, "  AWF ", ["TEXT", "OUTPUT"])
    assert list(hits) == [False, True, False, False, False, False]
    assert an.search_mask(table, "", ["TEXT"]).all()
    assert list(an.search_mask(table, "not sure", ["TEXT", "OUTPUT", "missing"])) == [
        False,
        False,
        False,
        True,
        False,
        False,
    ]


def test_top_confusions(label_job):
    built = an.analyse(label_job)
    top = an.top_confusions(built.report)
    assert list(top.columns) == ["gold", "predicted", "count", "share"]
    assert top.iloc[0].to_dict() == {
        "gold": "mixed",
        "predicted": INVALID,
        "count": 1,
        "share": 1.0,
    }


def test_per_class_table_weakest_first(label_job):
    table = an.per_class_table(an.analyse(label_job).report)
    assert table.iloc[0]["label"] == "mixed"  # F1 0
    assert table["f1"].iloc[:-1].is_monotonic_increasing or table["f1"].isna().iloc[-1]


def test_slices_and_subset(label_job):
    built = an.analyse(label_job)
    slices = an.slice_table(built, "TIER")
    assert list(slices["group"]) == ["1", "2", "3"]
    assert list(slices["n"]) == [2, 1, 1]
    assert an.slice_table(built, "missing").empty

    keep = (built.context.frame["CONTESTED"] != "1").tolist()
    subset = an.subset_report(built, keep)
    assert subset.n_scored == 2 and subset.accuracy == 1.0
    assert an.binary_values(built.context.frame["CONTESTED"]) == ["0", "1"]
    assert an.binary_values(built.context.frame["TIER"]) is None


def test_group_candidates(label_job):
    built = an.analyse(label_job)
    found = an.group_candidates(built)
    assert "TEXT" not in found and "GOLD" not in found and "ID" not in found
    assert found[0] == "TIER"
    assert "CONTESTED" in found


def test_statuses_aligned(label_job):
    built = an.analyse(label_job)
    assert list(built.statuses) == [
        "generated",
        "cached",
        "blank",
        "generated",
        "generated",
        "error",
    ]
    assert built.columns.added[:4] == ["OUTPUT", "LABEL", "CORRECT", "STATUS"]


def test_previous_evaluation_ignores_later_runs(label_job):
    built = an.analyse(label_job)
    later = (built.context.created_at + timedelta(hours=1)).isoformat()
    assert an.previous_evaluation(pd.DataFrame([_history_row(created_at=later)]), built) is None


# ---- partial runs, memoised candidates ---------------------------------------


@pytest.mark.parametrize(
    ("change", "partial"),
    [({}, False), ({"cancelled": True}, True), ({"aborted_reason": "boom"}, True)],
)
def test_partial_runs_are_not_logged(label_job, change, partial):
    for key, value in change.items():
        setattr(label_job.result, key, value)
    built = an.analyse(label_job)
    assert an.is_partial(built.result) is partial
    assert built.partial is partial
    store = FakeStore()
    an.finish_hook(store)(label_job)
    assert len(store.records) == (0 if partial else 1)
    assert (an.log_evaluation(store, built) is None) is partial


def test_group_candidates_memoised(label_job, monkeypatch):
    from batchtextreview import data

    built = an.analyse(label_job)
    calls = []
    real = data.guess_group_columns

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(data, "guess_group_columns", counting)
    first = an.group_candidates(built)
    first.append("mutated")  # callers get a copy
    assert an.group_candidates(built)[0] == "TIER"
    assert "mutated" not in an.group_candidates(built)
    assert an.flag_columns(built) == {"CONTESTED": "1"}
    assert an.flag_columns(built) == {"CONTESTED": "1"}
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (["0", "1", "1"], "1"),
        (["False", "True"], "True"),
        (["no", "YES"], "YES"),
        (["1", "2"], None),
        (["a", "b", "c"], None),
    ],
)
def test_flag_value(values, expected):
    assert an.flag_value(pd.Series(values)) == expected


# ---- copy --------------------------------------------------------------------


def test_display_label():
    assert an.display_label(INVALID) == "Invalid"
    assert an.display_label(INVALID, ["valid", "Invalid"]) == "Not in label set"
    assert an.display_label("positive", ["positive"]) == "positive"
    assert an.display_label(None) == ""


def test_run_summary_uses_ui_terms():
    result = make_result(
        ["a", "b", "", "", ""],
        ["generated", "cached", "blank", "error", "pending"],
        duplicates=2,
        truncated_inputs=1,
        hit_token_limit=3,
    )
    result.cancelled = True
    result.aborted_reason = "RuntimeError: boom"
    text = an.run_summary(result)
    assert text.startswith(
        "2 of 5 rows have outputs · 1 generated · 1 reused from saved outputs · "
        "1 blank text skipped · 1 failed · 1 not generated (run cancelled)"
    )
    assert "2 duplicate texts share answers" in text
    assert "1 text cut to 8,000 characters" in text
    assert "3 outputs reached the 8-token limit" in text
    # No "cache", and the abort reason is left to the page's cards.
    assert "cache" not in text.replace("saved outputs", "")
    assert "boom" not in text
    assert an.run_summary(make_result([], [])) == "No rows"
    done = make_result(["a", "b"], ["generated", "generated"], duplicates=1)
    assert an.run_summary(done) == (
        "All 2 rows have outputs · 2 generated · 1 duplicate text shares an answer"
    )


# Real texts the engine and backends produce (engine._describe_error and
# transformers_backend's ModelLoadError messages, which end with their hint).
_TOKENIZER = (
    "Couldn't load Demo: The tokenizer of org/model has no pad, end-of-sequence or unknown "
    "token, so prompts can't be batched."
)
_NO_TOKENIZER_FILES = (
    "Couldn't load Demo: C:/models/x has no tokenizer files, so it looks like an incomplete "
    "copy of the model. Copy the whole model folder, including tokenizer.json / "
    "tokenizer_config.json (or tokenizer.model), next to config.json."
)
_4BIT_CPU = (
    "Couldn't load Demo: 4-bit quantisation needs an NVIDIA GPU (CUDA), but this machine is "
    "using the CPU. Choose fp32 precision instead."
)
_DEVICE = (
    "RuntimeError: Expected all tensors to be on the same device, but found at least two "
    "devices, cuda:0 and cpu! (stopped after 3 failed batches in a row)"
)


@pytest.mark.parametrize("reason", [_TOKENIZER, _NO_TOKENIZER_FILES, _4BIT_CPU])
def test_explain_failure_keeps_backend_messages(reason):
    failure = an.explain_failure(reason, "Demo")
    assert failure.loading and not failure.technical
    # The backend's own words (and hint) are shown; no contradicting advice.
    assert failure.cause == reason.removeprefix("Couldn't load Demo: ")
    assert failure.action is None


def test_explain_failure_raw_exceptions():
    device = an.explain_failure(_DEVICE)
    assert device.technical and not device.loading
    assert device.cause.startswith("Expected all tensors to be on the same device")
    assert "stopped after" not in device.cause
    assert device.action is None  # "cuda" alone is not a bitsandbytes problem

    oom = an.explain_failure(
        "OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB (GPU 0)"
    )
    assert oom.cause == "CUDA out of memory."
    assert "batch size" in oom.action

    load_oom = an.explain_failure(
        "Couldn't load Demo (fake outputs): RuntimeError: CUDA out of memory. Tried to allocate",
        "Demo (fake outputs)",
    )
    assert load_oom.loading and "smaller model" in load_oom.action

    gated = an.explain_failure(
        "Couldn't load X: OSError: 401 Client Error. Cannot access gated repo for url https://x"
    )
    assert "HF_TOKEN" in gated.action

    tls = an.explain_failure(
        "Couldn't load X: SSLError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed"
    )
    assert "proxy" in tls.action

    bnb = an.explain_failure("ImportError: bitsandbytes is not installed")
    assert "bitsandbytes" in bnb.action

    # A tokenizer error is not a licence problem.
    tok = an.explain_failure("ValueError: tokenizer has no chat template")
    assert tok.action is None
    assert tok.cause == "Tokenizer has no chat template."


def test_explain_failure_other_texts():
    assert an.explain_failure(None).cause == "No error message was recorded."
    store = an.explain_failure(
        "Couldn't save outputs to /x/results.db: OSError: disk full. Rows finished so far are "
        "kept; run again to resume."
    )
    assert not store.technical and not store.loading
    assert store.cause.startswith("Couldn't save outputs")
    bare = an.explain_failure("KeyboardInterrupt")
    assert bare.technical and bare.cause == "KeyboardInterrupt (no message)."


def test_explain_failure_cuda_build_and_missing_model():
    cuda = an.explain_failure(
        "Couldn't load X: AssertionError: Torch not compiled with CUDA enabled", "X"
    )
    assert cuda.loading and "CUDA build" in cuda.action
    missing = an.explain_failure(
        "Couldn't load X: OSError: org/typo is not a local folder and is not a valid model "
        "identifier listed on 'https://huggingface.co/models'",
        "X",
    )
    assert "model name" in missing.action


def test_describe_failure_outright_failure():
    problem = an.describe_failure(None, "RuntimeError: worker died", "Demo")
    assert problem.headline == "The run failed before producing any output."
    assert problem.cause == "Worker died."
    assert problem.hint  # always a next step
    assert problem.details == "RuntimeError: worker died"


def test_describe_failure_model_load():
    result = make_result([""] * 3, ["pending"] * 3)
    result.aborted_reason = "Couldn't load Demo: Not enough GPU memory to load it."
    problem = an.describe_failure(result, result.aborted_reason, "Demo")
    assert problem.headline == "The model couldn't be loaded, so no rows were generated."
    assert problem.cause == "Not enough GPU memory to load it."
    assert problem.hint is None  # the backend's message is its own advice
    assert problem.details == result.aborted_reason


def test_describe_failure_stopped_after_some_rows():
    result = make_result(
        ["a", "", "", ""], ["generated", "error", "error", "pending"], ["", "boom", "boom", ""]
    )
    result.aborted_reason = "RuntimeError: boom (stopped after 3 failed batches in a row)"
    problem = an.describe_failure(result, None, "Demo")
    assert problem.headline == "The run stopped early: 2 rows failed, 1 not generated."
    assert problem.cause == (
        "Boom. The run stopped after 3 failed batches in a row, so 1 row wasn't tried."
    )
    # An unrecognised error still gets a next step.
    assert problem.hint.startswith("Retry the failed rows.") and an.KEPT_NOTE in problem.hint


def test_describe_failure_nothing_produced():
    result = make_result(["", ""], ["error", "pending"], ["boom", ""])
    result.aborted_reason = "RuntimeError: boom"
    problem = an.describe_failure(result, None, "Demo")
    assert problem.headline == "The run stopped before generating any rows."
    assert an.KEPT_NOTE not in (problem.hint or "")


def test_previous_evaluation_same_second(label_job):
    """History stores whole seconds: a line from the second this run started
    (possibly logged after it) is not "previous"."""
    built = an.analyse(label_job)
    started = built.context.created_at.replace(microsecond=0)
    built.context.created_at = started.replace(microsecond=400_000)
    same = pd.DataFrame([_history_row(created_at=started.isoformat())])
    assert an.previous_evaluation(same, built) is None
    before = (started - timedelta(seconds=1)).isoformat()
    assert an.previous_evaluation(pd.DataFrame([_history_row(created_at=before)]), built)


# ---- final review: history written once, failure wording --------------------------


def _done_job(job, finished: datetime | None = None):
    """``job`` with the snapshot() a finished Job has."""
    when = finished or datetime(2026, 10, 12, 10, 5, 0)
    job.snapshot = lambda: SimpleNamespace(finished_at=when, done=True)
    return job


def test_finish_hook_records_the_history_line(label_job):
    an.finish_hook(FakeStore())(label_job)
    assert label_job.request.context[an.HISTORY_ID_KEY] == 1
    nothing = make_job(make_result([""] * 6, ["error"] * 6), label_job.request.context["run"])
    an.finish_hook(FakeStore())(nothing)
    assert nothing.request.context[an.HISTORY_ID_KEY] is None  # nothing scored


def test_viewing_a_logged_run_never_writes_history_again(label_job, tmp_path):
    """A line deleted on the History page must not come back (nor move in
    time) because someone opened the run's Evaluation tab."""
    from batchtextreview.store import Store

    store = Store(tmp_path / "btr.db")
    an.finish_hook(store)(label_job)
    entry = label_job.request.context[an.HISTORY_ID_KEY]
    stamp = store.list_evaluations().iloc[0]["created_at"]
    assert an.history_entry(store, _done_job(label_job)) == entry
    assert store.list_evaluations().iloc[0]["created_at"] == stamp
    store.delete_evaluation(entry)
    assert an.history_entry(store, label_job) == entry  # what the hook wrote...
    assert store.list_evaluations().empty  # ...but nothing is written back


def test_history_entry_retries_once_after_a_failed_hook(label_job, tmp_path):
    from batchtextreview.store import Store

    an.finish_hook(FakeStore(fail=True))(label_job)
    assert an.FINISH_ERROR_KEY in label_job.request.context
    store = Store(tmp_path / "btr.db")
    finished = datetime(2026, 10, 12, 10, 7, 30)
    entry = an.history_entry(store, _done_job(label_job, finished))
    assert entry is not None
    assert an.FINISH_ERROR_KEY not in label_job.request.context
    history = store.list_evaluations()
    assert len(history) == 1
    assert str(history.iloc[0]["created_at"]).startswith("2026-10-12T10:07:30")  # not "now"
    # Once only, for every session: a deleted line stays deleted.
    store.delete_evaluation(entry)
    an.history_entry(store, label_job)
    assert store.list_evaluations().empty


def test_history_entry_never_overwrites_a_retry(label_job, frame, tmp_path):
    """The original run of a retried configuration must not put its older
    numbers back over the retry's line."""
    from batchtextreview.store import Store

    an.finish_hook(FakeStore(fail=True))(label_job)
    ctx = label_job.request.context["run"]
    retry_ctx = state.RunContext(
        **{**ctx.__dict__, "created_at": ctx.created_at + timedelta(minutes=2)}
    )
    retry = make_job(label_job.result, retry_ctx, job_id="job-2")
    store = Store(tmp_path / "btr.db")
    assert an.history_entry(store, _done_job(label_job), [retry, label_job]) is None
    assert store.list_evaluations().empty


def test_explain_failure_keeps_the_backend_hint_of_a_long_message():
    """A download failure quotes a long raw exception before its advice: the
    raw part is shortened, never the advice about proxies."""
    detail = (
        "OSError: Can't load the configuration of 'ibm-granite/granite-4.0-350m'. If you were "
        "trying to load it from 'https://huggingface.co/models', make sure you don't have a "
        "local directory with the same name. Otherwise, make sure 'ibm-granite/granite-4.0-350m'"
        " is the correct path to a directory containing a config.json file (or not)"
    )[:299] + "…"
    hint = (
        "Check your internet connection and proxy settings (HTTPS_PROXY). Corporate TLS "
        "inspection is trusted through the operating-system certificate store (truststore) "
        "unless BTR_NO_TRUSTSTORE is set."
    )
    reason = (
        "Couldn't load Granite-4.0-350M: Couldn't download ibm-granite/granite-4.0-350m from "
        f"huggingface.co ({detail}). {hint}"
    )
    assert len(reason) > 550
    failure = an.explain_failure(reason, "Granite-4.0-350M")
    assert failure.loading and not failure.technical and not failure.settings
    assert len(failure.cause) <= 400
    assert failure.cause.startswith("Couldn't download ibm-granite/granite-4.0-350m from")
    assert failure.cause.endswith(hint)
    assert "(OSError: Can't load the configuration" in failure.cause
    problem = an.describe_failure(
        make_result([""] * 2, ["pending"] * 2), reason, "Granite-4.0-350M"
    )
    assert "HTTPS_PROXY" in problem.cause and problem.retry_helps
    assert problem.details == reason  # the whole text stays in the details


@pytest.mark.parametrize(
    "reason",
    [
        "Couldn't load tinymodel: fp16 on the CPU is unusably slow (no GPU was detected). "
        "Use fp32 precision on the CPU.",
        "Couldn't load tinymodel: 4-bit quantisation needs an NVIDIA GPU (CUDA), but this "
        "machine is using the CPU. Choose fp32 precision instead.",
        "Couldn't load results: /tmp/results is not a model folder (it has no config.json). "
        "Choose the folder that contains config.json.",
        "Couldn't load Qwen2.5-1.5B-Instruct: Qwen/Qwen2.5-1.5B-Instruct isn't in the local "
        "Hugging Face cache, and offline mode is on. Unset HF_HUB_OFFLINE to allow the download.",
    ],
)
def test_settings_failures_offer_no_retry(reason):
    label = reason.split(":", 1)[0].removeprefix("Couldn't load ")
    failure = an.explain_failure(reason, label)
    assert failure.settings
    problem = an.describe_failure(make_result([""] * 2, ["pending"] * 2), reason, label)
    assert not problem.retry_helps
    assert problem.hint is None or "retry" not in problem.hint.lower()
    outright = an.describe_failure(None, reason, label)
    assert not outright.retry_helps and outright.hint is None


def test_transient_failures_keep_the_retry():
    oom = "Couldn't load X: Not enough GPU memory to load X at fp16. Choose 4-bit precision."
    assert an.describe_failure(make_result([""], ["pending"]), oom, "X").retry_helps
    rows = an.describe_failure(None, "RuntimeError: worker died", "X")
    assert rows.retry_helps


def test_lowercase_technical_words_keep_their_case():
    failure = an.explain_failure(
        "Couldn't load tinymodel: fp16 on the CPU is unusably slow (no GPU was detected).",
        "tinymodel",
    )
    assert failure.cause.startswith("fp16 on the CPU")
    assert an.explain_failure("Couldn't load X: bitsandbytes is missing", "X").cause.startswith(
        "bitsandbytes"
    )
    assert an.explain_failure("something odd happened").cause == "Something odd happened."
