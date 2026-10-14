"""Tests for batchtextreview.cli (demo model, real Store in tmp_path)."""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from batchtextreview import cli, engine
from batchtextreview.backends import base as backend_base
from batchtextreview.backends.base import DeviceInfo, ModelLoadError
from batchtextreview.backends.fake import FakeBackend, demo_responder
from batchtextreview.config import DB_FILENAME
from batchtextreview.labels import INVALID
from batchtextreview.presets import make_custom_preset
from batchtextreview.store import EVALUATION_COLUMNS, SCHEMA_VERSION, Store

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE = REPO_ROOT / "examples" / "sample_feedback.csv"
DEMO = "Demo (fake outputs)"
BOM = b"\xef\xbb\xbf"

SMALL_CSV = (
    "INDEX,TEXT,GOLD_SENTIMENT,GOLD_INTENT\n"
    "1,The staff were wonderful and so helpful. Thank you!,positive,praise\n"
    "2,I was charged a late fee twice and I want a refund.,negative,complaint\n"
    "3,What time does the library open on Sunday?,neutral,query\n"
    "4,Please renew my library card.,neutral,request\n"
    "5,,neutral,other\n"
    "6,The app is great but the website keeps crashing.,mixed,complaint\n"
)


# --------------------------------------------------------------------------
# Fixtures and helpers
# --------------------------------------------------------------------------


@pytest.fixture
def data_file(tmp_path: Path) -> Path:
    """The bundled sample (copied, so outputs land in tmp_path), or a small
    stand-in when the examples folder isn't there."""
    target = tmp_path / "sample_feedback.csv"
    if SAMPLE.is_file():
        shutil.copyfile(SAMPLE, target)
    else:  # pragma: no cover - the repository ships the sample
        target.write_text(SMALL_CSV, encoding="utf-8")
    return target


@pytest.fixture
def small_file(tmp_path: Path) -> Path:
    target = tmp_path / "small.csv"
    target.write_text(SMALL_CSV, encoding="utf-8")
    return target


@pytest.fixture
def results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Results folder passed with --results-dir (and exported, so nothing can
    fall back to ./results in the repository)."""
    path = tmp_path / "results"
    monkeypatch.setenv("BTR_RESULTS_DIR", str(path))
    return path


@pytest.fixture(autouse=True)
def restore_package_logger(monkeypatch):
    """main() configures the "batchtextreview" logger (a file in the results
    folder, propagation off): undo that after every test, closing the file.
    BTR_VERBOSE is cleared so a developer's own setting can't leak in."""
    monkeypatch.delenv("BTR_VERBOSE", raising=False)
    logger = logging.getLogger("batchtextreview")
    handlers, level, propagate = list(logger.handlers), logger.level, logger.propagate
    yield
    for handler in logger.handlers:
        if handler not in handlers:
            logger.removeHandler(handler)
            handler.close()
    logger.setLevel(level)
    logger.propagate = propagate


def log_file(results: Path) -> Path:
    return results / "logs" / "batchtextreview.log"


def run_cli(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    code = cli.main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def demo_run(
    capsys: pytest.CaptureFixture[str], data: Path, results: Path, *extra: str
) -> tuple[int, str, str]:
    return run_cli(capsys, "run", str(data), "--model", DEMO, "--results-dir", str(results), *extra)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)


def history(results: Path) -> pd.DataFrame:
    return Store(results / DB_FILENAME).list_evaluations()


def n_rows(path: Path) -> int:
    return len(pd.read_csv(path, dtype=str, keep_default_na=False))


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------


def test_run_scores_writes_csv_and_logs_history(capsys, data_file, results):
    code, out, err = demo_run(
        capsys, data_file, results, "--preset", "Sentiment", "--gold", "GOLD_SENTIMENT"
    )
    assert code == 0, err
    output = data_file.with_name("sample_feedback_sentiment_demo-fake-outputs_results.csv")
    assert output.is_file()
    assert output.read_bytes().startswith(BOM)
    frame = read_csv(output)
    for column in ("OUTPUT", "LABEL", "STATUS", "CORRECT"):
        assert column in frame.columns
    assert len(frame) == n_rows(data_file)
    # Original columns come first, in their original order.
    original = list(pd.read_csv(data_file, nrows=0).columns)
    assert list(frame.columns[: len(original)]) == original
    assert set(frame["STATUS"]) <= {"generated", "blank"}
    allowed = {"positive", "negative", "mixed", "neutral", INVALID, ""}
    assert set(frame["LABEL"]) <= allowed

    assert "generated" in out
    for metric in ("Accuracy", "Macro-F1", "Kappa"):
        assert metric in out
    assert "Logged to history" in out
    assert "Traceback" not in err
    lines = history(results)
    assert len(lines) == 1
    line = lines.iloc[0]
    assert line["source"] == "cli"
    assert line["preset"] == "Sentiment"
    assert line["model_id"] == "demo://fake"
    assert line["gold_column"] == "GOLD_SENTIMENT"
    assert line["text_column"] == "TEXT"
    assert line["dataset"] == "sample_feedback.csv"
    assert list(line["labels"]) == ["positive", "negative", "mixed", "neutral"]
    assert 0.0 <= float(line["accuracy"]) <= 1.0
    # The run is registered as a CLI run.
    runs = Store(results / DB_FILENAME).list_runs()
    assert list(runs["source"]) == ["cli"]

    # Same command again: everything comes from the cache and the history
    # line is updated, not duplicated.
    code, out2, err2 = demo_run(
        capsys, data_file, results, "--preset", "Sentiment", "--gold", "GOLD_SENTIMENT"
    )
    assert code == 0, err2
    assert "from cache" in out2
    assert "generated" not in out2.split("\n", 1)[0]
    assert "already saved" in err2
    assert len(history(results)) == 1
    assert read_csv(output)["OUTPUT"].tolist() == frame["OUTPUT"].tolist()


def test_run_logs_provenance_to_history(capsys, small_file, results):
    code, _out, err = demo_run(
        capsys, small_file, results, "--preset", "Sentiment", "--gold", "GOLD_SENTIMENT",
        "--limit", "4",
    )  # fmt: skip
    assert code == 0, err
    line = history(results).iloc[0]
    assert line["selection"] == "First 4 of 6 rows"
    assert line["preset_title"] == "Sentiment"
    assert line["template_edited"] is False
    assert line["app_version"] == cli.__version__
    assert line["elapsed_s"] > 0
    assert line["rows_per_s"] > 0

    code, _out, err = demo_run(
        capsys, small_file, results, "--preset", "Sentiment", "--gold", "GOLD_SENTIMENT",
        "--template", "Classify the sentiment: positive, negative, mixed or neutral.\n\nTEXT:\n{text}",
    )  # fmt: skip
    assert code == 0, err
    line = history(results).iloc[0]
    assert line["selection"] == "All 6 rows"
    assert line["template_edited"] is True


def test_rerun_from_cache_keeps_the_history_figures(capsys, small_file, results):
    """The same scored command again comes entirely from the cache: the
    history line keeps the first run's speed and duration."""
    args = ("--preset", "Sentiment", "--gold", "GOLD_SENTIMENT")
    code, _out, err = demo_run(capsys, small_file, results, *args)
    assert code == 0, err
    first = history(results).iloc[0]
    assert first["rows_per_s"] > 0 and first["elapsed_s"] > 0

    code, out, err = demo_run(capsys, small_file, results, *args)
    assert code == 0, err
    assert "from cache" in out
    lines = history(results)
    assert len(lines) == 1
    again = lines.iloc[0]
    for key in ("rows_per_s", "elapsed_s", "selection", "preset_title", "template_edited"):
        assert again[key] == first[key], key


def test_run_token_limit_advice_names_the_option(capsys, small_file, results):
    code, out, err = demo_run(
        capsys, small_file, results, "--preset", "Sentiment", "--max-new-tokens", "1"
    )
    assert code == 0, err
    assert "reached the 1-token limit and may be cut off - raise --max-new-tokens." in out
    assert "Max output tokens" not in out


def test_run_reports_truncated_texts(capsys, small_file, results):
    code, out, err = demo_run(
        capsys, small_file, results, "--preset", "Sentiment", "--max-input-chars", "30"
    )
    assert code == 0, err
    assert "note: 4 texts longer than 30 characters will be cut before sending" in err
    assert "4 texts were longer than 30 characters and were cut before sending." in out
    frame = read_csv(small_file.with_name("small_sentiment_demo-fake-outputs_results.csv"))
    cut = "input truncated"
    assert frame["NOTE"].tolist() == [cut, cut, cut, "", "", cut]  # row 4 fits, row 5 is blank


def test_run_without_truncation_has_no_note(capsys, small_file, results):
    code, out, err = demo_run(capsys, small_file, results, "--preset", "Sentiment")
    assert code == 0, err
    assert "cut before sending" not in out and "will be cut" not in err
    assert (
        "NOTE"
        not in read_csv(
            small_file.with_name("small_sentiment_demo-fake-outputs_results.csv")
        ).columns
    )


def test_selection_text():
    assert cli._selection(100, 2000) == "First 100 of 2,000 rows"
    assert cli._selection(5, 5) == "All 5 rows"
    assert cli._selection(1, 1) == "All 1 row"


def test_run_writes_a_log_file(capsys, small_file, results):
    code, _out, err = demo_run(capsys, small_file, results, "--preset", "Sentiment")
    assert code == 0, err
    text = log_file(results).read_text(encoding="utf-8")
    assert "INFO batchtextreview.engine: Run " in text
    assert "started (cli): Demo (fake outputs)" in text
    assert " finished: 6 rows - 5 generated, 1 blank;" in text
    assert "DEBUG" not in text
    assert "batchtextreview.engine" not in err  # the log stays out of the terminal


def test_failed_rows_are_logged_but_not_printed_as_tracebacks(
    capsys, small_file, results, monkeypatch
):
    def flaky(model_id: str, precision: str) -> FakeBackend:
        return FakeBackend(fail_when=lambda batch: any("refund" in p for p in batch))

    monkeypatch.setattr(cli, "default_factory", flaky)
    code, _out, err = demo_run(capsys, small_file, results, "--preset", "Sentiment")
    assert code == 0, err
    assert "Traceback" not in err and "fake failure" not in err
    text = log_file(results).read_text(encoding="utf-8")
    assert "WARNING batchtextreview.engine: Run " in text
    assert "(5 prompts) failed: RuntimeError: fake failure" in text
    assert "Traceback" in text  # the run's first failure only
    assert "retried one row at a time: 4 succeeded, 1 failed" in text
    frame = read_csv(small_file.with_name("small_sentiment_demo-fake-outputs_results.csv"))
    assert frame["STATUS"].tolist().count("error") == 1  # isolated, not the whole batch

    # --no-cache: no log file, and still nothing on the terminal.
    code, _out, err = demo_run(capsys, small_file, results, "--preset", "Sentiment", "--no-cache")
    assert code == 0, err
    assert "Traceback" not in err and "fake failure" not in err


def test_no_cache_writes_no_log(capsys, small_file, results):
    code, _out, err = demo_run(capsys, small_file, results, "--preset", "Sentiment", "--no-cache")
    assert code == 0, err
    assert not results.exists()


def test_verbose_logs_debug_to_stderr(capsys, small_file, results):
    code, _out, err = run_cli(
        capsys, "-v", "run", str(small_file), "--model", DEMO, "--preset", "Sentiment",
        "--results-dir", str(results),
    )  # fmt: skip
    assert code == 0, err
    assert "DEBUG batchtextreview.cli: batchtextreview" in err
    assert "INFO batchtextreview.engine: Run " in err
    assert "INFO batchtextreview.engine: Run " in log_file(results).read_text(encoding="utf-8")


def test_verbose_with_no_cache_logs_to_stderr_only(capsys, small_file, results):
    code, _out, err = run_cli(
        capsys, "--verbose", "run", str(small_file), "--model", DEMO, "--preset", "Sentiment",
        "--no-cache",
    )  # fmt: skip
    assert code == 0, err
    assert "INFO batchtextreview.engine: Run " in err
    assert not results.exists()


def test_verbose_flag_and_presets_v_are_separate(capsys, results):
    code, out, err = run_cli(capsys, "-v", "presets")
    assert code == 0
    assert "key:         Categorise (set {var})" not in out  # not "presets -v"
    assert "DEBUG batchtextreview.cli" in err
    code, out, err = run_cli(capsys, "presets", "-v")
    assert code == 0
    assert "key:         Categorise (set {var})" in out
    assert "DEBUG" not in err


def test_run_limit(capsys, data_file, results):
    code, out, err = demo_run(capsys, data_file, results, "--preset", "Intent", "--limit", "5")
    assert code == 0, err
    frame = read_csv(data_file.with_name("sample_feedback_intent_demo-fake-outputs_results.csv"))
    assert len(frame) == 5
    assert out.startswith("5 rows")


def test_run_template_file(capsys, data_file, results, tmp_path):
    prompt = tmp_path / "prompt.txt"
    # Windows line endings and a final newline are normalised away.
    prompt.write_bytes(b"Summarise this in one line:\r\n{text}\r\n")
    code, _out, err = demo_run(
        capsys, data_file, results, "--template-file", str(prompt), "--limit", "4"
    )
    assert code == 0, err
    output = data_file.with_name("sample_feedback_custom-prompt_demo-fake-outputs_results.csv")
    frame = read_csv(output)
    assert len(frame) == 4
    assert "LABEL" not in frame.columns  # open output: no label set
    assert frame.loc[frame["STATUS"] == "generated", "OUTPUT"].str.len().gt(0).all()
    runs = Store(results / DB_FILENAME).list_runs()
    assert list(runs["template"]) == ["Summarise this in one line:\n{text}"]


def test_run_template_file_utf16_from_notepad(capsys, small_file, results, tmp_path):
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Résumé of the text:\n{text}\n", encoding="utf-16")
    code, _out, err = demo_run(capsys, small_file, results, "--template-file", str(prompt))
    assert code == 0, err
    runs = Store(results / DB_FILENAME).list_runs()
    assert list(runs["template"]) == ["Résumé of the text:\n{text}"]


def test_run_template_text_with_preset_metadata(capsys, small_file, results):
    """--template with --preset keeps the preset's label set."""
    code, _out, err = demo_run(
        capsys,
        small_file,
        results,
        "--preset",
        "Sentiment",
        "--template",
        "Classify the sentiment of the text below. Reply with one word.\n\nTEXT:\n{text}",
    )
    assert code == 0, err
    frame = read_csv(small_file.with_name("small_sentiment_demo-fake-outputs_results.csv"))
    assert "LABEL" in frame.columns


def test_run_var_with_categorise(capsys, small_file, results):
    code, _out, err = demo_run(
        capsys,
        small_file,
        results,
        "--preset",
        "Categorise",
        "--var",
        "complaint, query, request, praise",
        "--gold",
        "GOLD_INTENT",
    )
    assert code == 0, err
    frame = read_csv(small_file.with_name("small_categorise_demo-fake-outputs_results.csv"))
    assert set(frame["LABEL"]) <= {"complaint", "query", "request", "praise", INVALID, ""}
    line = history(results).iloc[0]
    assert line["preset"] == "Categorise (set {var})"
    assert line["var"] == "complaint, query, request, praise"
    assert list(line["labels"]) == ["complaint", "query", "request", "praise"]


def test_run_labels_override_and_exact_match(capsys, small_file, results):
    code, out, err = demo_run(
        capsys,
        small_file,
        results,
        "--preset",
        "Sentiment",
        "--labels",
        "positive; negative",
        "--gold",
        "GOLD_SENTIMENT",
    )
    assert code == 0, err
    frame = read_csv(small_file.with_name("small_sentiment_demo-fake-outputs_results.csv"))
    assert set(frame["LABEL"]) <= {"positive", "negative", INVALID, ""}
    assert "labels: positive, negative" in out

    # An empty label set means open output, scored by exact match.
    code, out, err = demo_run(
        capsys,
        small_file,
        results,
        "--preset",
        "Sentiment",
        "--labels",
        "",
        "--gold",
        "GOLD_SENTIMENT",
        "--output",
        str(small_file.with_name("exact.csv")),
    )
    assert code == 0, err
    assert "exact match" in out
    assert "LABEL" not in read_csv(small_file.with_name("exact.csv")).columns


def test_exact_match_reports_empty_outputs_not_invalid_ones(
    capsys, small_file, results, monkeypatch
):
    # A think-only generation cleans to "": wrong, but with no label set it
    # isn't "not one of the labels".
    def responder(prompt: str) -> str:
        return "<think>hmm" if "library card" in prompt else demo_responder(prompt)

    monkeypatch.setattr(cli, "default_factory", lambda m, p: FakeBackend(responder=responder))
    code, out, err = demo_run(
        capsys,
        small_file,
        results,
        "--preset",
        "Sentiment",
        "--labels",
        "",
        "--gold",
        "GOLD_SENTIMENT",
    )
    assert code == 0, err
    assert "exact match" in out
    assert "1 empty output (counted as wrong)" in out
    assert "invalid output" not in out
    assert history(results).iloc[-1]["n_invalid"] == 0


def test_run_gold_outside_label_set_warns_and_skips_history(capsys, small_file, results):
    code, out, err = demo_run(
        capsys, small_file, results, "--preset", "Sentiment", "--gold", "GOLD_INTENT"
    )
    assert code == 0, err
    assert 'none of the values in "GOLD_INTENT"' in err
    assert "Nothing could be scored" in out
    assert "Not logged to history: nothing could be scored" in out
    assert history(results).empty


def test_run_json_preset_to_excel_expands_columns(capsys, data_file, results, tmp_path):
    output = tmp_path / "out" / "entities.xlsx"
    code, out, err = demo_run(
        capsys,
        data_file,
        results,
        "--preset",
        "Entity extraction",
        "--limit",
        "8",
        "--output",
        str(output),
    )
    assert code == 0, err
    assert output.is_file()  # parent folder created
    assert output.read_bytes()[:2] == b"PK"
    frame = pd.read_excel(output, dtype=object)
    assert len(frame) == 8
    assert "OUTPUT.valid_json" in frame.columns
    assert any(c.startswith("OUTPUT.") and c != "OUTPUT.valid_json" for c in frame.columns)
    assert frame["OUTPUT.valid_json"].map(lambda v: v is True or v == 1).any()
    assert str(output) in out


def test_run_on_exported_file_gets_free_column_names(capsys, small_file, results):
    code, _out, err = demo_run(capsys, small_file, results, "--preset", "Sentiment")
    assert code == 0, err
    exported = small_file.with_name("small_sentiment_demo-fake-outputs_results.csv")
    code, _out, err = demo_run(
        capsys, exported, results, "--preset", "Sentiment", "--text-column", "TEXT"
    )
    assert code == 0, err
    again = read_csv(
        small_file.with_name(
            "small_sentiment_demo-fake-outputs_results_sentiment_demo-fake-outputs_results.csv"
        )
    )
    assert {"OUTPUT", "OUTPUT_2", "STATUS", "STATUS_2"} <= set(again.columns)


def test_run_case_insensitive_columns_and_model_names(capsys, small_file, results):
    code, _out, err = run_cli(
        capsys,
        "run",
        str(small_file),
        "--model",
        "demo (FAKE outputs)",
        "--preset",
        "sentiment",
        "--text-column",
        "text",
        "--gold",
        "gold_sentiment",
        "--results-dir",
        str(results),
    )
    assert code == 0, err
    line = history(results).iloc[0]
    assert (line["text_column"], line["gold_column"]) == ("TEXT", "GOLD_SENTIMENT")


def test_run_no_cache(capsys, small_file, results):
    for _ in range(2):
        code, out, err = demo_run(
            capsys, small_file, results, "--preset", "Sentiment", "--no-cache", "--gold",
            "GOLD_SENTIMENT",
        )  # fmt: skip
        assert code == 0, err
        assert "from cache" not in out
        assert "Not logged to history (--no-cache)" in out
    assert not results.exists()


def test_run_reports_failed_rows_and_retries_them(capsys, small_file, results, monkeypatch):
    def flaky(model_id: str, precision: str) -> FakeBackend:
        return FakeBackend(fail_when=lambda batch: any("refund" in p for p in batch))

    monkeypatch.setattr(cli, "default_factory", flaky)
    code, out, err = demo_run(
        capsys, small_file, results, "--preset", "Sentiment", "--batch-size", "1"
    )
    assert code == 0, err
    frame = read_csv(small_file.with_name("small_sentiment_demo-fake-outputs_results.csv"))
    assert frame["STATUS"].tolist().count("error") == 1
    assert "ERROR" in frame.columns
    assert "1 row failed" in out

    monkeypatch.setattr(cli, "default_factory", lambda m, p: FakeBackend())
    code, out, err = demo_run(capsys, small_file, results, "--preset", "Sentiment")
    assert code == 0, err
    assert "1 generated" in out and "from cache" in out


def test_run_where_every_row_fails_exits_1(capsys, small_file, results, monkeypatch):
    monkeypatch.setattr(
        cli, "default_factory", lambda m, p: FakeBackend(fail_when=lambda batch: True)
    )
    code, _out, err = demo_run(
        capsys, small_file, results, "--preset", "Sentiment", "--batch-size", "100"
    )
    assert code == 1
    assert "no row could be generated" in err and "fake failure" in err
    frame = read_csv(small_file.with_name("small_sentiment_demo-fake-outputs_results.csv"))
    assert set(frame["STATUS"]) == {"error", "blank"}


def test_run_aborts_after_repeated_batch_failures(capsys, small_file, results, monkeypatch):
    calls = {"n": 0}

    def responder(prompt: str) -> str:
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("CUDA error: device-side assert triggered")
        return demo_responder(prompt)

    monkeypatch.setattr(cli, "default_factory", lambda m, p: FakeBackend(responder=responder))
    code, out, err = demo_run(
        capsys, small_file, results, "--preset", "Sentiment", "--batch-size", "1"
    )
    assert code == 1
    assert "stopped early" in err and "device-side assert" in err
    # The row that worked is written along with the failures.
    frame = read_csv(small_file.with_name("small_sentiment_demo-fake-outputs_results.csv"))
    assert frame["STATUS"].tolist().count("generated") == 1
    assert "Stopped early" in out


@pytest.mark.parametrize("order", [None, "longest", "shortest"])
def test_run_order_sets_which_texts_go_first(capsys, small_file, results, monkeypatch, order):
    backend = FakeBackend()
    monkeypatch.setattr(cli, "default_factory", lambda m, p: backend)
    seen = {}
    real = engine.execute_run

    def spy(config, texts, **kwargs):
        seen.update(kwargs)
        return real(config, texts, **kwargs)

    monkeypatch.setattr(engine, "execute_run", spy)
    extra = ["--order", order] if order else []
    code, _out, err = demo_run(
        capsys, small_file, results, "--preset", "Sentiment", "--batch-size", "1", *extra
    )
    assert code == 0, err
    expected = order or "longest"
    assert seen["order"] == expected
    texts = [t for t in pd.read_csv(small_file, dtype=str)["TEXT"].dropna()]
    first = (max if expected == "longest" else min)(texts, key=len)
    assert first in backend.calls[0][0]
    [header] = [line for line in err.splitlines() if line.startswith("Sentiment | ")]
    assert header.endswith(f"column TEXT | {expected} texts first")

    # The order isn't part of the run id: the other order is served from the cache.
    other = "shortest" if expected == "longest" else "longest"
    code, out, err = demo_run(
        capsys, small_file, results, "--preset", "Sentiment", "--order", other
    )
    assert code == 0, err
    assert "from cache" in out and "generated" not in out.split("\n", 1)[0]
    [header] = [line for line in err.splitlines() if line.startswith("Sentiment | ")]
    assert "texts first" not in header  # nothing to generate, so no order to tell


def test_run_order_rejects_other_values(capsys, small_file):
    with pytest.raises(SystemExit) as raised:
        cli.main(
            ["run", str(small_file), "--model", DEMO, "--preset", "Sentiment", "--order", "random"]
        )
    assert raised.value.code == 2
    err = capsys.readouterr().err
    assert "--order" in err and "'longest'" in err and "'shortest'" in err


# --------------------------------------------------------------------------
# run: friendly errors (exit 1, one message, no traceback)
# --------------------------------------------------------------------------


def assert_friendly_error(code: int, err: str, *fragments: str) -> None:
    assert code == 1
    assert err.startswith("error: ") or "\nerror: " in err
    assert "Traceback" not in err
    for fragment in fragments:
        assert fragment in err


def test_error_missing_file(capsys, tmp_path, results):
    code, _out, err = demo_run(capsys, tmp_path / "nope.csv", results, "--preset", "Sentiment")
    assert_friendly_error(code, err, "File not found", "nope.csv")


def test_error_folder_as_input(capsys, tmp_path, results):
    code, _out, err = demo_run(capsys, tmp_path, results, "--preset", "Sentiment")
    assert_friendly_error(code, err, "is a folder")


def test_error_unknown_preset_suggests(capsys, small_file, results):
    code, _out, err = demo_run(capsys, small_file, results, "--preset", "Sentimnet")
    assert_friendly_error(code, err, 'Unknown preset "Sentimnet"', 'Did you mean "Sentiment"')
    # Validation happens before anything is written.
    assert not results.exists()


def test_error_no_task(capsys, small_file, results):
    code, _out, err = demo_run(capsys, small_file, results)
    assert_friendly_error(code, err, "--preset", "--template")


def test_error_missing_var(capsys, small_file, results):
    code, _out, err = demo_run(capsys, small_file, results, "--preset", "Categorise")
    assert_friendly_error(code, err, "{var}", "--var", "invoice")


def test_error_var_containing_text_placeholder(capsys, small_file, results):
    code, _out, err = demo_run(
        capsys, small_file, results, "--preset", "Categorise", "--var", "a, {text}"
    )
    assert_friendly_error(code, err, "{text}")


def test_error_template_without_text(capsys, small_file, results):
    code, _out, err = demo_run(capsys, small_file, results, "--template", "Classify this.")
    assert_friendly_error(code, err, "{text}")


def test_error_template_file_missing(capsys, small_file, results, tmp_path):
    code, _out, err = demo_run(
        capsys, small_file, results, "--template-file", str(tmp_path / "missing.txt")
    )
    assert_friendly_error(code, err, "Template file not found")


def test_error_unknown_text_column_lists_columns(capsys, small_file, results):
    code, _out, err = demo_run(
        capsys, small_file, results, "--preset", "Sentiment", "--text-column", "BODY"
    )
    assert_friendly_error(
        code, err, 'No text column "BODY"', "Available columns: INDEX, TEXT, GOLD_SENTIMENT"
    )


def test_error_missing_gold_column(capsys, small_file, results):
    code, _out, err = demo_run(
        capsys, small_file, results, "--preset", "Sentiment", "--gold", "GOLD_SENTIMNET"
    )
    assert_friendly_error(
        code,
        err,
        'No gold column "GOLD_SENTIMNET"',
        'Did you mean "GOLD_SENTIMENT"',
        "Available columns:",
    )
    assert not small_file.with_name("small_sentiment_demo-fake-outputs_results.csv").exists()


def test_error_model_load_failure_prints_hint(capsys, small_file, results, monkeypatch):
    def broken(model_id: str, precision: str) -> FakeBackend:
        raise ModelLoadError(
            "Couldn't download the model.", hint="Set HF_TOKEN and accept the licence."
        )

    monkeypatch.setattr(cli, "default_factory", broken)
    code, _out, err = demo_run(capsys, small_file, results, "--preset", "Sentiment")
    assert_friendly_error(
        code, err, "Couldn't download the model.", "Set HF_TOKEN and accept the licence."
    )
    assert not small_file.with_name("small_sentiment_demo-fake-outputs_results.csv").exists()


def test_error_unknown_model_suggests(capsys, small_file, results):
    code, _out, err = run_cli(
        capsys, "run", str(small_file), "--preset", "Sentiment", "--model", "Qwen2.5-1.5B",
        "--results-dir", str(results),
    )  # fmt: skip
    assert_friendly_error(code, err, 'Did you mean "Qwen2.5-1.5B-Instruct"')


def test_error_bad_model_path(capsys, small_file, results, tmp_path):
    code, _out, err = run_cli(
        capsys, "run", str(small_file), "--preset", "Sentiment", "--model",
        str(tmp_path / "no" / "such" / "model"), "--results-dir", str(results),
    )  # fmt: skip
    assert_friendly_error(code, err, "Folder not found")


def test_error_all_blank_text(capsys, tmp_path, results):
    path = tmp_path / "blank.csv"
    path.write_text("TEXT,NOTE\n,a\n  ,b\n", encoding="utf-8")
    code, _out, err = demo_run(
        capsys, path, results, "--preset", "Sentiment", "--text-column", "TEXT"
    )
    assert_friendly_error(code, err, "is empty")


@pytest.mark.parametrize("name", ["out.parquet", "out.xls", "out.json"])
def test_error_unwritable_output_format(capsys, small_file, results, tmp_path, name):
    code, _out, err = demo_run(
        capsys, small_file, results, "--preset", "Sentiment", "--output", str(tmp_path / name)
    )
    assert_friendly_error(code, err, ".csv or .xlsx")


def test_error_output_would_overwrite_input(capsys, small_file, results):
    code, _out, err = demo_run(
        capsys, small_file, results, "--preset", "Sentiment", "--output", str(small_file)
    )
    assert_friendly_error(code, err, "overwrite the input")
    assert small_file.read_text(encoding="utf-8") == SMALL_CSV


def test_error_unreadable_data(capsys, tmp_path, results):
    path = tmp_path / "data.parquet"
    path.write_bytes(b"not a parquet file")
    code, _out, err = demo_run(capsys, path, results, "--preset", "Sentiment")
    assert_friendly_error(code, err, "data.parquet")


def test_usage_errors_exit_2(capsys, small_file):
    with pytest.raises(SystemExit) as raised:
        cli.main(["run", str(small_file), "--preset", "Sentiment"])  # no --model
    assert raised.value.code == 2
    with pytest.raises(SystemExit) as raised:
        cli.main(["run", str(small_file), "--model", DEMO, "--batch-size", "0"])
    assert raised.value.code == 2
    with pytest.raises(SystemExit) as raised:
        cli.main(["models", "--bogus"])
    assert raised.value.code == 2
    capsys.readouterr()


# --------------------------------------------------------------------------
# run: Ctrl+C
# --------------------------------------------------------------------------


def interrupting_factory(n_signals: int):
    """A demo backend that delivers ``n_signals`` SIGINTs during its first batch."""

    def responder(prompt: str) -> str:
        if not state["sent"]:
            state["sent"] = True
            for _ in range(n_signals):
                signal.raise_signal(signal.SIGINT)
        return demo_responder(prompt)

    state = {"sent": False}
    return lambda model_id, precision: FakeBackend(responder=responder)


def test_ctrl_c_cancels_after_batch_and_writes_partial_results(
    capsys, small_file, results, monkeypatch
):
    monkeypatch.setattr(cli, "default_factory", interrupting_factory(1))
    before = signal.getsignal(signal.SIGINT)
    code, out, err = demo_run(
        capsys, small_file, results, "--preset", "Sentiment", "--batch-size", "1", "--gold",
        "GOLD_SENTIMENT",
    )  # fmt: skip
    assert code == 130
    assert signal.getsignal(signal.SIGINT) is before
    assert "press Ctrl+C again" in err
    assert "Cancelled" in err
    frame = read_csv(small_file.with_name("small_sentiment_demo-fake-outputs_results.csv"))
    statuses = frame["STATUS"].tolist()
    assert statuses.count("generated") == 1
    assert "pending" in statuses
    assert "cancelled" in out
    assert "Not logged to history" in out
    assert history(results).empty

    # The finished batch was saved: running again resumes.
    monkeypatch.setattr(cli, "default_factory", lambda m, p: FakeBackend())
    code, out, err = demo_run(capsys, small_file, results, "--preset", "Sentiment")
    assert code == 0, err
    assert "1 from cache" in out


def test_second_ctrl_c_quits_at_once(capsys, small_file, results, monkeypatch):
    monkeypatch.setattr(cli, "default_factory", interrupting_factory(2))
    before = signal.getsignal(signal.SIGINT)
    code, _out, err = demo_run(
        capsys, small_file, results, "--preset", "Sentiment", "--batch-size", "1"
    )
    assert code == 130
    assert signal.getsignal(signal.SIGINT) is before
    assert "Stopped" in err
    assert "Traceback" not in err
    assert not small_file.with_name("small_sentiment_demo-fake-outputs_results.csv").exists()


# --------------------------------------------------------------------------
# Progress display
# --------------------------------------------------------------------------


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def progress(phase: str, done: int, total: int = 100, **kw) -> engine.Progress:
    return engine.Progress(phase=phase, total_rows=total, done_rows=done, **kw)  # type: ignore[arg-type]


def test_progress_plain_lines_are_occasional():
    stream = io.StringIO()
    clock = FakeClock()
    printer = cli.ProgressPrinter(stream, interactive=False, clock=clock)
    printer(progress("planning", 0, message="100 rows to generate (100 model calls)."))
    printer(progress("loading", 0, message="Loading Demo..."))
    for done in range(1, 101):
        clock.now += 0.5
        printer(
            progress(
                "generating",
                done,
                generated_rows=done,
                rows_per_s=2.0,
                generation_s=done * 0.5,
                eta_s=(100 - done) * 0.5,
            )
        )
    printer(progress("finished", 100, message="Finished."))
    printer.close()
    text = stream.getvalue()
    assert "\r" not in text
    lines = text.splitlines()
    assert lines[:2] == ["100 rows to generate (100 model calls).", "Loading Demo..."]
    generating = [line for line in lines if line.startswith("Generating")]
    # One line per 10% at most: 50 s of progress, never more than 10 lines.
    assert 5 <= len(generating) <= 10
    assert "2.0 rows/s" in generating[0]


def test_progress_percentage_measures_work_not_rows():
    # Longest first: 10 rows of 100 done, but 90 s of the 120 s of work.
    p = progress("generating", 10, eta_s=30.0, generation_s=90.0)
    assert p.fraction == 0.1
    assert cli._progress_text(p, bar=False) == "Generating: 75% - 10/100 rows, ETA 30s"
    bar = cli._progress_text(p, bar=True)
    assert bar.startswith("Generating " + "█" * 15 + "░" * 5 + "  75%  10/100 rows")
    # The engine's own figure wins; without an estimate, rows.
    assert "50%" in cli._progress_text(dataclasses.replace(p, work_done=0.5), bar=False)
    assert "10%" in cli._progress_text(dataclasses.replace(p, eta_s=None), bar=False)
    assert "100%" in cli._progress_text(dataclasses.replace(p, work_done=1.5), bar=False)


def test_progress_plain_lines_follow_the_work_done():
    stream = io.StringIO()
    clock = FakeClock()
    printer = cli.ProgressPrinter(stream, interactive=False, clock=clock)
    clock.now += 5
    printer(progress("generating", 2, generation_s=50.0, eta_s=50.0))  # 2% of rows
    assert stream.getvalue() == "Generating: 50% - 2/100 rows, ETA 50s\n"


def test_progress_quick_run_stays_quiet():
    stream = io.StringIO()
    printer = cli.ProgressPrinter(stream, interactive=False, clock=FakeClock())
    for done in range(1, 101):
        printer(progress("generating", done))
    assert stream.getvalue() == ""


def test_progress_interactive_redraws_one_line():
    stream = io.StringIO()
    clock = FakeClock()
    printer = cli.ProgressPrinter(stream, interactive=True, clock=clock)
    printer(progress("planning", 0, message="10 rows to generate."))
    printer(progress("loading", 0, message="Loading Demo..."))
    for done in range(1, 11):
        clock.now += 1
        printer(
            progress("generating", done, total=10, rows_per_s=1.0, eta_s=10 - done, error_rows=1)
        )
    printer.note("Stopping after the current batch.")
    printer.close()
    text = stream.getvalue()
    assert text.startswith("10 rows to generate.\n")
    assert "\rLoading Demo..." in text
    assert "100%" in text and "10/10 rows" in text
    assert "1 failed" in text
    assert "Stopping after the current batch.\n" in text
    # close() blanks the transient line and returns the cursor.
    assert text.endswith("\r")
    assert "\n" not in text.split("Stopping after the current batch.\n", 1)[1]


def test_progress_never_raises():
    class Broken(io.StringIO):
        def write(self, s: str) -> int:
            raise OSError("closed")

    printer = cli.ProgressPrinter(Broken(), interactive=True)
    printer(progress("generating", 1))
    printer.note("x")
    printer.close()


# --------------------------------------------------------------------------
# Other commands
# --------------------------------------------------------------------------


def test_models_command(capsys):
    code, out, _err = run_cli(capsys, "models")
    assert code == 0
    assert "Qwen2.5-1.5B-Instruct" in out
    assert "Qwen/Qwen2.5-1.5B-Instruct" in out
    assert "demo://fake" in out
    assert "gated" in out
    assert "1.5B" in out


def test_presets_command(capsys, results):
    code, out, _err = run_cli(capsys, "presets")
    assert code == 0
    assert "Sentiment" in out
    assert "positive, negative, mixed, neutral" in out
    assert "Categorise" in out and "--var" in out
    assert "JSON" in out
    assert not results.exists()  # listing never creates a database

    code, out, _err = run_cli(capsys, "presets", "-v")
    assert code == 0
    assert "key:         Categorise (set {var})" in out
    assert "{text}" in out


def test_saved_custom_preset_is_listed_and_runnable(capsys, small_file, results):
    store = Store(results / DB_FILENAME)
    store.save_preset(
        make_custom_preset(
            "Triage", "Is this a complaint? yes or no.\n\nTEXT:\n{text}", 4, ["yes", "no"]
        )
    )
    code, out, _err = run_cli(capsys, "presets", "--results-dir", str(results))
    assert code == 0
    assert "Triage" in out and "Saved" in out

    code, _out, err = demo_run(capsys, small_file, results, "--preset", "triage")
    assert code == 0, err
    frame = read_csv(small_file.with_name("small_triage_demo-fake-outputs_results.csv"))
    assert set(frame["LABEL"]) <= {"yes", "no", INVALID, ""}


def test_history_command(capsys, small_file, results):
    code, out, _err = run_cli(capsys, "history", "--results-dir", str(results))
    assert code == 0
    assert "No evaluation history yet" in out
    assert not results.exists()

    for preset, gold in (("Sentiment", "GOLD_SENTIMENT"), ("Intent", "GOLD_INTENT")):
        code, _out, err = demo_run(capsys, small_file, results, "--preset", preset, "--gold", gold)
        assert code == 0, err
    code, out, _err = run_cli(capsys, "history", "--results-dir", str(results))
    assert code == 0
    assert "ACCURACY" in out and "MACRO-F1" in out and "KAPPA" in out
    assert "GOLD_INTENT" in out and "GOLD_SENTIMENT" in out
    assert "cli" in out and "Demo (fake outputs)" in out

    code, out, _err = run_cli(capsys, "history", "--results-dir", str(results), "--limit", "1")
    assert code == 0
    assert "GOLD_INTENT" in out and "GOLD_SENTIMENT" not in out
    assert "Showing the newest 1 of 2" in out


def test_history_logs_a_database_upgrade(capsys, results):
    import sqlite3

    Store(results / DB_FILENAME)
    conn = sqlite3.connect(results / DB_FILENAME)
    try:
        with conn:
            conn.execute("UPDATE meta SET value = '1' WHERE key = 'schema_version'")
    finally:
        conn.close()
    code, _out, _err = run_cli(capsys, "history")
    assert code == 0
    assert Store(results / DB_FILENAME).get_meta("schema_version") == str(SCHEMA_VERSION)
    assert "INFO batchtextreview.store: Upgraded " in log_file(results).read_text(encoding="utf-8")


def _two_categorise_runs(capsys, small_file, results) -> None:
    for var in ("praise, complaint, query, request, other", "praise, complaint, other"):
        code, _out, err = demo_run(
            capsys, small_file, results, "--preset", "Categorise", "--var", var,
            "--gold", "GOLD_INTENT",
        )  # fmt: skip
        assert code == 0, err


def test_history_tells_runs_with_different_variables_apart(capsys, small_file, results):
    """Regression: two Categorise runs with different categories both read
    "Categorise (set {var})", and the invalid-output count v1's evals.csv
    kept wasn't shown."""
    _two_categorise_runs(capsys, small_file, results)
    code, out, _err = run_cli(capsys, "history")
    assert code == 0
    header, newest, oldest = out.splitlines()[:3]
    assert "TASK" in header and "INVALID" in header and "PRESET" not in header
    assert "Categorise · praise, complaint, other" in newest
    assert "Categorise · praise, complaint, query," in oldest
    assert "{var}" not in out
    # The INVALID column holds each line's n_invalid.
    invalid_at = header.index("INVALID") + len("INVALID")
    lines = history(results)
    for line, n_invalid in zip((newest, oldest), lines["n_invalid"], strict=True):
        assert line[:invalid_at].split()[-1] == f"{int(n_invalid):,}"


def test_history_task_names_custom_and_unknown_presets():
    saved = [make_custom_preset("Triage", "Urgent? {text}", 8, ["yes", "no"])]
    assert cli._task_text({"preset": "Sentiment", "var": None}, saved) == "Sentiment"
    assert cli._task_text({"preset": "Triage", "var": ""}, saved) == "Triage"
    # A deleted custom preset: the title logged with the line, else the key.
    gone = {"preset": "Old one", "preset_title": "Old title", "var": "a,\n b"}
    assert cli._task_text(gone, []) == "Old title · a, b"
    assert cli._task_text({"preset": "Old one"}, []) == "Old one"
    assert cli._task_text({"preset": None}, []) == "-"


@pytest.mark.parametrize("suffix", [".csv", ".xlsx"])
def test_history_output_saves_every_column(capsys, small_file, results, tmp_path, suffix):
    _two_categorise_runs(capsys, small_file, results)
    target = tmp_path / "exports" / f"history{suffix}"
    code, out, _err = run_cli(capsys, "history", "--output", str(target))
    assert code == 0
    assert out.strip() == f"Saved 2 evaluations to {target}"
    if suffix == ".csv":
        assert target.read_bytes().startswith(BOM)
        frame = read_csv(target)
    else:
        frame = pd.read_excel(target, dtype=str, keep_default_na=False)
    assert list(frame.columns) == list(EVALUATION_COLUMNS)
    assert frame["var"].tolist() == [
        "praise, complaint, other",
        "praise, complaint, query, request, other",
    ]
    assert frame["labels"].tolist() == frame["var"].tolist()  # not "['praise', ...]"
    assert frame["n_invalid"].tolist() == [str(v) for v in history(results)["n_invalid"]]

    # --limit applies to the file too; by default the file has every line.
    code, out, _err = run_cli(capsys, "history", "--output", str(target), "--limit", "1")
    assert code == 0 and "Saved 1 evaluation to" in out


def test_history_output_refuses_folders_and_other_formats(capsys, tmp_path, results):
    code, _out, err = run_cli(capsys, "history", "--output", str(tmp_path))
    assert code == 1 and "is a folder" in err
    code, _out, err = run_cli(capsys, "history", "--output", str(tmp_path / "h.json"))
    assert code == 1 and "use a .csv or .xlsx --output" in err
    assert not results.exists()  # checked before the database is opened


def test_history_empty_database(capsys, results):
    Store(results / DB_FILENAME)
    code, out, _err = run_cli(capsys, "history")
    assert code == 0
    assert "No evaluations logged yet" in out
    assert log_file(results).is_file()  # the database is there, so is the log


@pytest.mark.parametrize(
    ("info", "expected"),
    [
        (
            DeviceInfo(
                kind="cuda",
                name="NVIDIA Test GPU",
                total_memory_gb=8.0,
                bitsandbytes=True,
                torch_version="2.5.1",
                transformers_version="4.56.0",
                cuda_version="12.4",
            ),
            ["NVIDIA Test GPU - 8.0 GB", "4bit", "built for CUDA 12.4", "4.56.0", "available"],
        ),
        (
            DeviceInfo(kind="cpu", name="CPU (no CUDA GPU detected)", torch_version="2.5.1"),
            ["CPU (no CUDA GPU detected)", "fp32", "CPU-only build", "not available"],
        ),
        (
            DeviceInfo(kind="none", name="PyTorch not installed"),
            ["PyTorch not installed", "fp32"],
        ),
    ],
)
def test_device_command(capsys, monkeypatch, results, info, expected):
    monkeypatch.setattr(backend_base, "device_info", lambda: info)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    code, out, _err = run_cli(capsys, "device")
    assert code == 0
    for fragment in expected:
        assert fragment in out
    assert "Python" in out and "Streamlit" in out
    assert "not set" in out  # HF token
    assert str(results / DB_FILENAME) in out


def _v1_checkpoint(folder: Path) -> str:
    """One version-1 checkpoint + manifest + evals.csv line, as v1 wrote them."""
    label = "Qwen2.5-3B-Instruct (faster)"
    template = "Classify the sentiment.\n\nTEXT:\n{text}"
    payload = json.dumps(["v1", label, template, 8, 8000])
    run_id = hashlib.sha256(payload.encode()).hexdigest()[:16]
    folder.mkdir(parents=True, exist_ok=True)
    keys = [engine.content_key(t) for t in ("good", "bad")]
    pd.DataFrame({"key": keys, "output": ["positive", "negative"]}).to_parquet(
        folder / f"{run_id}.parquet", index=False
    )
    manifest = {
        "run_id": run_id,
        "model": label,
        "template": template,
        "max_new_tokens": 8,
        "max_input_chars": 8000,
        "created": "2025-06-01T12:00:00",
    }
    (folder / f"{run_id}.json").write_text(json.dumps(manifest), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "timestamp": "2025-06-01T12:30:00",
                "model": label,
                "preset": "Sentiment",
                "var": "",
                "rows_scored": 2,
                "accuracy": 0.5,
                "invalid_outputs": 0,
                "run_id": run_id,
            }
        ]
    ).to_csv(folder / "evals.csv", index=False)
    return run_id


def test_import_legacy_command(capsys, tmp_path, results):
    old = tmp_path / "v1"
    _v1_checkpoint(old)
    code, out, _err = run_cli(capsys, "import-legacy", str(old), "--results-dir", str(results))
    assert code == 0
    assert "Imported 1 run (2 outputs) and 1 history entry" in out
    assert (results / DB_FILENAME).is_file()
    lines = history(results)
    assert len(lines) == 1 and lines.iloc[0]["source"] == "legacy"

    code, out, _err = run_cli(capsys, "import-legacy", str(old), "--results-dir", str(results))
    assert code == 0
    assert "Nothing new to import" in out
    assert len(history(results)) == 1
    assert "INFO batchtextreview.store: Version-1 import from" in log_file(results).read_text(
        encoding="utf-8"
    )


def test_import_legacy_defaults_to_results_dir(capsys, results):
    _v1_checkpoint(results)
    code, out, _err = run_cli(capsys, "import-legacy")
    assert code == 0
    assert "Imported 1 run" in out


def test_import_legacy_nothing_found(capsys, tmp_path, results):
    empty = tmp_path / "empty"
    empty.mkdir()
    pd.DataFrame({"key": ["k"], "output": ["o"]}).to_parquet(empty / "0123456789abcdef.parquet")
    code, out, _err = run_cli(capsys, "import-legacy", str(empty))
    assert code == 0
    assert "No version-1 results found" in out
    assert "0123456789abcdef.parquet" in out
    assert not results.exists()

    code, _out, err = run_cli(capsys, "import-legacy", str(tmp_path / "missing"))
    assert_friendly_error(code, err, "Folder not found")


# --------------------------------------------------------------------------
# ui
# --------------------------------------------------------------------------


@pytest.fixture
def launched(monkeypatch):
    calls: list[tuple[list[str], dict[str, str]]] = []

    def fake_run(command, env=None, check=False, **kwargs):
        calls.append((list(command), dict(env or {})))
        return subprocess.CompletedProcess(command, 0)

    real_find_spec = cli.importlib.util.find_spec

    def find_spec(name: str, *args):
        # Launching is faked, so Streamlit needn't be importable here.
        return object() if name == "streamlit" else real_find_spec(name, *args)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    monkeypatch.setattr(cli.importlib.util, "find_spec", find_spec)
    return calls


def test_ui_builds_streamlit_command(capsys, launched, tmp_path, monkeypatch):
    monkeypatch.delenv("BTR_DEMO", raising=False)
    code, _out, err = run_cli(
        capsys, "ui", "--demo", "--results-dir", str(tmp_path / "res"), "--",
        "--server.port", "8502",
    )  # fmt: skip
    assert code == 0
    assert "Ctrl+C" in err
    (command, env) = launched[0]
    assert command[:4] == [sys.executable, "-m", "streamlit", "run"]
    script = Path(command[4])
    assert script.is_file()
    assert script.name in {"app.py", "streamlit_app.py"}
    assert "batchtextreview.ui.main" in script.read_text(encoding="utf-8")
    assert command[-2:] == ["--server.port", "8502"]
    assert env["BTR_DEMO"] == "1"
    assert env["BTR_RESULTS_DIR"] == str((tmp_path / "res").resolve())


def test_ui_passthrough_after_verbose(capsys, launched, monkeypatch):
    code, _out, _err = run_cli(capsys, "-v", "ui", "--", "--server.port", "8502")
    assert code == 0
    command, _env = launched[0]
    assert command[-2:] == ["--server.port", "8502"]


def test_ui_verbose_reaches_the_app_through_the_environment(capsys, launched):
    """Streamlit runs in its own process: -v travels as BTR_VERBOSE=1, which
    the app's configure_logging() reads."""
    assert run_cli(capsys, "ui")[0] == 0
    assert "BTR_VERBOSE" not in launched[0][1]
    assert run_cli(capsys, "-v", "ui")[0] == 0
    assert launched[1][1]["BTR_VERBOSE"] == "1"


def test_btr_verbose_environment_works_like_v(capsys, small_file, results, monkeypatch):
    monkeypatch.setenv("BTR_VERBOSE", "1")
    code, _out, err = run_cli(
        capsys, "run", str(small_file), "--model", DEMO, "--preset", "Sentiment",
        "--results-dir", str(results),
    )  # fmt: skip
    assert code == 0, err
    assert "DEBUG batchtextreview.cli: batchtextreview" in err
    assert "INFO batchtextreview.engine: Run " in err
    assert logging.getLogger("batchtextreview").level == logging.DEBUG
    assert "INFO batchtextreview.engine: Run " in log_file(results).read_text(encoding="utf-8")


def test_ui_passes_unknown_options_through(capsys, launched, monkeypatch):
    monkeypatch.delenv("BTR_DEMO", raising=False)
    code, _out, _err = run_cli(capsys, "ui", "--server.headless", "true")
    assert code == 0
    command, env = launched[0]
    assert command[-2:] == ["--server.headless", "true"]
    assert "BTR_DEMO" not in env


def test_ui_bundled_entry_point():
    bundled = Path(cli.__file__).resolve().parent / "ui" / "streamlit_app.py"
    assert bundled.is_file()
    assert "from batchtextreview.ui.main import main" in bundled.read_text(encoding="utf-8")


def test_ui_essentials_as_flags_without_a_project_config(monkeypatch, tmp_path):
    script = tmp_path / "streamlit_app.py"
    script.write_text("", encoding="utf-8")
    monkeypatch.setattr(cli, "_ui_script", lambda: script)
    command = cli._ui_command(["--server.port", "9000"])
    assert command[4] == str(script)
    assert "--server.fileWatcherType" in command and "--browser.gatherUsageStats" in command
    assert command[-2:] == ["--server.port", "9000"]
    # With the config next to the script (a checkout, or a wheel, which ships
    # it beside ui/streamlit_app.py), Streamlit reads it instead.
    (tmp_path / ".streamlit").mkdir()
    (tmp_path / ".streamlit" / "config.toml").write_text("", encoding="utf-8")
    assert "--server.fileWatcherType" not in cli._ui_command()


@pytest.mark.parametrize(
    ("passthrough", "env_address", "expected"),
    [
        ([], None, "localhost"),
        (["--server.port", "9000"], None, "localhost"),
        # Sharing on purpose: Streamlit takes an option's last occurrence.
        (["--server.address", "0.0.0.0"], None, "0.0.0.0"),
        ([], "192.0.2.7", "192.0.2.7"),
    ],
)
def test_ui_listens_on_this_computer_only_by_default(
    monkeypatch, passthrough, env_address, expected
):
    # Streamlit's own default is every network interface, and the app can
    # list and read files on this disk.
    streamlit_cli = pytest.importorskip("streamlit.web.cli")
    if env_address is None:
        monkeypatch.delenv("STREAMLIT_SERVER_ADDRESS", raising=False)
    else:
        monkeypatch.setenv("STREAMLIT_SERVER_ADDRESS", env_address)
    command = cli._ui_command(passthrough)
    assert command[len(command) - len(passthrough) :] == passthrough
    # Parse it the way "streamlit run" will (nothing is started).
    context = streamlit_cli.main_run.make_context("run", command[4:])
    assert context.params["server_address"] == expected


def test_ui_prefers_checkout_app_py(monkeypatch, tmp_path):
    checkout = tmp_path / "checkout"
    (checkout / "src" / "batchtextreview").mkdir(parents=True)
    (checkout / "app.py").write_text("", encoding="utf-8")
    (checkout / "pyproject.toml").write_text("", encoding="utf-8")
    monkeypatch.setattr(cli, "__file__", str(checkout / "src" / "batchtextreview" / "cli.py"))
    assert cli._ui_script() == (checkout / "app.py").resolve()


def test_ui_installed_package_uses_its_bundled_entry_point(monkeypatch, tmp_path):
    # Where the package lives decides, not where the tests run from: a
    # checkout's tests can run against an installed wheel.
    package = tmp_path / "venv" / "Lib" / "site-packages" / "batchtextreview"
    package.mkdir(parents=True)
    monkeypatch.setattr(cli, "__file__", str(package / "cli.py"))
    assert cli._ui_script() == (package / "ui" / "streamlit_app.py").resolve()


def test_ui_source_checkout_uses_its_streamlit_config():
    if not Path(cli.__file__).resolve().is_relative_to(REPO_ROOT / "src"):
        pytest.skip("batchtextreview is installed from a build, not imported from this checkout")
    assert cli._ui_script() == (REPO_ROOT / "app.py").resolve()
    # The repository's .streamlit/config.toml applies as script-level config.
    assert "--server.fileWatcherType" not in cli._ui_command()


def test_ui_without_streamlit(capsys, launched, monkeypatch):
    real_find_spec = cli.importlib.util.find_spec
    monkeypatch.setattr(
        cli.importlib.util,
        "find_spec",
        lambda name, *a: None if name == "streamlit" else real_find_spec(name, *a),
    )
    code, _out, err = run_cli(capsys, "ui")
    assert_friendly_error(code, err, "Streamlit isn't installed")
    assert launched == []


# --------------------------------------------------------------------------
# Parser and helpers
# --------------------------------------------------------------------------


def test_version_and_help(capsys):
    with pytest.raises(SystemExit) as raised:
        cli.main(["--version"])
    assert raised.value.code == 0
    assert "batchtextreview" in capsys.readouterr().out

    code, out, _err = run_cli(capsys)
    assert code == 0
    for command in ("ui", "run", "models", "presets", "history", "import-legacy", "device"):
        assert command in out


def test_run_help_lists_options(capsys):
    with pytest.raises(SystemExit):
        cli.main(["run", "--help"])
    out = capsys.readouterr().out
    for option in (
        "--text-column", "--preset", "--template", "--template-file", "--var", "--labels",
        "--model", "--precision", "--max-new-tokens", "--max-input-chars", "--batch-size",
        "--order", "--limit", "--gold", "--output", "--no-cache", "--results-dir",
    ):  # fmt: skip
        assert option in out
    assert "{longest,shortest}" in out


def test_precision_help_matches_what_auto_resolves_to(capsys):
    """Regression: the help said auto meant 4bit on any NVIDIA GPU, fp32
    otherwise - but without bitsandbytes, and on Apple silicon, it is fp16."""
    from batchtextreview.backends.base import DeviceInfo, resolve_precision

    with pytest.raises(SystemExit):
        cli.main(["run", "--help"])
    text = " ".join(capsys.readouterr().out.split())
    assert "4bit on an NVIDIA GPU with bitsandbytes" in text
    assert "fp16 on an NVIDIA GPU without it or on Apple silicon" in text
    assert "fp32 on CPU" in text
    cases = {("cuda", True): "4bit", ("cuda", False): "fp16", ("mps", False): "fp16"}
    for (kind, bnb), expected in cases.items():
        assert resolve_precision("auto", DeviceInfo(kind, kind, bitsandbytes=bnb)) == expected
    assert resolve_precision("auto", DeviceInfo("cpu", "cpu")) == "fp32"


@pytest.mark.parametrize(
    ("name", "key"),
    [
        ("Sentiment", "Sentiment"),
        ("sentiment", "Sentiment"),
        ("Categorise", "Categorise (set {var})"),
        ("Categorise (set {var})", "Categorise (set {var})"),
        ("Yes/No question", "Yes/No flag (set {var})"),
        ("yes/no flag", "Yes/No flag (set {var})"),
        ("one line summary", "One-line summary"),
        ("extract field", "Extract field (set {var})"),
        ("Keywords", "Keyword extraction"),
        ("urgency (1-5)", "Urgency (1-5)"),
        ("custom", "Custom"),
    ],
)
def test_find_preset(name, key):
    assert cli.find_preset(name).key == key


def test_find_preset_unknown():
    with pytest.raises(cli.CliError, match=r'Did you mean "Emotion"'):
        cli.find_preset("Emotoin")
    with pytest.raises(cli.CliError, match="presets"):
        cli.find_preset("zzzz")


def test_default_output_path():
    from batchtextreview.presets import get_preset

    preset = get_preset("Yes/No flag (set {var})")
    assert preset is not None
    path = cli.default_output_path(Path("data") / "my file.xlsx", preset)
    assert path == Path("data") / "my file_yes-no-question_results.csv"
    # With the model, so comparing models doesn't replace the earlier results.
    model = cli._resolve_model("Qwen2.5-1.5B-Instruct")
    path = cli.default_output_path(Path("data") / "my file.xlsx", preset, model)
    assert path == Path("data") / "my file_yes-no-question_qwen2.5-1.5b-instruct_results.csv"


def test_output_into_folder(capsys, small_file, results, tmp_path):
    folder = tmp_path / "exports"
    folder.mkdir()
    code, _out, err = demo_run(
        capsys, small_file, results, "--preset", "Sentiment", "--output", str(folder)
    )
    assert code == 0, err
    assert (folder / "small_sentiment_demo-fake-outputs_results.csv").is_file()


# --------------------------------------------------------------------------
# run: writing the output
# --------------------------------------------------------------------------


def never_loaded(model_id: str, precision: str) -> FakeBackend:
    raise AssertionError("the model must not load when the output can't be written")


def test_unwritable_output_fails_before_the_model_loads(capsys, small_file, results, monkeypatch):
    monkeypatch.setattr(cli, "default_factory", never_loaded)
    blocker = small_file.with_name("blocker")
    blocker.write_text("a file, not a folder", encoding="utf-8")
    code, _out, err = demo_run(
        capsys, small_file, results, "--preset", "Sentiment", "--no-cache",
        "--output", str(blocker / "out.csv"),
    )  # fmt: skip
    assert_friendly_error(code, err, "Can't write --output", "blocker is a file, not a folder")
    assert "rows to generate" not in err
    assert not results.exists()


def test_output_locked_by_excel_fails_before_the_model_loads(
    capsys, small_file, results, monkeypatch
):
    target = small_file.with_name("locked.csv")
    target.write_text("previous results", encoding="utf-8")

    def excel_lock(path, mode="r", *args, **kwargs):
        if Path(path) == target and mode != "r":
            raise PermissionError(13, "Permission denied", str(path))
        return open(path, mode, *args, **kwargs)

    monkeypatch.setattr(cli, "default_factory", never_loaded)
    monkeypatch.setattr(cli, "open", excel_lock, raising=False)
    code, _out, err = demo_run(
        capsys, small_file, results, "--preset", "Sentiment", "--output", str(target)
    )
    assert_friendly_error(code, err, "Permission denied", "close it")
    assert target.read_text(encoding="utf-8") == "previous results"


def test_output_with_a_stray_quote_from_a_trailing_backslash(capsys, small_file, results):
    # cmd.exe turns --output "D:\Results\" into D:\Results"
    code, _out, err = demo_run(
        capsys, small_file, results, "--preset", "Sentiment", "--output", 'D:\\Results"'
    )
    assert_friendly_error(code, err, "double quote", "backslash")


@pytest.mark.parametrize("no_cache", [False, True])
def test_output_locked_at_the_end_goes_to_a_free_name(
    capsys, small_file, results, monkeypatch, no_cache
):
    target = small_file.with_name("out.csv")
    target.write_text("previous results", encoding="utf-8")
    replace = cli._replace_file

    def locked_after_the_check(path: Path, payload: bytes) -> None:
        if path == target:  # opened in Excel while the model ran
            raise PermissionError(13, "Permission denied", str(path))
        replace(path, payload)

    monkeypatch.setattr(cli, "_replace_file", locked_after_the_check)
    extra = ["--no-cache"] if no_cache else []
    code, out, err = demo_run(
        capsys, small_file, results, "--preset", "Sentiment", "--output", str(target), *extra
    )
    spare = small_file.with_name("out (1).csv")
    assert code == 1
    assert "note: " + str(target) + " exists - it will be replaced." in err
    assert f"Saved the results as {spare} instead" in err
    assert "is it open in Excel?" in err
    assert f"Results: {spare}" in out
    assert err.rstrip().endswith(f"error: the results are in {spare}, not {target}.")
    assert len(read_csv(spare)) == 6
    assert target.read_text(encoding="utf-8") == "previous results"


@pytest.mark.parametrize(
    ("extra", "advice"),
    [
        ([], "finished rows are saved, so the re-run is quick"),
        (["--no-cache"], "nothing was saved (--no-cache)"),
    ],
)
def test_output_nowhere_writable_says_whether_anything_was_kept(
    capsys, small_file, results, monkeypatch, extra, advice
):
    def disk_full(path: Path, payload: bytes) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(cli, "_replace_file", disk_full)
    code, _out, err = demo_run(capsys, small_file, results, "--preset", "Sentiment", *extra)
    assert_friendly_error(code, err, "No space left on device", advice)
    other = "nothing was saved" if not extra else "finished rows are saved"
    assert other not in err


def test_failed_write_leaves_the_previous_file_whole(tmp_path, monkeypatch):
    target = tmp_path / "out.csv"
    target.write_bytes(b"previous results")

    def interrupted(src, dst):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(cli.os, "replace", interrupted)
    with pytest.raises(PermissionError):
        cli._replace_file(target, b"new results")
    assert target.read_bytes() == b"previous results"
    assert [p.name for p in tmp_path.iterdir()] == ["out.csv"]  # no temporary file left


def test_comparing_models_keeps_each_models_results(capsys, small_file, results):
    for model in (DEMO, "demo://fake2"):
        code, _out, err = run_cli(
            capsys, "run", str(small_file), "--preset", "Sentiment", "--model", model,
            "--no-cache",
        )  # fmt: skip
        assert code == 0, err
        assert "will be replaced" not in err
    assert small_file.with_name("small_sentiment_demo-fake-outputs_results.csv").is_file()
    assert small_file.with_name("small_sentiment_demo-fake2_results.csv").is_file()
    # The same model again replaces its own results, and says so up front.
    code, _out, err = run_cli(
        capsys, "run", str(small_file), "--preset", "Sentiment", "--model", DEMO, "--no-cache"
    )
    assert code == 0, err
    assert "small_sentiment_demo-fake-outputs_results.csv exists - it will be replaced" in err


# --------------------------------------------------------------------------
# run: Excel sheets, column names, encodings
# --------------------------------------------------------------------------


def two_sheet_workbook(path: Path) -> Path:
    import openpyxl

    book = openpyxl.Workbook()
    summary = book.active
    summary.title = "Summary"
    summary.append(["Metric", "Value"])
    summary.append(["Total tickets", "3"])
    tickets = book.create_sheet("Tickets")
    tickets.append(["Ticket\nID", "Customer\nComment", "Gold\xa0Sentiment"])
    for row in SMALL_CSV.splitlines()[1:4]:
        number, text, gold, _intent = row.split(",")
        tickets.append([number, text, gold])
    book.save(path)
    return path


def test_sheet_option_reads_the_chosen_sheet(capsys, tmp_path, results):
    book = two_sheet_workbook(tmp_path / "two_sheets.xlsx")
    code, _out, err = demo_run(capsys, book, results, "--preset", "Sentiment", "--no-cache")
    assert code == 0, err
    assert 'note: reading sheet "Summary" - the workbook has 2: Summary, Tickets' in err
    assert '1 row of two_sheets.xlsx, sheet "Summary"' in err

    code, out, err = demo_run(
        capsys, book, results, "--preset", "Sentiment", "--no-cache", "--sheet", "tickets",
        "--gold", "Gold Sentiment", "--output", str(tmp_path / "tickets.csv"),
    )  # fmt: skip
    assert code == 0, err
    assert "reading sheet" not in err
    assert '3 rows of two_sheets.xlsx, sheet "Tickets", column Customer Comment' in err
    assert "Evaluation against Gold\xa0Sentiment" in out
    assert len(read_csv(tmp_path / "tickets.csv")) == 3


def test_sheet_errors_and_notes(capsys, tmp_path, small_file, results):
    book = two_sheet_workbook(tmp_path / "two_sheets.xlsx")
    code, _out, err = demo_run(capsys, book, results, "--preset", "Sentiment", "--sheet", "Nope")
    assert_friendly_error(code, err, "'Nope' not found", "Summary, Tickets")
    code, _out, err = demo_run(
        capsys, small_file, results, "--preset", "Sentiment", "--no-cache", "--sheet", "Tickets"
    )
    assert code == 0, err
    assert "note: --sheet is ignored - small.csv isn't an Excel workbook." in err


def test_wrapped_excel_headers_can_be_named_with_spaces(capsys, tmp_path, results):
    book = two_sheet_workbook(tmp_path / "wrapped.xlsx")
    code, _out, err = demo_run(
        capsys, book, results, "--preset", "Sentiment", "--sheet", "Tickets", "--no-cache",
        "--text-column", "customer comment", "--gold", "Gold Sentimnet",
    )  # fmt: skip
    # Suggestions and the column list stay on one line.
    assert_friendly_error(
        code,
        err,
        'No gold column "Gold Sentimnet" in wrapped.xlsx. Did you mean "Gold Sentiment"? '
        "Available columns: Ticket ID, Customer Comment, Gold Sentiment.",
    )


def test_header_line_names_a_guessed_encoding(capsys, tmp_path, results):
    path = tmp_path / "ansi.csv"
    path.write_bytes(SMALL_CSV.replace("wonderful", "caf\xe9-wonderful").encode("cp1252"))
    code, _out, err = demo_run(capsys, path, results, "--preset", "Sentiment", "--no-cache")
    assert code == 0, err
    assert "6 rows of ansi.csv (Windows-1252), column TEXT" in err


def test_python_dash_m_entry_point(small_file, results):
    """``python -m batchtextreview`` reaches the same main()."""
    done = subprocess.run(
        [sys.executable, "-m", "batchtextreview", "presets"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONUTF8": "1"},
        timeout=60,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert "Sentiment" in done.stdout


def test_broken_pipe_exits_quietly(monkeypatch, capsys):
    """``batchtextreview presets -v | head -1`` must not print a traceback."""

    def closed_pipe(*_args, **_kwargs):
        raise BrokenPipeError(32, "Broken pipe")

    monkeypatch.setattr(cli, "_cmd_presets", closed_pipe)
    assert cli.main(["presets"]) == cli.EXIT_OK
    assert "Traceback" not in capsys.readouterr().err
