"""History page: pure helpers plus AppTest coverage of the page itself.

History is seeded straight into the store with ``Store.log_evaluation`` (the
same call the Results page's finish hook makes), and version-1 artefacts are
written the way version 1 wrote them: ``{run_id}.parquet`` with key/output
columns, a ``{run_id}.json`` manifest and an append-only ``evals.csv``.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from batchtextreview import models, presets
from batchtextreview.backends.fake import FakeBackend
from batchtextreview.backends.manager import ModelManager
from batchtextreview.engine import RunConfig
from batchtextreview.jobs import JobRequest
from batchtextreview.ui import nav, resources
from batchtextreview.ui.pages import history

APP = Path(__file__).resolve().parents[1] / "app.py"
TIMEOUT = 30

SENTIMENT = presets.get_preset("Sentiment")
INTENT = presets.get_preset("Intent")


@pytest.fixture(autouse=True)
def demo_env(tmp_path, monkeypatch):
    monkeypatch.setenv("BTR_DEMO", "1")
    monkeypatch.setenv("BTR_RESULTS_DIR", str(tmp_path / "results"))
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


# ---- seeding -----------------------------------------------------------------


def record(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "created_at": "2026-10-10T12:00:00",
        "run_id": "run-a",
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "model_label": "Qwen2.5-3B-Instruct",
        "precision": "4bit",
        "preset": "Sentiment",
        "var": None,
        "labels": ["positive", "negative", "mixed", "neutral"],
        "dataset": "feedback.csv",
        "dataset_fingerprint": "fp-all",
        "text_column": "TEXT",
        "gold_column": "GOLD_SENTIMENT",
        "n_rows": 60,
        "n_scored": 59,
        "n_correct": 44,
        "accuracy": 0.75,
        "macro_f1": 0.70,
        "kappa": 0.62,
        "n_invalid": 1,
        "n_excluded": 1,
        "n_gold_unscorable": 0,
        "source": "app",
    }
    values.update(overrides)
    return values


SEED = [
    # oldest first, so the last one is the newest evaluation
    record(created_at="2026-10-01T09:00:00", run_id="r1", accuracy=0.60, macro_f1=0.55),
    record(
        created_at="2026-10-02T09:00:00",
        run_id="r2",
        model_id="Qwen/Qwen2.5-7B-Instruct",
        model_label="Qwen2.5-7B-Instruct",
        accuracy=0.82,
        macro_f1=0.79,
    ),
    record(
        created_at="2026-10-03T09:00:00",
        run_id="r3",
        model_id="microsoft/Phi-4-mini-instruct",
        model_label="Phi-4-mini-Instruct",
        accuracy=0.77,
        macro_f1=0.74,
    ),
    record(created_at="2026-10-04T09:00:00", run_id="r4", accuracy=0.78, macro_f1=0.72),
    record(
        created_at="2026-10-05T09:00:00",
        run_id="r5",
        dataset_fingerprint="fp-first20",
        n_rows=20,
        n_scored=20,
        accuracy=0.95,
        macro_f1=0.93,
    ),
    record(
        created_at="2026-10-06T09:00:00",
        run_id="r6",
        dataset="other.csv",
        dataset_fingerprint="fp-other",
        preset="Intent",
        gold_column="GOLD_INTENT",
        labels=list(INTENT.labels),
        accuracy=0.66,
        macro_f1=0.61,
    ),
    record(created_at="2026-10-07T09:00:00", run_id="r7", accuracy=0.71, macro_f1=0.66),
]


def seed(store, records=SEED) -> list[int]:
    return [store.log_evaluation(r) for r in records]


def v1_run_id(label: str, template: str, tokens: int, chars: int = 8000) -> str:
    payload = json.dumps(["v1", label, template, int(tokens), int(chars)])
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def write_legacy(folder: Path) -> str:
    """One v1 checkpoint + manifest, an orphan checkpoint and an evals.csv."""
    folder.mkdir(parents=True, exist_ok=True)
    label = "Qwen2.5-3B-Instruct (faster)"
    run_id = v1_run_id(label, SENTIMENT.template, SENTIMENT.max_new_tokens)
    pd.DataFrame({"key": [f"k{i}" for i in range(5)], "output": ["neutral"] * 5}).to_parquet(
        folder / f"{run_id}.parquet", index=False
    )
    manifest = {
        "run_id": run_id,
        "model": label,
        "template": SENTIMENT.template,
        "max_new_tokens": SENTIMENT.max_new_tokens,
        "max_input_chars": 8000,
        "created": "2026-09-20T10:15:00",
    }
    (folder / f"{run_id}.json").write_text(json.dumps(manifest), encoding="utf-8")
    pd.DataFrame({"key": ["x"], "output": ["y"]}).to_parquet(
        folder / "0123456789abcdef.parquet", index=False
    )
    pd.DataFrame(
        [
            {
                "timestamp": "2026-09-20T10:30:00",
                "model": label,
                "preset": "Sentiment",
                "var": "",
                "rows_scored": 20,
                "accuracy": 0.7,
                "invalid_outputs": 1,
                "run_id": run_id,
            },
            {
                "timestamp": "2026-09-21T11:30:00",
                "model": "Llama-3.2-3B-Instruct (gated)",
                "preset": "Sentiment",
                "var": "",
                "rows_scored": 20,
                "accuracy": 0.65,
                "invalid_outputs": 0,
                "run_id": "",
            },
            {
                "timestamp": "2026-09-22T09:00:00",
                "model": "Phi-4-mini-Instruct (3.8B)",
                "preset": "Intent",
                "var": "",
                "rows_scored": 15,
                "accuracy": 0.8,
                "invalid_outputs": 0,
                "run_id": "",
            },
        ]
    ).to_csv(folder / "evals.csv", index=False)
    return run_id


# ---- AppTest helpers -------------------------------------------------------------


def open_history(query: dict[str, str] | None = None, **state: Any) -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT)
    for key, value in state.items():
        at.session_state[key] = value
    at.run()
    # Pages built from callables have no file to switch_page() to; point the
    # test session at the History page's hash directly.
    at._page_hash = nav.page("history")._script_hash
    if query:
        at.query_params.update(query)
    return at.run()


def table_key(at: AppTest) -> str:
    """Key of the full-history table as last rendered (it names the rows in view)."""
    (key,) = [df.key for df in at.dataframe if (df.key or "").startswith(history.K_TABLE)]
    return key


def full_table(at: AppTest) -> pd.DataFrame:
    return at.dataframe(key=table_key(at)).value


def delete_button(at: AppTest):
    (button,) = [b for b in at.button if (b.key or "").startswith(history.K_DELETE + "::")]
    return button


def select(at: AppTest, rows: list[int]) -> None:
    """Tick rows of the full-history table, as a click in the browser would."""
    at.session_state[table_key(at)] = {"selection": {"rows": rows, "columns": []}}


def url(at: AppTest) -> dict[str, str]:
    """The page's own query parameters (the filters), as the browser shows them."""
    return {k: v for k, v in at.query_params.items() if k in history.URL_PARAMS.values()}


def texts(at: AppTest, kind: str = "markdown") -> str:
    return "\n".join(str(e.value) for e in getattr(at, kind))


def no_exceptions(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]


def board(at: AppTest) -> pd.DataFrame:
    """The leaderboard (the page's first table; read-only tables carry no key)."""
    return at.dataframe[0].value


# ==========================================================================
# Pure helpers
# ==========================================================================


def frame_of(records: list[dict[str, Any]]) -> pd.DataFrame:
    """A history_frame as the page builds it (newest first, like the store)."""
    raw = pd.DataFrame(list(reversed(records)))
    return history.history_frame(raw)


def test_history_frame_display_columns():
    legacy = {
        "id": 9,
        "created_at": "2026-09-20T10:30:00",
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "model_label": "Qwen2.5-3B-Instruct (faster)",
        "preset": "Yes/No flag (set {var})",
        "var": "does it mention parking?",
        "accuracy": 0.7,
        "source": "legacy",
    }
    frame = history.history_frame(pd.DataFrame([legacy]))
    row = frame.iloc[0]
    # Known repos read with the catalogue's name, not the v1 label.
    assert row["model"] == "Qwen2.5-3B-Instruct"
    assert row["model_key"] == "Qwen/Qwen2.5-3B-Instruct"
    assert row["task"] == "Yes/No question · does it mention parking?"
    assert row["task_name"] == "Yes/No question"
    assert row["dataset_key"] == history.UNKNOWN
    assert row["gold_key"] == history.UNKNOWN
    assert row["source_word"] == "v1"
    assert row["precision_word"] == "–"
    assert math.isnan(row["macro_f1"])
    assert row["accuracy"] == pytest.approx(0.7)


def test_task_title_unknown_and_custom():
    assert history.task_title(None) == "Unknown task"
    assert history.task_title("No longer exists") == "No longer exists"
    custom = presets.make_custom_preset("My triage", "Triage: {text}", 8, ["a", "b"])
    assert history.task_title("My triage", "", [custom]) == "My triage"


def test_leaderboard_best_per_model_ranks_by_f1_then_accuracy():
    records = [
        *SEED[:5],
        # same macro-F1 as r2 but lower accuracy: ranks right after it
        record(
            run_id="tie",
            model_id="meta-llama/Llama-3.2-3B-Instruct",
            model_label="Llama-3.2-3B-Instruct",
            accuracy=0.80,
            macro_f1=0.79,
        ),
        # version 1: accuracy only - ranks after every line with a macro-F1
        record(
            run_id=None,
            model_id="ibm-granite/granite-4.0-1b",
            model_label="Granite-4.0-1B",
            accuracy=0.99,
            macro_f1=None,
            source="legacy",
        ),
    ]
    frame = frame_of(records)
    best = history.leaderboard(frame, best_per_model=True)
    assert list(best["model"]) == [
        "Qwen2.5-3B-Instruct",  # its first-20 run (0.93) is its best
        "Qwen2.5-7B-Instruct",
        "Llama-3.2-3B-Instruct",
        "Phi-4-mini-Instruct",
        "Granite-4.0-1B",
    ]
    assert best.loc[0, "runs"] == 3
    every = history.leaderboard(frame, best_per_model=False)
    assert len(every) == len(records)
    assert every.iloc[-1]["model"] == "Granite-4.0-1B"
    assert history.leaderboard(frame.iloc[0:0]).empty


def test_filters_and_defaults():
    frame = frame_of(SEED)
    assert history.default_filter(frame, "dataset_key") == "feedback.csv"  # newest line
    assert history.default_filter(frame, "gold_key") == "GOLD_SENTIMENT"
    feedback = history.apply_filters(frame, dataset="feedback.csv")
    assert len(feedback) == 6
    assert history.default_filter(feedback, "task_key") == "Sentiment"  # names the one task
    assert history.default_filter(feedback.iloc[0:0], "task_key") == history.ALL
    assert len(history.apply_filters(frame, task="Intent")) == 1
    assert len(history.apply_filters(frame, gold="GOLD_SENTIMENT", selection="fp-first20")) == 1
    assert len(history.apply_filters(frame)) == len(SEED)
    # Rows follow the newest evaluation, like the other filters ...
    assert history.default_selection(feedback) == "fp-all"
    assert history.default_selection(feedback.iloc[0:0]) == history.ALL
    # ... even when most runs were on other rows, so the run just scored shows.
    newer = frame_of(
        [*SEED, record(created_at="2026-10-08T09:00:00", dataset_fingerprint="fp-new")]
    )
    assert history.default_selection(newer) == "fp-new"


def test_selection_options_labels():
    frame = frame_of(
        [
            record(created_at="2026-10-01T09:00", dataset_fingerprint="aaa", n_rows=60),
            record(created_at="2026-10-08T09:00", dataset_fingerprint="bbb", n_rows=20),
            record(created_at="2026-10-09T10:00", dataset_fingerprint="ccc", n_rows=20),
            record(created_at="2026-10-12T09:00", dataset_fingerprint="bbb", n_rows=20),
        ]
    )
    options = history.selection_options(frame)
    assert list(options) == ["bbb", "ccc", "aaa"]  # newest first
    assert options["aaa"] == "60 rows"
    # Two selections of the same size are told apart by when they were last
    # scored - never by a hash.
    assert options["bbb"] == "20 rows · last scored 12 Oct"
    assert options["ccc"] == "20 rows · last scored 9 Oct"
    # "All 62 rows" (recorded by the app) and "62 rows" (an older line) read the same.
    recorded = frame_of(
        [
            record(created_at="2026-10-08T09:00", dataset_fingerprint="old", n_rows=62),
            record(
                created_at="2026-10-13T00:23",
                dataset_fingerprint="new",
                n_rows=62,
                selection="All 62 rows",
            ),
        ]
    )
    assert history.selection_options(recorded) == {
        "new": "62 rows · last scored 13 Oct",
        "old": "62 rows · last scored 8 Oct",
    }
    same_day = frame_of(
        [
            record(created_at="2026-10-09T09:00", dataset_fingerprint="ddd", n_rows=20),
            record(created_at="2026-10-09T10:30", dataset_fingerprint="eee", n_rows=20),
        ]
    )
    assert set(history.selection_options(same_day).values()) == {
        "20 rows · last scored 9 Oct 09:00",
        "20 rows · last scored 9 Oct 10:30",
    }
    same_minute = frame_of(
        [record(dataset_fingerprint="fff", n_rows=5), record(dataset_fingerprint="ggg", n_rows=5)]
    )
    assert list(history.selection_options(same_minute).values()) == [
        "5 rows · last scored 10 Oct 12:00 (1)",
        "5 rows · last scored 10 Oct 12:00 (2)",
    ]


def test_constant_columns_are_folded_into_the_caption():
    board = history.leaderboard(frame_of(SEED[:4]))
    folded, words = history.constant_columns(board)
    assert folded == {"Weights", "Prompt", "Rows scored", "Invalid"}
    assert words == ["4-bit weights", "59 rows scored", "1 invalid answer each"]
    assert history.constant_columns(history.leaderboard(frame_of(SEED[:1])))[1][-1] == (
        "1 invalid answer"
    )
    mixed = frame_of([*SEED[:2], record(run_id="x", precision="fp16", n_invalid=0)])
    folded, words = history.constant_columns(history.leaderboard(mixed, best_per_model=False))
    assert folded == {"Prompt", "Rows scored"}  # Prompt: not recorded for any line
    legacy = frame_of(
        [record(precision=None, n_invalid=0, macro_f1=None, kappa=None, source="legacy")]
    )
    folded, words = history.constant_columns(history.leaderboard(legacy))
    assert {"Macro-F1", "Cohen's κ"} <= folded  # nothing in them
    assert words == [
        "weights not recorded",
        "59 rows scored",
        "no invalid answers",
    ]
    # The full-history table also folds Excluded.
    folded, words = history.constant_columns(frame_of(SEED[:4]), excluded=True)
    assert "Excluded" in folded


def test_prompt_word_and_prompt_variants_on_the_leaderboard():
    assert history.prompt_word(None, "abc") == "–"
    assert history.prompt_word(0, "abc") == "Preset"
    assert history.prompt_word(1, "3f2a91d0") == "Edited · 3f2a91"
    preset = record(run_id="aaaaaa11", template_edited=0, accuracy=0.6, macro_f1=0.5)
    edited = record(run_id="bbbbbb22", template_edited=1, accuracy=0.7, macro_f1=0.6)
    again = record(run_id="bbbbbb22", template_edited=1, accuracy=0.65, macro_f1=0.55)
    frame = frame_of([preset, edited, again])
    best = history.leaderboard(frame, best_per_model=True)
    # One line per prompt of the same model, each counting its own runs.
    assert list(best["prompt_word"]) == ["Edited · bbbbbb", "Preset"]
    assert list(best["runs"]) == [2, 1]
    folded, _ = history.constant_columns(best)
    assert "Prompt" not in folded


def test_skip_reason_says_saved_outputs():
    reason = "abc.parquet: unreadable checkpoint (ArrowInvalid: bad file)."
    assert history.skip_reason(reason) == (
        "abc.parquet: unreadable saved-outputs file (ArrowInvalid: bad file)."
    )


def test_history_table_keeps_fractions():
    table = history.history_table(frame_of(SEED[:2]))
    assert list(table["Model"]) == ["Qwen2.5-7B-Instruct", "Qwen2.5-3B-Instruct"]
    assert table["Accuracy"].tolist() == pytest.approx([0.82, 0.60])
    assert {"Date", "Task", "Dataset", "Gold column", "Run id"} <= set(table.columns)


# ==========================================================================
# Page
# ==========================================================================


@pytest.fixture
def start_run(monkeypatch):
    """Starts a run that stays in progress until the test ends (its model
    blocks on an event). Call it before or after opening the page."""
    release = threading.Event()
    jobs = []

    def slow(prompt: str) -> str:
        release.wait(30)
        return "positive"

    monkeypatch.setattr(
        resources,
        "default_manager",
        lambda: ModelManager(lambda mid, p: FakeBackend(model_id=mid, responder=slow)),
    )

    def start():
        spec = models.find_model(models.DEMO_REPO)
        config = RunConfig.build(spec, SENTIMENT.template, 4)
        job = resources.get_job_manager().submit(JobRequest(config, ["a", "b"], batch_size=1))
        jobs.append(job)
        return job

    yield start
    release.set()
    for job in jobs:
        job.wait(30)


@pytest.mark.ui
def test_empty_state():
    at = open_history()
    no_exceptions(at)
    assert "No evaluations yet" in texts(at)
    assert "Pick a gold-label column on the Run page" in texts(at, "caption")
    assert not at.dataframe
    assert not url(at)


@pytest.mark.ui
def test_leaderboard_defaults_to_latest_comparable_view():
    seed(resources.get_store())
    at = open_history()
    no_exceptions(at)
    # Newest evaluation: feedback.csv / Sentiment / GOLD_SENTIMENT on its rows.
    assert at.selectbox(key=history.K_DATASET).value == "feedback.csv"
    assert at.selectbox(key=history.K_SELECTION).value == "fp-all"
    best = board(at)
    assert list(best["Model"]) == [
        "Qwen2.5-7B-Instruct",
        "Phi-4-mini-Instruct",
        "Qwen2.5-3B-Instruct",  # its best on these rows is r4 (F1 0.72)
    ]
    assert "Rank" not in best.columns  # the sort order is the rank
    assert best["Accuracy"].tolist() == pytest.approx([82.0, 77.0, 78.0])
    assert list(best["Runs"]) == [1, 1, 3]
    # Columns with one value for every run are said once, not repeated.
    assert "All runs here: 4-bit weights · 59 rows scored · 1 invalid answer each." in texts(
        at, "caption"
    )
    assert len(full_table(at)) == 5
    assert "5 evaluations of 7" in texts(at, "caption")

    at.segmented_control(key=history.K_VIEW).set_value(history.VIEW_ALL).run()
    no_exceptions(at)
    assert len(board(at)) == 5


@pytest.mark.ui
def test_newest_evaluation_is_in_the_default_view():
    """Most runs were on all rows, but the newest one scored the first 20:
    the page opens on the newest run's rows, so it is on the leaderboard."""
    newest = record(
        created_at="2026-10-09T09:00:00",
        run_id="new",
        model_id="ibm-granite/granite-4.0-1b",
        model_label="Granite-4.0-1B",
        dataset_fingerprint="fp-first20",
        n_rows=20,
        accuracy=0.5,
        macro_f1=0.4,
    )
    seed(resources.get_store(), [*SEED, newest])
    at = open_history()
    no_exceptions(at)
    assert at.selectbox(key=history.K_SELECTION).value == "fp-first20"
    assert "Granite-4.0-1B" in list(board(at)["Model"])
    assert "new" in list(full_table(at)["Run id"])


@pytest.mark.ui
def test_all_datasets_warns_about_mixed_scores():
    seed(resources.get_store())
    at = open_history()
    at.selectbox(key=history.K_DATASET).set_value(history.ALL).run()
    # The task filter keeps its value (Sentiment is in both views) ...
    assert at.selectbox(key=history.K_TASK).value == "Sentiment"
    assert len(full_table(at)) == len(SEED) - 1
    # ... until it is cleared too.
    at.selectbox(key=history.K_TASK).set_value(history.ALL).run()
    at.selectbox(key=history.K_GOLD).set_value(history.ALL).run()
    no_exceptions(at)
    assert "This view mixes 2 datasets" in texts(at, "caption")
    # No row-selection filter without a dataset; every line is listed.
    assert not [s for s in at.selectbox if s.key == history.K_SELECTION]
    assert len(full_table(at)) == len(SEED)
    assert len(board(at)) == 3  # three models
    assert "Model quality" in [s.value for s in at.subheader]
    assert "Macro-F1 over time" in [s.value for s in at.subheader]
    assert url(at) == {"dataset": "all", "task": "all", "gold": "all"}


@pytest.mark.ui
def test_filters_read_from_the_url():
    seed(resources.get_store())
    at = open_history({"dataset": "other.csv"})
    no_exceptions(at)
    assert at.selectbox(key=history.K_DATASET).value == "other.csv"
    assert at.selectbox(key=history.K_TASK).options == ["All tasks", "Intent"]
    shown = board(at)
    assert list(shown["Task"]) == ["Intent"]
    # Filters the link left out are filled in, so the URL names the whole view.
    assert url(at) == {
        "dataset": "other.csv",
        "task": "Intent",
        "gold": "GOLD_INTENT",
        "rows": "fp-other",
    }


@pytest.mark.ui
def test_a_link_keeps_meaning_the_same_view():
    store = resources.get_store()
    seed(store)
    at = open_history()
    at.selectbox(key=history.K_SELECTION).set_value("fp-first20").run()
    link = url(at)
    assert link == {
        "dataset": "feedback.csv",
        "task": "Sentiment",
        "gold": "GOLD_SENTIMENT",
        "rows": "fp-first20",
    }
    # Later, a newer evaluation on other data becomes the default view ...
    store.log_evaluation(
        record(created_at="2026-10-11T09:00:00", run_id="later", dataset="third.csv")
    )
    # ... but the link still opens the view it was copied from.
    again = open_history(link)
    no_exceptions(again)
    assert again.selectbox(key=history.K_DATASET).value == "feedback.csv"
    assert again.selectbox(key=history.K_SELECTION).value == "fp-first20"
    assert list(full_table(again)["Run id"]) == ["r5"]
    assert url(again) == link


@pytest.mark.ui
def test_unknown_url_value_falls_back_to_the_newest_view():
    seed(resources.get_store())
    at = open_history({"dataset": "garbage.csv"})
    no_exceptions(at)
    assert at.selectbox(key=history.K_DATASET).value == "feedback.csv"
    assert "mixes" not in texts(at, "caption")
    assert url(at)["dataset"] == "feedback.csv"


@pytest.mark.ui
def test_latest_evaluation_outside_chosen_filters_is_offered():
    store = resources.get_store()
    seed(store)
    at = open_history()
    at.selectbox(key=history.K_DATASET).set_value("other.csv").run()
    assert "latest evaluation" not in texts(at, "caption")
    # A run finishes (say in another tab) on data outside the chosen view.
    store.log_evaluation(record(created_at="2026-10-11T09:00:00", run_id="fresh"))
    at.run()
    no_exceptions(at)
    assert at.selectbox(key=history.K_DATASET).value == "other.csv"  # the choice stands
    note = texts(at, "caption").replace("\\", "")  # unescape Markdown
    assert "The latest evaluation, Qwen2.5-3B-Instruct · Sentiment on feedback.csv" in note
    assert "isn't in this view" in note
    at.button(key="history.show_latest").click().run()
    no_exceptions(at)
    assert at.selectbox(key=history.K_DATASET).value == "feedback.csv"
    assert "fresh" in list(full_table(at)["Run id"])
    assert "latest evaluation" not in texts(at, "caption")


@pytest.mark.ui
def test_delete_selected_evaluations():
    store = resources.get_store()
    seed(store)
    at = open_history()
    assert delete_button(at).disabled
    first_two = set(full_table(at)["Run id"].iloc[:2])

    select(at, [0, 1])
    at.run()
    delete = delete_button(at)
    assert delete.label == "Delete 2 evaluations"
    assert not delete.disabled
    delete.click().run()
    no_exceptions(at)
    assert "Remove 2 evaluations from history?" in texts(at)
    at.button(key="history.delete_confirm").click().run()
    no_exceptions(at)

    remaining = store.list_evaluations()
    assert len(remaining) == len(SEED) - 2
    assert not first_two & set(remaining["run_id"])
    # The table is rebuilt with nothing selected.
    assert len(full_table(at)) == 3
    assert delete_button(at).disabled


@pytest.mark.ui
def test_selection_follows_evaluations_when_rows_change():
    store = resources.get_store()
    seed(store)
    at = open_history()
    picked = list(full_table(at)["Run id"].iloc[:2])  # r7, r4
    select(at, [0, 1])
    at.run()
    assert delete_button(at).label == "Delete 2 evaluations"

    # A new evaluation lands at the top of this view (another tab finished a run).
    store.log_evaluation(record(created_at="2026-10-11T09:00:00", run_id="fresh"))
    at.run()
    no_exceptions(at)
    assert list(full_table(at)["Run id"].iloc[:3]) == ["fresh", *picked]
    delete = delete_button(at)
    assert delete.label == "Delete 2 evaluations"
    delete.click().run()
    no_exceptions(at)
    dialog = texts(at).replace("\\", "")  # unescape Markdown
    assert dialog.count("Qwen2.5-3B-Instruct · Sentiment") == 2
    assert "10 Oct" not in dialog  # "fresh" (logged for 11 Oct) wasn't picked ...
    assert "7 Oct 2026" in dialog and "4 Oct 2026" in dialog  # ... r7 and r4 were
    at.button(key="history.delete_confirm").click().run()
    assert set(picked).isdisjoint(store.list_evaluations()["run_id"])
    assert "fresh" in set(store.list_evaluations()["run_id"])


@pytest.mark.ui
def test_view_follows_new_evaluations_but_not_under_a_selection():
    store = resources.get_store()
    seed(store)
    at = open_history()
    assert at.selectbox(key=history.K_DATASET).value == "feedback.csv"
    # No filter chosen yet: a newer evaluation elsewhere becomes the view.
    store.log_evaluation(record(created_at="2026-10-11T09:00:00", run_id="n1", dataset="third.csv"))
    at.run()
    assert at.selectbox(key=history.K_DATASET).value == "third.csv"
    # With rows ticked the view stays put; the newcomer is offered in a note.
    select(at, [0])
    at.run()
    assert delete_button(at).label == "Delete 1 evaluation"
    store.log_evaluation(
        record(created_at="2026-10-12T09:00:00", run_id="n2", dataset="fourth.csv")
    )
    at.run()
    no_exceptions(at)
    assert at.selectbox(key=history.K_DATASET).value == "third.csv"
    assert delete_button(at).label == "Delete 1 evaluation"
    assert "isn't in this view" in texts(at, "caption")


@pytest.mark.ui
def test_selection_is_dropped_when_a_filter_hides_it():
    seed(resources.get_store())
    at = open_history()
    select(at, [0, 1])
    at.run()
    assert delete_button(at).label == "Delete 2 evaluations"
    at.selectbox(key=history.K_DATASET).set_value("other.csv").run()
    no_exceptions(at)
    delete = delete_button(at)
    assert delete.disabled and delete.label == "Delete selected"
    # Back on the first view, nothing is ticked behind the user's back either.
    at.selectbox(key=history.K_DATASET).set_value("feedback.csv").run()
    assert delete_button(at).disabled


@pytest.mark.ui
def test_delete_cancel_keeps_history():
    store = resources.get_store()
    seed(store)
    at = open_history()
    select(at, [0])
    at.run()
    delete_button(at).click().run()
    at.button(key="history.delete_cancel").click().run()
    no_exceptions(at)
    assert len(store.list_evaluations()) == len(SEED)


@pytest.mark.ui
def test_delete_is_disabled_while_a_run_is_in_progress(start_run):
    store = resources.get_store()
    seed(store)
    start_run()
    at = open_history()
    no_exceptions(at)
    select(at, [0])
    at.run()
    delete = delete_button(at)
    assert delete.disabled
    assert delete.help == "Wait for the current run to finish."


@pytest.mark.ui
def test_delete_confirm_rechecks_for_a_run(start_run):
    """The dialog was opened while idle, then a run started before Delete."""
    store = resources.get_store()
    seed(store)
    at = open_history()
    select(at, [0])
    at.run()
    delete_button(at).click().run()
    assert "Remove 1 evaluation from history?" in texts(at)
    start_run()
    at.button(key="history.delete_confirm").click().run()
    no_exceptions(at)
    assert len(store.list_evaluations()) == len(SEED)
    # The dialog shows this on its own rerun (AppTest reruns the whole page).
    assert "A run is in progress" in at.session_state[history._K_DELETE_MSG]


@pytest.mark.ui
def test_legacy_import_card(tmp_path):
    results_dir = tmp_path / "results"
    v1_id = write_legacy(results_dir)
    at = open_history()
    no_exceptions(at)
    assert "Found 1 saved run and an evaluation history from the previous version" in texts(at)

    at.button(key="history.legacy_import").click().run()
    no_exceptions(at)
    store = resources.get_store()
    assert store.get_meta("legacy_imported_at")
    evaluations = store.list_evaluations()
    assert len(evaluations) == 3
    assert set(evaluations["source"]) == {"legacy"}
    runs = store.list_runs()
    assert len(runs) == 1 and runs.iloc[0]["n_outputs"] == 5
    assert "Imported 1 saved run (5 outputs) and 3 evaluations" in texts(at)
    assert "Not imported · 1" in [e.label for e in at.expander]
    assert "0123456789abcdef" in texts(at)  # the orphan checkpoint, with its reason
    assert "checkpoint" not in texts(at)
    assert (results_dir / f"{v1_id}.parquet").exists()  # files are left alone

    # Version-1 lines have no dataset or gold column: "Show imported scores"
    # points the filters there, one task at a time (never Intent ranked
    # against Sentiment).
    at.button(key="history.import_show").click().run()
    no_exceptions(at)
    assert at.selectbox(key=history.K_DATASET).value == history.UNKNOWN
    assert at.selectbox(key=history.K_TASK).value == "Intent"  # the newest v1 line
    shown = board(at)
    assert list(shown["Model"]) == ["Phi-4-mini-Instruct"]
    assert set(shown["Source"]) == {"v1"}
    assert url(at) == {"dataset": "not-recorded", "task": "Intent", "gold": "not-recorded"}
    at.selectbox(key=history.K_TASK).set_value(history.ALL).run()
    assert "This view mixes 2 tasks" in texts(at, "caption")
    # The link to that view opens it again.
    again = open_history({"dataset": "not-recorded", "task": "Sentiment"})
    assert again.selectbox(key=history.K_DATASET).value == history.UNKNOWN
    assert set(board(again)["Model"]) == {"Qwen2.5-3B-Instruct", "Llama-3.2-3B-Instruct"}

    at.button(key="history.import_dismiss").click().run()
    no_exceptions(at)
    assert "previous version" not in texts(at)


@pytest.mark.ui
def test_legacy_card_not_now(tmp_path):
    write_legacy(tmp_path / "results")
    at = open_history()
    at.button(key="history.legacy_hide").click().run()
    no_exceptions(at)
    assert "previous version" not in texts(at)
    assert resources.get_store().get_meta("legacy_imported_at") is None


# ---- final review fixes ----------------------------------------------------------


def test_same_named_model_folders_are_told_apart():
    """Fine-tune folders often share a name (checkpoint-500, final): each
    gets its own row label, colour and chart line."""
    folders = ["/work/runs/a/tinymodel", "/work/runs/other/tinymodel", "/work/third/tinymodel"]
    records = [
        record(
            id=i + 1,
            run_id=f"run-{i}",
            model_id=path,
            model_label="tinymodel",
            macro_f1=0.5 + i / 10,
            accuracy=0.5 + i / 10,
        )
        for i, path in enumerate(folders)
    ]
    records.append(record(id=9, run_id="run-q"))  # a catalogue model keeps its name
    frame = frame_of(records)
    names = dict(zip(frame["model_key"], frame["model"], strict=True))
    assert names == {
        "/work/runs/a/tinymodel": "a/tinymodel",
        "/work/runs/other/tinymodel": "other/tinymodel",
        "/work/third/tinymodel": "third/tinymodel",
        "Qwen/Qwen2.5-3B-Instruct": "Qwen2.5-3B-Instruct",
    }
    ranked = history.leaderboard(frame)
    assert len(ranked) == 4 and ranked["model"].is_unique


def test_repeated_v1_log_lines_are_merges_not_skips():
    from batchtextreview.store import LegacyImportReport

    merged = (
        "evals.csv: 3 rows repeat an earlier evaluation of the same run (version 1 logged it "
        "again on every rerun) - kept once, at its first time."
    )
    orphan = "0123456789abcdef.parquet: no manifest - not imported."
    report = LegacyImportReport(runs_imported=3, evaluations_imported=3, skipped=[merged])
    assert history.import_notes(report) == ([], [merged])
    note = history.merged_note([merged])
    assert note is not None and note.startswith("3 repeated log lines from version 1 were merged")
    both = LegacyImportReport(skipped=[orphan, merged])
    assert history.import_notes(both) == ([orphan], [merged])
    assert history.merged_note([]) is None


@pytest.mark.ui
def test_legacy_import_of_repeated_evaluations_reports_nothing_lost(tmp_path):
    """Version 1 logged an evaluation again on every rerun: importing merges
    the repeats, and the card must not call them items that couldn't be
    imported."""
    results_dir = tmp_path / "results"
    write_legacy(results_dir)
    (results_dir / "0123456789abcdef.parquet").unlink()  # no orphan this time
    evals = pd.read_csv(results_dir / "evals.csv", keep_default_na=False)
    pd.concat([evals, evals.iloc[[0]]]).to_csv(results_dir / "evals.csv", index=False)
    at = open_history()
    at.button(key="history.legacy_import").click().run()
    no_exceptions(at)
    captions = texts(at, "caption")
    assert "1 repeated log line from version 1 was merged" in captions
    assert "couldn't be imported" not in captions
    assert not [e for e in at.expander if e.label.startswith("Not imported")]
