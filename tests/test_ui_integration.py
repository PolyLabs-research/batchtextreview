"""End-to-end tests of the whole app (every page, nothing stubbed) in demo
mode, driven with Streamlit's AppTest.

The page-level test files stub the other pages; these walk the user's path
across them: load the sample, run, land on Results, score, then History and
Models & settings - once for a good run and once for runs whose backend
fails (failing batches, a model that can't load), which the demo backend
can't be made to do from the UI.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from batchtextreview.backends import manager as manager_module
from batchtextreview.backends.base import ModelLoadError
from batchtextreview.backends.fake import FakeBackend
from batchtextreview.ui import nav, resources, state
from batchtextreview.ui.pages import run as run_page

APP = Path(__file__).resolve().parents[1] / "app.py"
SAMPLE = Path(__file__).resolve().parents[1] / "examples" / "sample_feedback.csv"
TIMEOUT = 30

pytestmark = [
    pytest.mark.ui,
    pytest.mark.skipif(not SAMPLE.is_file(), reason="examples/sample_feedback.csv missing"),
]


@pytest.fixture
def app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BTR_DEMO", "1")
    monkeypatch.setenv("BTR_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("BTR_DATA_DIR", str(tmp_path))
    # A fresh model manager per test (the process-wide one outlives
    # st.cache_resource and would keep another test's backend loaded).
    monkeypatch.setattr(manager_module, "_DEFAULT", manager_module.ModelManager())
    st.cache_resource.clear()
    st.cache_data.clear()
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT)
    yield at
    jobs = resources.get_job_manager()
    active = jobs.active()
    if active is not None:
        active.cancel()
        active.wait(10)
    st.cache_resource.clear()


def _use_backend(monkeypatch: pytest.MonkeyPatch, **fake_kwargs) -> None:
    load_error = fake_kwargs.pop("load_error", None)

    def factory(model_id: str, precision: str) -> FakeBackend:
        if load_error is not None:
            raise load_error
        return FakeBackend(model_id=model_id, precision="none", **fake_kwargs)

    monkeypatch.setattr(manager_module, "_DEFAULT", manager_module.ModelManager(factory))


def _open(at: AppTest, page: str) -> AppTest:
    """Show ``page`` on the next runs (AppTest can't switch to a page that
    is a function; it opens whatever page hash it is given)."""
    at._page_hash = nav.page(page)._script_hash
    return at.run()


def _text(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption]
    parts += [str(e.value) for e in at.error] + [str(i.value) for i in at.info]
    return "\n".join(parts)


def _run_sample(at: AppTest, preset: tuple[str, str] | None = None) -> tuple[str, list[str]]:
    """Load the built-in sample, click the run button, wait for the job and
    let the app react to its end. Returns the job id and the toasts shown
    (the click's own rerun may already have seen an instant run end).
    ``preset`` = (category, preset key) picks another task than Sentiment."""
    at.run()
    at.segmented_control(key=run_page.SOURCE_KEY).set_value("sample").run()
    assert not at.exception
    if preset is not None:
        category, key = preset
        at.segmented_control(key=run_page.CATEGORY_KEY).set_value(category).run()
        at.pills(key=f"run.preset::{category}").set_value(key).run()
        assert not at.exception
    at.button(key=run_page.RUN_BUTTON_KEY).click().run()
    assert not at.exception
    toasts = [t.value for t in at.toast]
    job_id = at.session_state[state.SESSION_JOBS][0]
    job = resources.get_job_manager().get(job_id)
    assert job is not None and job.wait(20)
    at.run()  # the strip notices the end: toast, switch to Results
    assert not at.exception
    return job_id, toasts + [t.value for t in at.toast]


def test_sample_run_across_every_page(app: AppTest) -> None:
    at = app.run()
    at.segmented_control(key=run_page.SOURCE_KEY).set_value("sample").run()
    gold = next(s for s in at.selectbox if s.label == "Gold labels")
    assert gold.value == "GOLD_SENTIMENT"  # guessed from the task's answers
    assert "61 / 62 gold values match the allowed answers" in _text(at)

    job_id, toasts = _run_sample(at)
    assert any(t.startswith("Run complete: 62 rows") for t in toasts)
    assert at.session_state[state.ACTIVE_JOB] == job_id

    # Results: header, outputs, then the evaluation (logged to history).
    at = _open(at, "results")
    assert not at.exception
    assert at.header[0].value == "Results"
    text = _text(at)
    assert "Complete" in text and "Scored" in text
    assert "61 of 62 rows have outputs · 61 generated · 1 blank text skipped" in text.replace(
        "\\", ""
    )
    assert [b.label for b in at.get("download_button")][:2] == [
        "All rows · CSV",
        "All rows · Excel",
    ]
    at.session_state["results.tabs"] = "Evaluation"
    at.run()
    assert not at.exception
    assert [m.label for m in at.metric][:5] == [
        "Accuracy",
        "Macro-F1",
        "Cohen's κ",
        "Valid answers",
        "Coverage",
    ]
    assert "Saved to history" in _text(at)
    slice_by = next(s for s in at.selectbox if s.label == "Slice by")
    assert slice_by.value == "TIER"

    # History lists the evaluation; Models & settings lists the saved outputs.
    at = _open(at, "history")
    assert not at.exception
    assert [s.value for s in at.subheader][:2] == ["Leaderboard", "Evaluations"]
    assert "No evaluations yet" not in _text(at)
    at = _open(at, "settings")
    assert not at.exception
    assert "No saved outputs yet" not in _text(at)

    # Back on the Run page, the same run opens its results (no duplicate run).
    at = _open(at, "run")
    assert not at.exception
    assert at.button(key=run_page.RUN_BUTTON_KEY).label == "Open results"
    assert "Last run" in [s.value for s in at.subheader]


@pytest.mark.parametrize(
    ("backend", "headline", "preset"),
    [
        (
            {"fail_when": lambda batch: True},
            "The run stopped before generating any rows.",
            None,
        ),
        (
            {"load_error": ModelLoadError("Not enough GPU memory to load it.")},
            "The model couldn't be loaded, so no rows were generated.",
            None,
        ),
        # JSON output with nothing to check used to crash the Outputs tab.
        (
            {"fail_when": lambda batch: True},
            "The run stopped before generating any rows.",
            ("Extract", "Entity extraction"),
        ),
    ],
    ids=["failing-batches", "model-load", "failing-batches-json"],
)
def test_failed_run_renders_on_every_page(
    app: AppTest,
    monkeypatch: pytest.MonkeyPatch,
    backend: dict,
    headline: str,
    preset: tuple[str, str] | None,
) -> None:
    _use_backend(monkeypatch, **backend)
    job_id, toasts = _run_sample(app, preset)
    at = app
    # A failed run stays on the Run page; the toast points at the card.
    assert "Run failed before producing any output. Details are in the Last run card." in toasts
    job = resources.get_job_manager().get(job_id)
    assert job is not None and job.snapshot().state == "failed"

    # Results: the red card (what -> why -> what to do) and a retry.
    at = _open(at, "results")
    assert not at.exception
    errors = "\n".join(str(e.value) for e in at.error)
    assert headline in errors
    assert "Failed" in _text(at)
    assert any(b.label.startswith("Retry ") for b in at.button)
    assert "Technical details" in [e.label for e in at.expander]

    # Run page: the "Last run" card explains it in the same words.
    at = _open(at, "run")
    assert not at.exception
    assert headline in "\n".join(str(e.value) for e in at.error)
    assert at.button(key="run.open_results")

    # A failed run is never logged as an evaluation.
    at = _open(at, "history")
    assert not at.exception
    assert "No evaluations yet" in _text(at)
    at = _open(at, "settings")
    assert not at.exception
