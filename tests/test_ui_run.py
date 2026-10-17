"""Tests for the Run page (ui/pages/run.py) and the shared UI components.

Pure helpers are unit-tested directly; the page itself is driven with
Streamlit's AppTest in demo mode (``@pytest.mark.ui``). AppTest runs the app
in-process, so ``st.cache_resource`` singletons (store, job manager) are
shared with the test: they are cleared before every test, and the test waits
on the job manager for submitted runs.

The Results page is replaced by a stub here so these tests only depend on the
Run page; it has its own tests.
"""

from __future__ import annotations

import math
import re
import threading
import time
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from batchtextreview import models
from batchtextreview.engine import Progress
from batchtextreview.jobs import JobSnapshot
from batchtextreview.ui import components, state
from batchtextreview.ui.pages import run as run_page

APP = Path(__file__).resolve().parents[1] / "app.py"
TIMEOUT = 30

ROWS = [
    ("101", "Thank you so much, the staff were excellent and very helpful.", "positive", "praise"),
    ("102", "This is not acceptable. My reservation was lost again.", "negative", "complaint"),
    ("103", "Attached is the signed room booking form, as requested.", "neutral", "other"),
    ("104", "", "neutral", "other"),
    ("105", "Can you tell me when my payment will arrive?", "neutral", "query"),
    ("106", "Thank you so much, the staff were excellent and very helpful.", "positive", "praise"),
]


# =============================================================================
# Pure helpers
# =============================================================================


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('  "C:\\Users\\me\\data\\feedback.csv"  ', "C:\\Users\\me\\data\\feedback.csv"),
        ("'/home/me/data'", "/home/me/data"),
        ('""/tmp/x.csv""', "/tmp/x.csv"),
        ('"/tmp/unbalanced.csv', "/tmp/unbalanced.csv"),
        ("   ", ""),
        ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"),
    ],
)
def test_strip_quotes(raw: str, expected: str) -> None:
    assert run_page.strip_quotes(raw) == expected


def test_clean_path_expands_user_and_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BTR_TEST_DIR", str(tmp_path))
    assert run_page.clean_path('"$BTR_TEST_DIR/data.csv"') == tmp_path / "data.csv"
    assert run_page.clean_path("~/x.csv") == Path("~/x.csv").expanduser()
    assert run_page.clean_path('  ""  ') is None


def test_select_rows_modes() -> None:
    frame = pd.DataFrame({"TEXT": [f"t{i}" for i in range(50)]})

    everything = run_page.select_rows(frame, "all", 5)
    assert len(everything.frame) == 50
    assert everything.description == "All 50 rows"

    first = run_page.select_rows(frame, "first", 5)
    assert first.frame["TEXT"].tolist() == ["t0", "t1", "t2", "t3", "t4"]
    assert first.description == "First 5 of 50 rows"

    random_a = run_page.select_rows(frame, "random", 5)
    random_b = run_page.select_rows(frame, "random", 5)
    assert random_a.frame.index.tolist() == random_b.frame.index.tolist()  # seeded
    assert random_a.frame.index.is_monotonic_increasing  # original order kept
    assert len(random_a.frame) == 5
    assert random_a.description == "Random 5 of 50 rows (seed 42)"


def test_select_rows_n_larger_than_table_means_all() -> None:
    frame = pd.DataFrame({"TEXT": ["a", "b"]})
    for mode in ("first", "random"):
        picked = run_page.select_rows(frame, mode, 10)
        assert len(picked.frame) == 2
        assert picked.mode == "all"


def test_progress_text() -> None:
    p = Progress(phase="generating", total_rows=1000, done_rows=312, rows_per_s=4.1, eta_s=170)
    assert run_page.progress_text(p) == "312 / 1,000 rows · 4.1 rows/s · about 2 min 50 s left"
    early = Progress(phase="generating", total_rows=1000, done_rows=8)
    assert run_page.progress_text(early) == "8 / 1,000 rows · estimating time left…"
    assert run_page.progress_text(Progress(phase="loading", total_rows=1)) == "0 / 1 row"


def test_gold_match_counts_rows_resolving_to_labels() -> None:
    golds = pd.Series(["positive", "Negative ", "3.0", None, "angry", "positive"])
    matched, total = run_page._gold_match(golds, ["positive", "negative", "neutral"])
    assert (matched, total) == (3, 6)
    numeric = pd.Series([3.0, 5.0, float("nan")])
    assert run_page._gold_match(numeric, ["1", "2", "3", "4", "5"]) == (2, 3)


def test_ready_badge() -> None:
    assert components.ready_badge([]) == ("Ready", "green")
    assert components.ready_badge(iter(["x"])) == ("Needs attention", "orange")


def _snapshot(state_: str, has_result: bool = True, generated: int = 2) -> JobSnapshot:
    return JobSnapshot(
        id="j1",
        state=state_,  # type: ignore[arg-type]
        progress=Progress(phase="finished", total_rows=3, done_rows=3, generated_rows=generated),
        created_at=datetime(2026, 1, 1),
        started_at=None,
        finished_at=None,
        error=None,
        has_result=has_result,
    )


def test_job_status_words() -> None:
    assert components.job_status(_snapshot("succeeded"))[0] == "Complete"
    assert components.job_status(_snapshot("cancelled"))[0] == "Cancelled"
    assert components.job_status(_snapshot("failed", has_result=True))[0] == "Stopped early"
    assert components.job_status(_snapshot("failed", has_result=False))[0] == "Failed"
    # An aborted run carries a result even when nothing was produced (the
    # model never loaded): that's a failure, not "stopped early".
    assert components.job_status(_snapshot("failed", generated=0))[0] == "Failed"
    assert components.job_status(_snapshot("running"))[0] == "Running"


def test_split_load_error_moves_the_cause_to_details() -> None:
    from zipfile import BadZipFile

    from batchtextreview.data import DataLoadError

    try:
        try:
            raise BadZipFile("File is not a zip file")
        except BadZipFile as cause:
            message = "Couldn't read broken.xlsx as an Excel workbook: File is not a zip file."
            raise DataLoadError(message) from cause
    except DataLoadError as exc:
        plain, details = run_page.split_load_error(exc)
    assert plain == "Couldn't read broken.xlsx as an Excel workbook."
    assert details == "BadZipFile: File is not a zip file"

    # Messages written for people (no embedded cause) stay whole.
    hint = DataLoadError("Couldn't parse x.csv: line 3 has 5 fields. Check the delimiter.")
    assert run_page.split_load_error(hint) == (str(hint), None)


def test_word_problem_avoids_template_jargon() -> None:
    from batchtextreview import presets

    categorise = presets.get_preset("Categorise (set {var})")
    chip, reason = run_page._word_problem(
        "This task uses {var} - fill in the variable first.", categorise
    )
    assert (chip, reason) == ("Enter the categories above", "Enter the categories in step 2.")
    assert run_page._var_label(categorise) == "Categories"
    chip, reason = run_page._word_problem(
        "The prompt template must include the {text} placeholder.", categorise
    )
    assert "{text}" in chip and "{text}" in reason


# =============================================================================
# Components (AppTest.from_function)
# =============================================================================


def _components_app() -> None:
    import streamlit as st

    from batchtextreview.backends.base import DeviceInfo
    from batchtextreview.models import DEMO_MODEL, find_model
    from batchtextreview.ui import components

    components.page_header("Title", "One-line caption.")
    box = components.section("Data", step=1)
    with box:
        st.write("content")
    components.set_section_badge(box, components.ready_badge(["problem"]))
    components.section("Task", step=2, badge=("Ready", "green"))
    components.empty_state(":material/table_chart:", "Nothing yet", "Load a table.")
    components.error_with_details("It broke.", details="Traceback ...", hint="Try again.")
    cpu = DeviceInfo(kind="cpu", name="CPU")
    components.model_badges(find_model("meta-llama/Llama-3.2-3B-Instruct"), cpu)
    components.model_badges(DEMO_MODEL, DeviceInfo(kind="none", name="PyTorch not installed"))


@pytest.mark.ui
def test_components_render() -> None:
    at = AppTest.from_function(_components_app, default_timeout=TIMEOUT).run()
    assert not at.exception
    assert at.header[0].value == "Title"
    assert [s.value for s in at.subheader] == ["1 · Data", "2 · Task"]
    markdown = " ".join(m.value for m in at.markdown)
    assert "Needs attention" in markdown and "Ready" in markdown
    assert "Nothing yet" in markdown
    # The empty state's icon is decorative: hidden from screen readers, and
    # not a heading named after the icon.
    assert not any(m.value.lstrip().startswith("#") for m in at.markdown)
    icons = [h.proto.body for h in at.get("html") if "btr-empty-icon" in h.proto.body]
    assert icons and 'aria-hidden="true"' in icons[0] and ">table_chart<" in icons[0]
    assert "Gated · needs HF_TOKEN" in markdown or "Gated · token set" in markdown
    assert "CPU only" in markdown
    assert "Instant fake outputs" in markdown
    assert "It broke." in at.error[0].value and "Try again." in at.error[0].value
    assert at.expander[0].label == "Technical details"


# =============================================================================
# Run page (AppTest on the real app)
# =============================================================================


def _results_stub() -> None:
    st.write("results stub")


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "data"
    folder.mkdir()
    frame = pd.DataFrame(ROWS, columns=["INDEX", "TEXT", "GOLD_SENTIMENT", "GOLD_INTENT"])
    frame.to_csv(folder / "feedback.csv", index=False)
    return folder


@pytest.fixture
def app(tmp_path: Path, data_dir: Path, monkeypatch: pytest.MonkeyPatch):
    from batchtextreview.ui.pages import results as results_page

    monkeypatch.setenv("BTR_DEMO", "1")
    monkeypatch.setenv("BTR_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("BTR_DATA_DIR", str(data_dir))
    monkeypatch.setattr(results_page, "render", _results_stub)
    st.cache_resource.clear()
    st.cache_data.clear()
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT)
    yield at
    _wait_for_jobs()
    st.cache_resource.clear()


def _job_manager():
    from batchtextreview.ui import resources

    return resources.get_job_manager()


def _wait_for_jobs() -> None:
    try:
        jobs = _job_manager()
    except Exception:
        return
    active = jobs.active()
    if active is not None:
        active.cancel()
        active.wait(10)


def _load_feedback(at: AppTest, data_dir: Path) -> AppTest:
    at.run()
    at.segmented_control(key=run_page.SOURCE_KEY).set_value("path").run()
    at.selectbox(key=run_page.FOLDER_FILE_KEY).set_value(str(data_dir / "feedback.csv")).run()
    return at


def _by_label(widgets, label: str):
    return next(w for w in widgets if w.label == label)


def _markdown(at: AppTest) -> str:
    return "\n".join(m.value for m in at.markdown) + "\n".join(c.value for c in at.caption)


@pytest.mark.ui
def test_run_page_renders_empty(app: AppTest) -> None:
    app.run()
    assert not app.exception
    assert app.header[0].value == "Run"
    assert [s.value for s in app.subheader][:3] == ["1 · Data", "2 · Task", "Run plan"]
    assert "No table loaded yet" in _markdown(app)
    # Nothing is wrong yet: a calm badge, not "Needs attention".
    assert "No data yet" in _markdown(app)
    assert "Needs attention" not in _markdown(app)
    button = app.button(key=run_page.RUN_BUTTON_KEY)
    assert button.disabled
    assert "Load a table first" in _markdown(app)
    # demo mode puts the demo model first
    assert app.selectbox(key=run_page.MODEL_KEY).value == "demo://fake"


@pytest.mark.ui
def test_load_from_folder_and_run(app: AppTest, data_dir: Path) -> None:
    at = _load_feedback(app, data_dir)
    assert not at.exception
    text = _markdown(at)
    assert "feedback.csv" in text and "6 rows" in text and "4 columns" in text
    # The context strip (rendered before the page) shows the table in the same run.
    assert "feedback.csv · 6 rows" in text and "No data loaded" not in text
    assert _by_label(at.selectbox, "Text column").value == "TEXT"
    assert _by_label(at.selectbox, "ID column").value == "INDEX"
    # Sentiment is the default task; its answers match GOLD_SENTIMENT.
    assert _by_label(at.selectbox, "Gold labels").value == "GOLD_SENTIMENT"
    assert "6 / 6 gold values match the allowed answers" in text
    assert "1 blank text" in text and "1 duplicate text" in text

    button = at.button(key=run_page.RUN_BUTTON_KEY)
    assert not button.disabled
    assert button.label == "Generate 5 rows"
    button.click().run()
    assert not at.exception
    assert "results stub" not in _markdown(at)

    job_id = at.session_state[state.SESSION_JOBS][0]
    assert at.session_state[state.ACTIVE_JOB] == job_id
    job = _job_manager().get(job_id)
    assert job is not None and job.wait(15)
    assert job.snapshot().state == "succeeded"

    context = job.request.context[state.CONTEXT_KEY]
    assert isinstance(context, state.RunContext)
    assert context.dataset_name == "feedback.csv"
    assert context.text_column == "TEXT"
    assert context.id_column == "INDEX"
    assert context.gold_column == "GOLD_SENTIMENT"
    assert context.labels == ["positive", "negative", "mixed", "neutral"]
    assert context.output_kind == "label"
    assert context.var is None
    assert context.selection == "All 6 rows"
    assert len(context.frame) == 6
    assert job.request.batch_size == 8
    assert job.request.on_finish is not None

    # The finished run is noticed once - in the click's rerun when the demo
    # model was quick enough, else on the next run - with a toast. The page
    # stays where it is: someone may be typing the next prompt.
    toasts = [t.value for t in at.toast]
    if not any(t.startswith("Run complete") for t in toasts):
        at.run()
        assert not at.exception
        toasts = [t.value for t in at.toast]
    assert any(
        t.startswith("Run complete: 6 rows") and t.endswith("Open results to see them.")
        for t in toasts
    ), toasts
    assert "results stub" not in _markdown(at)
    at.run()
    assert not at.exception
    # The same run just finished: the button opens it instead of adding an
    # identical copy, and the Last run card doesn't repeat that button.
    assert at.button(key=run_page.RUN_BUTTON_KEY).label == "Open results"
    assert "Complete" in _markdown(at)  # last-run card
    assert not [b for b in at.button if b.key == "run.open_results"]
    at.button(key=run_page.RUN_BUTTON_KEY).click().run()
    assert "results stub" in _markdown(at)
    assert at.session_state[state.ACTIVE_JOB] == job_id
    # Back to the Run page. From Streamlit 1.66, AppTest's next run stays on
    # the page st.switch_page opened, so open Run explicitly.
    at._page_hash = _page_hash("run")
    at.run()
    assert not at.exception
    assert "results stub" not in _markdown(at)
    # Any change to the setup makes it a new run: saved outputs are reused.
    at.slider(key=run_page.BATCH_KEY).set_value(4).run()
    assert at.button(key=run_page.RUN_BUTTON_KEY).label == "Show saved results"
    assert at.button(key="run.open_results")


def _on_results(at: AppTest) -> bool:
    on_results = "results stub" in _markdown(at)
    if on_results:
        assert any("Run complete: 6 rows" in t.value for t in at.toast)
    return on_results


def _page_hash(url_path: str) -> str:
    """The script hash of an ``st.navigation`` page (callable pages hash
    their url_path) - setting ``AppTest._page_hash`` to it opens that page."""
    from streamlit.util import calc_hash

    return calc_hash(url_path)


@pytest.mark.ui
def test_pasted_file_path_with_quotes(app: AppTest, data_dir: Path) -> None:
    app.run()
    app.segmented_control(key=run_page.SOURCE_KEY).set_value("path").run()
    app.text_input(key=run_page.FOLDER_KEY).input(f'  "{data_dir / "feedback.csv"}" ').run()
    assert not app.exception
    assert "feedback.csv" in _markdown(app)
    assert not app.button(key=run_page.RUN_BUTTON_KEY).disabled


@pytest.mark.ui
def test_unreadable_file_shows_error(app: AppTest, data_dir: Path) -> None:
    (data_dir / "broken.xlsx").write_bytes(b"not really a workbook")
    app.run()
    app.segmented_control(key=run_page.SOURCE_KEY).set_value("path").run()
    app.selectbox(key=run_page.FOLDER_FILE_KEY).set_value(str(data_dir / "broken.xlsx")).run()
    assert not app.exception
    assert "broken.xlsx" in app.error[0].value
    assert "zip" not in app.error[0].value  # the low-level reason is in the details
    assert [e.label for e in app.expander if e.label == "Technical details"]
    button = app.button(key=run_page.RUN_BUTTON_KEY)
    assert button.disabled
    assert button.help == "Fix the file problem in step 1."


@pytest.mark.ui
def test_upload(app: AppTest, data_dir: Path) -> None:
    app.run()
    payload = (data_dir / "feedback.csv").read_bytes()
    app.file_uploader(key=run_page.UPLOAD_KEY).upload("upload.csv", payload, "text/csv").run()
    assert not app.exception
    assert "upload.csv" in _markdown(app)
    assert app.session_state[state.LOADED_TABLE].name == "upload.csv"
    assert app.session_state[state.DATA_SOURCE] == "upload"


@pytest.mark.ui
def test_variable_task_needs_categories(app: AppTest, data_dir: Path) -> None:
    at = _load_feedback(app, data_dir)
    at.pills(key="run.preset::Classify").set_value("Categorise (set {var})").run()
    button = at.button(key=run_page.RUN_BUTTON_KEY)
    assert button.disabled
    assert button.help == "Enter the categories in step 2."
    assert "Enter the categories above" in _markdown(at)
    assert "Scorable" not in _markdown(at)  # no answers yet
    # Picking the gold column suggests its values as categories.
    _by_label(at.selectbox, "Gold labels").set_value("GOLD_INTENT").run()
    categories = at.multiselect(key="var::Categorise (set {var})")
    assert set(categories.options) >= {"praise", "complaint", "other", "query"}
    categories.set_value(["praise", "complaint", "other", "query"]).run()
    assert not at.exception
    assert not at.button(key=run_page.RUN_BUTTON_KEY).disabled
    assert "6 / 6 gold values match" in _markdown(at)
    assert "Scorable" in _markdown(at)


@pytest.mark.ui
def test_open_ended_task_offers_exact_match(app: AppTest, data_dir: Path) -> None:
    at = _load_feedback(app, data_dir)
    at.segmented_control(key=run_page.CATEGORY_KEY).set_value("Generate").run()
    assert _by_label(at.selectbox, "Gold labels").value == run_page.NO_COLUMN
    assert not [c for c in at.checkbox if c.label == "Score by exact match"]
    _by_label(at.selectbox, "Gold labels").set_value("GOLD_INTENT").run()
    exact = [c for c in at.checkbox if c.label == "Score by exact match"]
    # Choosing a gold column for open-ended output means "score against it",
    # as in version 1 and the command line; it can still be switched off.
    assert exact and exact[0].value is True
    assert "compared with the gold column by exact match" in _markdown(at)
    exact[0].uncheck().run()
    assert "Tick “Score by exact match” in step 2" in _markdown(at)


@pytest.mark.ui
def test_template_edit_survives_preset_switch(app: AppTest, data_dir: Path) -> None:
    at = _load_feedback(app, data_dir)
    edited = "Is this text happy? Answer positive or negative.\n\n{text}"
    at.text_area(key="tmpl::Sentiment").input(edited).run()
    assert "Edited" in _markdown(at)
    at.pills(key="run.preset::Classify").set_value("Emotion").run()
    at.pills(key="run.preset::Classify").set_value("Sentiment").run()
    assert at.text_area(key="tmpl::Sentiment").value == edited
    at.button(key="run.reset::Sentiment").click().run()
    assert "{text}" in at.text_area(key="tmpl::Sentiment").value
    assert "Edited" not in _markdown(at)


@pytest.mark.ui
def test_template_without_text_placeholder_blocks_run(app: AppTest, data_dir: Path) -> None:
    at = _load_feedback(app, data_dir)
    at.text_area(key="tmpl::Sentiment").input("No placeholder here").run()
    button = at.button(key=run_page.RUN_BUTTON_KEY)
    assert button.disabled
    assert "{text}" in (button.help or "")


@pytest.mark.ui
def test_row_scope_first_n(app: AppTest, data_dir: Path) -> None:
    at = _load_feedback(app, data_dir)
    scope = next(s for s in at.segmented_control if s.label == "Rows")
    scope.set_value("first").run()
    _by_label(at.number_input, "Number of rows").set_value(2).run()
    assert not at.exception
    assert at.button(key=run_page.RUN_BUTTON_KEY).label == "Generate 2 rows"


@pytest.mark.ui
def test_save_and_delete_preset(app: AppTest, data_dir: Path) -> None:
    at = _load_feedback(app, data_dir)
    at.button(key="run.save_preset").click().run()  # no name yet
    assert not at.exception
    assert any("Enter a name for the preset" in e.value for e in at.error)
    assert at.segmented_control(key=run_page.CATEGORY_KEY).value == "Classify"

    at.text_input(key=run_page.PRESET_NAME_KEY).input("Triage")
    at.button(key="run.save_preset").click().run()
    assert not at.exception
    assert at.segmented_control(key=run_page.CATEGORY_KEY).value == "Saved"
    assert at.pills(key="run.preset::Saved").value == "Triage"

    at.button(key="run.delete_preset").click().run()
    at.button(key="run.delete_confirm").click().run()
    assert not at.exception
    assert "Saved" not in at.segmented_control(key=run_page.CATEGORY_KEY).options


@pytest.mark.ui
def test_custom_model_needs_a_value(app: AppTest, data_dir: Path) -> None:
    at = _load_feedback(app, data_dir)
    at.selectbox(key=run_page.MODEL_KEY).set_value("__custom__").run()
    assert at.button(key=run_page.RUN_BUTTON_KEY).disabled
    at.text_input(key=run_page.CUSTOM_MODEL_KEY).input('"demo://fake"').run()
    assert not at.exception
    assert not at.button(key=run_page.RUN_BUTTON_KEY).disabled


# ---- running, finishing and failing runs -------------------------------------


@pytest.fixture
def hold(app: AppTest) -> Iterator[threading.Event]:
    """Pass to ``_use_backend(..., hold=hold)``: every model call then waits
    until the test sets it, so the page can be checked mid-run however slow
    the AppTest runs are (a sleep-paced run can end before a slow run even
    draws it). Set at teardown too, before ``app`` waits for the jobs."""
    event = threading.Event()
    yield event
    event.set()


def _use_backend(
    monkeypatch: pytest.MonkeyPatch, hold: threading.Event | None = None, **fake_kwargs
) -> None:
    """Serve the demo model from a FakeBackend with ``fake_kwargs`` (e.g. a
    per-batch delay, or a load failure via ``load_error``); with ``hold``,
    each model call first waits for that event (see the ``hold`` fixture)."""
    from batchtextreview.backends import manager as manager_module
    from batchtextreview.backends.fake import FakeBackend, demo_responder

    if hold is not None:
        respond = fake_kwargs.pop("responder", demo_responder)

        def held(prompt: str) -> str:
            hold.wait(TIMEOUT)
            return respond(prompt)

        fake_kwargs["responder"] = held
    load_error = fake_kwargs.pop("load_error", None)

    def factory(model_id: str, precision: str) -> FakeBackend:
        if load_error is not None:
            raise load_error
        return FakeBackend(model_id=model_id, precision="none", **fake_kwargs)

    monkeypatch.setattr(manager_module, "_DEFAULT", manager_module.ModelManager(factory))


def _start_slow_run(at: AppTest, data_dir: Path) -> str:
    """Load the table, one row per batch, click Generate; returns the job id."""
    at = _load_feedback(at, data_dir)
    at.slider(key=run_page.BATCH_KEY).set_value(1).run()
    at.button(key=run_page.RUN_BUTTON_KEY).click().run()
    assert not at.exception
    return at.session_state[state.SESSION_JOBS][0]


@pytest.mark.ui
def test_progress_panel_and_cancel(
    app: AppTest, data_dir: Path, monkeypatch: pytest.MonkeyPatch, hold: threading.Event
) -> None:
    _use_backend(monkeypatch, hold=hold)
    job_id = _start_slow_run(app, data_dir)
    job = _job_manager().get(job_id)
    assert job is not None and not job.snapshot().done
    deadline = time.monotonic() + 5
    while job.snapshot().progress.phase != "generating" and time.monotonic() < deadline:
        time.sleep(0.05)
    app.run()

    # The progress panel replaces the plan: a status that stays "running"
    # (leaving a ``with st.status`` block would flip it to "complete").
    assert len(app.status) == 1
    assert app.status[0].state == "running"
    assert app.status[0].label == "Generating"
    assert [m.label for m in app.metric] == ["Generated", "Reused", "Errors"]
    assert "Includes 1 blank row, skipped without a model call." in _markdown(app)
    assert "Run in progress" in _markdown(app)  # the strip's static chip
    cancel = app.button(key="run.cancel")
    assert cancel.label == "Cancel run"

    cancel.click().run()
    assert not app.exception
    hold.set()  # batch 1 ends only now: the cancel always lands before batch 2
    assert job.wait(15)
    assert job.snapshot().state == "cancelled"
    app.run()  # the first run to see the end: a toast, and the page stays on Run
    assert not app.exception
    assert any(
        t.value.startswith("Run cancelled after") and t.value.endswith("in the Last run card.")
        for t in app.toast
    )
    # People cancel to change something here: no switch to Results.
    assert "results stub" not in _markdown(app)
    assert "Cancelled" in _markdown(app)  # last-run card
    assert "not generated (run cancelled)" in _markdown(app)


@pytest.mark.ui
def test_editing_during_a_run_keeps_the_user_on_the_run_page(
    app: AppTest, data_dir: Path, monkeypatch: pytest.MonkeyPatch, hold: threading.Event
) -> None:
    """Someone preparing the next run while one finishes isn't pulled away."""
    _use_backend(monkeypatch, hold=hold)
    job_id = _start_slow_run(app, data_dir)
    assert "Changes here apply to your next run." in _markdown(app)
    app.slider(key=run_page.BATCH_KEY).set_value(2).run()  # an edit mid-run
    assert not app.exception
    hold.set()
    job = _job_manager().get(job_id)
    assert job is not None and job.wait(15) and job.snapshot().state == "succeeded"
    app.run()
    assert not app.exception
    assert any(t.value.endswith("Open results to see them.") for t in app.toast)
    assert "results stub" not in _markdown(app)
    assert app.button(key="run.open_results")


@pytest.mark.ui
def test_run_finishing_on_another_page_toasts_without_switching(
    app: AppTest, data_dir: Path, monkeypatch: pytest.MonkeyPatch, hold: threading.Event
) -> None:
    from batchtextreview.ui.pages import history as history_page

    monkeypatch.setattr(history_page, "render", lambda: st.write("history stub"))
    _use_backend(monkeypatch, hold=hold)
    job_id = _start_slow_run(app, data_dir)
    app._page_hash = _page_hash("history")
    app.run()
    assert "history stub" in _markdown(app)
    hold.set()
    job = _job_manager().get(job_id)
    assert job is not None and job.wait(15)

    app.run()
    assert not app.exception
    assert any(t.value.startswith("Run complete: 6 rows") for t in app.toast)
    assert "history stub" in _markdown(app)
    assert "results stub" not in _markdown(app)
    assert app.session_state[state.ACTIVE_JOB] == job_id


@pytest.mark.ui
def test_failed_model_load_shows_failed_card(
    app: AppTest, data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from batchtextreview.backends.base import ModelLoadError

    _use_backend(monkeypatch, load_error=ModelLoadError("Not enough GPU memory to load it."))
    at = _load_feedback(app, data_dir)
    at.button(key=run_page.RUN_BUTTON_KEY).click().run()
    job = _job_manager().get(at.session_state[state.SESSION_JOBS][0])
    assert job is not None and job.wait(15)
    assert job.snapshot().state == "failed" and job.result is not None
    toasts = [t.value for t in at.toast]  # the click's rerun may have seen the end
    if not toasts:
        at.run()  # the toast, on the Run page: a failed run doesn't switch pages
        toasts = [t.value for t in at.toast]
    assert "Run failed before producing any output. Details are in the Last run card." in toasts
    assert "results stub" not in _markdown(at)
    at.run()
    assert not at.exception
    text = _markdown(at)
    assert "Failed" in text and "Stopped early" not in text
    # The same wording as the Results page's card (analysis.describe_failure).
    card = next(e.value for e in at.error if "couldn't be loaded" in e.value)
    assert "The model couldn't be loaded, so no rows were generated." in card
    assert "Not enough GPU memory to load it\\." in card  # the backend's own message
    assert "Technical details" in [e.label for e in at.expander]
    assert at.button(key="run.open_results")  # the aborted run still has results


@pytest.mark.ui
def test_gold_column_that_does_not_fit_warns(app: AppTest, data_dir: Path) -> None:
    at = _load_feedback(app, data_dir)
    _by_label(at.selectbox, "Gold labels").set_value("GOLD_INTENT").run()
    assert not at.exception
    text = _markdown(at)
    assert "Gold column may not fit this task" in text
    assert "0 / 6 gold values match the allowed answers" in text
    assert "Needs attention" in text  # the Data section's badge
    assert "None of the 6 gold values match" in text  # under the run button
    assert not at.button(key=run_page.RUN_BUTTON_KEY).disabled  # a warning only


@pytest.mark.ui
def test_plan_notices_deleted_saved_outputs(app: AppTest, data_dir: Path) -> None:
    from batchtextreview.ui import resources

    at = _load_feedback(app, data_dir)
    at.button(key=run_page.RUN_BUTTON_KEY).click().run()
    job = _job_manager().get(at.session_state[state.SESSION_JOBS][0])
    assert job is not None and job.wait(15)
    at.run()
    at.run()
    assert at.button(key=run_page.RUN_BUTTON_KEY).label == "Open results"

    resources.get_store().delete_run(job.request.config.run_id)  # e.g. from Settings
    at.run()
    assert not at.exception
    assert at.button(key=run_page.RUN_BUTTON_KEY).label == "Generate 5 rows"


# =============================================================================
# Final review fixes
# =============================================================================


def _path_to(at: AppTest, key: str) -> tuple[int, ...] | None:
    """Where the widget keyed ``key`` sits in the element tree (child
    indexes from the root): the browser keeps a widget - and text still
    being typed into it - only while this stays the same."""

    def walk(node, path):
        if getattr(node, "key", None) == key and not getattr(node, "children", None):
            return path
        for index, child in (getattr(node, "children", None) or {}).items():
            found = walk(child, (*path, index))
            if found is not None:
                return found
        return None

    return walk(at._tree, ())


@pytest.mark.ui
def test_prompt_editor_keeps_its_place_when_a_run_ends(
    app: AppTest, data_dir: Path, monkeypatch: pytest.MonkeyPatch, hold: threading.Event
) -> None:
    """When a run ends the page reruns: if anything above the prompt editor
    appeared or vanished, the browser would rebuild it and lose text still
    being typed. And the page never switches away by itself."""
    _use_backend(monkeypatch, hold=hold)
    job_id = _start_slow_run(app, data_dir)
    assert "Changes here apply to your next run." in _markdown(app)
    during = _path_to(app, "tmpl::Sentiment")
    model_during = _path_to(app, run_page.MODEL_KEY)
    assert during is not None and model_during is not None
    hold.set()
    job = _job_manager().get(job_id)
    assert job is not None and job.wait(15)
    app.run()
    assert not app.exception
    assert "Changes here apply to your next run." not in _markdown(app)
    assert any(t.value.endswith("Open results to see them.") for t in app.toast)
    assert "results stub" not in _markdown(app)
    assert _path_to(app, "tmpl::Sentiment") == during
    assert _path_to(app, run_page.MODEL_KEY) == model_during


@pytest.mark.ui
def test_open_results_after_a_retry_opens_the_retry(app: AppTest, data_dir: Path) -> None:
    """A retry ran exactly the original's setup: "Open results" opens the
    retry, never the original it replaced."""
    import dataclasses

    at = _load_feedback(app, data_dir)
    at.button(key=run_page.RUN_BUTTON_KEY).click().run()
    original_id = at.session_state[state.SESSION_JOBS][0]
    original = _job_manager().get(original_id)
    assert original is not None and original.wait(15)
    at.run()
    assert at.button(key=run_page.RUN_BUTTON_KEY).label == "Open results"
    # What the Results page's retry button does (see test_ui_results).
    context = {**original.request.context, state.RETRY_OF: original_id}
    retry = _job_manager().submit(dataclasses.replace(original.request, context=context))
    assert retry.wait(15)
    at.session_state[state.SESSION_JOBS] = [retry.id, original_id]
    setups = dict(at.session_state[state.JOB_SETUPS])
    setups[retry.id] = setups[original_id]
    at.session_state[state.JOB_SETUPS] = setups
    at.run()
    at.button(key=run_page.RUN_BUTTON_KEY).click().run()
    assert "results stub" in _markdown(at)
    assert at.session_state[state.ACTIVE_JOB] == retry.id


def test_finish_toast_mentions_failed_rows() -> None:
    from types import SimpleNamespace

    def job(errors: int):
        stats = SimpleNamespace(errors=errors)
        return SimpleNamespace(result=SimpleNamespace(stats=stats))

    snap = _snapshot("succeeded")
    message, icon = components._finish_message(job(2), snap)
    assert message.startswith("Run complete: 3 rows in")
    assert "2 rows failed: retry them on the Results page." in message
    assert icon == ":material/warning:"
    message, icon = components._finish_message(job(0), snap)
    assert message.endswith(".") and "failed" not in message
    assert icon == ":material/check_circle:"


def test_progress_text_never_says_about_under() -> None:
    p = Progress(phase="generating", total_rows=62, done_rows=50, rows_per_s=168.2, eta_s=0.4)
    assert run_page.progress_text(p) == "50 / 62 rows · 168.2 rows/s · under 1 s left"


def _step(frame: pd.DataFrame):
    from types import SimpleNamespace

    return SimpleNamespace(frame=frame, text_col="TEXT", id_col="INDEX")


def test_default_gold_needs_a_gold_like_name() -> None:
    """A 1-3 TIER column fits Urgency's 1-5 answers, but nobody said it holds
    the correct answers: no gold column is chosen (version 1 chose none)."""
    from batchtextreview import presets

    frame = pd.DataFrame(
        {
            "INDEX": [str(i) for i in range(12)],
            "TEXT": [f"text number {i}" for i in range(12)],
            "GOLD_SENTIMENT": ["positive", "negative", "neutral"] * 4,
            "TIER": ["1", "2", "3"] * 4,
            "CONTESTED": ["0", "1"] * 6,
        }
    )
    urgency = presets.get_preset("Urgency (1-5)")
    sentiment = presets.get_preset("Sentiment")
    assert urgency is not None and sentiment is not None
    labels = urgency.resolve_labels(None)
    assert run_page._default_gold(_step(frame), urgency, labels) is None
    assert (
        run_page._default_gold(_step(frame), sentiment, sentiment.resolve_labels(None))
        == "GOLD_SENTIMENT"
    )
    named = frame.assign(URGENCY=["1", "4", "5"] * 4)
    assert run_page._default_gold(_step(named), urgency, labels) == "URGENCY"
    # An exported results file: its LABEL column is an earlier run's answers.
    exported = frame.drop(columns="GOLD_SENTIMENT").assign(
        OUTPUT=["positive"] * 12, LABEL=["positive", "negative", "neutral"] * 4
    )
    assert (
        run_page._default_gold(_step(exported), sentiment, sentiment.resolve_labels(None)) is None
    )


def _plan_inputs(model: str, precision: str = "auto"):
    from types import SimpleNamespace

    from batchtextreview import models

    spec = models.find_model(model) or models.custom_model(model)
    no_problems = SimpleNamespace(problems=[])
    plan_settings = SimpleNamespace(model=spec, model_problem=None, precision=precision)
    return no_problems, no_problems, plan_settings


def test_a_fully_saved_run_needs_no_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every row saved (e.g. imported from version 1): nothing is loaded, so
    a gated model, a missing token or 4-bit on a CPU don't block it."""
    from types import SimpleNamespace

    from batchtextreview.backends.base import DeviceInfo
    from batchtextreview.ui import resources

    monkeypatch.setattr(resources, "downloaded", lambda repo: False)
    monkeypatch.setattr(resources, "hf_token_available", lambda: False)
    cpu = DeviceInfo("cpu", "CPU", torch_version="2")
    data_step, task, plan_settings = _plan_inputs("meta-llama/Llama-3.2-3B-Instruct", "4bit")
    saved = SimpleNamespace(n_rows=40, n_blank=0, fully_cached=True)
    todo = SimpleNamespace(n_rows=40, n_blank=0, fully_cached=False)
    assert run_page._first_problem(data_step, task, plan_settings, saved, cpu, None) is None
    blocked = run_page._first_problem(data_step, task, plan_settings, todo, cpu, None)
    assert blocked is not None and blocked.startswith("4-bit weights need an NVIDIA GPU")
    plan_settings.precision = "auto"
    gated = run_page._first_problem(data_step, task, plan_settings, todo, cpu, None)
    assert gated is not None and "is gated" in gated and "hf auth login" in gated
    monkeypatch.setattr(resources, "hf_token_available", lambda: True)
    assert run_page._first_problem(data_step, task, plan_settings, todo, cpu, None) is None


def test_saved_login_token_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    huggingface_hub = pytest.importorskip("huggingface_hub")
    from batchtextreview.ui import resources

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: "hf_saved_by_login")
    resources.hf_token_available.clear()
    try:
        assert resources.hf_token_available() is True
        monkeypatch.setattr(huggingface_hub, "get_token", lambda: None)
        resources.hf_token_available.clear()
        assert resources.hf_token_available() is False
    finally:
        resources.hf_token_available.clear()


def test_model_problems_before_running(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings that are certain to fail at load are flagged before the run."""
    from batchtextreview import models
    from batchtextreview.backends.base import DeviceInfo
    from batchtextreview.ui import resources

    cpu = DeviceInfo("cpu", "CPU", torch_version="2")
    gpu = DeviceInfo("cuda", "RTX", total_memory_gb=8, bitsandbytes=False, torch_version="2")
    folder = tmp_path / "tinymodel"
    folder.mkdir()
    local = models.custom_model(str(folder))
    fp16 = run_page.model_problem(local, "fp16", cpu)
    assert fp16 is not None and fp16.startswith("fp16 weights on the CPU are unusably slow")
    four_bit = run_page.model_problem(local, "4bit", cpu)
    assert four_bit is not None and "an NVIDIA GPU" in four_bit and "fp32" in four_bit
    no_bnb = run_page.model_problem(local, "4bit", gpu)
    assert no_bnb is not None and "bitsandbytes" in no_bnb and "fp16" in no_bnb
    not_a_model = run_page.model_problem(local, "fp32", cpu)
    assert not_a_model is not None and "no config.json" in not_a_model
    (folder / "config.json").write_text("{}")
    assert run_page.model_problem(local, "fp32", cpu) is None
    assert run_page.model_problem(local, "auto", cpu) is None

    qwen = models.find_model("Qwen/Qwen2.5-1.5B-Instruct")
    assert qwen is not None
    monkeypatch.setattr(resources, "downloaded", lambda repo: False)
    monkeypatch.setattr(resources, "offline_mode", lambda: True)
    offline = run_page.model_problem(qwen, "auto", cpu)
    assert offline is not None and "offline mode is on" in offline
    assert run_page._first_download_gb(qwen.repo) is None  # no "downloads about 3 GB"
    monkeypatch.setattr(resources, "offline_mode", lambda: False)
    assert run_page.model_problem(qwen, "auto", cpu) is None
    assert run_page.model_problem(qwen, "auto", DeviceInfo("none", "No torch")) is not None


@pytest.mark.ui
def test_offline_badge(app: AppTest, monkeypatch: pytest.MonkeyPatch) -> None:
    from batchtextreview.ui import resources

    monkeypatch.setattr(resources, "downloaded", lambda repo: False)
    monkeypatch.setattr(resources, "offline_mode", lambda: True)
    app.run()
    app.selectbox(key=run_page.MODEL_KEY).set_value("Qwen/Qwen2.5-1.5B-Instruct").run()
    assert not app.exception
    text = _markdown(app)
    assert "Not downloaded · offline mode" in text
    assert "First run downloads" not in text


@pytest.mark.ui
def test_last_local_model_folder_is_restored(app: AppTest, tmp_path: Path) -> None:
    from batchtextreview.ui import resources

    folder = tmp_path / "my-finetune"
    folder.mkdir()
    (folder / "config.json").write_text("{}")
    resources.get_store().set_setting("last_model", str(folder))
    app.run()
    assert not app.exception
    assert app.selectbox(key=run_page.MODEL_KEY).value == "__custom__"
    assert app.text_input(key=run_page.CUSTOM_MODEL_KEY).value == str(folder)
    assert "Local folder" in _markdown(app)


@pytest.mark.ui
def test_last_model_folder_that_is_gone_is_not_restored(app: AppTest, tmp_path: Path) -> None:
    from batchtextreview.ui import resources

    resources.get_store().set_setting("last_model", str(tmp_path / "deleted-model"))
    app.run()
    assert not app.exception
    assert app.selectbox(key=run_page.MODEL_KEY).value == "demo://fake"


@pytest.mark.ui
def test_discard_saved_outputs_from_the_run_page(app: AppTest, data_dir: Path) -> None:
    """Version 1's "Clear cache", next to the counts it changes."""
    from batchtextreview.ui import resources

    at = _load_feedback(app, data_dir)
    assert not [b for b in at.button if b.key == "run.discard_saved"]  # nothing saved yet
    at.button(key=run_page.RUN_BUTTON_KEY).click().run()
    job = _job_manager().get(at.session_state[state.SESSION_JOBS][0])
    assert job is not None and job.wait(15)
    at.run()
    run_id = job.result.run_id
    assert resources.get_store().count_outputs(run_id) == 4  # one per distinct text
    at.button(key="run.discard_saved").click().run()
    assert not at.exception
    assert "Delete 4 saved outputs for Demo" in _markdown(at)
    at.button(key="settings.delete_run_confirm").click().run()
    assert not at.exception
    assert resources.get_store().count_outputs(run_id) == 0
    at.run()
    assert at.button(key=run_page.RUN_BUTTON_KEY).label == "Generate 5 rows"


# =============================================================================
# Processing order
# =============================================================================


def test_bar_fraction_measures_work_and_falls_back_to_rows() -> None:
    from types import SimpleNamespace

    assert run_page.bar_fraction(SimpleNamespace(fraction=0.1, work_fraction=0.4)) == 0.4
    # No work measure (or a broken one): the share of rows done.
    assert run_page.bar_fraction(SimpleNamespace(fraction=0.1)) == 0.1
    assert run_page.bar_fraction(SimpleNamespace(fraction=0.1, work_fraction=None)) == 0.1
    assert run_page.bar_fraction(SimpleNamespace(fraction=0.1, work_fraction=float("nan"))) == 0.1
    assert run_page.bar_fraction(SimpleNamespace(fraction=0.5, work_fraction=1.2)) == 1.0
    assert run_page.bar_fraction(SimpleNamespace(fraction=-0.5)) == 0.0
    finished = Progress(phase="finished", total_rows=4, done_rows=4)
    assert run_page.bar_fraction(finished) == 1.0


def test_remembered_order_ignores_unknown_values() -> None:
    from types import SimpleNamespace

    from batchtextreview import engine

    def store(value):
        return SimpleNamespace(get_setting=lambda key, default=None: value)

    def broken(key, default=None):
        raise OSError("database is locked")

    assert run_page.remembered_order(store("shortest")) == "shortest"
    assert run_page.remembered_order(store("longest")) == "longest"
    assert run_page.remembered_order(store(None)) == engine.DEFAULT_ORDER
    assert run_page.remembered_order(store("random")) == engine.DEFAULT_ORDER
    assert run_page.remembered_order(store(["shortest"])) == engine.DEFAULT_ORDER
    assert run_page.remembered_order(SimpleNamespace(get_setting=broken)) == engine.DEFAULT_ORDER


def _advanced_caption(at: AppTest) -> str:
    return next(c.value for c in at.caption if c.value.startswith("Up to "))


@pytest.mark.ui
def test_processing_order_control(app: AppTest, data_dir: Path) -> None:
    from batchtextreview import engine

    at = _load_feedback(app, data_dir)
    assert not at.exception
    order = at.segmented_control(key=run_page.ORDER_KEY)
    assert order.label == "Processing order"
    assert order.options == ["Longest first", "Shortest first"]
    assert order.value == engine.DEFAULT_ORDER == "longest"
    assert order.help == run_page.ORDER_HELP
    assert "runs out of memory fails in the first batch" in order.help
    assert _advanced_caption(at) == "Up to 8 output tokens · batch size 8 · longest first"

    order.set_value("shortest").run()
    assert not at.exception
    assert _advanced_caption(at) == "Up to 8 output tokens · batch size 8 · shortest first"


@pytest.mark.ui
def test_processing_order_is_remembered(app: AppTest, data_dir: Path) -> None:
    from batchtextreview.ui import resources

    at = _load_feedback(app, data_dir)
    at.segmented_control(key=run_page.ORDER_KEY).set_value("shortest").run()
    assert resources.get_store().get_setting(run_page.ORDER_SETTING) == "shortest"
    # A later session (another tab, or after restarting the app) starts
    # from the last choice.
    later = AppTest.from_file(str(APP), default_timeout=TIMEOUT).run()
    assert not later.exception
    assert later.segmented_control(key=run_page.ORDER_KEY).value == "shortest"
    assert "batch size 8 · shortest first" in _advanced_caption(later)
    # A stored value that isn't an order (any more) is ignored.
    resources.get_store().set_setting(run_page.ORDER_SETTING, "random")
    fresh = AppTest.from_file(str(APP), default_timeout=TIMEOUT).run()
    assert not fresh.exception
    assert fresh.segmented_control(key=run_page.ORDER_KEY).value == "longest"


@pytest.mark.ui
def test_processing_order_goes_to_the_job(app: AppTest, data_dir: Path) -> None:
    at = _load_feedback(app, data_dir)
    at.button(key=run_page.RUN_BUTTON_KEY).click().run()
    first = _job_manager().get(at.session_state[state.SESSION_JOBS][0])
    assert first is not None and first.wait(15)
    assert first.request.order == "longest"
    assert first.request.context[state.CONTEXT_KEY].extra["order"] == "longest"
    at.run()
    at.run()
    assert at.button(key=run_page.RUN_BUTTON_KEY).label == "Open results"

    # Another order is another setup (the button no longer opens the run
    # that just finished), but the same run id: every row is saved.
    at.segmented_control(key=run_page.ORDER_KEY).set_value("shortest").run()
    button = at.button(key=run_page.RUN_BUTTON_KEY)
    assert button.label == "Show saved results"
    button.click().run()
    assert not at.exception
    second = _job_manager().get(at.session_state[state.SESSION_JOBS][0])
    assert second is not None and second.id != first.id and second.wait(15)
    assert second.request.order == "shortest"
    assert second.request.context[state.CONTEXT_KEY].extra["order"] == "shortest"
    assert second.request.config.run_id == first.request.config.run_id


@pytest.mark.ui
def test_progress_bar_measures_work(
    app: AppTest, data_dir: Path, monkeypatch: pytest.MonkeyPatch, hold: threading.Event
) -> None:
    """The bar shows the share of the work done (so it moves with the time
    estimate), while the text beside it still counts rows."""
    monkeypatch.setattr(Progress, "work_fraction", property(lambda self: 0.37))
    _use_backend(monkeypatch, hold=hold)  # the run stays in batch 1 throughout
    job_id = _start_slow_run(app, data_dir)
    job = _job_manager().get(job_id)
    assert job is not None
    deadline = time.monotonic() + 5
    while job.snapshot().progress.phase != "generating" and time.monotonic() < deadline:
        time.sleep(0.05)
    app.run()
    assert not app.exception
    bars = app.get("progress")
    assert len(bars) == 1
    assert bars[0].proto.value == 37
    assert re.match(r"\d / 6 rows", bars[0].proto.text)


# ---- cloud models -------------------------------------------------------------


@pytest.fixture
def no_api_keys(monkeypatch: pytest.MonkeyPatch):
    from batchtextreview import api_keys

    for name in (
        *api_keys.ENV_VARS.values(),
        "OPENAI_BASE_URL",
        *api_keys.ANTHROPIC_OTHER_CREDENTIALS,
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(api_keys, "_keyring_backend", lambda: None)
    api_keys._reset_for_tests()
    yield
    api_keys._reset_for_tests()


HAIKU = "anthropic:claude-haiku-5-5"


@pytest.mark.ui
def test_cloud_model_without_a_key_cannot_run(app: AppTest, data_dir: Path, no_api_keys) -> None:
    pytest.importorskip("anthropic")
    at = _load_feedback(app, data_dir)
    assert "Claude Haiku 5.5 · Anthropic API" in at.selectbox(key=run_page.MODEL_KEY).options
    at.selectbox(key=run_page.MODEL_KEY).set_value(HAIKU).run()
    assert not at.exception
    text = _markdown(at)
    assert "Anthropic API" in text and "No API key" in text
    assert "$0.10 / $0.50 per M tokens" in text
    button = at.button(key=run_page.RUN_BUTTON_KEY)
    assert button.disabled
    assert "No Anthropic API key is set" in text
    assert at.segmented_control(key=run_page.PRECISION_KEY).disabled
    assert "requests at once" in text and "weights" not in text.split("requests at once")[1][:40]


@pytest.mark.ui
def test_cloud_model_asks_before_sending_the_texts(
    app: AppTest, data_dir: Path, monkeypatch: pytest.MonkeyPatch, no_api_keys
) -> None:
    pytest.importorskip("anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key-0000")
    _use_backend(monkeypatch)  # no request leaves the test
    at = _load_feedback(app, data_dir)
    at.selectbox(key=run_page.MODEL_KEY).set_value(HAIKU).run()
    assert not at.exception
    text = _markdown(at)
    assert "Key from ANTHROPIC_API_KEY" in text
    # 6 rows: 1 blank and 1 duplicate, so 4 distinct texts are sent.
    assert "Sends 4 texts to Anthropic: about" in text and "at list prices" in text
    button = at.button(key=run_page.RUN_BUTTON_KEY)
    assert button.disabled and "Tick “Send these texts to Anthropic”" in text
    consent = at.checkbox(key=f"{run_page.API_CONSENT_KEY}::anthropic")
    assert consent.value is False
    consent.check().run()
    assert not at.button(key=run_page.RUN_BUTTON_KEY).disabled
    at.button(key=run_page.RUN_BUTTON_KEY).click().run()
    assert not at.exception
    job = _job_manager().get(at.session_state[state.SESSION_JOBS][0])
    assert job is not None and job.wait(15) and job.snapshot().state == "succeeded"
    assert job.request.config.model_id == HAIKU
    assert job.request.config.precision == "none"


def test_api_usage_estimate_counts_texts_not_saved_yet() -> None:
    from batchtextreview import engine

    spec = models.find_model(HAIKU)
    assert spec is not None
    config = engine.RunConfig.build(spec, "Label: {text}", 8)
    texts = ["abcd" * 10, "", "abcd" * 10, "xy" * 50]
    plan = engine.plan_run(config, texts, None)
    calls, sent, answers = run_page.api_usage_estimate(plan, config)
    assert calls == 2 and answers == 16
    # "Label: " (7 characters) + the text, at 4 characters a token.
    assert sent == math.ceil((7 + 40 + 7 + 100) / run_page.CHARS_PER_TOKEN)
