"""AppTest coverage for the Results page.

Jobs are created directly through the shared JobManager (``ui.resources``)
with the demo model, the same way the Run page submits them; AppTest runs the
app in-process, so the page sees the same singletons.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from batchtextreview import models, presets
from batchtextreview.engine import Progress, RunConfig, RunResult, RunStats
from batchtextreview.ui import analysis, nav, resources, state
from batchtextreview.ui import format as fmt
from batchtextreview.ui.pages import results as results_page

pytestmark = pytest.mark.ui

APP = Path(__file__).resolve().parents[1] / "app.py"
TIMEOUT = 30

ROWS = [
    ("1001", "Thank you so much, the staff were excellent and very helpful.", "positive", "1", "0"),
    (
        "1002",
        "This is not acceptable, my reservation has been lost again.",
        "negative",
        "1",
        "0",
    ),
    ("1003", "Attached is the signed room booking form, as requested.", "neutral", "1", "0"),
    ("1004", "", "neutral", "1", "0"),
    ("1005", "Great, another week of waiting. Just what I needed.", "negative", "3", "1"),
    ("1006", "The payment arrived but the letter was confusing.", "mixed", "2", "1"),
    ("1007", "Can you tell me when my refund will be processed?", "neutral", "2", "0"),
    ("1008", "Brilliant help from the call centre today, thank you.", "positive", "1", "0"),
]


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame(ROWS, columns=["INDEX", "TEXT", "GOLD_SENTIMENT", "TIER", "CONTESTED"])


@pytest.fixture(autouse=True)
def demo_env(tmp_path, monkeypatch):
    monkeypatch.setenv("BTR_DEMO", "1")
    monkeypatch.setenv("BTR_RESULTS_DIR", str(tmp_path / "results"))
    st.cache_resource.clear()
    yield
    job = resources.get_job_manager().active()
    if job is not None:
        job.cancel()
        job.wait(10)
    st.cache_resource.clear()


def _request(**kwargs: Any):
    from batchtextreview.jobs import JobRequest

    names = {f.name for f in dataclasses.fields(JobRequest)}
    if "on_finish" not in names:
        kwargs.pop("on_finish", None)
    return JobRequest(**kwargs)


def make_context(frame: pd.DataFrame, preset: presets.Preset, **overrides: Any) -> state.RunContext:
    from batchtextreview.data import frame_fingerprint

    values: dict[str, Any] = {
        "name": state.run_name(preset.title, "feedback.csv"),
        "dataset_name": "feedback.csv",
        "dataset_fingerprint": frame_fingerprint(frame),
        "frame": frame,
        "text_column": "TEXT",
        "preset_key": preset.key,
        "preset_title": preset.title,
        "template_raw": preset.template,
        "template_edited": False,
        "var": None,
        "labels": preset.resolve_labels(None),
        "output_kind": preset.output_kind,
        "selection": f"All {len(frame)} rows",
        "id_column": "INDEX",
        "gold_column": "GOLD_SENTIMENT",
    }
    values.update(overrides)
    return state.RunContext(**values)


def submit(
    frame: pd.DataFrame, preset_key: str = "Sentiment", order: str | None = None, **overrides: Any
):
    """Run the demo model over ``frame`` like the Run page does; wait for it.
    ``order``: the processing order (the request's default when None)."""
    preset = presets.get_preset(preset_key)
    assert preset is not None
    ctx = make_context(frame, preset, **overrides)
    config = RunConfig.build(models.DEMO_MODEL, preset.template, preset.max_new_tokens)
    manager = resources.get_job_manager()
    job = manager.submit(
        _request(
            config=config,
            texts=frame["TEXT"].tolist(),
            batch_size=4,
            context={state.CONTEXT_KEY: ctx},
            on_finish=analysis.finish_hook(resources.get_store()),
            **({"order": order} if order is not None else {}),
        )
    )
    assert job.wait(TIMEOUT), "demo job did not finish"
    return job


def open_results(job_ids: list[str] | None = None, **state_values: Any) -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT)
    if job_ids:
        at.session_state[state.SESSION_JOBS] = list(job_ids)
        at.session_state[state.ACTIVE_JOB] = job_ids[0]
    for key, value in state_values.items():
        at.session_state[key] = value
    at.run()
    # st.navigation pages built from callables have no file to switch_page()
    # to; point the test session at the Results page's hash directly.
    at._page_hash = nav.page("results")._script_hash
    return at.run()


def texts(at: AppTest, kind: str = "markdown") -> str:
    return "\n".join(str(e.value) for e in getattr(at, kind))


def no_exceptions(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]


# ---- tests ---------------------------------------------------------------------


def test_empty_state():
    at = open_results()
    no_exceptions(at)
    assert "No results yet" in texts(at)
    assert any(b.label == "Go to the Run page" for b in at.button)


def test_label_run_outputs(frame):
    job = submit(frame)
    at = open_results([job.id])
    no_exceptions(at)
    ctx = job.request.context[state.CONTEXT_KEY]
    assert fmt.md(ctx.name) in texts(at)  # markdown-escaped in the header
    filters = at.segmented_control(key=f"results.filter::{job.id}")
    assert filters.options[0] == "All · 8"
    assert any(o.startswith("Blank · 1") for o in filters.options)
    table = at.dataframe(key=f"results.table::{job.id}::0")
    assert len(table.value) == 8
    # Every download sits in one Export menu in the header.
    assert [b.label for b in at.download_button] == [
        "All rows · CSV",
        "All rows · Excel",
        "Filtered view · CSV (8 rows)",
    ]
    # Unscored rows read blank, scored ones as words (never booleans).
    words = table.value["CORRECT"]
    assert set(words.dropna()) <= {"Correct", "Wrong"}
    assert words.isna().sum() == 1  # the blank row
    # Columns that only repeat what another says are left out by default:
    # every row was simply generated (the blank one has its own filter).
    assert "STATUS" not in table.value.columns
    built = analysis.get_analysis(job)
    assert ("OUTPUT" in table.value.columns) == (not built.output_matches_label())

    filters.set_value(next(o for o in filters.options if o.startswith("Blank"))).run()
    no_exceptions(at)
    assert len(at.dataframe(key=f"results.table::{job.id}::0").value) == 1
    # The Export menu's filtered view follows the filter.
    assert "Filtered view · CSV (1 row)" in [b.label for b in at.download_button]

    filters = at.segmented_control(key=f"results.filter::{job.id}")
    filters.set_value("All · 8").run()
    at.text_input(key=f"results.search::{job.id}").set_value("REFUND").run()
    no_exceptions(at)
    shown = at.dataframe(key=f"results.table::{job.id}::0").value
    assert list(shown["INDEX"]) == ["1007"]


def test_results_survive_reruns(frame):
    job = submit(frame)
    at = open_results([job.id])
    at.run()
    at.run()
    no_exceptions(at)
    assert len(at.dataframe(key=f"results.table::{job.id}::0").value) == 8


def test_evaluation_tab(frame):
    job = submit(frame)
    at = open_results([job.id], **{"results.tabs": "Evaluation"})
    no_exceptions(at)
    labels = [m.label for m in at.metric]
    assert labels[:5] == ["Accuracy", "Macro-F1", "Cohen's κ", "Valid answers", "Coverage"]
    assert "left out: 1 blank text" in texts(at, "caption")
    assert "Saved to history" in texts(at, "caption")
    # No earlier run to compare with: κ reads as an agreement band.
    kappa = next(m for m in at.metric if m.label == "Cohen's κ")
    assert kappa.delta and "agreement" in kappa.delta
    assert "would score" in texts(at, "caption")  # the majority-class baseline
    per_label = next(d.value for d in at.dataframe if "precision" in d.value.columns)
    # Undefined precision / recall read "–", never the grid's "None".
    assert all(v == "–" or v.endswith("%") for v in per_label["precision"])
    assert all(v == "–" or v.endswith("%") for v in per_label["recall"])
    history = resources.get_store().list_evaluations()
    assert len(history) == 1
    assert history.iloc[0]["n_scored"] == 7
    # Slicing and the flag-column exclusion toggle.
    assert at.selectbox(key=f"results.slice::{job.id}").value == "TIER"
    toggle = at.toggle(key=f"results.exclude::{job.id}::CONTESTED")
    toggle.set_value(True).run()
    no_exceptions(at)
    captions = texts(at, "caption")
    # Rows left out by the toggle are counted on their own, not as "excluded".
    assert "5 rows scored · left out: 1 blank text" in captions
    assert "2 scored rows with CONTESTED = 1 left out" in captions
    assert "Excluding CONTESTED = 1" in texts(at)  # next to the numbers it changes
    coverage = next(m for m in at.metric if m.label == "Coverage")
    assert not coverage.delta  # leaving rows out isn't a regression
    accuracy = next(m for m in at.metric if m.label == "Accuracy")
    assert accuracy.delta and "changes are against all rows" in captions
    # History keeps the full evaluation.
    assert len(resources.get_store().list_evaluations()) == 1


def test_details_tab(frame):
    job = submit(frame)
    at = open_results([job.id], **{"results.tabs": "Details"})
    no_exceptions(at)
    assert f"`{job.result.run_id}`" in texts(at)  # inline code in the Run card
    assert "{text}" in texts(at, "code")
    assert [s.value for s in at.subheader][:4] == ["Run", "Rows", "Model", "Data"]
    assert "Not generated" in texts(at)
    assert "Processing order" in texts(at)
    assert "Longest first" in texts(at)  # the default


def test_details_tab_shows_the_processing_order(frame):
    job = submit(frame, order="shortest", extra={"order": "shortest"})
    at = open_results([job.id], **{"results.tabs": "Details"})
    no_exceptions(at)
    lines = texts(at).splitlines()
    at_label = lines.index(":gray[Processing order]")
    assert lines[at_label + 1] == "Shortest first"
    assert "Longest first" not in texts(at)


def test_open_ended_run_is_not_scored(frame):
    job = submit(frame, "Language", labels=None)
    at = open_results([job.id])
    no_exceptions(at)
    assert "Median length" in [m.label for m in at.metric]
    at.session_state["results.tabs"] = "Evaluation"
    at.run()
    no_exceptions(at)
    assert "Open-ended output isn't scored" in texts(at)


def test_row_drawer_navigation(frame):
    job = submit(frame)
    at = open_results([job.id])
    # As if a row had been clicked (a drawer is only opened on this page).
    at.session_state["results.drawer"] = {"job": job.id, "rows": [0, 1, 2], "pos": 0}
    at.run()
    no_exceptions(at)
    assert "Row 1001" in texts(at) and "1 of 3 shown" in texts(at)
    assert ROWS[0][1] in texts(at, "text")
    next_button = next(b for b in at.button if b.label == "Next")
    next_button.click().run()
    no_exceptions(at)
    assert "2 of 3 shown" in texts(at)
    assert next(b for b in at.button if b.label == "Next").proto.shortcut
    assert ROWS[1][1] in texts(at, "text")


def test_several_runs_selectable(frame):
    first = submit(frame)
    second = submit(
        frame.head(4), "Intent", labels=["complaint", "query", "request", "praise", "other"]
    )
    at = open_results([second.id, first.id])
    no_exceptions(at)
    picker = at.selectbox(key="results.run")
    assert picker.value == second.id
    picker.set_value(first.id).run()
    no_exceptions(at)
    assert at.session_state[state.ACTIVE_JOB] == first.id
    assert len(at.dataframe(key=f"results.table::{first.id}::0").value) == 8


# ---- synthetic outcomes (failed / partial) through a fake manager --------------


class FakeJob:
    def __init__(
        self,
        request,
        result: RunResult | None,
        snap_state: str,
        error: str | None = None,
        job_id: str | None = None,
        done_rows: int | None = None,
    ):
        self.id = job_id or f"fake-{snap_state}"
        self.request = request
        self.result = result
        self._state = snap_state
        self._error = error
        self._done_rows = done_rows

    def cancel(self) -> None:
        self._state = "cancelled"

    def wait(self, timeout: float | None = None) -> bool:
        return True

    def snapshot(self):
        from batchtextreview.jobs import JobSnapshot

        total = len(self.request.texts)
        now = datetime.now()
        stats = self.result.stats if self.result is not None else RunStats()
        running = self._state in ("queued", "running")
        done = self._done_rows if self._done_rows is not None else total
        return JobSnapshot(
            id=self.id,
            state=self._state,
            progress=Progress(
                phase="generating" if running else "finished",
                total_rows=total,
                done_rows=done,
                generated_rows=stats.generated,
                cached_rows=stats.cached,
            ),
            created_at=now,
            started_at=now,
            finished_at=None if running else now,
            error=self._error,
            has_result=self.result is not None,
        )


class FakeManager:
    def __init__(self, jobs: list[FakeJob]) -> None:
        self.jobs_by_id = {j.id: j for j in jobs}
        self.submitted: list[Any] = []

    def get(self, job_id):
        return self.jobs_by_id.get(job_id)

    def active(self):
        return next((j for j in self.jobs_by_id.values() if j._state == "running"), None)

    def latest(self):
        return next(iter(self.jobs_by_id.values()), None)

    def jobs(self):
        return list(self.jobs_by_id.values())

    def submit(self, request):
        self.submitted.append(request)
        job = FakeJob(request, None, "running")
        job.id = "fake-retry"
        self.jobs_by_id = {job.id: job, **self.jobs_by_id}  # held, like JobManager
        return job


def test_stopped_early_run_offers_retry(frame, monkeypatch):
    preset = presets.get_preset("Sentiment")
    config = RunConfig.build(models.DEMO_MODEL, preset.template, preset.max_new_tokens)
    outputs = ["positive", "negative", "", "", "", "", "", ""]
    statuses = [
        "generated",
        "generated",
        "error",
        "blank",
        "pending",
        "pending",
        "pending",
        "pending",
    ]
    errors = [None, None, "RuntimeError: CUDA out of memory", None, None, None, None, None]
    result = RunResult(
        run_id=config.run_id,
        config=config,
        outputs=outputs,
        statuses=statuses,
        errors=errors,
        stats=RunStats(total=8, generated=2, blank=1, errors=1, pending=4, elapsed_s=3.0),
        aborted_reason="RuntimeError: CUDA out of memory",
    )
    request = _request(
        config=config,
        texts=frame["TEXT"].tolist(),
        context={state.CONTEXT_KEY: make_context(frame, preset)},
    )
    job = FakeJob(request, result, "failed", error=result.aborted_reason)
    fake = FakeManager([job])
    monkeypatch.setattr(resources, "get_job_manager", lambda: fake)

    at = open_results([job.id])
    no_exceptions(at)
    errors = "\n".join(str(e.value) for e in at.error)
    assert "The run stopped early: 1 row failed, 4 not generated." in errors
    assert "so 4 rows weren't tried" in errors
    assert "CUDA out of memory" in errors and "batch size" in errors  # cause + advice
    assert "Stopped early" in texts(at)  # status badge
    # The header sentence never says "cache" and leaves the reason to the card.
    summary = texts(at, "caption")
    assert (
        "2 of 8 rows have outputs · 2 generated · 1 blank text skipped · 1 failed · "
        "4 not generated (run stopped)"
    ) in summary.replace("\\", "")
    assert "CUDA" not in summary and "cache" not in summary
    # The red card covers the failed row too: no second "1 row failed" card.
    assert "1 row failed** —" not in texts(at)
    assert "Not saved to history" not in texts(at, "caption")  # Outputs tab is open
    retry = next(b for b in at.button if b.label.startswith("Retry"))
    assert retry.label == "Retry 5 rows (1 failed, 4 not tried)"
    retry.click().run()
    assert len(fake.submitted) == 1
    resubmitted = fake.submitted[0]
    assert resubmitted.config == config
    assert resubmitted.texts == request.texts
    new_ctx = resubmitted.context[state.CONTEXT_KEY]
    assert new_ctx is not request.context[state.CONTEXT_KEY]
    assert new_ctx.frame is request.context[state.CONTEXT_KEY].frame
    assert at.session_state[state.ACTIVE_JOB] == "fake-retry"


def test_failed_without_result(frame, monkeypatch):
    preset = presets.get_preset("Sentiment")
    config = RunConfig.build(models.DEMO_MODEL, preset.template, preset.max_new_tokens)
    request = _request(
        config=config,
        texts=frame["TEXT"].tolist(),
        context={state.CONTEXT_KEY: make_context(frame, preset)},
    )
    job = FakeJob(request, None, "failed", error="ValueError: bad things")
    monkeypatch.setattr(resources, "get_job_manager", lambda: FakeManager([job]))
    at = open_results([job.id])
    no_exceptions(at)
    error = "\n".join(str(e.value) for e in at.error)
    assert "failed before producing any output.** Bad things" in error
    assert "Check the settings on the Run page, then retry." in error
    assert "Nothing was lost" not in error and "were generated" not in error
    assert "ValueError: bad things" in texts(at, "code")
    assert any(b.label == "Retry run" for b in at.button)


def test_row_errors_card_and_retry(frame, monkeypatch):
    from batchtextreview.backends.fake import FakeBackend
    from batchtextreview.backends.manager import ModelManager
    from batchtextreview.jobs import JobManager

    def factory(model_id, precision):
        return FakeBackend(
            model_id, precision, fail_when=lambda batch: any("refund" in p for p in batch)
        )

    manager = JobManager(resources.get_store(), ModelManager(factory=factory), history=5)
    monkeypatch.setattr(resources, "get_job_manager", lambda: manager)
    preset = presets.get_preset("Sentiment")
    job = manager.submit(
        _request(
            config=RunConfig.build(models.DEMO_MODEL, preset.template, preset.max_new_tokens),
            texts=frame["TEXT"].tolist(),
            batch_size=1,
            context={state.CONTEXT_KEY: make_context(frame, preset)},
        )
    )
    assert job.wait(TIMEOUT)
    assert job.result.stats.errors == 1

    at = open_results([job.id])
    no_exceptions(at)
    assert "1 row failed" in texts(at)
    filters = at.segmented_control(key=f"results.filter::{job.id}")
    assert "Errors · 1" in filters.options
    retry = next(b for b in at.button if b.label == "Retry 1 failed row")
    retry.click().run()
    new_job = manager.latest()
    assert new_job is not None and new_job.id != job.id
    assert at.session_state[state.ACTIVE_JOB] == new_job.id
    assert new_job.request.texts == job.request.texts
    new_job.wait(TIMEOUT)


def test_retry_keeps_the_processing_order(frame, monkeypatch):
    """Retrying failed rows (or generating the remaining ones) sends them in
    the order the original run used, whatever the Run page is set to now."""
    from batchtextreview.backends.fake import FakeBackend
    from batchtextreview.backends.manager import ModelManager
    from batchtextreview.jobs import JobManager

    def factory(model_id, precision):
        return FakeBackend(
            model_id, precision, fail_when=lambda batch: any("refund" in p for p in batch)
        )

    manager = JobManager(resources.get_store(), ModelManager(factory=factory), history=5)
    monkeypatch.setattr(resources, "get_job_manager", lambda: manager)
    preset = presets.get_preset("Sentiment")
    job = manager.submit(
        _request(
            config=RunConfig.build(models.DEMO_MODEL, preset.template, preset.max_new_tokens),
            texts=frame["TEXT"].tolist(),
            batch_size=1,
            order="shortest",
            context={state.CONTEXT_KEY: make_context(frame, preset, extra={"order": "shortest"})},
        )
    )
    assert job.wait(TIMEOUT)
    at = open_results([job.id], **{"run.order": "longest"})
    no_exceptions(at)
    next(b for b in at.button if b.label == "Retry 1 failed row").click().run()
    no_exceptions(at)
    retry = manager.latest()
    assert retry is not None and retry.id != job.id
    assert retry.request.order == "shortest"
    assert retry.request.context[state.CONTEXT_KEY].extra["order"] == "shortest"
    assert retry.wait(TIMEOUT)
    at.session_state["results.tabs"] = "Details"
    at.run()
    no_exceptions(at)
    assert "Shortest first" in texts(at)


# ---- review fixes ----------------------------------------------------------------


def test_same_minute_runs_get_distinct_picker_labels(frame):
    """Two runs whose labels would match (same model, task, file, minute) must
    stay selectable: the selectbox maps a choice back through its label."""
    name = "Demo (fake outputs) · Sentiment · 10:00"
    first = submit(frame, name=name)
    regraded = frame.assign(GOLD_SENTIMENT=["negative"] * len(frame))
    second = submit(regraded, name=name)
    jobs = [resources.get_job_manager().get(i) for i in (second.id, first.id)]
    labels = results_page._run_labels(jobs)
    assert len(set(labels.values())) == 2
    assert all(label.startswith(name) for label in labels.values())

    at = open_results([second.id, first.id])
    no_exceptions(at)
    picker = at.selectbox(key="results.run")
    assert picker.value == second.id
    assert at.session_state[state.ACTIVE_JOB] == second.id
    assert at.dataframe(key=f"results.table::{second.id}::0") is not None

    picker.set_value(first.id).run()
    no_exceptions(at)
    assert at.selectbox(key="results.run").value == first.id
    assert at.session_state[state.ACTIVE_JOB] == first.id
    assert len(at.dataframe(key=f"results.table::{first.id}::0").value) == 8

    at.selectbox(key="results.run").set_value(second.id).run()
    assert at.session_state[state.ACTIVE_JOB] == second.id


def test_run_labels_prefer_meaningful_differences(frame):
    preset = presets.get_preset("Sentiment")
    config = RunConfig.build(models.DEMO_MODEL, preset.template, preset.max_new_tokens)

    def job(job_id: str, **ctx: Any) -> FakeJob:
        request = _request(
            config=config,
            texts=frame["TEXT"].tolist(),
            context={
                state.CONTEXT_KEY: make_context(
                    frame, preset, name="Demo · Sentiment · 10:00", **ctx
                )
            },
        )
        return FakeJob(request, None, "succeeded", job_id=job_id)

    when = datetime(2026, 10, 12, 10, 0, 5)
    jobs = [
        job("a", selection="First 20 of 62 rows", created_at=when),
        job("b", selection="All 62 rows", created_at=when),
        job("c", created_at=when),
        job("d", created_at=when),
    ]
    labels = results_page._run_labels(jobs)
    assert labels["a"].endswith("· First 20 of 62 rows")
    assert labels["b"].endswith("· All 62 rows")
    # c and d differ in nothing a person would recognise: numbered, oldest first.
    assert labels["c"] == labels["d"] + " · #2"
    assert len(set(labels.values())) == 4


def test_tab_switches_keep_filters_and_toggles(frame):
    job = submit(frame)
    at = open_results([job.id])
    at.text_input(key=f"results.search::{job.id}").set_value("refund").run()
    filters = at.segmented_control(key=f"results.filter::{job.id}")
    filters.set_value(next(o for o in filters.options if o.startswith("Correct"))).run()
    shown = list(at.dataframe(key=f"results.table::{job.id}::0").value["INDEX"])

    at.session_state["results.tabs"] = "Evaluation"
    at.run()
    at.toggle(key=f"results.exclude::{job.id}::CONTESTED").set_value(True).run()
    at.segmented_control(key="results.cm_mode").set_value("% of gold row").run()
    excluded_accuracy = next(m for m in at.metric if m.label == "Accuracy").value

    at.session_state["results.tabs"] = "Outputs"
    at.run()
    no_exceptions(at)
    assert at.text_input(key=f"results.search::{job.id}").value == "refund"
    assert at.segmented_control(key=f"results.filter::{job.id}").value.startswith("Correct")
    assert list(at.dataframe(key=f"results.table::{job.id}::0").value["INDEX"]) == shown

    at.session_state["results.tabs"] = "Evaluation"
    at.run()
    no_exceptions(at)
    assert at.toggle(key=f"results.exclude::{job.id}::CONTESTED").value is True
    assert at.segmented_control(key="results.cm_mode").value == "% of gold row"
    assert next(m for m in at.metric if m.label == "Accuracy").value == excluded_accuracy
    assert "CONTESTED = 1 left out" in texts(at, "caption")

    # A trip to another page and back keeps the open tab and the toggle.
    at._page_hash = nav.page("history")._script_hash
    at.run()
    at._page_hash = nav.page("results")._script_hash
    at.run()
    no_exceptions(at)
    assert at.session_state["results.tabs"] == "Evaluation"
    assert at.toggle(key=f"results.exclude::{job.id}::CONTESTED").value is True


def test_remembered_confusion_cell_filters_examples(frame):
    job = submit(frame)
    built = analysis.get_analysis(job)
    gold, predicted = next(
        (g, p)
        for g, p, ok in zip(
            built.report.rows["gold"],
            built.report.rows["predicted"],
            built.report.rows["correct"],
            strict=True,
        )
        if not ok
    )
    at = open_results(
        [job.id],
        **{"results.tabs": "Evaluation", f"results.cell::{job.id}": (gold, predicted)},
    )
    no_exceptions(at)
    assert f"Gold **{gold}** predicted as" in texts(at)
    # The pair picker shows the remembered cell and clears it.
    pair = at.selectbox(key=f"results.pair::{job.id}")
    assert pair.value == f"{gold}\x1f{predicted}"
    pair.set_value("").run()
    no_exceptions(at)
    assert "Misclassified rows ·" in texts(at)
    assert f"results.cell::{job.id}" not in at.session_state
    assert at.session_state["results.cm_nonce"] == 1


def _load_failure_job(frame: pd.DataFrame) -> FakeJob:
    preset = presets.get_preset("Sentiment")
    config = RunConfig.build(models.DEMO_MODEL, preset.template, preset.max_new_tokens)
    n = len(frame)
    reason = f"Couldn't load {config.model_label}: RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB"
    result = RunResult(
        run_id=config.run_id,
        config=config,
        outputs=[""] * n,
        statuses=["pending"] * n,
        errors=[None] * n,
        stats=RunStats(total=n, pending=n, elapsed_s=1.0),
        aborted_reason=reason,
    )
    request = _request(
        config=config,
        texts=frame["TEXT"].tolist(),
        context={state.CONTEXT_KEY: make_context(frame, preset)},
    )
    return FakeJob(request, result, "failed", error=reason, job_id="loadfail")


def test_run_that_never_loaded_the_model(frame, monkeypatch):
    job = _load_failure_job(frame)
    monkeypatch.setattr(resources, "get_job_manager", lambda: FakeManager([job]))
    at = open_results([job.id])
    no_exceptions(at)
    error = "\n".join(str(e.value) for e in at.error)
    assert "The model couldn't be loaded, so no rows were generated.** CUDA out of memory" in error
    assert "smaller model" in error
    assert "Saved outputs are kept" not in error  # nothing was generated
    badges = texts(at)
    assert "Failed" in badges and "Scored" not in badges
    assert "Predicted labels" not in texts(at)  # no distribution card without answers
    assert not at.get("button_group") or all(
        "Predicted label" not in str(getattr(g, "label", "")) for g in at.get("button_group")
    )
    # Nothing was produced, so "remaining" would mislead: a plain retry.
    assert any(b.label == "Retry 8 rows" for b in at.button)

    at.session_state["results.tabs"] = "Evaluation"
    at.run()
    no_exceptions(at)
    body = texts(at) + texts(at, "caption")
    assert "Nothing to score yet" in body
    assert "gold column belongs" not in body
    assert "Not saved to history" not in body


def test_partial_run_is_not_saved_to_history(frame, monkeypatch):
    preset = presets.get_preset("Sentiment")
    config = RunConfig.build(models.DEMO_MODEL, preset.template, preset.max_new_tokens)
    statuses = ["generated"] * 4 + ["pending"] * 4
    result = RunResult(
        run_id=config.run_id,
        config=config,
        outputs=["positive", "negative", "neutral", "neutral"] + [""] * 4,
        statuses=statuses,
        errors=[None] * 8,
        stats=RunStats(total=8, generated=4, pending=4, elapsed_s=1.0),
        cancelled=True,
    )
    request = _request(
        config=config,
        texts=frame["TEXT"].tolist(),
        context={state.CONTEXT_KEY: make_context(frame, preset)},
    )
    job = FakeJob(request, result, "cancelled", job_id="cancelled")
    monkeypatch.setattr(resources, "get_job_manager", lambda: FakeManager([job]))
    at = open_results([job.id], **{"results.tabs": "Evaluation"})
    no_exceptions(at)
    assert "Not saved to history: this run is incomplete" in texts(at, "caption")
    assert resources.get_store().list_evaluations().empty
    assert any("Cancelled with 4 rows not generated" in str(i.value) for i in at.info)


def test_notice_while_a_run_is_in_progress(frame, monkeypatch):
    preset = presets.get_preset("Sentiment")
    config = RunConfig.build(models.DEMO_MODEL, preset.template, preset.max_new_tokens)
    finished = _load_failure_job(frame)
    running_ctx = make_context(frame, preset, name="Demo · Emotion · 10:05")
    running = FakeJob(
        _request(
            config=config, texts=frame["TEXT"].tolist(), context={state.CONTEXT_KEY: running_ctx}
        ),
        None,
        "running",
        job_id="running",
        done_rows=3,
    )
    monkeypatch.setattr(resources, "get_job_manager", lambda: FakeManager([running, finished]))
    at = open_results([running.id, finished.id])
    no_exceptions(at)
    notice = texts(at)
    assert "is running: 3 of 8 rows done." in notice
    assert "Its results open here when it finishes." in notice
    assert any(link.label == "View progress" for link in at.get("page_link"))
    # The earlier run is still shown below, and the running one stays active.
    assert at.session_state[state.ACTIVE_JOB] == running.id
    assert "The model couldn't be loaded" in "\n".join(str(e.value) for e in at.error)


def test_deltas_against_previous_run_use_blue_and_orange(frame, monkeypatch):
    import zlib

    from streamlit.proto.Metric_pb2 import Metric as MetricProto

    from batchtextreview.backends.fake import FakeBackend, demo_responder
    from batchtextreview.backends.manager import ModelManager
    from batchtextreview.jobs import JobManager

    noisy = {"on": True}

    def responder(prompt: str) -> str:
        if noisy["on"] and zlib.crc32(prompt.encode()) % 2 == 0:
            return "mixed"
        return demo_responder(prompt)

    store = resources.get_store()
    manager = JobManager(
        store, ModelManager(factory=lambda m, p: FakeBackend(m, p, responder=responder)), 5
    )
    monkeypatch.setattr(resources, "get_job_manager", lambda: manager)
    preset = presets.get_preset("Sentiment")

    def run(tokens: int, **ctx: Any):
        job = manager.submit(
            _request(
                config=RunConfig.build(models.DEMO_MODEL, preset.template, tokens),
                texts=frame["TEXT"].tolist(),
                context={state.CONTEXT_KEY: make_context(frame, preset, **ctx)},
                on_finish=analysis.finish_hook(store),
            )
        )
        assert job.wait(TIMEOUT)
        return job

    worse = run(9)
    noisy["on"] = False
    # History keeps whole seconds: start the second run in a later one.
    better = run(10, created_at=datetime.now() + timedelta(seconds=2))
    assert len(store.list_evaluations()) == 2

    at = open_results([better.id, worse.id], **{"results.tabs": "Evaluation"})
    no_exceptions(at)
    accuracy = next(m for m in at.metric if m.label == "Accuracy")
    assert accuracy.delta.startswith("+")
    assert accuracy.proto.color == MetricProto.MetricColor.BLUE
    assert "changes are against the previous run of this task on this data" in texts(at, "caption")

    at.selectbox(key="results.run").set_value(worse.id).run()
    no_exceptions(at)
    # The noisy run started first, so it has no earlier run to compare with.
    assert not next(m for m in at.metric if m.label == "Accuracy").delta


def test_cell_count_treats_missing_and_empty_cells_as_zero():
    matrix = pd.DataFrame([[2, 0], [1, 3]], index=["a", "b"], columns=["a", "b"])
    assert results_page._cell_count(matrix, ("a", "a")) == 2
    assert results_page._cell_count(matrix, ("a", "b")) == 0
    assert results_page._cell_count(matrix, ("zzz", "a")) == 0


# ---- final review fixes ----------------------------------------------------------


def test_viewing_the_evaluation_never_writes_history(frame):
    """The finish hook logs a run once. Opening its Evaluation tab must not
    bring back a line deleted on the History page, nor move its date."""
    job = submit(frame)
    store = resources.get_store()
    history = store.list_evaluations()
    assert len(history) == 1
    entry, stamp = int(history.iloc[0]["id"]), history.iloc[0]["created_at"]
    assert job.request.context[analysis.HISTORY_ID_KEY] == entry

    at = open_results([job.id], **{"results.tabs": "Evaluation"})
    no_exceptions(at)
    assert "Saved to history" in texts(at, "caption")
    assert store.list_evaluations().iloc[0]["created_at"] == stamp

    store.delete_evaluation(entry)
    at.run()
    no_exceptions(at)
    assert store.list_evaluations().empty, "deleted evaluation was written back to history"
    captions = texts(at, "caption")
    assert "Not in history: its line was deleted" in captions
    assert "Saved to history" not in captions
    # A new browser session that only views the run doesn't write either.
    fresh = open_results([job.id], **{"results.tabs": "Evaluation"})
    no_exceptions(fresh)
    assert store.list_evaluations().empty


def test_a_retried_run_does_not_overwrite_the_retry_in_history(frame):
    first = submit(frame)
    store = resources.get_store()
    entry = first.request.context[analysis.HISTORY_ID_KEY]
    # Its retry, finished later, updated the same history line.
    store.log_evaluation({**analysis.history_record(analysis.get_analysis(first)), "n_scored": 99})
    at = open_results([first.id], **{"results.tabs": "Evaluation"})
    no_exceptions(at)
    line = store.list_evaluations()
    assert len(line) == 1 and int(line.iloc[0]["id"]) == entry
    assert int(line.iloc[0]["n_scored"]) == 99  # not put back to this run's 7
    assert "History holds a later run of this configuration (99 rows scored)" in texts(
        at, "caption"
    )


@pytest.mark.parametrize("ends_mid_render", [False, True], ids=["normal", "ends-mid-render"])
def test_retry_follows_the_new_run_here_with_the_same_setup(frame, monkeypatch, ends_mid_render):
    """A retry stays on the Results page (its running card shows progress)
    and carries the original's Run-page setup, so the Run page knows it ran
    exactly the same thing. ``ends_mid_render``: the retry finishes just
    after the page has listed the runs, so the list doesn't have it yet -
    the page must still follow it, not fall back to the original."""
    from batchtextreview.backends.fake import FakeBackend
    from batchtextreview.backends.manager import ModelManager
    from batchtextreview.jobs import JobManager

    gate = {"open": False}

    def factory(model_id, precision):
        return FakeBackend(
            model_id,
            precision,
            fail_when=lambda batch: not gate["open"] and any("refund" in p for p in batch),
            delay_s=0.0,
        )

    manager = JobManager(resources.get_store(), ModelManager(factory=factory), history=5)
    monkeypatch.setattr(resources, "get_job_manager", lambda: manager)
    preset = presets.get_preset("Sentiment")
    job = manager.submit(
        _request(
            config=RunConfig.build(models.DEMO_MODEL, preset.template, preset.max_new_tokens),
            texts=frame["TEXT"].tolist(),
            batch_size=1,
            context={state.CONTEXT_KEY: make_context(frame, preset)},
        )
    )
    assert job.wait(TIMEOUT)
    setup = ("some", "setup")
    at = open_results([job.id], **{state.JOB_SETUPS: {job.id: setup}})
    no_exceptions(at)
    if ends_mid_render:
        list_jobs = results_page._selectable_jobs

        def list_then_finish(jobs_manager):
            listed = list_jobs(jobs_manager)
            running = jobs_manager.active()
            if running is not None:
                running.wait(TIMEOUT)
            return listed

        monkeypatch.setattr(results_page, "_selectable_jobs", list_then_finish)
    gate["open"] = True
    next(b for b in at.button if b.label == "Retry 1 failed row").click().run()
    no_exceptions(at)
    retry = manager.latest()
    assert retry is not None and retry.id != job.id
    assert retry.request.context[state.RETRY_OF] == job.id
    assert at.session_state[state.JOB_SETUPS][retry.id] == setup
    assert at.session_state[state.ACTIVE_JOB] == retry.id
    assert retry.wait(TIMEOUT)
    at.run()
    no_exceptions(at)
    assert at.selectbox(key="results.run").value == retry.id  # its results, here
    assert "1 row failed" not in texts(at)


def test_retry_of_a_run_set_up_elsewhere_matches_no_setup(frame, monkeypatch):
    job = _load_failure_job(frame)
    job.request.context[state.CONTEXT_KEY] = make_context(frame, presets.get_preset("Sentiment"))
    job._error = job.result.aborted_reason = (
        f"Couldn't load {job.request.config.model_label}: RuntimeError: CUDA error: unknown"
    )
    fake = FakeManager([job])
    monkeypatch.setattr(resources, "get_job_manager", lambda: fake)
    at = open_results([job.id])
    no_exceptions(at)
    next(b for b in at.button if b.label.startswith("Retry")).click().run()
    no_exceptions(at)
    recorded = at.session_state[state.JOB_SETUPS]["fake-retry"]
    assert recorded[0] == state.NOT_SET_UP_HERE
    # While it runs, the original's retry button says so.
    assert any(b.label == "Retrying…" and b.disabled for b in at.button)


def test_drawer_of_another_run_does_not_reopen(frame):
    """A drawer that went away without being dismissed (the page switched to
    a run that just finished) must not reopen when its run is picked again."""
    first = submit(frame)
    second = submit(frame.head(4))
    at = open_results([second.id, first.id])
    at.session_state["results.drawer"] = {"job": first.id, "rows": [0, 1], "pos": 0}
    at.run()
    no_exceptions(at)
    assert "results.drawer" not in at.session_state
    at.selectbox(key="results.run").set_value(first.id).run()
    no_exceptions(at)
    assert "1 of 2 shown" not in texts(at)


def test_running_card_follows_the_picker(frame, monkeypatch):
    """Picking another run while one is in progress keeps the page on the
    picked run, so the card no longer promises to open the new one."""
    preset = presets.get_preset("Sentiment")
    config = RunConfig.build(models.DEMO_MODEL, preset.template, preset.max_new_tokens)
    finished = _load_failure_job(frame)
    running = FakeJob(
        _request(
            config=config,
            texts=frame["TEXT"].tolist(),
            context={state.CONTEXT_KEY: make_context(frame, preset, name="Demo · Emotion · 10:05")},
        ),
        None,
        "running",
        job_id="running",
        done_rows=3,
    )
    monkeypatch.setattr(resources, "get_job_manager", lambda: FakeManager([running, finished]))
    at = open_results([running.id, finished.id])
    assert "Its results open here when it finishes." in texts(at)
    at.session_state[state.ACTIVE_JOB] = finished.id  # what picking it does
    at.run()
    no_exceptions(at)
    notice = texts(at)
    assert "It joins the run list when it finishes." in notice
    assert "Its results open here when it finishes." not in notice


def test_settings_failure_offers_to_change_settings(frame, monkeypatch):
    job = _load_failure_job(frame)
    reason = (
        f"Couldn't load {job.request.config.model_label}: fp16 on the CPU is unusably slow "
        "(no GPU was detected). Use fp32 precision on the CPU."
    )
    job._error = job.result.aborted_reason = reason
    monkeypatch.setattr(resources, "get_job_manager", lambda: FakeManager([job]))
    at = open_results([job.id])
    no_exceptions(at)
    assert "fp16 on the CPU is unusably slow" in "\n".join(str(e.value) for e in at.error)
    labels = [b.label for b in at.button]
    assert "Change settings on the Run page" in labels
    assert not [label for label in labels if label.startswith("Retry")]
