"""Results page: browse, score and export what a run produced.

Everything here renders from the job objects held by the process-wide
JobManager (plus session state for UI choices), never from a button's return
value, so results survive every rerun, download and widget change.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any, Final

import pandas as pd
import streamlit as st

from batchtextreview import engine, models, results
from batchtextreview.labels import INVALID, label_key
from batchtextreview.prompting import render_prompt
from batchtextreview.ui import analysis as an
from batchtextreview.ui import charts, components, nav, resources, state
from batchtextreview.ui import format as fmt

if TYPE_CHECKING:
    from batchtextreview.evaluation import EvalReport
    from batchtextreview.jobs import Job, JobManager, JobSnapshot

# ---- session-state keys (page-local) --------------------------------------
_K_SELECT = "results.run"
_K_TABS = "results.tabs"
_K_TAB_MEMORY = "results.last_tab"
"""Plain copy of the open tab: the tabs widget forgets it on a page switch."""
_K_SHOWN = "results.shown_job"
"""Job id the page showed last (a different run may reset the open tab)."""
_K_DRAWER = "results.drawer"
"""dict(job=..., rows=[table positions in view order], pos=index into rows)."""
_K_TABLE_NONCE = "results.table_nonce"
_K_CM_MODE = "results.cm_mode"
_K_CELL = "results.cell"
"""Prefix of the per-job plain key remembering the clicked confusion cell."""
_K_CM_NONCE = "results.cm_nonce"

_TABS = ("Outputs", "Evaluation", "Details")

# Widgets in a closed tab aren't drawn, and Streamlit drops the state of
# widgets that aren't drawn; "session" keeps filters, search and toggles
# across tab and page switches.
_PERSIST: Final = "session"

STATUS_WORDS = {
    "generated": "Generated",
    "cached": "Saved",
    "blank": "Blank text",
    "error": "Error",
    "pending": "Not generated",
}
"""How row statuses read in the table ("saved" = reused saved output)."""
_STATUS_COLORS = {"Error": "orange"}
"""Status chips that stand out (the rest are grey)."""
RESULT_WORDS = ("Correct", "Wrong")
JSON_WORDS = ("Valid", "Invalid")

ROW_PX = 35
"""Row height of st.dataframe: table heights are whole rows plus the header."""
SMALL_SLICE = 5
"""Slices with fewer scored rows than this are noisy (said under the table)."""

_TOP_CONFUSIONS_OVER = 12
"""Label sets larger than this get a ranked table instead of a heatmap."""


# ==========================================================================
# Entry point
# ==========================================================================


def render() -> None:
    components.page_header("Results", "Browse, score and export what a run produced.")
    manager = resources.get_job_manager()
    # Asked before listing: a run that ends between the two is missing from
    # the list yet no longer running, and must still be the one followed.
    pending = _is_running(st.session_state.get(state.ACTIVE_JOB))
    jobs = _selectable_jobs(manager)
    if not jobs:
        _empty_page(manager)
        return

    running = manager.active()
    # Always the same element here, running or not: the page's elements keep
    # their places when a run ends (so the search box keeps what was typed).
    with st.container():
        if running is not None:
            _running_notice(running.id)

    job = _current_job(jobs, pending)
    _forget_other_drawer(job.id)
    snap = job.snapshot()
    built = an.get_analysis(job) if job.result is not None else None
    if built is not None and an.FINISH_ERROR_KEY in job.request.context:
        _retry_history(job, manager)
    _run_header(job, snap, built, manager, jobs)
    if built is None:
        return

    _choose_tab(job.id, built)
    outputs_tab, eval_tab, details_tab = st.tabs(list(_TABS), key=_K_TABS, on_change="rerun")
    st.session_state[_K_TAB_MEMORY] = st.session_state.get(_K_TABS) or _TABS[0]
    # Lazy: only the open tab runs (``open`` is None if state isn't tracked).
    if outputs_tab.open is not False:
        with outputs_tab:
            _outputs_tab(built)
    if eval_tab.open:
        with eval_tab:
            _evaluation_tab(built)
    if details_tab.open:
        with details_tab:
            _details_tab(job, snap, built)


def _choose_tab(job_id: str, built: an.JobAnalysis) -> None:
    """Restore the open tab after a page switch (the tabs widget forgets it),
    but open a newly shown run that has nothing to score on Outputs rather
    than on an empty Evaluation tab."""
    if _K_TABS not in st.session_state and st.session_state.get(_K_TAB_MEMORY) in _TABS:
        st.session_state[_K_TABS] = st.session_state[_K_TAB_MEMORY]
    new_run = st.session_state.get(_K_SHOWN) != job_id
    st.session_state[_K_SHOWN] = job_id
    if new_run and built.report is None and st.session_state.get(_K_TABS) == _TABS[1]:
        st.session_state[_K_TABS] = _TABS[0]


# ==========================================================================
# Run selection
# ==========================================================================


def _selectable_jobs(manager: JobManager) -> list[Job]:
    """Finished jobs worth showing: this session's first (newest first), then
    the manager's other finished jobs. A job qualifies when it has a result,
    or failed outright (so its error can be shown and retried)."""
    seen: set[str] = set()
    ordered: list[Job] = []
    session_ids = list(st.session_state.get(state.SESSION_JOBS, []))
    candidates = [manager.get(i) for i in session_ids] + manager.jobs()
    for job in candidates:
        if job is None or job.id in seen:
            continue
        snap = job.snapshot()
        if not snap.done:
            continue
        if job.result is None and snap.state != "failed":
            continue
        if not isinstance(job.request.context.get(state.CONTEXT_KEY), state.RunContext):
            continue  # e.g. a CLI-style job without the UI's context
        seen.add(job.id)
        ordered.append(job)
    return ordered


def _run_label(job: Job) -> str:
    ctx: state.RunContext = job.request.context[state.CONTEXT_KEY]
    word = components.job_status(job.snapshot())[0]
    return ctx.name if word == "Complete" else f"{ctx.name} · {word}"


def _run_labels(jobs: list[Job]) -> dict[str, str]:
    """Picker label per job id, all different.

    The selectbox sends the chosen *label* back to the server, and equal
    labels map to the last of them - so two runs of the same model and task
    on one file in the same minute would open the wrong run. Runs that share
    a label get whatever tells them apart (rows, gold column, edited prompt,
    start time to the second), and a counter as the last resort."""
    labels = {job.id: _run_label(job) for job in jobs}
    groups: dict[str, list[Job]] = {}
    for job in jobs:
        groups.setdefault(labels[job.id], []).append(job)

    def ctx_of(job: Job) -> state.RunContext:
        return job.request.context[state.CONTEXT_KEY]

    details: tuple[Callable[[Job], str], ...] = (
        lambda j: j.request.config.model_label,
        lambda j: ctx_of(j).selection,
        lambda j: f"gold {ctx_of(j).gold_column}" if ctx_of(j).gold_column else "no gold column",
        lambda j: "edited prompt" if ctx_of(j).template_edited else "preset prompt",
        lambda j: f"{j.request.config.max_new_tokens:,} max tokens",
        lambda j: f"started {ctx_of(j).created_at:%H:%M:%S}",
    )
    for base, group in groups.items():
        if len(group) < 2:
            continue
        extra: dict[str, list[str]] = {job.id: [] for job in group}
        for describe in details:
            values = {job.id: describe(job) for job in group}
            if len(set(values.values())) > 1:
                for job in group:
                    extra[job.id].append(values[job.id])
        for job in group:
            labels[job.id] = " · ".join([base, *extra[job.id]])

    seen: Counter[str] = Counter()
    for job in reversed(jobs):  # oldest first, so "#2" is the later run
        label = labels[job.id]
        seen[label] += 1
        if seen[label] > 1:
            labels[job.id] = f"{label} · #{seen[label]}"
    return labels


def _current_job(jobs: list[Job], pending: bool) -> Job:
    """The run to show: ACTIVE_JOB when it is selectable, else the run
    selector's last choice, else the newest. Syncs the selector's state (the
    widget itself is drawn in the header row by :func:`_run_picker`).
    ``pending``: ACTIVE_JOB was still running when ``jobs`` was listed."""
    by_id = {job.id: job for job in jobs}
    ids = list(by_id)
    wanted = st.session_state.get(state.ACTIVE_JOB)
    if wanted in by_id and st.session_state.get(_K_SELECT) != wanted:
        st.session_state[_K_SELECT] = wanted
    elif st.session_state.get(_K_SELECT) not in by_id:
        st.session_state[_K_SELECT] = ids[0]
    chosen = st.session_state[_K_SELECT]
    # ACTIVE_JOB may name a run still in progress (e.g. after a retry): leave
    # it alone so that run is shown as soon as it finishes (the context
    # strip's poller reruns the app then).
    if st.session_state.get(state.ACTIVE_JOB) not in by_id and not pending:
        st.session_state[state.ACTIVE_JOB] = chosen
    return by_id[chosen]


def _run_picker(jobs: list[Job]) -> None:
    labels = _run_labels(jobs)

    def _remember() -> None:
        st.session_state[state.ACTIVE_JOB] = st.session_state[_K_SELECT]

    st.selectbox(
        "Run",
        list(labels),
        format_func=lambda i: labels.get(i, str(i)),
        key=_K_SELECT,
        on_change=_remember,
        label_visibility="collapsed",
        help="Switch to another run from this session.",
        width=300,
    )


def _is_running(job_id: str | None) -> bool:
    if not job_id:
        return False
    job = resources.get_job_manager().get(job_id)
    return job is not None and not job.snapshot().done


def _running_notice(job_id: str) -> None:
    """Slim card above the run picker while a run is in progress, so an
    earlier run's results aren't mistaken for the new one's."""
    if st.session_state.get(state.ACTIVE_JOB) == job_id:
        follow = "Its results open here when it finishes."
    elif job_id in st.session_state.get(state.SESSION_JOBS, []):
        # The user picked another run meanwhile: the page stays on that one.
        follow = "It joins the run list when it finishes."
    else:
        follow = "Started in another tab or window; it joins the run list when it finishes."
    _running_card(job_id, follow)


@st.fragment(run_every="2s")
def _running_card(job_id: str, follow: str) -> None:
    # Display only: the context strip's poller reruns the app when the job
    # ends (two pollers reacting to the same end can leave a blank page).
    job = resources.get_job_manager().get(job_id)
    if job is None:
        return
    snap = job.snapshot()
    ctx = job.request.context.get(state.CONTEXT_KEY)
    name = ctx.name if isinstance(ctx, state.RunContext) else job.request.config.model_label
    p = snap.progress
    if snap.done:
        doing = "has finished. Loading its results…"
    elif snap.state == "queued" or p.phase == "planning":
        doing = "is starting."
    elif p.phase == "loading":
        doing = "is loading the model."
    else:
        doing = f"is running: {p.done_rows:,} of {fmt.plural(p.total_rows, 'row')} done."
    with st.container(
        border=True, horizontal=True, vertical_alignment="center", gap="medium", wrap=False
    ):
        st.markdown(
            f":blue[:material/progress_activity:] **{fmt.md(name)}** {doing} :gray[{follow}]",
            width="stretch",
        )
        st.page_link(nav.page("run"), label="View progress", icon=":material/monitoring:")


def _empty_page(manager: JobManager) -> None:
    running = manager.active() is not None
    with components.empty_card():
        if running:
            components.empty_state(
                ":material/progress_activity:",
                "A run is in progress",
                "Its outputs, evaluation and exports appear here as soon as it finishes. "
                "Follow its progress on the Run page.",
            )
        else:
            components.empty_state(
                ":material/table_view:",
                "No results yet",
                "Run a task on the Run page. Its outputs, evaluation and exports appear here, "
                "and stay available while you explore.",
            )
        with st.container(horizontal=True, horizontal_alignment="center"):
            if running:  # the same words as the running notice's link
                components.page_button("run", "View progress", ":material/monitoring:")
            else:
                components.page_button("run", "Go to the Run page", ":material/play_circle:")


# ==========================================================================
# Header strip
# ==========================================================================


def _run_header(
    job: Job,
    snap: JobSnapshot,
    built: an.JobAnalysis | None,
    manager: JobManager,
    jobs: list[Job],
) -> None:
    """The run's title row - the same shape whatever the number of runs:
    name and status badges on the left; the run picker (with more than one
    run) and the Export menu on the right - then what the name doesn't say."""
    ctx: state.RunContext = job.request.context[state.CONTEXT_KEY]
    result = job.result
    word, icon, color = components.job_status(snap)

    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.markdown(f"#### {fmt.md(ctx.name)}", width="content", anchors=False)
        st.badge(word, icon=icon, color=color)  # type: ignore[arg-type]
        if result is not None and result.stats.errors:
            st.badge(
                fmt.plural(result.stats.errors, "error"), icon=":material/warning:", color="orange"
            )
        if built is not None and built.report is not None and built.report.n_scored > 0:
            st.badge("Scored", icon=":material/rule:", color="blue")
        st.space("stretch")
        if len(jobs) > 1:
            _run_picker(jobs)
        if built is not None:
            _export_menu(built)

    meta = [job.request.config.model_label, ctx.selection]
    if ctx.template_edited:
        meta.append("edited prompt")
    if snap.finished_at is not None:
        meta.append(f"finished {fmt.timestamp(snap.finished_at)}")
    lines = [" · ".join(meta)]
    if result is not None:
        lines.append(an.run_summary(result))
    st.caption("  \n".join(fmt.md(line) for line in lines))

    _problem_cards(job, snap, manager)
    finish_error = job.request.context.get(an.FINISH_ERROR_KEY)
    if finish_error:
        st.caption(
            f":material/info: This run's evaluation couldn't be saved to history ({finish_error})."
        )


def _problem_cards(job: Job, snap: JobSnapshot, manager: JobManager) -> None:
    """What went wrong (if anything) -> why -> what to do, with a retry."""
    result = job.result
    model_label = job.request.config.model_label
    if result is None:
        problem = an.describe_failure(None, snap.error, model_label)
        _failure_card(problem)
        _retry_or_change(problem, job, manager, "Retry run")
        return
    stats = result.stats
    unfinished = stats.errors + stats.pending
    if not unfinished:
        return
    if stats.errors and stats.pending:
        label = (
            f"Retry {fmt.plural(unfinished, 'row')} "
            f"({stats.errors:,} failed, {stats.pending:,} not tried)"
        )
    elif not (stats.generated or stats.cached):
        label = f"Retry {fmt.plural(unfinished, 'row')}"
    elif stats.errors:
        label = f"Retry {fmt.plural(stats.errors, 'failed row')}"
    else:
        label = f"Generate {fmt.plural(stats.pending, 'remaining row')}"

    messages = Counter(e for e in result.errors if e)
    if result.aborted_reason:
        problem = an.describe_failure(result, snap.error, model_label)
        details = problem.details or ""
        other = [m for m in messages if m.strip() not in details]
        if other:
            details += "\n\nRow errors:\n" + _error_counts(Counter({m: messages[m] for m in other}))
        _failure_card(problem, details.strip() or None)
        _retry_or_change(problem, job, manager, label)
        return
    if result.cancelled and stats.pending:
        st.info(
            f"Cancelled with {fmt.plural(stats.pending, 'row')} not generated. Saved outputs are "
            "kept, so continuing only generates the missing rows.",
            icon=":material/pause_circle:",
        )
    if not stats.errors:
        _retry_button(job, manager, label)
        return

    top, _ = messages.most_common(1)[0] if messages else ("unknown error", 0)
    advice = an.explain_failure(top, model_label).action
    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center", gap="medium"):
            with st.container(gap="xsmall"):
                st.markdown(
                    f":orange[:material/warning:] **{fmt.plural(stats.errors, 'row')} failed** — "
                    f"{fmt.md(fmt.truncate(top, 140))}"
                )
                st.caption(
                    " ".join(
                        filter(
                            None,
                            [
                                "Failed rows weren't saved and are left out of the evaluation.",
                                advice,
                                "Retrying generates just those rows.",
                            ],
                        )
                    )
                )
            _retry_button(job, manager, label)
        if len(messages) > 1 or len(top) > 140:
            with st.expander("Technical details", icon=":material/code:"):
                st.code(_error_counts(messages), language=None, wrap_lines=True)


def _failure_card(problem: an.Problem, details: str | None = None) -> None:
    """The red card of a run that failed or stopped early."""
    components.error_with_details(
        f"**{problem.headline}** {fmt.md(problem.cause)}",
        details=details if details is not None else problem.details,
        hint=problem.hint,
    )


def _error_counts(messages: Counter[str]) -> str:
    return "\n".join(f"{n:,} × {m}" for m, n in messages.most_common())


def _retry_or_change(problem: an.Problem, job: Job, manager: JobManager, label: str) -> None:
    """A retry - unless the same settings are certain to fail again (e.g.
    fp16 on a CPU, a folder that isn't a model): then a way to change them."""
    if problem.retry_helps:
        _retry_button(job, manager, label)
        return
    components.page_button(
        "run",
        "Change settings on the Run page",
        ":material/tune:",
        type="secondary",
        key=f"results.change::{job.id}",
    )


def _retry_button(job: Job, manager: JobManager, label: str) -> None:
    active = manager.active()
    if active is not None and active.request.context.get(state.RETRY_OF) == job.id:
        st.button(
            "Retrying…",
            key=f"results.retry::{job.id}",
            icon=":material/progress_activity:",
            disabled=True,
            help="This run is being retried: the card at the top of the page shows how far "
            "it has got.",
        )
        return
    busy = active is not None
    if st.button(
        label,
        key=f"results.retry::{job.id}",
        icon=":material/refresh:",
        disabled=busy,
        help="Another run is in progress. Wait for it to finish or cancel it on the Run page."
        if busy
        else "Runs the same configuration again. Saved outputs are reused.",
    ):
        _resubmit(job, manager)


def _resubmit(job: Job, manager: JobManager) -> None:
    """Submit ``job``'s request again and follow the new run on this page
    (the running card shows its progress; its results replace these when it
    finishes)."""
    from batchtextreview.jobs import JobBusyError

    request = job.request
    context = {k: v for k, v in request.context.items() if not str(k).startswith("_")}
    context.pop(an.FINISH_ERROR_KEY, None)
    context[state.RETRY_OF] = job.id
    ctx = request.context.get(state.CONTEXT_KEY)
    if isinstance(ctx, state.RunContext):
        now = datetime.now()
        context[state.CONTEXT_KEY] = dataclasses.replace(
            ctx,
            created_at=now,
            name=state.run_name(ctx.preset_title, ctx.dataset_name, now),
        )
    try:
        new_job = manager.submit(dataclasses.replace(request, context=context))
    except JobBusyError:
        st.warning("Another run started in the meantime. Try again when it finishes.")
        return
    jobs = [
        new_job.id,
        *[i for i in st.session_state.get(state.SESSION_JOBS, []) if i != new_job.id],
    ]
    st.session_state[state.SESSION_JOBS] = jobs
    # The same setup as the run it retries: the Run page then knows the
    # retry ran exactly what the original did ("Open results" opens the
    # retry, not the stale original). A run this session didn't set up gets
    # a setup that matches nothing on the Run page.
    setups = st.session_state.setdefault(state.JOB_SETUPS, {})
    setups[new_job.id] = setups.get(job.id) or (state.NOT_SET_UP_HERE, job.id)
    st.session_state[state.ACTIVE_JOB] = new_job.id
    st.rerun()


def _export_menu(built: an.JobAnalysis) -> None:
    """One "Export" menu with every download: all rows as CSV or Excel, and
    the rows the Outputs tab currently shows."""
    ctx = built.context
    cfg = built.result.config
    when = ctx.created_at

    def name(ext: str, suffix: str = "") -> str:
        return fmt.export_filename(
            ctx.dataset_name, ctx.preset_title + suffix, cfg.model_label, ext, when
        )

    rows = _view_rows(built)
    with st.popover("Export", icon=":material/download:"):
        st.download_button(
            "All rows · CSV",
            data=_export_callable(built, "csv"),
            file_name=name("csv"),
            mime="text/csv",
            icon=":material/table:",
            on_click="ignore",
            width="stretch",
            key=f"results.dl_csv::{built.job_id}",
            help="Every row with the result columns. UTF-8 with BOM, opens cleanly in Excel.",
        )
        st.download_button(
            "All rows · Excel",
            data=_export_callable(built, "xlsx"),
            file_name=name("xlsx"),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            icon=":material/grid_on:",
            on_click="ignore",
            width="stretch",
            key=f"results.dl_xlsx::{built.job_id}",
        )
        st.download_button(
            f"Filtered view · CSV ({fmt.plural(len(rows), 'row')})",
            data=lambda: results.to_csv_bytes(_export_frame(built, rows)),
            file_name=name("csv", " filtered"),
            mime="text/csv",
            icon=":material/filter_alt:",
            on_click="ignore",
            width="stretch",
            disabled=len(rows) == 0,
            key=f"results.dl_view::{built.job_id}",
            help="The rows the Outputs tab shows with its current filter and search.",
        )


def _export_callable(built: an.JobAnalysis, kind: results.ExportFormat) -> Callable[[], bytes]:
    """Deferred export: built only when the button is clicked, then memoised
    on the job's context."""

    def _make() -> bytes:
        if kind not in built.exports:
            payload, _mime = results.export_bytes(_export_frame(built), kind)
            built.exports[kind] = payload
        return built.exports[kind]

    return _make


def _export_frame(built: an.JobAnalysis, rows: pd.Index | None = None) -> pd.DataFrame:
    frame = built.table if rows is None else built.table.loc[rows]
    return frame.reset_index(drop=True)


# ==========================================================================
# Outputs tab
# ==========================================================================


def _default_columns(built: an.JobAnalysis) -> list[str]:
    """The columns the table opens with: the ones that carry information
    for this run. OUTPUT is left out of label tasks when it only repeats the
    resolved label, STATUS when every row was simply generated (blank rows
    have their own filter), ERROR when nothing failed. The Columns menu
    offers them all."""
    ctx = built.context
    cols = built.columns
    statuses = set(str(v) for v in built.result.statuses)
    wanted = [ctx.id_column, ctx.text_column, ctx.gold_column]
    if cols.label:
        wanted += [cols.label, cols.correct]
        if not built.output_matches_label():
            wanted.append(cols.output)
    elif cols.json_fields:
        # JSON tasks: validity first, then a few expanded fields (they read
        # better than the raw JSON string).
        wanted += [cols.json_valid, cols.correct, *cols.json_fields[:4]]
    else:
        wanted += [cols.output, cols.correct, cols.json_valid]
    if statuses & {"error", "pending"} or {"generated", "cached"} <= statuses:
        wanted.append(cols.status)
    if "error" in statuses:
        wanted.append(cols.error)
    present = [c for c in wanted if c and c in built.table.columns]
    return list(dict.fromkeys(present))


def _display_frame(built: an.JobAnalysis, rows: pd.Index, columns: list[str]) -> pd.DataFrame:
    view = built.table.loc[rows, columns].copy()
    cols = built.columns
    if cols.label and cols.label in view.columns:
        labels = built.context.labels
        view[cols.label] = view[cols.label].map(lambda v: an.display_label(v, labels))
    if cols.correct in view.columns:
        view[cols.correct] = view[cols.correct].map(
            lambda v: None if an.is_blank(v) else RESULT_WORDS[0 if bool(v) else 1]
        )
    if cols.status in view.columns:
        view[cols.status] = view[cols.status].map(lambda v: STATUS_WORDS.get(str(v), str(v)))
    if cols.json_valid and cols.json_valid in view.columns:
        view[cols.json_valid] = view[cols.json_valid].map(
            lambda v: None if an.is_blank(v) else JSON_WORDS[0 if an.truthy(v) else 1]
        )
    return view


def _column_names(built: an.JobAnalysis) -> dict[str, str]:
    """Header of each table column, also used by the Columns menu: the run's
    own columns in words ("Result", not CORRECT), the role columns by role
    ("Text", "Gold", "ID")."""
    ctx = built.context
    cols = built.columns
    names = {str(c): str(c) for c in built.table.columns}
    for column, name in (
        (ctx.id_column, "ID"),
        (ctx.text_column, "Text"),
        (ctx.gold_column, "Gold"),
        (cols.output, "Output"),
        (cols.label, "Label"),
        (cols.correct, "Result"),
        (cols.status, "Status"),
        (cols.error, "Error"),
        (cols.json_valid, "JSON"),
    ):
        if column:
            names[str(column)] = name
    for column in cols.json_fields:
        names[column] = column.split(".", 1)[-1]
    return names


def _chips(label: str, words: Sequence[str], colors: Sequence[str], help_text: str) -> Any:
    """A read-only column whose words are drawn as coloured chips (the word
    is the signal; the colour helps scanning)."""
    return st.column_config.MultiselectColumn(
        label,
        options=list(words),
        color=list(colors),
        disabled=True,
        width=96,
        help=help_text,
    )


def _column_config(built: an.JobAnalysis, columns: list[str]) -> dict[str, Any]:
    ctx = built.context
    cols = built.columns
    names = _column_names(built)
    config: dict[str, Any] = {}
    for name in columns:
        label = names.get(name, name)
        if name == ctx.text_column:
            # Pixel widths are stretched in proportion to fill the table, so
            # a wide Text column takes most of the room the short ones leave.
            config[name] = st.column_config.TextColumn(label, width=640 if cols.label else 520)
        elif name == ctx.id_column:
            config[name] = st.column_config.TextColumn(
                label, help=f"From the {name} column.", pinned=True, width=80
            )
        elif name == cols.output:
            config[name] = st.column_config.TextColumn(
                label, width=120 if cols.label else 360, help="The model's raw answer."
            )
        elif name == cols.label:
            config[name] = st.column_config.TextColumn(
                label, width=96, help="The output resolved to the label set."
            )
        elif name == ctx.gold_column:
            config[name] = st.column_config.TextColumn(
                label, width=96, help=f"Expected label, from the {name} column."
            )
        elif name == cols.correct:
            config[name] = _chips(
                label,
                RESULT_WORDS,
                ("blue", "orange"),
                "Label compared with the gold value. Blank when the row wasn't scored.",
            )
        elif name == cols.status:
            words = list(STATUS_WORDS.values())
            config[name] = _chips(
                label,
                words,
                [_STATUS_COLORS.get(w, "gray") for w in words],
                "Generated now, Saved (reused from an earlier identical run), Blank text, "
                "Error or Not generated.",
            )
        elif name == cols.error:
            config[name] = st.column_config.TextColumn(label, width="medium")
        elif name == cols.json_valid:
            config[name] = _chips(
                label, JSON_WORDS, ("blue", "orange"), "Whether the output parses as JSON."
            )
        elif name in cols.json_fields:
            config[name] = st.column_config.TextColumn(label, width="medium")
    return config


def _filter_keys(job_id: str) -> tuple[str, str, str]:
    """Session keys of the Outputs filters: (show, predicted labels, search)."""
    return (
        f"results.filter::{job_id}",
        f"results.labels::{job_id}",
        f"results.search::{job_id}",
    )


def _row_filters(built: an.JobAnalysis) -> dict[str, pd.Series]:
    cached = built.memo.get("row_filters")
    if cached is None:
        cached = built.memo["row_filters"] = an.row_filters(built)
    return cached


def _view_rows(built: an.JobAnalysis) -> pd.Index:
    """Table rows the Outputs filters currently select (read from session
    state, so the header's Export menu can offer the same rows)."""
    table = built.table
    cols = built.columns
    filters = _row_filters(built)
    show_key, labels_key, search_key = _filter_keys(built.job_id)
    choice = st.session_state.get(show_key) or an.FILTER_ALL
    mask = filters.get(choice, filters[an.FILTER_ALL])
    picked = list(st.session_state.get(labels_key) or [])
    if picked and cols.label:
        mask = mask & table[cols.label].astype(str).isin(picked)
    searchable = [
        built.context.text_column,
        cols.output,
        *([cols.label] if cols.label else []),
        *cols.json_fields,
    ]
    mask = mask & an.search_mask(table, str(st.session_state.get(search_key) or ""), searchable)
    return table.index[mask.to_numpy()]


def _outputs_tab(built: an.JobAnalysis) -> None:
    """One layout for every task type: a summary card (numbers left, chart
    right), a toolbar (filter, labels, search, columns), the table."""
    job_id = built.job_id
    cols = built.columns
    table = built.table
    filters = _row_filters(built)
    counts = built.label_counts()
    show_key, labels_key, search_key = _filter_keys(job_id)

    if cols.label:
        _label_summary(built, counts)
    else:
        _output_diagnostics(built)

    names = _column_names(built)
    with st.container(horizontal=True, vertical_alignment="bottom", gap="small"):
        st.segmented_control(
            "Show",
            list(filters),
            format_func=lambda k: f"{k} · {int(filters[k].sum()):,}",
            default=an.FILTER_ALL,
            required=True,
            key=show_key,
            persist_state=_PERSIST,
        )
        options = [lab for lab, n in zip(counts["label"], counts["count"], strict=True) if n]
        if cols.label and options:
            labels = built.context.labels
            st.pills(
                "Predicted label",
                options,
                selection_mode="multi",
                key=labels_key,
                format_func=lambda lab: an.display_label(lab, labels),
                persist_state=_PERSIST,
            )
        st.space("stretch")
        st.text_input(
            "Search",
            key=search_key,
            placeholder="Search text and outputs",
            icon=":material/search:",
            label_visibility="collapsed",
            width=260,
            persist_state=_PERSIST,
        )
        with st.popover("Columns", icon=":material/view_column:"):
            shown = st.multiselect(
                "Columns to show",
                [str(c) for c in table.columns],
                default=_default_columns(built),
                format_func=lambda c: names.get(c, c),
                key=f"results.columns::{job_id}",
                persist_state=_PERSIST,
            )
    _results_table(built, _view_rows(built), shown or _default_columns(built))


def _table_height(n_rows: int, visible: int) -> int:
    """Whole rows only: header + up to ``visible`` rows (+3 px of borders),
    so no row is ever cut in half at the bottom edge."""
    return (min(n_rows, visible) + 1) * ROW_PX + 3


def _results_table(built: an.JobAnalysis, rows: pd.Index, columns: list[str]) -> None:
    job_id = built.job_id
    total = len(built.table)
    shown = "all" if len(rows) == total else f"{len(rows):,} of"
    st.caption(f"Showing {shown} {fmt.plural(total, 'row')}. Click a row to read it in full.")

    if len(rows) == 0:
        with components.empty_card():
            components.empty_state(
                ":material/filter_alt_off:", "No rows match", "Change the filter or the search."
            )
        return

    nonce = st.session_state.get(_K_TABLE_NONCE, 0)
    key = f"results.table::{job_id}::{nonce}"
    view = _display_frame(built, rows, columns)

    def _opened() -> None:
        event = st.session_state.get(key)
        selection = getattr(event, "selection", None)
        picked = list(getattr(selection, "rows", []) or [])
        cells = list(getattr(selection, "cells", []) or [])
        position = picked[0] if picked else (cells[0][0] if cells else None)
        if position is None:
            return
        st.session_state[_K_DRAWER] = {
            "job": job_id,
            "rows": [int(r) for r in rows],
            "pos": int(position),
            "table": key,
        }

    st.dataframe(
        view,
        hide_index=True,
        column_config=_column_config(built, columns),
        on_select=_opened,
        # Clicking any cell opens the row (a row checkbox alone is a 16 px
        # target). Closing the drawer clears the selection in place rather
        # than rebuilding the grid, so it keeps its scroll position.
        selection_mode=["single-row", "single-cell"],
        key=key,
        height=_table_height(len(view), 15),
        alt=f"Run outputs, {len(view)} rows",
    )
    drawer = st.session_state.get(_K_DRAWER)
    if drawer and drawer.get("job") == job_id:
        _row_drawer(built)


def _forget_other_drawer(job_id: str) -> None:
    """Close a row drawer that went away without being dismissed - the page
    switched to another run (one that just finished) or the user went to
    another page - so it doesn't reopen by itself on the next visit."""
    drawer = st.session_state.get(_K_DRAWER)
    if drawer is None:
        return
    if not isinstance(drawer, dict) or drawer.get("job") != job_id or components.arrived_on_page():
        _close_drawer()


def _close_drawer() -> None:
    drawer = st.session_state.pop(_K_DRAWER, None)
    table = drawer.get("table") if isinstance(drawer, dict) else None
    if table:
        # Deselect, so clicking the same row again opens it again.
        st.session_state[table] = {"selection": {"rows": [], "columns": [], "cells": []}}


def _step_drawer(delta: int) -> None:
    drawer = st.session_state.get(_K_DRAWER)
    if drawer:
        drawer["pos"] = max(0, min(len(drawer["rows"]) - 1, drawer["pos"] + delta))


def _row_drawer(built: an.JobAnalysis) -> None:
    @st.dialog("Row details", width="medium", position="right", on_dismiss=_close_drawer)
    def _drawer() -> None:
        drawer = st.session_state.get(_K_DRAWER)
        if not drawer or not drawer["rows"]:
            return
        position = drawer["pos"]
        row = drawer["rows"][position]
        _row_detail(built, row, position, len(drawer["rows"]))

    _drawer()


def _row_detail(built: an.JobAnalysis, row: int, position: int, count: int) -> None:
    ctx = built.context
    cols = built.columns
    record = built.table.loc[row]
    status = str(built.result.statuses[row])

    with st.container(horizontal=True, vertical_alignment="center"):
        ident = f"Row {record[ctx.id_column]}" if ctx.id_column else f"Row {row + 1:,}"
        st.markdown(f"**{fmt.md(ident)}** · {position + 1:,} of {count:,} shown", width="content")
        st.space("stretch")
        st.button(
            "Previous",
            icon=":material/chevron_left:",
            on_click=_step_drawer,
            args=(-1,),
            disabled=position == 0,
            shortcut="Left",
            key="results.drawer_prev",
        )
        st.button(
            "Next",
            icon=":material/chevron_right:",
            icon_position="right",
            on_click=_step_drawer,
            args=(1,),
            disabled=position >= count - 1,
            shortcut="Right",
            key="results.drawer_next",
        )

    with st.container(horizontal=True, gap="small"):
        word = STATUS_WORDS.get(status, status)
        st.badge(word, color=_STATUS_COLORS.get(word, "gray"))  # type: ignore[arg-type]
        if cols.correct and not an.is_blank(record.get(cols.correct)):
            if bool(record[cols.correct]):
                st.badge("Correct", icon=":material/check:", color="blue")
            else:
                st.badge("Wrong", icon=":material/close:", color="orange")
        if cols.label and str(record.get(cols.label, "")) == INVALID:
            st.badge("Answer not in label set", icon=":material/help:", color="gray")

    if cols.label or ctx.gold_column:
        with st.container(border=True, horizontal=True, gap="large"):
            if cols.label:
                label = an.display_label(record.get(cols.label), ctx.labels) or fmt.DASH
                st.markdown(f":gray[Predicted label]  \n**{fmt.md(label)}**", width="content")
            if ctx.gold_column:
                gold = record.get(ctx.gold_column)
                gold_text = fmt.DASH if an.is_blank(gold) else str(gold)
                st.markdown(
                    f":gray[Gold · {fmt.md(ctx.gold_column)}]  \n**{fmt.md(gold_text)}**",
                    width="content",
                )

    text = record.get(ctx.text_column)
    st.markdown("**Text**")
    with st.container(border=True):
        st.text("" if an.is_blank(text) else str(text))

    output = built.result.outputs[row]
    parsed = built.parsed_json(row) if ctx.output_kind == "json" and output else None
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.markdown("**Model output**", width="content")
        if ctx.output_kind == "json" and output:
            if parsed is None:
                st.badge("Not valid JSON", icon=":material/data_object:", color="orange")
            else:
                st.badge("Valid JSON", icon=":material/data_object:", color="blue")
    if status == "error":
        st.error(
            built.result.errors[row] or "Generation failed for this row.", icon=":material/error:"
        )
    elif not output:
        st.caption("No output for this row.")
    elif parsed is not None:
        st.json(parsed, expanded=2)
        with st.expander("Raw output"):
            st.code(output, language=None, wrap_lines=True)
    else:
        st.code(output, language=None, wrap_lines=True)

    if not an.is_blank(text):
        with st.expander("Prompt sent to the model"):
            snippet = str(text)[: built.result.config.max_input_chars]
            st.code(
                render_prompt(built.result.config.template, snippet), language=None, wrap_lines=True
            )
            st.caption("The model's chat template wraps this prompt at run time.")

    shown = {ctx.text_column, ctx.gold_column, ctx.id_column, *built.columns.added}
    other = [c for c in built.table.columns if c not in shown]
    if other:
        st.markdown("**Other columns**")
        values = pd.DataFrame(
            {
                "Column": [str(c) for c in other],
                "Value": ["" if an.is_blank(record[c]) else str(record[c]) for c in other],
            }
        )
        st.table(values, hide_index=True, hide_header=True, border="horizontal")


def _label_summary(built: an.JobAnalysis, counts: pd.DataFrame) -> None:
    """Label tasks: how the answers spread over the labels (all rows, before
    the filters below), with the gold counts outlined when there are any."""
    total = int(counts["count"].sum()) if not counts.empty else 0
    if not total:
        return  # nothing answered yet; the table and the header say why
    invalid = int(counts.loc[counts["label"] == INVALID, "count"].sum())
    valid = counts.loc[counts["label"] != INVALID]
    top = valid.sort_values("count", ascending=False).iloc[0] if not valid.empty else None
    gold = built.gold_counts()
    with st.container(border=True):
        numbers, chart = st.columns([2, 3], gap="large", vertical_alignment="center")
        with numbers:
            st.markdown(f"**Predicted labels** · all {fmt.plural(total, 'answer')}")
            with st.container(horizontal=True, gap="small"):
                st.metric(
                    "In label set",
                    fmt.percent((total - invalid) / total, 0),
                    help="Answers that resolved to exactly one allowed label.",
                )
                if top is not None and int(top["count"]):
                    st.metric(
                        "Most common",
                        an.display_label(top["label"], built.context.labels),
                        help=f"{int(top['count']):,} of {total:,} answers "
                        f"({int(top['count']) / total:.0%}).",
                    )
            notes = ["Before the filters below."]
            if invalid:
                notes.append(
                    f"{fmt.plural(invalid, 'answer')} didn't match the label set "
                    f"({fmt.md(', '.join(built.context.labels or []))})."
                )
            if gold:
                notes.append("Dashed outline: rows with that gold label.")
            st.caption(" ".join(notes))
        with chart:
            st.altair_chart(
                charts.label_distribution(counts, dark=_dark(), gold=gold or None),
                width="stretch",
                alt="Bar chart of predicted label counts"
                + (", with gold label counts outlined" if gold else ""),
            )


def _output_diagnostics(built: an.JobAnalysis) -> None:
    """Open-ended / JSON tasks: length and validity numbers beside a small
    length histogram, in one full-width card. Nothing until there are
    outputs (the table and the header already say why)."""
    diag = built.output_diagnostics()
    if not diag["n_outputs"]:
        return
    with st.container(border=True):
        numbers, chart = st.columns([2, 3], gap="large", vertical_alignment="center")
        with numbers:
            st.markdown(f"**Outputs** · all {fmt.plural(int(diag['n_outputs']), 'row')}")
            with st.container(horizontal=True, gap="small"):
                st.metric(
                    "Median length",
                    f"{fmt.number(diag['median_chars'], 0)} chars",
                    help="Of the non-empty outputs.",
                )
                st.metric("Empty", fmt.percent(diag["empty_rate"], 0))
                if diag["json_valid_rate"] is not None:
                    st.metric(
                        "Valid JSON",
                        fmt.percent(diag["json_valid_rate"], 0),
                        help="Outputs that parse as JSON (code fences and surrounding prose are "
                        "tolerated).",
                    )
            st.caption("Before the filters below.")
        lengths = built.output_lengths()
        if lengths:
            with chart:
                st.altair_chart(
                    charts.length_histogram(lengths, dark=_dark()),
                    width="stretch",
                    alt="Histogram of output lengths",
                )


# ==========================================================================
# Evaluation tab
# ==========================================================================


def _evaluation_tab(built: an.JobAnalysis) -> None:
    ctx = built.context
    report = built.report
    if report is None:
        with components.empty_card():
            if ctx.labels and not ctx.gold_column:
                components.empty_state(
                    ":material/rule:",
                    "This run wasn't scored",
                    "Choose a gold-label column on the Run page to compare outputs with "
                    "expected labels.",
                )
            else:
                components.empty_state(
                    ":material/rule:",
                    "Open-ended output isn't scored",
                    "Scoring needs a closed label set: pick a classification task. To compare "
                    "free text with a gold column anyway, turn on “Score by exact match” on the "
                    "Run page. Output statistics are on the Outputs tab.",
                )
            with st.container(horizontal=True, horizontal_alignment="center"):
                st.page_link(
                    nav.page("run"), label="Go to the Run page", icon=":material/play_circle:"
                )
        return

    if report.n_scored == 0:
        _nothing_scored(built, report)
        return

    flags = an.flag_columns(built)
    excluded = _exclusion_toggles(built, flags)
    effective = report
    if excluded:
        keep = pd.Series(True, index=ctx.frame.index)
        for column in excluded:
            keep &= ctx.frame[column].map(str) != flags[column]
        effective = an.subset_report(built, keep.tolist()) or report

    try:
        history = resources.get_store().list_evaluations()
    except Exception:
        history = None
    _kpi_row(built, effective, excluded, history)
    st.space("small")
    cell = _breakdown_section(built, effective)
    _examples(built, effective, cell)
    _slice_section(built, effective)
    _history_note(built, history)


def _exclusion_toggles(built: an.JobAnalysis, flags: dict[str, str]) -> list[str]:
    """Flag columns (0/1, yes/no) whose "true" rows can be left out of every
    number on the tab - right above the numbers they change. Returns the
    flags switched on."""
    if not flags:
        return []
    frame = built.context.frame
    on: list[str] = []
    with st.container(horizontal=True, vertical_alignment="center", gap="medium"):
        for flag, value in flags.items():
            n = int((frame[flag].map(str) == value).sum())
            if st.toggle(
                f"Exclude {fmt.md(flag)} = {fmt.md(value)} ({fmt.plural(n, 'row')})",
                key=_exclude_key(built, flag),
                help="Recomputes every number on this tab without those rows. History keeps "
                "the full evaluation.",
                persist_state=_PERSIST,
            ):
                on.append(flag)
        for flag in on:
            n = int((frame[flag].map(str) == flags[flag]).sum())
            st.badge(
                f"Excluding {flag} = {flags[flag]} · {fmt.plural(n, 'row')}",
                icon=":material/filter_alt:",
                color="orange",
            )
    return on


def _nothing_scored(built: an.JobAnalysis, report: EvalReport) -> None:
    ctx = built.context
    stats = built.result.stats
    with components.empty_card():
        if report.n_excluded >= report.n_input:
            # No row has an output: the gold column isn't the problem.
            if stats.pending or stats.errors:
                body = (
                    "The run stopped before any row produced an output. Generate the remaining "
                    "rows to score them."
                )
            else:
                body = "Every text in this run is blank, so there is nothing to compare."
            components.empty_state(":material/hourglass_empty:", "Nothing to score yet", body)
            return
        components.empty_state(
            ":material/rule:",
            "No rows could be scored",
            f"None of the gold values in {fmt.md(ctx.gold_column)} matched the label set "
            f"({fmt.md(', '.join(report.labels))}). Check that the gold column belongs to this task.",
        )


def _exclude_key(built: an.JobAnalysis, column: str) -> str:
    return f"results.exclude::{built.job_id}::{column}"


def _pp(delta: float | None) -> str | None:
    return None if delta is None else f"{delta * 100:+.1f} pp"


def _kpi_row(
    built: an.JobAnalysis,
    report: EvalReport,
    excluded: list[str],
    history: pd.DataFrame | None = None,
) -> None:
    ctx = built.context
    n_rows = ctx.n_rows
    now = an.kpis(report, n_rows)
    base: an.Kpis | None = None
    # What the deltas compare with, said once in the caption below: a
    # per-card "vs previous run" is cut off in five columns at 1024 px.
    baseline = ""
    if excluded:
        base = an.kpis(built.report, n_rows) if built.report is not None else None
        baseline = "changes are against all rows"
    else:
        previous = an.previous_evaluation(history if history is not None else pd.DataFrame(), built)
        if previous is not None:
            base = an.history_kpis(previous)
            when = fmt.timestamp(previous.get("created_at"))
            baseline = "changes are against the previous run of this task on this data" + (
                f" ({when})" if when else ""
            )

    def delta(attr: str, points: bool = True) -> tuple[str | None, str]:
        """(delta text, colour): blue = better, orange = worse (colour-blind
        safe); grey against the all-rows baseline, which isn't a better or
        worse run, just a different subset."""
        if base is None:
            return None, "off"
        a, b = getattr(now, attr), getattr(base, attr)
        if a is None or b is None or abs(a - b) < 1e-9:
            return None, "off"
        text = _pp(a - b) if points else f"{a - b:+.2f}"
        if excluded:
            return text, "gray"
        return text, "blue" if a > b else "orange"

    kappa_delta = delta("kappa", points=False)
    kappa_arrow = True
    if kappa_delta[0] is None:
        # No earlier run to compare with: say how strong the agreement is.
        band = an.kappa_band(now.kappa)
        if band:
            kappa_delta, kappa_arrow = (band.capitalize(), "off"), False

    cards = [
        ("Accuracy", fmt.percent(now.accuracy), delta("accuracy"), components.HELP_ACCURACY, True),
        ("Macro-F1", fmt.percent(now.macro_f1), delta("macro_f1"), components.HELP_MACRO_F1, True),
        (
            "Cohen's κ",
            fmt.number(now.kappa),
            kappa_delta,
            components.HELP_KAPPA + " Bands after Landis & Koch.",
            kappa_arrow,
        ),
        (
            "Valid answers",
            fmt.percent(now.valid_rate),
            delta("valid_rate"),
            "Scored rows whose output resolved to exactly one allowed label.",
            True,
        ),
        (
            "Coverage",
            fmt.percent(now.coverage),
            # Leaving rows out lowers coverage by design: no delta for that.
            (None, "off") if excluded else delta("coverage"),
            "Scored rows out of all rows in the run. Blank texts, failed rows and gold values "
            "outside the label set aren't scored.",
            True,
        ),
    ]
    if report.mode == "exact":
        cards = [cards[0], cards[2], cards[4]]
    # Equal columns: five bordered metrics never wrap unevenly at 1024 px.
    shown_any = False
    for column, (label, value, (change, colour), help_text, arrow) in zip(
        st.columns(len(cards), gap="small"), cards, strict=True
    ):
        with column:
            st.metric(
                label,
                value,
                delta=change,
                delta_color=colour,  # type: ignore[arg-type]
                delta_arrow="auto" if arrow else "off",
                help=help_text,
                border=True,
                height="stretch",
            )
            shown_any = shown_any or (bool(change) and arrow)

    # The note describes the run as a whole; rows left out by a toggle are
    # counted separately rather than lumped in with blank / failed rows.
    full = built.report or report
    notes = [f"{fmt.plural(report.n_scored, 'row')} scored"]
    if report.mode == "exact":
        notes.append("exact match after normalising case, spacing and punctuation")
    exclusion = an.exclusion_note(full, built.result.statuses)
    if exclusion:
        notes.append(exclusion)
    if excluded:
        flags = an.flag_columns(built)
        left_out = max(0, full.n_scored - report.n_scored)
        where = " or ".join(f"{fmt.md(c)} = {fmt.md(flags[c])}" for c in excluded)
        notes.append(f"{fmt.plural(left_out, 'scored row')} with {where} left out")
    majority = an.majority_baseline(report)
    if majority is not None:
        label, share = majority
        notes.append(f"always answering “{fmt.md(label)}” would score {share:.0%}")
    if shown_any and baseline:
        notes.append(baseline)
    st.caption(" · ".join(notes))


def _breakdown_section(built: an.JobAnalysis, report: EvalReport) -> tuple[str, str] | None:
    """Confusion matrix (or top confusions) beside the per-label table - or
    above it, full width, when there are too many labels for half a page.
    Returns the chosen matrix cell, if any."""
    per_label = report.mode == "labels"
    heatmap = per_label and len(report.labels) <= _TOP_CONFUSIONS_OVER
    wide = per_label and len(report.labels) > charts.COMPACT_LABELS
    cell = None
    if not per_label or wide:
        if heatmap:
            cell = _confusion_matrix(built, report)
        else:
            _top_confusions(built, report)
        if per_label:
            st.space("small")
            _per_class_section(report)
        return cell
    # The table gets the larger share: at 1024 px all its columns still fit.
    left, right = st.columns([5, 6], gap="large")
    with left:
        if heatmap:
            cell = _confusion_matrix(built, report)
        else:
            _top_confusions(built, report)
    with right:
        _per_class_section(report)
    return cell


def _section_heading(text: str, control: Callable[[], Any] | None = None) -> Any:
    """A subsection heading row of fixed height, with an optional control
    beside it, so headings side by side share one baseline."""
    with st.container(
        horizontal=True, vertical_alignment="center", gap="medium", height=44, border=False
    ):
        components.subheading(text, width="content")
        return control() if control is not None else None


def _top_confusions(built: an.JobAnalysis, report: EvalReport) -> None:
    _section_heading("Top confusions")
    top = an.top_confusions(report)
    if top.empty:
        st.caption("No wrong answers among the scored rows.")
        return
    labels = built.context.labels
    key = f"results.top_confusions::{built.job_id}"

    def _picked() -> None:
        event = st.session_state.get(key)
        rows = list(getattr(getattr(event, "selection", None), "rows", []) or [])
        if rows and 0 <= rows[0] < len(top):
            _set_cell(built, (str(top["gold"].iloc[rows[0]]), str(top["predicted"].iloc[rows[0]])))

    st.dataframe(
        top.assign(
            share=top["share"] * 100,
            predicted=[an.display_label(p, labels) for p in top["predicted"]],
        ),
        hide_index=True,
        on_select=_picked,
        selection_mode="single-row",
        key=key,
        height=_table_height(len(top), 10),
        column_config={
            "gold": st.column_config.TextColumn("Gold"),
            "predicted": st.column_config.TextColumn("Predicted"),
            "count": st.column_config.NumberColumn("Rows", format="%d"),
            "share": st.column_config.ProgressColumn(
                "Share of gold label", min_value=0, max_value=100, format="%.0f%%"
            ),
        },
    )
    st.caption("The most frequent wrong answers, by gold label. Click one to list its rows.")


def _cell_key(built: an.JobAnalysis) -> str:
    return f"{_K_CELL}::{built.job_id}"


def _set_cell(built: an.JobAnalysis, cell: tuple[str, str] | None) -> None:
    """Choose the (gold, predicted) cell whose rows are listed (None: all
    misclassified rows). A fresh chart key drops the chart's own selection
    outline; the chosen cell is outlined from the session key instead."""
    if cell is None:
        st.session_state.pop(_cell_key(built), None)
    else:
        st.session_state[_cell_key(built)] = cell
    st.session_state[_K_CM_NONCE] = st.session_state.get(_K_CM_NONCE, 0) + 1


def _clear_cell(built: an.JobAnalysis) -> None:
    _set_cell(built, None)


def _confusion_matrix(built: an.JobAnalysis, report: EvalReport) -> tuple[str, str] | None:
    """The heatmap; returns the chosen (gold, predicted) cell, if any.

    The chart's own selection is lost whenever the tab or page changes, so
    the chosen cell is also kept in a plain session key (updated by the
    selection callback) and outlined again from there."""
    chart_key = f"results.cm::{built.job_id}::{st.session_state.get(_K_CM_NONCE, 0)}"
    cell_key = _cell_key(built)

    confusion = report.confusion()

    def _remember_cell() -> None:
        cell = _selected_cell(st.session_state.get(chart_key))
        # An empty cell has no rows to list: clicking one clears the choice.
        if cell is not None and not _cell_count(confusion, cell):
            cell = None
        st.session_state[cell_key] = cell

    # Control on the left, next to the heading: the chart's hover toolbar
    # sits above its top-right corner and would cover a right-aligned one.
    mode = _section_heading(
        "Confusion matrix",
        lambda: st.segmented_control(
            "Cell values",
            ["Counts", "% of gold row"],
            default="Counts",
            required=True,
            key=_K_CM_MODE,
            label_visibility="collapsed",
            persist_state=_PERSIST,
        ),
    )
    remembered = st.session_state.get(cell_key)
    compact = len(confusion.columns) > charts.COMPACT_LABELS
    chart = charts.confusion_heatmap(
        confusion,
        normalize=mode == "% of gold row",
        dark=_dark(),
        selected=remembered,
    )
    st.altair_chart(
        chart,
        width="stretch",
        on_select=_remember_cell,
        selection_mode=charts.CONFUSION_SELECTION,
        key=chart_key,
        alt="Confusion matrix of gold labels against predicted labels",
    )
    # Below the chart: its hover toolbar would cover a caption above it.
    unit = " Cell values: % of the gold row." if compact and mode == "% of gold row" else ""
    st.caption(
        "Rows are gold labels, columns are predictions. Indigo cells are correct answers, "
        "orange cells errors; a stronger colour is a larger share of the gold row."
        f"{unit} Click a cell to list its rows; double-click to clear."
    )
    return st.session_state.get(cell_key)


def _cell_count(confusion: pd.DataFrame, cell: tuple[str, str]) -> int:
    """Rows in the (gold, predicted) cell of a confusion matrix (0 if absent)."""
    matrix = confusion.copy()
    matrix.index = [str(v) for v in matrix.index]
    matrix.columns = [str(v) for v in matrix.columns]
    try:
        return int(matrix.loc[cell[0], cell[1]])
    except (KeyError, TypeError, ValueError):
        return 0


def _dark() -> bool:
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return False


def _selected_cell(event: Any) -> tuple[str, str] | None:
    try:
        points = event.selection[charts.CONFUSION_SELECTION]
    except (AttributeError, KeyError, TypeError):
        return None
    if isinstance(points, dict):
        points = [points]
    for point in points or []:
        if isinstance(point, dict) and "gold" in point and "predicted" in point:
            return str(point["gold"]), str(point["predicted"])
    return None


_ALL_PAIRS = ""
"""Value of the pair picker meaning "all misclassified rows"."""
_PAIR_SEP = "\x1f"


def _pair_picker(built: an.JobAnalysis, report: EvalReport, cell: tuple[str, str] | None) -> None:
    """A keyboard-friendly way to choose a confusion: every wrong gold ->
    predicted pair, most frequent first. Kept in step with chart clicks."""
    labels = built.context.labels
    rows = report.rows
    wrong = rows.loc[~rows["correct"].astype(bool)]
    pairs = (
        wrong.groupby(["gold", "predicted"]).size().sort_values(ascending=False)
        if not wrong.empty
        else pd.Series(dtype="int64")
    )
    options = [_ALL_PAIRS] + [f"{g}{_PAIR_SEP}{p}" for g, p in pairs.index]
    names = {
        f"{g}{_PAIR_SEP}{p}": f"{g} → {an.display_label(p, labels)} · {n:,}"
        for (g, p), n in pairs.items()
    }
    key = f"results.pair::{built.job_id}"
    wanted = f"{cell[0]}{_PAIR_SEP}{cell[1]}" if cell is not None else _ALL_PAIRS
    if wanted not in options:  # e.g. a correct (diagonal) cell
        options.append(wanted)
        names[wanted] = f"{cell[0]} → {an.display_label(cell[1], labels)}" if cell else ""
    st.session_state[key] = wanted

    def _picked() -> None:
        value = st.session_state.get(key) or _ALL_PAIRS
        _set_cell(built, tuple(value.split(_PAIR_SEP, 1)) if value else None)

    st.selectbox(
        "Show rows for",
        options,
        key=key,
        format_func=lambda v: "All misclassified rows" if v == _ALL_PAIRS else names.get(v, v),
        on_change=_picked,
        label_visibility="collapsed",
        width=320,
    )


def _examples(built: an.JobAnalysis, report: EvalReport, cell: tuple[str, str] | None) -> None:
    ctx = built.context
    cols = built.columns
    scored = report.rows
    labels = ctx.labels
    if cell is not None:
        gold, pred = cell
        hit = (scored["gold"].astype(str) == gold) & (scored["predicted"].astype(str) == pred)
        shown_pred = an.display_label(pred, labels)
        title = f"Gold **{fmt.md(gold)}** predicted as **{fmt.md(shown_pred)}**"
    else:
        hit = ~scored["correct"].astype(bool)
        title = "Misclassified rows"
    picked = scored.loc[hit]
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        components.subheading(f"{title} · {len(picked):,}", width="content")
        st.space("stretch")
        if report.mode == "labels":
            _pair_picker(built, report, cell)
        elif cell is not None:
            st.button(
                "Show all misclassified rows",
                icon=":material/filter_alt_off:",
                type="tertiary",
                key=f"results.clear_cell::{built.job_id}",
                on_click=_clear_cell,
                args=(built,),
            )
    if picked.empty:
        st.caption("No rows." if cell is not None else "Every scored row is correct.")
        return
    positions = picked["row"].astype(int).tolist()
    frame = built.table.loc[positions]
    groups = set(an.group_candidates(built))
    flags = set(an.flag_columns(built))
    extra = [
        c
        for c in ctx.frame.columns
        if c not in (ctx.text_column, ctx.gold_column)
        and (c == ctx.id_column or c in groups or "rationale" in str(c).lower())
    ]
    predicted = [an.display_label(p, labels) for p in picked["predicted"]]
    view = pd.DataFrame(
        {
            "Text": frame[ctx.text_column].astype(object).tolist(),
            "Gold": picked["gold"].astype(str).tolist(),
            "Predicted": predicted,
            "Output": frame[cols.output].astype(object).tolist(),
        }
    )
    # The raw output only earns a column when it says more than the label.
    same = all(label_key(o) == label_key(p) for o, p in zip(view["Output"], predicted, strict=True))
    for column in extra:
        view[str(column)] = frame[column].astype(object).tolist()
    order = [*(str(c) for c in extra if c == ctx.id_column), "Text", "Gold", "Predicted"]
    order += [] if same else ["Output"]
    order += [str(c) for c in extra if c != ctx.id_column]
    config: dict[str, Any] = {
        "Text": st.column_config.TextColumn(width=340),
        "Gold": st.column_config.TextColumn(width=96),
        "Predicted": st.column_config.TextColumn(width=96),
        "Output": st.column_config.TextColumn(width="medium"),
    }
    for column in extra:
        name = str(column)
        if column == ctx.id_column:
            config[name] = st.column_config.TextColumn(
                "ID", help=f"From the {name} column.", pinned=True, width=80
            )
        elif "rationale" in name.lower():
            config[name] = st.column_config.TextColumn(width=240)
        elif column in flags or column in groups:
            config[name] = st.column_config.TextColumn(width=96)
    st.dataframe(
        view[order],
        hide_index=True,
        height=_table_height(len(view), 10),
        column_config=config,
    )


def _percent_column(label: str, help_text: str | None = None, width: int | None = None) -> Any:
    """A 0-100 progress bar printed as a whole percentage (values pre-scaled)."""
    return st.column_config.ProgressColumn(
        label, min_value=0, max_value=100, format="%.0f%%", help=help_text, width=width
    )


def _per_class_section(report: EvalReport) -> None:
    _section_heading("Per-label metrics")
    table = an.per_class_table(report)

    def percent_text(value: Any) -> str:
        number = pd.to_numeric(value, errors="coerce")
        return fmt.DASH if pd.isna(number) else f"{number * 100:.0f}%"

    # Compact enough for a half-width column at 1024 px: F1 (the sort key)
    # right after the label as a bar, precision / recall as plain numbers
    # ("–" when undefined, rather than the grid's "None"). Predicted counts
    # are the heatmap's column totals.
    view = table.assign(
        f1=pd.to_numeric(table["f1"], errors="coerce") * 100,
        precision=table["precision"].map(percent_text),
        recall=table["recall"].map(percent_text),
    )
    st.dataframe(
        view,
        hide_index=True,
        column_order=("label", "f1", "precision", "recall", "support"),
        height=_table_height(len(view), 12),
        column_config={
            "label": st.column_config.TextColumn("Label"),
            "f1": _percent_column("F1", "Harmonic mean of precision and recall.", 104),
            "precision": st.column_config.TextColumn(
                "Precision",
                width=76,
                alignment="right",
                help="Of the rows predicted as this label, the share that were right.",
            ),
            "recall": st.column_config.TextColumn(
                "Recall",
                width=64,
                alignment="right",
                help="Of the rows with this gold label, the share the model found.",
            ),
            "support": st.column_config.NumberColumn(
                "Rows", format="%d", width=56, help="Rows with this gold label."
            ),
        },
    )
    st.caption(
        "Weakest labels first. A dash means undefined: the label was never predicted "
        "(precision) or never in the gold column (recall)."
    )


def _slice_section(built: an.JobAnalysis, report: EvalReport) -> None:
    candidates = an.group_candidates(built)
    if not candidates:
        return
    st.space("small")
    with st.container(horizontal=True, vertical_alignment="center", gap="medium"):
        components.subheading("By slice", width="content")
        column = st.selectbox(
            "Slice by",
            candidates,
            key=f"results.slice::{built.job_id}",
            help="Columns with 2-20 distinct values. Accuracy is shown per value.",
            label_visibility="collapsed",
            width=240,
            persist_state=_PERSIST,
        )
    slices = an.slice_table(built, column, report)
    if slices.empty:
        st.caption("No scored rows to slice.")
        return
    st.dataframe(
        slices.assign(accuracy=slices["accuracy"] * 100),
        hide_index=True,
        width=720,
        height=_table_height(len(slices), 12),
        column_config={
            "group": st.column_config.TextColumn(column),
            "n": st.column_config.NumberColumn("Rows scored", format="%d", width=104),
            "correct": st.column_config.NumberColumn("Correct", format="%d", width=88),
            "accuracy": _percent_column("Accuracy"),
        },
    )
    small = int((pd.to_numeric(slices["n"], errors="coerce") < SMALL_SLICE).sum())
    notes = [f"Overall accuracy {fmt.percent(report.accuracy)}."]
    if small:
        notes.append(
            f"{fmt.plural(small, 'slice')} {'has' if small == 1 else 'have'} fewer than "
            f"{SMALL_SLICE} scored rows: too few to read much into."
        )
    st.caption(" ".join(notes))


def _retry_history(job: Job, manager: JobManager) -> None:
    """The job's finish hook couldn't save its evaluation: try once more (at
    most once per job, whichever session gets here first)."""
    try:
        an.history_entry(resources.get_store(), job, manager.jobs())
    except Exception as exc:
        job.request.context[an.FINISH_ERROR_KEY] = f"{type(exc).__name__}: {exc}"


def _history_note(built: an.JobAnalysis, history: pd.DataFrame | None) -> None:
    """Whether this evaluation is in history - as the run's finish hook
    logged it; viewing a run never writes history (see an.history_entry)."""
    if built.partial:
        # Partial scores would sit on the leaderboard as if the run were done.
        st.space("small")
        st.caption(
            ":material/info: Not saved to history: this run is incomplete. Its evaluation is "
            "saved once the remaining rows are generated."
        )
        return
    job = resources.get_job_manager().get(built.job_id)
    context = job.request.context if job is not None else {}
    entry = context.get(an.HISTORY_ID_KEY)
    if entry is None or history is None or "id" not in history.columns:
        return
    line = history.loc[history["id"].astype("int64") == int(entry)]
    if line.empty or str(line.iloc[0].get("run_id")) != built.result.run_id:
        note = (
            ":material/info: Not in history: its line was deleted on the History page. "
            "Run this configuration again to add it back."
        )
        st.space("small")
        st.caption(note)
        return
    st.space("small")
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        scored = pd.to_numeric(line.iloc[0].get("n_scored"), errors="coerce")
        report = built.report
        if report is not None and pd.notna(scored) and int(scored) != report.n_scored:
            # A later run of the same configuration (a retry) updated the line.
            st.caption(
                ":material/info: History holds a later run of this configuration "
                f"({fmt.plural(int(scored), 'row')} scored)",
                width="content",
            )
        else:
            st.caption(":material/check: Saved to history", width="content")
        st.page_link(nav.page("history"), label="Compare runs", icon=":material/leaderboard:")


# ==========================================================================
# Details tab
# ==========================================================================


def _card(title: str, rows: list[tuple[str, Any]]) -> None:
    """A Details card: title plus aligned label / value lines. Cards in one
    row share a height."""
    with st.container(border=True, height="stretch"):
        st.subheader(title, anchor=False)
        components.facts(rows, label_width=190)


def tokens_used(stats: Any, spec: models.ModelSpec | None) -> str:
    """ "12k in · 300 out · ≈ $0.01 at list prices" for a cloud run (the
    cost only when the model's price is known); "None" when nothing was
    sent."""
    sent, generated = int(stats.input_tokens or 0), int(stats.output_tokens or 0)
    if not (sent or generated):
        return "None (all rows from saved outputs)" if stats.cached else "None"
    words = f"{fmt.token_count(sent)} in · {fmt.token_count(generated)} out"
    if spec is not None and spec.price:
        cost = (sent * spec.price[0] + generated * spec.price[1]) / 1e6
        words += f" · ≈ {fmt.money(cost)} at list prices"
    return words


def _details_tab(job: Job, snap: JobSnapshot, built: an.JobAnalysis) -> None:
    ctx = built.context
    result = built.result
    cfg = result.config
    stats = result.stats
    request = job.request
    gen_s = max(0.0, stats.elapsed_s - stats.load_s)
    rate = (stats.generated / gen_s) if stats.generated and gen_s > 0 else None

    def code(value: Any) -> str | None:
        return None if value in (None, "") else f"`{str(value).replace('`', '')}`"

    def text(value: Any) -> str | None:
        return None if value in (None, "") else fmt.md(value)

    # Row by row, so the cards beside each other line up.
    run_col, rows_col = st.columns(2, gap="medium")
    with run_col:
        _card(
            "Run",
            [
                ("Run id", code(result.run_id)),
                ("Status", components.job_status(snap)[0]),
                ("Started", fmt.timestamp(snap.started_at)),
                ("Finished", fmt.timestamp(snap.finished_at)),
                ("Model load", fmt.duration(stats.load_s) if stats.load_s else "Not needed"),
                ("Generation", fmt.duration(gen_s)),
                ("Total", fmt.duration(stats.elapsed_s)),
                ("Throughput", fmt.rate(rate)),
            ],
        )
    with rows_col:
        _card(
            "Rows",
            [
                ("Rows", f"{stats.total:,}"),
                ("Generated", f"{stats.generated:,}"),
                ("From saved outputs", f"{stats.cached:,}"),
                ("Duplicate texts", f"{stats.duplicates:,} (answered once)"),
                ("Blank texts", f"{stats.blank:,}"),
                ("Errors", f"{stats.errors:,}"),
                ("Not generated", f"{stats.pending:,}"),
                ("Model calls", f"{stats.model_calls:,}"),
            ],
        )
    model_col, data_col = st.columns(2, gap="medium")
    with model_col:
        provider = models.API_PROVIDERS.get(models.api_provider(cfg.model_id) or "")
        if provider:  # a cloud model: the same number of rows, so the cards line up
            who = ctx.extra.get("recipient") or provider
            weights = ("Runs on", f"{provider}'s servers" if who == provider else text(who))
            batch = ("Requests at once", f"{request.batch_size:,}")
            used = tokens_used(stats, models.find_model(cfg.model_id))
            last = ("Tokens used", fmt.no_math(used))
        else:
            weights = ("Weights", fmt.precision(cfg.precision))
            batch = ("Batch size", f"{request.batch_size:,}")
            last = ("Max characters per batch", f"{request.max_batch_chars:,}")
        _card(
            "Model",
            [
                ("Model", text(cfg.model_label)),
                ("Model id", code(cfg.model_id)),
                weights,
                ("Max output tokens", f"{cfg.max_new_tokens:,}"),
                ("Max input characters", f"{cfg.max_input_chars:,}"),
                batch,
                ("Processing order", engine.ORDER_LABELS.get(request.order)),
                last,
            ],
        )
    with data_col:
        _card(
            "Data",
            [
                ("Dataset", text(ctx.dataset_name)),
                ("Selection", text(ctx.selection)),
                ("Fingerprint", code(ctx.dataset_fingerprint)),
                ("Text column", code(ctx.text_column)),
                ("ID column", code(ctx.id_column)),
                ("Gold column", code(ctx.gold_column)),
            ],
        )

    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center", gap="small"):
            st.subheader(f"Prompt · {ctx.preset_title}", anchor=False, width="content")
            if ctx.template_edited:
                st.badge("Edited", color="orange")
            if ctx.labels:
                st.badge(", ".join(ctx.labels), icon=":material/rule:", color="blue")
        if ctx.var and "{var}" in ctx.template_raw:
            st.caption(f"Variable: {fmt.md(ctx.var)}")
        st.code(cfg.template, language=None, wrap_lines=True)
        st.caption(
            "{text} is replaced by each row. Saved outputs are reused when you run the same model, "
            "prompt and settings again."
        )
