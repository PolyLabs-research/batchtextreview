"""Derived views of a finished job: results table, evaluation, history logging.

Pure Python (no Streamlit import): used by the Results page and by the job's
``on_finish`` hook, which runs on the worker thread so a run's evaluation is
logged to history even if nobody is looking at the page when it ends.
"""

from __future__ import annotations

import contextlib
import inspect
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

import pandas as pd

from batchtextreview import evaluation, results
from batchtextreview.labels import INVALID, canonical_labels, label_key, parse_json_output
from batchtextreview.ui import state

if TYPE_CHECKING:
    from batchtextreview.engine import RunResult
    from batchtextreview.evaluation import EvalReport
    from batchtextreview.jobs import Job
    from batchtextreview.store import Store
    from batchtextreview.ui.state import RunContext

CORRECT_COL = "CORRECT"

FINISH_ERROR_KEY = "finish_error"
"""``job.request.context`` key where :func:`finish_hook` records a failure."""
HISTORY_ID_KEY = "_history_id"
"""``job.request.context`` key where :func:`finish_hook` records the history
line it wrote (None when there was nothing to log). Private (leading
underscore), so a retry of the job doesn't inherit it."""
_FALLBACK_KEY = "_history_fallback"
"""Set on a job's context once :func:`history_entry` has retried logging."""
_CACHE_KEY = "_analysis"
"""``job.request.context`` key holding the memoised :class:`JobAnalysis`."""

# Row filters, in display order (see :func:`row_filters`).
FILTER_ALL = "All"
FILTER_CORRECT = "Correct"
FILTER_WRONG = "Wrong"
FILTER_INVALID = "Invalid"
FILTER_ERRORS = "Errors"
FILTER_BLANK = "Blank"
FILTER_NOT_RUN = "Not generated"
FILTER_NO_JSON = "Invalid JSON"


@dataclass
class ResultColumns:
    """Actual names of the result columns in :attr:`JobAnalysis.table`.

    They differ from the defaults when the input already had such a column
    (re-running on an exported file gives OUTPUT_2, LABEL_2, ...).
    """

    output: str = results.OUTPUT_COL
    status: str = results.STATUS_COL
    label: str | None = None
    error: str | None = None
    correct: str | None = None
    json_valid: str | None = None
    json_fields: list[str] = field(default_factory=list)

    @property
    def added(self) -> list[str]:
        """Every column the run added, in table order of importance."""
        names = [self.output, self.label, self.correct, self.status, self.error]
        names += self.json_fields
        names.append(self.json_valid)
        return [n for n in names if n]


@dataclass
class JobAnalysis:
    job_id: str
    context: RunContext
    result: RunResult
    table: pd.DataFrame
    """``results.build_results_frame(context.frame, result, labels=context.labels,
    expand_json=context.output_kind == "json")`` plus, when a report exists,
    a CORRECT column (True / False / None for unscored rows) - the table the
    Results page shows and exports."""
    scorable: list[bool]
    """``results.scorable_mask(result)``."""
    report: EvalReport | None
    """Evaluation against ``context.gold_column`` - labels mode when the task
    has labels; exact-match mode only when ``context.extra.get("exact_match")``
    is true; else None."""
    columns: ResultColumns = field(default_factory=ResultColumns)
    """Where the result columns ended up in :attr:`table`."""
    exports: dict[str, bytes] = field(default_factory=dict, repr=False, compare=False)
    """Memoised export payloads by format ("csv", "xlsx"), filled on demand."""
    memo: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)
    """Other derived values that never change for a finished job (slice
    candidates, flag columns), so page reruns don't recompute them."""

    @property
    def partial(self) -> bool:
        """The run was cancelled or stopped early (rows were left unrun)."""
        return is_partial(self.result)

    @property
    def statuses(self) -> pd.Series:
        """Row statuses aligned with :attr:`table`."""
        return pd.Series(list(self.result.statuses), index=self.table.index, dtype=object)

    def label_counts(self) -> pd.DataFrame:
        """Predicted label distribution over scorable rows: columns
        ``label``, ``count`` (label-set order, INVALID last; zero rows kept
        for labels never predicted). Empty frame for open-ended tasks."""
        column = self.columns.label
        if not self.context.labels or column is None or column not in self.table.columns:
            return pd.DataFrame(
                {"label": pd.Series([], dtype=object), "count": pd.Series([], dtype="int64")}
            )
        values = self.table.loc[pd.Series(self.scorable, index=self.table.index), column]
        counts = values.astype(str).value_counts()
        order = canonical_labels(self.context.labels)
        # Defensive: anything the label column holds outside the set (it
        # shouldn't) is listed after the set, before INVALID.
        order += [v for v in counts.index if v not in order and v not in (INVALID, "")]
        if counts.get(INVALID, 0):
            order.append(INVALID)
        return pd.DataFrame(
            {
                "label": pd.Series(order, dtype=object),
                "count": pd.Series([int(counts.get(v, 0)) for v in order], dtype="int64"),
            }
        )

    def gold_counts(self) -> dict[str, int]:
        """Scored rows per gold label (labels mode only; empty otherwise), so
        the predicted-label chart can show what the gold column expected."""
        report = self.report
        if report is None or report.mode != "labels" or report.rows.empty:
            return {}
        counts = report.rows["gold"].astype(str).value_counts()
        return {str(k): int(v) for k, v in counts.items()}

    def output_matches_label(self) -> bool:
        """True when every output on a scorable row reads the same as its
        resolved label (case, spacing and wrapping punctuation aside): the
        raw OUTPUT column then repeats LABEL and adds nothing to the table."""
        column = self.columns.label
        if column is None or column not in self.table.columns:
            return False
        labels = self.table[column].tolist()
        for keep, output, label in zip(self.scorable, self.result.outputs, labels, strict=True):
            if not keep:
                continue
            if is_blank(label) or str(label) == INVALID:
                return False
            if label_key(output) != label_key(label):
                return False
        return True

    def output_lengths(self) -> list[int]:
        """Character length of each non-empty output on a scorable row."""
        return [len(text) for text in self._scored_outputs() if text]

    def output_diagnostics(self) -> dict[str, float | int | None]:
        """For open-ended / JSON tasks: n_outputs, empty_rate, mean_chars,
        median_chars, json_valid_rate (JSON tasks only, else None).

        All computed over scorable rows (generated or saved outputs); the
        character statistics ignore empty outputs, which ``empty_rate``
        already counts. Rates are None when there are no outputs."""
        outputs = self._scored_outputs()
        n = len(outputs)
        lengths = pd.Series([len(t) for t in outputs if t], dtype="float64")
        json_rate: float | None = None
        if self.context.output_kind == "json" and n:
            json_rate = sum(self.json_valid()) / n
        return {
            "n_outputs": n,
            "empty_rate": (sum(1 for t in outputs if not t) / n) if n else None,
            "mean_chars": float(lengths.mean()) if len(lengths) else None,
            "median_chars": float(lengths.median()) if len(lengths) else None,
            "json_valid_rate": json_rate,
        }

    def json_valid(self) -> list[bool]:
        """For each scorable row (in order), whether its output parsed as JSON."""
        column = self.columns.json_valid
        mask = pd.Series(self.scorable, index=self.table.index)
        if column and column in self.table.columns:
            return [truthy(v) for v in self.table.loc[mask, column]]
        return [parse_json_output(t) is not None for t in self._scored_outputs()]

    def parsed_json(self, row: int) -> dict | list | None:
        """The parsed JSON output of table row ``row`` (None if it doesn't parse)."""
        return parse_json_output(self.result.outputs[row])

    def _scored_outputs(self) -> list[str]:
        return [
            str(o or "") for o, keep in zip(self.result.outputs, self.scorable, strict=True) if keep
        ]


# --------------------------------------------------------------------------
# Building the analysis
# --------------------------------------------------------------------------


def analyse(job: Job) -> JobAnalysis | None:
    """Build the analysis for a job that has a result and a RunContext in
    ``job.request.context[state.CONTEXT_KEY]``; None otherwise."""
    result = getattr(job, "result", None)
    request = getattr(job, "request", None)
    context = getattr(request, "context", None) or {}
    ctx = context.get(state.CONTEXT_KEY)
    if result is None or not isinstance(ctx, state.RunContext):
        return None

    frame = ctx.frame.reset_index(drop=True)
    table = results.build_results_frame(
        frame, result, labels=ctx.labels, expand_json=ctx.output_kind == "json"
    )
    columns = _locate_columns(frame, table, labels=bool(ctx.labels))
    scorable = list(results.scorable_mask(result))
    report = _evaluate(ctx, result, scorable)
    if report is not None:
        correct: list[bool | None] = [None] * len(table)
        for row, ok in zip(report.rows["row"], report.rows["correct"], strict=True):
            correct[int(row)] = bool(ok)
        name = results.free_column_name(table, CORRECT_COL)
        table[name] = pd.Series(correct, index=table.index, dtype=object)
        columns.correct = name
    return JobAnalysis(
        job_id=str(getattr(job, "id", "")),
        context=ctx,
        result=result,
        table=table,
        scorable=scorable,
        report=report,
        columns=columns,
    )


def get_analysis(job: Job) -> JobAnalysis | None:
    """:func:`analyse`, memoised on the job (results never change once a job
    has one), so every rerun and every browser session shares one copy."""
    result = getattr(job, "result", None)
    request = getattr(job, "request", None)
    context = getattr(request, "context", None)
    if result is None or context is None:
        return None
    cached = context.get(_CACHE_KEY)
    if isinstance(cached, JobAnalysis) and cached.result is result:
        return cached
    built = analyse(job)
    if built is not None:
        context[_CACHE_KEY] = built
    return built


def _evaluate(ctx: RunContext, result: RunResult, scorable: Sequence[bool]) -> EvalReport | None:
    gold = ctx.gold_column
    if not gold or gold not in ctx.frame.columns:
        return None
    golds = ctx.frame[gold].tolist()
    if ctx.labels:
        return evaluation.evaluate(result.outputs, golds, allowed=ctx.labels, include=scorable)
    if ctx.extra.get("exact_match"):
        return evaluation.evaluate(result.outputs, golds, allowed=None, include=scorable)
    return None


def _locate_columns(frame: pd.DataFrame, table: pd.DataFrame, labels: bool) -> ResultColumns:
    """Find the columns build_results_frame added (they may carry a _N suffix)."""
    original = {str(c) for c in frame.columns}
    added = [str(c) for c in table.columns if str(c) not in original]

    def find(wanted: str) -> str | None:
        pattern = re.compile(rf"{re.escape(wanted)}(?:_\d+)?")
        return next((c for c in added if pattern.fullmatch(c)), None)

    output = find(results.OUTPUT_COL) or results.OUTPUT_COL
    found = ResultColumns(
        output=output,
        status=find(results.STATUS_COL) or results.STATUS_COL,
        label=find(results.LABEL_COL) if labels else None,
        error=find(results.ERROR_COL),
    )
    json_cols = [
        c for c in added if c.startswith(f"{results.OUTPUT_COL}.") or c.startswith(f"{output}.")
    ]
    found.json_valid = next((c for c in json_cols if c.endswith(".valid_json")), None)
    found.json_fields = [c for c in json_cols if c != found.json_valid]
    return found


# --------------------------------------------------------------------------
# History
# --------------------------------------------------------------------------


def is_partial(result: RunResult) -> bool:
    """True for a run that was cancelled or stopped early: its scores cover
    only the rows that ran, so they don't belong on the leaderboard yet."""
    return bool(result.cancelled or result.aborted_reason)


def log_evaluation(
    store: Store, analysis: JobAnalysis, created_at: datetime | None = None
) -> int | None:
    """Write the analysis' evaluation to history (deduplicated by the store).
    Returns the history row id, or None when there is no report, nothing was
    scored, or the run is partial (:func:`is_partial`) - finishing the
    remaining rows produces a complete run with the same run id, which is
    logged then. ``created_at`` dates the line (default: now)."""
    report = analysis.report
    if report is None or report.n_scored == 0 or analysis.partial:
        return None
    record = history_record(analysis)
    if created_at is not None:
        record["created_at"] = created_at.isoformat(timespec="seconds")
    return int(store.log_evaluation(record))


def history_entry(store: Store, job: Job, jobs: Sequence[Job] = ()) -> int | None:
    """The history line holding ``job``'s evaluation, as recorded by its
    finish hook, or None.

    The finish hook is the only writer: viewing a run never writes it again
    (that would bring back a line deleted on the History page, overwrite a
    retry's line with the older numbers, and move the line's date). Only
    when the hook failed (``FINISH_ERROR_KEY``) is logging tried once more,
    once per job for every browser session (marked on the shared job
    context), dated when the job finished - and never when ``jobs`` holds a
    newer run of the same configuration on the same rows (a retry), whose
    line is the newer one."""
    context = getattr(getattr(job, "request", None), "context", None)
    if context is None:
        return None
    if FINISH_ERROR_KEY not in context or context.get(_FALLBACK_KEY):
        return context.get(HISTORY_ID_KEY)
    context[_FALLBACK_KEY] = True
    built = get_analysis(job)
    if built is None or _superseded(job, built, jobs):
        return None
    finished = _finished_at(job)
    logged = log_evaluation(store, built, created_at=finished)
    context[HISTORY_ID_KEY] = logged
    context.pop(FINISH_ERROR_KEY, None)
    return logged


def _finished_at(job: Job) -> datetime | None:
    try:
        return job.snapshot().finished_at
    except Exception:
        return None


def _superseded(job: Job, built: JobAnalysis, jobs: Sequence[Job]) -> bool:
    """True when ``jobs`` holds a run started after ``job`` with the same
    run id and dataset fingerprint (e.g. its retry)."""
    run_id = built.result.run_id
    fingerprint = built.context.dataset_fingerprint
    started = built.context.created_at
    for other in jobs:
        if other is job or getattr(other, "id", None) == getattr(job, "id", None):
            continue
        result = getattr(other, "result", None)
        ctx = getattr(getattr(other, "request", None), "context", {}).get(state.CONTEXT_KEY)
        if result is None or not isinstance(ctx, state.RunContext):
            continue
        if (
            result.run_id == run_id
            and ctx.dataset_fingerprint == fingerprint
            and ctx.created_at > started
        ):
            return True
    return False


def history_record(analysis: JobAnalysis) -> dict[str, Any]:
    """The ``Store.log_evaluation`` record for an analysis with a report.

    Built by ``results.history_record``; run-context fields it doesn't take
    as arguments (selection, timings, edited flag, app version) are added to
    the record for stores that have those columns (others ignore them)."""
    from batchtextreview import __version__

    report = analysis.report
    if report is None:
        raise ValueError("This run has no evaluation to record.")
    ctx = analysis.context
    result = analysis.result
    stats = result.stats
    gen_seconds = max(0.0, float(stats.elapsed_s) - float(stats.load_s))
    rows_per_s = (
        round(stats.generated / gen_seconds, 4) if stats.generated and gen_seconds > 0 else None
    )
    extras: dict[str, Any] = {
        "selection": ctx.selection,
        "template_edited": bool(ctx.template_edited),
        "preset_title": ctx.preset_title,
        "rows_per_s": rows_per_s,
        "elapsed_s": round(float(stats.elapsed_s), 3),
        "app_version": __version__,
    }
    kwargs: dict[str, Any] = {
        "preset_key": ctx.preset_key,
        # Only meaningful when the template actually uses it.
        "var": ctx.var if "{var}" in (ctx.template_raw or "") else None,
        "dataset": ctx.dataset_name,
        "dataset_fingerprint": ctx.dataset_fingerprint,
        "text_column": ctx.text_column,
        "gold_column": ctx.gold_column or "",
        "n_rows": ctx.n_rows,
        "source": "app",
    }
    accepted = _accepted_kwargs(results.history_record)
    for key, value in extras.items():
        if accepted is None or key in accepted:
            kwargs[key] = value
    try:
        record = dict(results.history_record(report, result, **kwargs))
    except TypeError:
        # Older signature: retry with the documented arguments only.
        for key in extras:
            kwargs.pop(key, None)
        record = dict(results.history_record(report, result, **kwargs))
    for key, value in extras.items():
        record.setdefault(key, value)
    return record


def _accepted_kwargs(func: Callable[..., Any]) -> set[str] | None:
    """Keyword names ``func`` accepts; None when it takes ``**kwargs``."""
    try:
        params = inspect.signature(func).parameters.values()
    except (TypeError, ValueError):
        return set()
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params):
        return None
    return {p.name for p in params}


def finish_hook(store: Store) -> Callable[[Job], None]:
    """A ``JobRequest.on_finish`` callback: analyse + log_evaluation, never
    raising (exceptions are swallowed after being recorded on
    ``job.request.context["finish_error"]``). The history line's id (or
    None) is recorded on the context under :data:`HISTORY_ID_KEY`."""

    def _on_finish(job: Job) -> None:
        try:
            built = get_analysis(job)
            logged = log_evaluation(store, built) if built is not None else None
            job.request.context[HISTORY_ID_KEY] = logged
        except Exception as exc:  # the worker must never die here
            with contextlib.suppress(Exception):
                job.request.context[FINISH_ERROR_KEY] = f"{type(exc).__name__}: {exc}"

    return _on_finish


def previous_evaluation(history: pd.DataFrame, analysis: JobAnalysis) -> dict[str, Any] | None:
    """The newest earlier history line comparable with this analysis: same
    dataset fingerprint, gold column, task (preset) and label set, from a
    different configuration (run id). None when there isn't one."""
    if analysis.report is None or history is None or history.empty:
        return None
    ctx = analysis.context
    needed = {"dataset_fingerprint", "gold_column", "preset", "run_id"}
    if not needed.issubset(history.columns):
        return None
    same = (
        (history["dataset_fingerprint"].astype(object) == ctx.dataset_fingerprint)
        & (history["gold_column"].astype(object) == (ctx.gold_column or ""))
        & (history["preset"].astype(object) == ctx.preset_key)
        & (history["run_id"].astype(object) != analysis.result.run_id)
    )
    candidates = history.loc[same.fillna(False).astype(bool)]
    if analysis.report.mode == "labels" and "labels" in candidates.columns:
        wanted = list(analysis.report.labels)
        keep = [_label_list(v) in (None, wanted) for v in candidates["labels"]]
        candidates = candidates.loc[keep]
    if candidates.empty:
        return None
    if "created_at" in candidates.columns:
        # Only lines logged before this run started count as "previous". The
        # store keeps whole seconds, so a line logged in the same second as
        # this run started may be either; leave it out rather than compare
        # against a run that finished after this one began.
        when = pd.to_datetime(candidates["created_at"], errors="coerce")
        started = pd.Timestamp(ctx.created_at).floor("s")
        earlier = (when < started).fillna(False).astype(bool)
        candidates = candidates.loc[earlier].assign(_when=when[earlier])
        candidates = candidates.sort_values("_when", ascending=False)
        if candidates.empty:
            return None
    return candidates.iloc[0].drop(labels=["_when"], errors="ignore").to_dict()


def _label_list(value: Any) -> list[str] | None:
    if isinstance(value, (list, tuple)):
        return canonical_labels([str(v) for v in value])
    return None


# --------------------------------------------------------------------------
# KPIs
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Kpis:
    accuracy: float | None
    macro_f1: float | None
    kappa: float | None
    valid_rate: float | None
    """Share of scored rows whose answer resolved to the label set."""
    coverage: float | None
    """Scored rows / rows in the run."""


def kpis(report: EvalReport, n_rows: int) -> Kpis:
    """Headline numbers for the Evaluation tab."""
    valid = (1 - report.n_pred_invalid / report.n_scored) if report.n_scored else None
    return Kpis(
        accuracy=report.accuracy,
        macro_f1=report.macro_f1,
        kappa=report.kappa,
        valid_rate=valid if report.mode == "labels" else None,
        coverage=(report.n_scored / n_rows) if n_rows else None,
    )


def history_kpis(row: dict[str, Any]) -> Kpis:
    """The same numbers recomputed from a history line."""
    n_scored = _num(row.get("n_scored"))
    n_invalid = _num(row.get("n_invalid"))
    n_rows = _num(row.get("n_rows"))
    labelled = _label_list(row.get("labels")) is not None
    return Kpis(
        accuracy=_num(row.get("accuracy")),
        macro_f1=_num(row.get("macro_f1")),
        kappa=_num(row.get("kappa")),
        valid_rate=(1 - (n_invalid or 0) / n_scored) if n_scored and labelled else None,
        coverage=(n_scored / n_rows) if n_scored is not None and n_rows else None,
    )


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def exclusion_note(report: EvalReport, statuses: Sequence[str] | None = None) -> str:
    """Why rows weren't scored, naming only the reasons that apply: "left
    out: 1 blank text" or "left out: 25 failed, 36 not generated · 1 row has a
    gold label outside the label set". Empty when nothing was left out.

    ``statuses`` are the run's row statuses (``RunResult.statuses``); without
    them the reasons can't be told apart and the note says so generically."""
    parts: list[str] = []
    if report.n_excluded:
        reasons: list[str] = []
        if statuses is not None:
            counts = {s: sum(1 for v in statuses if str(v) == s) for s in _EXCLUDED_WORDS}
            for status, (one, many) in _EXCLUDED_WORDS.items():
                n = counts[status]
                if n:
                    reasons.append(f"{n:,} {one if n == 1 else many}")
        if reasons:
            parts.append("left out: " + ", ".join(reasons))
        else:
            verb = "row" if report.n_excluded == 1 else "rows"
            parts.append(f"{report.n_excluded:,} {verb} left out (no output to score)")
    if report.n_gold_unscorable:
        n = report.n_gold_unscorable
        has = "row has" if n == 1 else "rows have"
        if report.mode == "labels":
            parts.append(f"{n:,} {has} a missing gold label or one outside the label set")
        else:
            parts.append(f"{n:,} {has} no gold value")
    return " · ".join(parts)


_EXCLUDED_WORDS: dict[str, tuple[str, str]] = {
    "blank": ("blank text", "blank texts"),
    "error": ("failed", "failed"),
    "pending": ("not generated", "not generated"),
}
"""Row statuses that leave a row out of scoring, as (one, many) phrases."""


def majority_baseline(report: EvalReport) -> tuple[str, float] | None:
    """(label, accuracy) of always answering the most common gold label:
    the floor a classifier has to beat. None outside labels mode."""
    if report.mode != "labels" or report.rows.empty:
        return None
    counts = report.rows["gold"].astype(str).value_counts()
    if counts.empty:
        return None
    return str(counts.index[0]), float(counts.iloc[0]) / float(counts.sum())


_KAPPA_BANDS: tuple[tuple[float, str], ...] = (
    (0.0, "worse than chance"),
    (0.2, "slight agreement"),
    (0.4, "fair agreement"),
    (0.6, "moderate agreement"),
    (0.8, "substantial agreement"),
)


def kappa_band(kappa: float | None) -> str | None:
    """Landis & Koch's reading of Cohen's κ ("fair agreement", ...)."""
    value = _num(kappa)
    if value is None:
        return None
    for upper, words in _KAPPA_BANDS:
        if value < upper:
            return words
    return "almost perfect agreement"


# --------------------------------------------------------------------------
# Copy: run summary, failure explanations, label names
# --------------------------------------------------------------------------

INVALID_NAME = "Invalid"
"""How :data:`labels.INVALID` reads in the UI."""
INVALID_NAME_ALT = "Not in label set"
"""Used instead when the label set itself has an "invalid" label."""


def display_label(value: Any, labels: Sequence[str] | None = None) -> str:
    """A label as the UI shows it: INVALID becomes "Invalid" (or "Not in
    label set" when ``labels`` already has an "invalid" label); anything
    else is unchanged."""
    text = "" if is_blank(value) else str(value)
    if text != INVALID:
        return text
    taken = {str(v).strip().casefold() for v in (labels or [])}
    return INVALID_NAME_ALT if INVALID_NAME.casefold() in taken else INVALID_NAME


def run_summary(result: RunResult) -> str:
    """The run's one-line summary, shared by the Results header and the Run
    page's "Last run" card, e.g. "61 of 62 rows have outputs · 61 generated ·
    1 blank text skipped · 1 duplicate text shares an answer".

    Only the counts that apply are listed, in the UI's terms ("saved
    outputs", never "cache"). Texts cut to ``max_input_chars`` and outputs
    that hit ``max_new_tokens`` are mentioned too; the cancellation or abort
    reason is left to the page's status badge and problem card."""
    stats = result.stats
    total = stats.total
    if not total:
        return "No rows"
    done = stats.generated + stats.cached
    if done == total:
        head = "The row has an output" if total == 1 else f"All {total:,} rows have outputs"
    else:
        verb = "row has" if total == 1 else "rows have"
        head = f"{done:,} of {total:,} {verb} {'an output' if total == 1 else 'outputs'}"
    stopped = "run cancelled" if result.cancelled else "run stopped"
    parts = [head]
    parts += [
        text
        for count, text in (
            (stats.generated, f"{stats.generated:,} generated"),
            (stats.cached, f"{stats.cached:,} reused from saved outputs"),
            (stats.blank, f"{_count(stats.blank, 'blank text')} skipped"),
            (stats.errors, f"{stats.errors:,} failed"),
            (stats.pending, f"{stats.pending:,} not generated ({stopped})"),
        )
        if count
    ]
    if stats.duplicates:
        parts.append(duplicates_note(stats.duplicates))
    if stats.truncated_inputs:
        parts.append(
            f"{_count(stats.truncated_inputs, 'text')} cut to "
            f"{result.config.max_input_chars:,} characters"
        )
    if stats.hit_token_limit:
        parts.append(
            f"{_count(stats.hit_token_limit, 'output')} reached the "
            f"{result.config.max_new_tokens:,}-token limit and may be cut off"
        )
    return " · ".join(parts)


def duplicates_note(n: int) -> str:
    """ "1 duplicate text shares an answer" / "3 duplicate texts share answers":
    duplicates get no model call of their own."""
    if n == 1:
        return "1 duplicate text shares an answer"
    return f"{n:,} duplicate texts share answers"


def _count(n: int, noun: str) -> str:
    return f"{n:,} {noun}" if n == 1 else f"{n:,} {noun}s"


@dataclass(frozen=True)
class Failure:
    """A failed or stopped run's error text, split for display."""

    cause: str
    """Why, in the user's terms: the backend's own message when it wrote
    one for users (its hint included), else the first sentence of the
    exception message."""
    action: str | None
    """What to do, when the error is recognised; None otherwise (the page
    adds a generic next step)."""
    loading: bool = False
    """The model couldn't be loaded."""
    technical: bool = False
    """The text was a raw exception ("SomeError: ...") rather than a
    message written for users."""
    settings: bool = False
    """The model couldn't be loaded with these settings (precision for this
    device, a folder that isn't a model, offline without a download...):
    running the same configuration again fails the same way."""


_SETTINGS_PHRASES = (
    "4-bit quantisation needs",
    "on the cpu is unusably slow",
    "unknown precision",
    "is not a model folder",
    "model folder not found",
    "has no tokenizer files",
    "offline mode is on",
    "model not found:",
    "isn't a text-generation",
    "needs custom code from its repository",
    "is a gated model and this machine doesn't have access",
)
"""Phrases of the backend's load errors that the same settings will always
hit again (compared case-insensitively)."""

_LOAD_PREFIX = re.compile(r"^Couldn't load (?P<label>.+?): ")
_EXC_PREFIX = re.compile(
    r"^(?P<name>[A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt|Warning))(?::\s*|$)"
)
_STREAK_SUFFIX = re.compile(r"\s*\(stopped after (?P<n>[\d,]+) failed batch(?:es)? in a row\)\s*$")

_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("memory", ("out of memory", "outofmemoryerror", "cuda oom")),
    (
        "access",
        ("gated repo", "gatedrepoerror", "401 client error", "403 client error", "hf_token"),
    ),
    (
        "network",
        (
            "certificate verify failed",
            "sslerror",
            "ssl:",
            "connectionerror",
            "connection error",
            "max retries exceeded",
            "read timed out",
            "connect timeout",
            "failed to resolve",
            "name resolution",
            "proxyerror",
        ),
    ),
    ("bitsandbytes", ("bitsandbytes",)),
    ("cuda_build", ("not compiled with cuda",)),
    (
        "not_found",
        (
            "repository not found",
            "repositorynotfounderror",
            "404 client error",
            "does not appear to have a file",
            "is not a local folder and is not a valid model identifier",
        ),
    ),
)


def explain_failure(reason: str | None, model_label: str | None = None) -> Failure:
    """Split a run's failure text (``RunResult.aborted_reason`` or
    ``JobSnapshot.error``) into a cause and, when recognised, an action.

    Messages the backends wrote for users (``ModelLoadError`` texts, which
    already end with their hint, or the engine's own "Couldn't save outputs
    ..." sentence) are kept whole and get no extra advice that could
    contradict them. Only raw exceptions ("RuntimeError: ...") are matched
    against known phrases - CUDA out of memory, gated repos / HTTP 401/403,
    network and TLS failures, bitsandbytes."""
    text = " ".join(str(reason or "").split())
    if not text:
        return Failure("No error message was recorded.", None)
    loading = False
    prefix = f"Couldn't load {model_label}: " if model_label else None
    if prefix and text.startswith(prefix):
        text, loading = text[len(prefix) :], True
    else:
        match = _LOAD_PREFIX.match(text)
        if match:
            text, loading = text[match.end() :], True
    text = _STREAK_SUFFIX.sub("", text)
    exc = _EXC_PREFIX.match(text)
    if exc is None:
        return Failure(
            _fit_message(text, _MESSAGE_LIMIT),
            None,
            loading=loading,
            settings=loading and any(p in text.casefold() for p in _SETTINGS_PHRASES),
        )

    message = text[exc.end() :].strip()
    cause = _first_sentence(message) if message else f"{exc.group('name')} (no message)."
    lowered = text.casefold()
    kind = next((k for k, phrases in _HINTS if any(p in lowered for p in phrases)), None)
    return Failure(cause, _hint(kind, loading), loading=loading, technical=True)


@dataclass(frozen=True)
class Problem:
    """A failed or stopped run, worded for its error card: what happened
    (``headline``), why (``cause``), what to do (``hint``), and the raw
    error for a collapsed "Technical details" block."""

    headline: str
    cause: str
    hint: str | None
    details: str | None
    retry_helps: bool = True
    """False when the same configuration is certain to fail again (see
    :attr:`Failure.settings`): offer to change the settings, not a retry."""


KEPT_NOTE = "Saved outputs are kept, so a retry only generates the missing rows."


def describe_failure(
    result: RunResult | None, error: str | None, model_label: str | None = None
) -> Problem:
    """The error card of a run that failed outright (``result`` None, the
    job's ``error`` says why) or stopped early (``result.aborted_reason``).
    Shared by the Run page's "Last run" card and the Results page, so both
    explain a failure in the same words."""
    if result is None:
        failure = explain_failure(error, model_label)
        return Problem(
            "The run failed before producing any output.",
            failure.cause,
            failure.action
            or (None if failure.settings else "Check the settings on the Run page, then retry."),
            error,
            retry_helps=not failure.settings,
        )
    stats = result.stats
    reason = result.aborted_reason or error
    failure = explain_failure(reason, model_label)
    produced = stats.generated + stats.cached
    cause = failure.cause
    if stats.pending and result.aborted_reason and not failure.loading:
        untried = "1 row wasn't" if stats.pending == 1 else f"{stats.pending:,} rows weren't"
        streak = _STREAK_SUFFIX.search(" ".join(str(result.aborted_reason).split()))
        if streak:
            batches = "batch" if streak.group("n") == "1" else "batches"
            cause += (
                f" The run stopped after {streak.group('n')} failed {batches} in a row, "
                f"so {untried} tried."
            )
        else:
            cause += f" The run stopped there, so {untried} tried."
    if failure.loading:
        headline = (
            "The model couldn't be loaded, so no rows were generated."
            if not produced
            else "The model couldn't be loaded, so the run stopped early."
        )
    elif not produced:
        headline = "The run stopped before generating any rows."
    else:
        left = []
        if stats.errors:
            left.append(f"{_count(stats.errors, 'row')} failed")
        if stats.pending:
            pending = f"{stats.pending:,}" if stats.errors else _count(stats.pending, "row")
            left.append(f"{pending} not generated")
        headline = (
            f"The run stopped early: {', '.join(left)}." if left else "The run stopped early."
        )
    if failure.action:
        action: str | None = failure.action
    elif failure.loading:
        action = None if not failure.technical else _LOAD_FALLBACK
    else:
        action = _RETRY_FALLBACK
    # Other settings make another configuration, which doesn't reuse these outputs.
    hints = [action, KEPT_NOTE if produced and not failure.settings else None]
    return Problem(
        headline,
        cause,
        " ".join(h for h in hints if h) or None,
        str(reason) if reason else None,
        retry_helps=not failure.settings,
    )


_RETRY_FALLBACK = (
    "Retry the failed rows. If they fail again, lower Batch size under Advanced on the Run "
    "page, or pick a smaller model."
)
_LOAD_FALLBACK = "Check the model name and its settings on the Run page, then retry."


def _hint(kind: str | None, loading: bool) -> str | None:
    if kind == "memory":
        if loading:
            return (
                "There isn't enough GPU memory for this model. Pick a smaller model or 4-bit "
                "precision, and close other programs that use the GPU."
            )
        return (
            "The GPU ran out of memory. Lower the batch size or Max input characters under "
            "Advanced on the Run page, or pick a smaller model."
        )
    if kind == "access":
        return (
            "This model needs access: accept its licence on huggingface.co, set HF_TOKEN, "
            "then restart the app."
        )
    if kind == "network":
        return (
            "Couldn't reach huggingface.co. Check your network and proxy settings, or use a "
            "model folder that is already on this machine."
        )
    if kind == "cuda_build":
        return (
            "This PyTorch build can't use the GPU. Install the CUDA build of PyTorch (see the "
            "README), or choose fp32 under Advanced on the Run page to run on the CPU."
        )
    if kind == "not_found":
        return "Check the model name (a Hugging Face repo id) or the local folder path."
    if kind == "bitsandbytes":
        return (
            "4-bit precision needs the bitsandbytes package and an NVIDIA GPU. Choose fp16 on a "
            "GPU or fp32 on the CPU under Advanced on the Run page."
        )
    return None


_MESSAGE_LIMIT = 400
"""Longest cause shown on an error card (the full text is in its details)."""
_RAW_DETAIL = re.compile(r"\s*\((?=[A-Za-z_][\w.]*(?:Error|Exception|Warning|Exit|Interrupt)\b)")
"""Start of a raw exception quoted in parentheses: " (OSError: ...)"."""


def _fit_message(text: str, limit: int) -> str:
    """A message written for users, at most ``limit`` characters. A raw
    exception it quotes in parentheses ("... (OSError: ...). Check your
    proxy ...") is shortened, or left out, before anything else: the advice
    after it (the backend's hint) must never be the part that is cut off.
    The full text stays in the card's technical details."""
    if len(text) <= limit:
        return _sentence(text, limit)
    match = _RAW_DETAIL.search(text)
    end = _closing_paren(text, match.end() - 1) if match else None
    if match is not None and end is not None:
        head, detail, tail = text[: match.start()], text[match.end() : end], text[end + 1 :]
        room = limit - len(head) - len(tail) - len(" (…)")
        if room >= 40:
            detail = detail.rstrip("…").rstrip()
            text = f"{head} ({detail[:room].rstrip()}…){tail}"
        else:
            text = head + tail
    return _sentence(text, limit)


def _closing_paren(text: str, start: int) -> int | None:
    """Index of the ")" closing the "(" at ``start``. A detail the backend
    cut short ends with "…)" and may hold an unmatched "(", so that marker
    wins."""
    marker = text.find("…)", start)
    if marker >= 0:
        return marker + 1
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return index
    return None


def _first_sentence(text: str, limit: int = 200) -> str:
    """The first sentence of ``text`` (split at ". " / "! " / "? "),
    capitalised and ending in a full stop, at most ``limit`` characters."""
    head = re.split(r"(?<=[.!?])\s+(?=[A-Z(])", text.strip(), maxsplit=1)[0]
    return _sentence(head, limit)


_LOWERCASE_NAMES = frozenset(
    {"bitsandbytes", "torch", "transformers", "accelerate", "sentencepiece", "protobuf", "pip"}
)
"""Names written in lower case even at the start of a sentence."""


def _sentence(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    first = text.split(" ", 1)[0]
    # Only a plain word: "fp16", a package name or a repo id keep their case.
    if first.isalpha() and first.islower() and first not in _LOWERCASE_NAMES:
        text = text[0].upper() + text[1:]
    if text and text[-1] not in ".!?…)":
        text += "."
    return text


# --------------------------------------------------------------------------
# Table views
# --------------------------------------------------------------------------


def row_filters(analysis: JobAnalysis) -> dict[str, pd.Series]:
    """Boolean masks over :attr:`JobAnalysis.table`, in display order: "All"
    always, then only the filters that match at least one row (Wrong,
    Correct, Invalid, Invalid JSON, Errors, Blank, Not generated)."""
    table = analysis.table
    statuses = analysis.statuses
    masks: dict[str, pd.Series] = {FILTER_ALL: pd.Series(True, index=table.index)}
    cols = analysis.columns
    if cols.correct and cols.correct in table.columns:
        flags = table[cols.correct].map(lambda v: None if is_blank(v) else bool(v))
        masks[FILTER_WRONG] = flags.map(lambda v: v is False)
        masks[FILTER_CORRECT] = flags.map(lambda v: v is True)
    if cols.label and cols.label in table.columns:
        masks[FILTER_INVALID] = table[cols.label].astype(object) == INVALID
    if analysis.context.output_kind == "json":
        scorable = pd.Series(analysis.scorable, index=table.index, dtype=bool)
        valid = pd.Series(False, index=table.index, dtype=bool)
        # Assigning through an all-False mask fails on pandas 3 (an empty list
        # isn't a bool): a run with no output yet has nothing to check.
        if scorable.any():
            valid.loc[scorable] = analysis.json_valid()
        masks[FILTER_NO_JSON] = scorable & ~valid
    masks[FILTER_ERRORS] = statuses == "error"
    masks[FILTER_BLANK] = statuses == "blank"
    masks[FILTER_NOT_RUN] = statuses == "pending"
    return {
        name: mask.fillna(False).astype(bool)
        for name, mask in masks.items()
        if name == FILTER_ALL or bool(mask.fillna(False).astype(bool).any())
    }


def search_mask(table: pd.DataFrame, query: str, columns: Sequence[str]) -> pd.Series:
    """Case-insensitive substring match of ``query`` over ``columns``."""
    needle = (query or "").strip().casefold()
    if not needle:
        return pd.Series(True, index=table.index)
    hit = pd.Series(False, index=table.index)
    for column in columns:
        if column in table.columns:
            values = table[column].map(lambda v: "" if is_blank(v) else str(v).casefold())
            hit |= values.str.contains(needle, regex=False)
    return hit


def is_blank(value: Any) -> bool:
    """True for None / NaN / NA / NaT cells."""
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def truthy(value: Any) -> bool:
    """A cell read as a yes/no flag (True, 1, "true", "yes")."""
    if is_blank(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def top_confusions(report: EvalReport, limit: int = 15) -> pd.DataFrame:
    """Most frequent wrong gold -> predicted pairs: columns ``gold``,
    ``predicted``, ``count``, ``share`` (of that gold label's scored rows)."""
    rows = report.rows
    if rows.empty:
        return pd.DataFrame(columns=["gold", "predicted", "count", "share"])
    support = rows.groupby("gold").size()
    wrong = rows.loc[~rows["correct"].astype(bool)]
    if wrong.empty:
        return pd.DataFrame(columns=["gold", "predicted", "count", "share"])
    pairs = wrong.groupby(["gold", "predicted"]).size().rename("count").reset_index()
    pairs["share"] = [c / support[g] for g, c in zip(pairs["gold"], pairs["count"], strict=True)]
    pairs = pairs.sort_values(["count", "gold", "predicted"], ascending=[False, True, True])
    return pairs.head(limit).reset_index(drop=True)


def per_class_table(report: EvalReport) -> pd.DataFrame:
    """Per-class metrics, weakest F1 first (labels never seen in gold or
    predictions, whose F1 is undefined, last)."""
    frame = report.per_class_frame()
    if frame.empty:
        return frame
    return frame.sort_values(
        ["f1", "support"], ascending=[True, False], na_position="last"
    ).reset_index(drop=True)


def group_candidates(analysis: JobAnalysis) -> list[str]:
    """Columns worth slicing the evaluation by, TIER-like names first.
    Computed once per analysis (``data.guess_group_columns`` scans every
    column) and memoised on it."""
    cached = analysis.memo.get("groups")
    if cached is None:
        cached = analysis.memo["groups"] = _group_candidates(analysis)
    return list(cached)


def flag_columns(analysis: JobAnalysis) -> dict[str, str]:
    """Slice candidates that are 0/1, true/false or yes/no flags, mapped to
    their "true" value (:func:`flag_value`), in candidate order. Memoised."""
    cached = analysis.memo.get("flags")
    if cached is None:
        frame = analysis.context.frame
        cached = {}
        for column in group_candidates(analysis):
            value = flag_value(frame[column])
            if value is not None:
                cached[column] = value
        analysis.memo["flags"] = cached
    return dict(cached)


def flag_value(series: pd.Series) -> str | None:
    """The "true" value of a 0/1, true/false or yes/no column (as the column
    spells it), else None."""
    values = binary_values(series)
    if values is None:
        return None
    lowered = [v.strip().lower() for v in values]
    for pair in (("0", "1"), ("false", "true"), ("no", "yes"), ("n", "y"), ("0.0", "1.0")):
        if sorted(lowered) == sorted(pair):
            return values[lowered.index(pair[1])]
    return None


def _group_candidates(analysis: JobAnalysis) -> list[str]:
    from batchtextreview import data

    ctx = analysis.context
    exclude = tuple(c for c in (ctx.text_column, ctx.id_column, ctx.gold_column) if c)
    try:
        found = data.guess_group_columns(ctx.frame, exclude=exclude)
    except Exception:
        found = []
    preferred = [c for c in ctx.group_columns if c in ctx.frame.columns and c not in exclude]
    ordered = preferred + [c for c in found if c not in preferred]

    def rank(name: str) -> int:
        lowered = name.lower()
        if any(k in lowered for k in ("tier", "level", "difficulty", "segment", "group")):
            return 0
        return 1

    return sorted(ordered, key=lambda c: (c not in preferred, rank(c)))


def slice_table(
    analysis: JobAnalysis, column: str, report: EvalReport | None = None
) -> pd.DataFrame:
    """``evaluation.slice_report`` for one context column."""
    report = report or analysis.report
    if report is None or column not in analysis.context.frame.columns:
        return pd.DataFrame(columns=["group", "n", "correct", "accuracy"])
    return evaluation.slice_report(report, analysis.context.frame[column].tolist())


def subset_report(analysis: JobAnalysis, keep: Sequence[bool]) -> EvalReport | None:
    """Re-score with only the rows where ``keep`` is True (and scorable)."""
    ctx = analysis.context
    report = analysis.report
    if report is None or not ctx.gold_column:
        return None
    include = [bool(a) and bool(b) for a, b in zip(analysis.scorable, keep, strict=True)]
    allowed = ctx.labels if report.mode == "labels" else None
    return evaluation.evaluate(
        analysis.result.outputs,
        ctx.frame[ctx.gold_column].tolist(),
        allowed=allowed,
        include=include,
    )


def binary_values(series: pd.Series) -> list[str] | None:
    """The two distinct values of a flag-like column (as strings, sorted), or
    None when it doesn't have exactly two."""
    values = sorted({str(v) for v in series if not is_blank(v) and str(v).strip()})
    return values if len(values) == 2 else None
