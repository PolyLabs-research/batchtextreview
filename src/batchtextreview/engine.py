"""Run engine: plan, batch, generate, checkpoint.

Work is content-addressed. A :class:`RunConfig` (model, precision, final
template, output length, truncation) hashes to a ``run_id``; each row's
(truncated) text hashes to a ``key`` within it. Outputs are written to the
:class:`~batchtextreview.store.Store` after every successful batch, so:

* an interrupted or cancelled run resumes where it stopped;
* duplicate rows are generated once and shared;
* re-running an identical configuration is instant (no model load);
* rows that errored are *not* stored, so a re-run retries exactly those.

A batch that fails is retried one prompt at a time, so one poisoned row
costs only itself rather than the whole batch.

Batch size, processing order (longest or shortest texts first) and row
limits are deliberately *not* part of the run id - they change speed and
scope, not the output for a given text.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from batchtextreview import labels, prompting
from batchtextreview.backends.base import BackendError
from batchtextreview.models import api_provider

if TYPE_CHECKING:
    from batchtextreview.backends.manager import ModelManager
    from batchtextreview.models import ModelSpec
    from batchtextreview.store import Store

log = logging.getLogger(__name__)

RUN_ID_VERSION = "v2"
DEFAULT_MAX_INPUT_CHARS = 8000
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_BATCH_CHARS = 20000
DEFAULT_MAX_CONSECUTIVE_FAILURES = 3
ORDERS: tuple[str, ...] = ("longest", "shortest")
"""Processing orders for :func:`execute_run`: which texts go to the model
first. "longest": the time estimate starts high and counts down, and a model
that runs out of memory fails in the first batch. "shortest": the first
results appear sooner."""
DEFAULT_ORDER = "longest"
ORDER_LABELS: dict[str, str] = {"longest": "Longest first", "shortest": "Shortest first"}
"""Sentence-case name of each of :data:`ORDERS`, for display."""
ETA_SMOOTHING = 0.3
"""Weight of a fresh time-left estimate against the previous one counted
down (see :class:`_Eta`)."""
RECENT_LIMIT = 5
"""How many of the latest (text, output) pairs :attr:`Progress.recent` holds."""
RECENT_TEXT_CHARS = 120
RECENT_OUTPUT_CHARS = 200

RowStatus = Literal["generated", "cached", "blank", "error", "pending"]
"""Per-row outcome.

generated  produced by the model in this run
cached     filled from the store (earlier run with the same configuration)
blank      empty / missing text - no model call, empty output
error      the row failed on its own (or with the rest of a batch whose
           failure was systemic); output is empty, see errors[]
pending    never attempted (run cancelled or aborted first)
"""

Phase = Literal["planning", "loading", "generating", "finished"]


def compute_run_id(
    model_id: str,
    precision: str,
    template: str,
    max_new_tokens: int,
    max_input_chars: int = DEFAULT_MAX_INPUT_CHARS,
    model_revision: str = "",
) -> str:
    """Stable 16-hex-char id for everything that changes generated output.

    ``model_revision`` (see :func:`batchtextreview.models.model_revision`)
    only enters the hash when set, so ids without one are unchanged.
    """
    parts: list[Any] = [
        RUN_ID_VERSION,
        model_id,
        precision,
        template,
        int(max_new_tokens),
        int(max_input_chars),
    ]
    if model_revision:
        parts.append(model_revision)
    payload = json.dumps(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def content_key(snippet: str) -> str:
    """Key for one (already truncated) text within a run. sha256 hex.

    Identical to version 1's keys, so imported checkpoints still match.
    """
    return hashlib.sha256(snippet.encode("utf-8")).hexdigest()


def is_blank(value: Any) -> bool:
    """True for None, NaN / pd.NA / NaT, and whitespace-only strings."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        import pandas as pd

        if value is pd.NA or value is pd.NaT:
            return True
    except Exception:  # pragma: no cover - pandas is a hard dependency
        pass
    return not str(value).strip()


@dataclass(frozen=True)
class RunConfig:
    """Everything that determines what is generated for a given text."""

    model_id: str
    """ModelSpec.repo (HF repo id, local path, or demo://fake)."""
    model_label: str
    precision: str
    """Concrete precision (never "auto"); "none" for the demo model."""
    template: str
    """Final template - ``{var}`` already applied, must contain ``{text}``."""
    max_new_tokens: int
    max_input_chars: int = DEFAULT_MAX_INPUT_CHARS
    model_revision: str = ""
    """Fingerprint of a local model folder's files ("" for hub repos and the
    demo model), so re-saving a fine-tune into the same folder starts a new
    run instead of serving the old model's outputs."""

    @property
    def run_id(self) -> str:
        return compute_run_id(
            self.model_id,
            self.precision,
            self.template,
            self.max_new_tokens,
            self.max_input_chars,
            self.model_revision,
        )

    @classmethod
    def build(
        cls,
        model: ModelSpec,
        template: str,
        max_new_tokens: int,
        precision: str = "auto",
        max_input_chars: int = DEFAULT_MAX_INPUT_CHARS,
    ) -> RunConfig:
        """Convenience constructor that resolves "auto" precision.

        The demo model and cloud API models always get precision "none". A
        local model folder gets its current ``model_revision``.
        """
        from batchtextreview.backends.base import resolve_precision
        from batchtextreview.models import model_revision

        resolved = "none" if model.is_demo or model.is_api else resolve_precision(precision)
        return cls(
            model_id=model.repo,
            model_label=model.label,
            precision=resolved,
            template=template,
            max_new_tokens=int(max_new_tokens),
            max_input_chars=int(max_input_chars),
            model_revision=model_revision(model.repo),
        )


@dataclass
class RunPlan:
    """What a run would do, computed without loading a model."""

    run_id: str
    n_rows: int
    n_blank: int
    n_unique: int
    """Distinct non-blank texts (after truncation)."""
    n_duplicates: int
    """Non-blank rows beyond the first occurrence of each text."""
    n_cached_rows: int
    """Rows whose output is already stored."""
    n_cached_unique: int
    n_todo_rows: int
    """Rows that still need the model."""
    n_todo_unique: int
    """Model calls (prompts) still needed."""
    keys: list[str | None]
    """Per input row: content key, or None for blank rows."""
    snippets: dict[str, str]
    """key -> truncated text."""
    cached: dict[str, str]
    """key -> stored output, for keys already in the store."""
    n_truncated_rows: int = 0
    """Non-blank rows whose text is longer than ``max_input_chars`` (they
    are cut before sending)."""
    cached_hit_limit: set[str] = field(default_factory=set)
    """Keys in ``cached`` whose output reached ``max_new_tokens`` when it
    was generated (as stored with it)."""

    @property
    def fully_cached(self) -> bool:
        return self.n_todo_unique == 0


@dataclass(frozen=True)
class Progress:
    """Immutable progress snapshot passed to the progress callback."""

    phase: Phase
    total_rows: int
    done_rows: int = 0
    generated_rows: int = 0
    cached_rows: int = 0
    blank_rows: int = 0
    error_rows: int = 0
    elapsed_s: float = 0.0
    rows_per_s: float | None = None
    """Generation throughput (generated rows / generation seconds)."""
    eta_s: float | None = None
    """Estimated generation seconds left (from the prompts still to send,
    not the rows, and smoothed - see :class:`_Eta`), or None before the
    first successful model call."""
    message: str = ""
    recent: tuple[tuple[str, str], ...] = ()
    """The last few (<= :data:`RECENT_LIMIT`) (text, output) pairs generated
    in this run, newest last - so a broken prompt shows early. The text is
    whitespace-collapsed and at most 120 characters, the output at most 200."""
    generation_s: float = 0.0
    """Seconds spent generating so far in this run (model load excluded)."""
    work_done: float | None = None
    """Share of the run's work done (0-1), reported as :attr:`work_fraction`;
    None to work it out from the other fields. :func:`execute_run` fills it
    in: rows that need no model (blank, cached) count as done from the
    start, the rest by generation time as in :attr:`work_fraction`, and the
    figure never goes down within a run (1.0 once a complete run has
    finished)."""

    @property
    def fraction(self) -> float:
        """Share of the rows done (0-1)."""
        return (self.done_rows / self.total_rows) if self.total_rows else 1.0

    @property
    def work_fraction(self) -> float:
        """Share of the work done (0-1), for a progress bar.

        :attr:`work_done` when set. Otherwise, while generating with a time
        estimate, the generation time so far over that plus the time left
        (``generation_s / (generation_s + eta_s)``), so a long text weighs
        more than a short one; else :attr:`fraction` (rows).
        """
        if self.work_done is not None:
            return self.work_done
        if self.phase == "generating" and self.eta_s is not None:
            whole = self.generation_s + self.eta_s
            if whole > 0:
                return self.generation_s / whole
        return self.fraction


@dataclass
class RunStats:
    total: int = 0
    blank: int = 0
    cached: int = 0
    duplicates: int = 0
    """Generated rows that repeated the text of another row generated with
    it, so shared its model call (generated rows minus distinct texts
    generated). Repeats served from the cache, failed or never run are not
    counted - see ``RunPlan.n_duplicates`` for all repeats in the data."""
    generated: int = 0
    errors: int = 0
    pending: int = 0
    model_calls: int = 0
    """Unique prompts sent to the model (successfully or not)."""
    elapsed_s: float = 0.0
    load_s: float = 0.0
    """Seconds spent loading the model (0 when already loaded / not needed)."""
    generation_s: float = 0.0
    """Seconds spent inside the model's generate calls."""
    truncated_inputs: int = 0
    """Non-blank rows whose text was longer than ``max_input_chars``."""
    hit_token_limit: int = 0
    """Generated or cached rows whose output reached ``max_new_tokens`` (as
    reported by the backend, and stored with the output) and may be cut
    off."""
    input_tokens: int = 0
    """Tokens sent to a cloud API model in this run, as the provider counted
    them (0 for models that run on this computer)."""
    output_tokens: int = 0
    """Tokens a cloud API model generated in this run, reasoning included."""


@dataclass
class RunResult:
    run_id: str
    config: RunConfig
    outputs: list[str]
    """One per input row; "" for blank / error / pending rows."""
    statuses: list[RowStatus]
    errors: list[str | None]
    """Error message for "error" rows, else None."""
    stats: RunStats = field(default_factory=RunStats)
    cancelled: bool = False
    aborted_reason: str | None = None
    """Set when the run stopped early (model failed to load,
    ``max_consecutive_failures`` batches failed in a row, or a finished batch
    could not be saved to the store)."""
    input_truncated: list[bool] = field(default_factory=list)
    """Per row: the text was longer than ``max_input_chars`` and was cut
    before sending (False for blank rows). May be empty for results made
    without it."""
    hit_token_limit: list[bool] = field(default_factory=list)
    """Per row: the output reached ``max_new_tokens`` and may be cut off.
    Known for rows generated by a backend that reports it
    (``last_hit_limit``), in this run or - for cached rows - when the output
    was stored; False otherwise (including outputs saved by older versions).
    May be empty for results made without it."""

    @property
    def complete(self) -> bool:
        return not self.cancelled and self.aborted_reason is None and self.stats.pending == 0


ProgressCallback = Callable[[Progress], None]


ERROR_MESSAGE_LIMIT = 300
"""Longest per-row error message ("ExcType: message")."""
_LOAD_ERROR_LIMIT = 1000
"""Load failures carry install / licence hints, so they may run longer."""


def plan_run(config: RunConfig, texts: Sequence[Any], store: Store | None) -> RunPlan:
    """Classify rows into blank / cached / to-do without touching the model.

    Texts are converted with ``str()`` and truncated to
    ``config.max_input_chars`` before hashing. ``store=None`` means "nothing
    cached" (useful in tests and the CLI's ``--no-cache``). Cached outputs
    stored as reaching ``max_new_tokens`` are listed in
    ``cached_hit_limit``.
    """
    run_id = config.run_id
    limit = int(config.max_input_chars)
    keys: list[str | None] = []
    snippets: dict[str, str] = {}
    rows_per_key: dict[str, int] = {}
    n_blank = 0
    n_truncated = 0
    for value in texts:
        if is_blank(value):
            keys.append(None)
            n_blank += 1
            continue
        text = value if isinstance(value, str) else str(value)
        # A non-positive limit would hash every row to the empty string;
        # treat it as "no truncation" instead.
        snippet = text[:limit] if limit > 0 else text
        if len(snippet) < len(text):
            n_truncated += 1
        key = content_key(snippet)
        keys.append(key)
        snippets.setdefault(key, snippet)
        rows_per_key[key] = rows_per_key.get(key, 0) + 1

    cached: dict[str, str] = {}
    cached_hit_limit: set[str] = set()
    if store is not None and snippets:
        stored = store.get_outputs(run_id, list(snippets))
        cached = {k: v for k, v in stored.items() if k in snippets and v is not None}
        if cached:
            cached_hit_limit = _stored_hit_limits(store, run_id, list(cached))

    n_nonblank = len(keys) - n_blank
    n_cached_rows = sum(rows_per_key[k] for k in cached)
    return RunPlan(
        run_id=run_id,
        n_rows=len(keys),
        n_blank=n_blank,
        n_unique=len(snippets),
        n_duplicates=n_nonblank - len(snippets),
        n_cached_rows=n_cached_rows,
        n_cached_unique=len(cached),
        n_todo_rows=n_nonblank - n_cached_rows,
        n_todo_unique=len(snippets) - len(cached),
        keys=keys,
        snippets=snippets,
        cached=cached,
        n_truncated_rows=n_truncated,
        cached_hit_limit=cached_hit_limit,
    )


def pack_batches(
    keys: Sequence[str],
    lengths: dict[str, int],
    batch_size: int,
    max_batch_chars: int,
) -> list[list[str]]:
    """Greedy packing of keys (already sorted by length, longest or
    shortest first) into batches.

    A batch closes when adding the next key would exceed ``batch_size`` rows
    or ``max_batch_chars`` total characters. A single key longer than the
    char budget still gets a batch of its own. ``batch_size`` < 1 is treated
    as 1.
    """
    size = max(1, int(batch_size))
    budget = int(max_batch_chars)
    batches: list[list[str]] = []
    current: list[str] = []
    current_chars = 0
    for key in keys:
        n_chars = lengths[key]
        if current and (len(current) >= size or current_chars + n_chars > budget):
            batches.append(current)
            current, current_chars = [], 0
        current.append(key)
        current_chars += n_chars
    if current:
        batches.append(current)
    return batches


def execute_run(
    config: RunConfig,
    texts: Sequence[Any],
    *,
    store: Store | None,
    manager: ModelManager,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_batch_chars: int = DEFAULT_MAX_BATCH_CHARS,
    order: str = DEFAULT_ORDER,
    progress: ProgressCallback | None = None,
    cancel: threading.Event | None = None,
    max_consecutive_failures: int = DEFAULT_MAX_CONSECUTIVE_FAILURES,
    plan: RunPlan | None = None,
    source: str = "app",
) -> RunResult:
    """Run ``config`` over ``texts``. Never raises for model/generation errors.

    Steps:

    1. ``plan_run`` (unless ``plan`` is given and its run_id matches).
       Blank rows -> status "blank", output "". Cached keys -> "cached".
    2. If anything is left, pending keys are sorted by snippet length -
       ``order`` "longest" (the default) longest first, "shortest" shortest
       first; stable either way, so ties keep first-seen order - and packed
       with :func:`pack_batches`. Longest first, the first batch is the
       heaviest, so a model that runs out of memory fails at once and the
       time estimate counts down; shortest first, the first results appear
       sooner. The run's manifest is registered in the store
       (``store.upsert_run``) with the first saved batch.
    3. The model is obtained lazily via ``manager.acquire(...)`` (a context
       manager that loads/swaps the model and holds the generation lock) only
       if at least one batch must run. A :class:`ModelLoadError` (or any
       exception while loading) ends the run: ``aborted_reason`` is set, and
       all not-yet-done rows stay "pending".
    4. For each batch: check ``cancel`` first (set -> stop, ``cancelled=True``,
       remaining rows "pending"). Render prompts with
       :func:`~batchtextreview.prompting.render_prompt` and call
       ``backend.generate(prompts, config.max_new_tokens)``. The backend
       itself handles OOM by splitting the batch. On success: every row
       sharing a key gets the output (status "generated"), outputs are
       cleaned with ``labels.clean_output`` and saved with
       ``store.put_outputs(run_id, {key: output})`` immediately. On any
       exception: rows get status "error" with a short message
       ("ExcType: message", max 300 chars), nothing is stored, and the run
       continues - unless ``max_consecutive_failures`` batches in a row have
       failed (0 disables the limit), in which case the run aborts with that
       error as ``aborted_reason`` and remaining rows stay "pending".

       Failed-batch isolation: when ``generate`` raises for a batch of more
       than one prompt, its prompts are retried one at a time. Rows whose
       own call succeeds get their output (stored as usual); only rows that
       fail alone become "error". If the first two single retries both fail
       with the batch's exception type and message, the failure is taken to
       be systemic (CUDA error, unloaded model...) and the batch's remaining
       rows are marked "error" without calling the model again. The cancel
       event is checked before every single retry. A batch counts towards
       ``max_consecutive_failures`` only if none of its rows succeeded. (A
       batch whose output count is wrong is not retried: the backend broke
       its contract rather than choking on one row.)
    5. ``progress`` is called with phase "planning" once, "loading" before
       the model load (if any), "generating" after every batch (and after
       every single retry of a failed batch), and "finished" at the end.
       Exceptions raised by the callback propagate (that is how a UI can
       stop a run), but the store is always left consistent because outputs
       are saved before the callback runs. Snapshots carry the latest
       generated (text, output) pairs in ``recent``.

    ``rows_per_s`` counts generated rows (duplicates included) per second of
    generation time (excluding model load). ``eta_s`` is estimated from the
    work left, not from rows: the batches not started yet (and the prompts
    of a failed batch still to retry alone) and their prompt characters,
    priced by the seconds per call and per character of the calls so far,
    and smoothed from one snapshot to the next (see :class:`_Eta`).
    Duplicate rows cost nothing extra, and batches longer or shorter than
    those so far are priced as such. ``work_done`` (what
    :attr:`Progress.work_fraction` reports) measures work rather than rows:
    the rows that need no model, plus the generation time so far over that
    time plus ``eta_s`` for the rest; it never goes down within a run, is
    below 1.0 while any call is still to come, and is 1.0 once a complete
    run has finished.

    Truncation: ``input_truncated`` flags non-blank rows whose text was cut
    to ``max_input_chars``. ``hit_token_limit`` flags generated rows whose
    output reached ``max_new_tokens``, read after each successful generate
    call from the backend's optional ``last_hit_limit`` (a list[bool]
    aligned with that call's prompts; ignored when missing or the wrong
    length). The flag is stored with the output (``put_outputs(...,
    hit_limit=keys)``), so cached rows keep it on a resume or re-run. Both
    are counted in the stats.

    Logging (``batchtextreview.engine``): one INFO line when the run starts
    and one when it ends; a WARNING per failed batch (with the traceback
    for the run's first failure only).

    Notes: a given ``plan`` is only reused when its run_id *and* row count
    match ``texts``. The manifest is registered just before the first batch
    is saved, so a run whose model never loads leaves no empty manifest
    behind. The cancel event is also checked before the model load, so a
    run cancelled while still queued never loads a model. A template
    without ``{text}``, or an ``order`` not in :data:`ORDERS`, raises
    ValueError up front (it is a caller bug, not a model error). ``source``
    ("app" or "cli") is recorded with the manifest. ``order`` is not part of
    the run id, so a run resumed in the other order reuses every saved
    output.

    If saving a finished batch fails (disk full, locked database), the run
    stops like any other abort: ``aborted_reason`` says what happened, that
    batch's rows keep their outputs (status "generated", but not stored) and
    everything saved before is kept, so a re-run resumes. A run deleted from
    the store mid-run is simply registered again.
    """
    if prompting.TEXT not in config.template:
        raise ValueError("The run's template must contain the {text} placeholder.")
    if order not in ORDERS:
        allowed = " or ".join(repr(name) for name in ORDERS)
        raise ValueError(f"Unknown processing order {order!r}: use {allowed}.")
    run = _Run(config, texts, store, progress, source)
    run.prepare(plan)
    run.log_start(order)
    if run.todo:
        run.generate(
            manager,
            batch_size=batch_size,
            max_batch_chars=max_batch_chars,
            order=order,
            cancel=cancel,
            max_consecutive_failures=max_consecutive_failures,
        )
    return run.finish()


# --------------------------------------------------------------------------
# Implementation
# --------------------------------------------------------------------------


class _Run:
    """Mutable state of one :func:`execute_run` call."""

    def __init__(
        self,
        config: RunConfig,
        texts: Sequence[Any],
        store: Store | None,
        progress: ProgressCallback | None,
        source: str = "app",
    ) -> None:
        self.config = config
        self.texts = texts
        self.store = store
        self.progress = progress
        self.started = time.perf_counter()
        n = len(texts)
        self.outputs: list[str] = [""] * n
        self.statuses: list[RowStatus] = ["pending"] * n
        self.errors: list[str | None] = [None] * n
        self.input_truncated: list[bool] = [False] * n
        self.hit_limit: list[bool] = [False] * n
        self.stats = RunStats(total=n)
        self.plan: RunPlan | None = None
        self.todo: dict[str, list[int]] = {}
        """key -> row positions still needing the model (first-seen order)."""
        self.recent: deque[tuple[str, str]] = deque(maxlen=RECENT_LIMIT)
        self.cancelled = False
        self.aborted_reason: str | None = None
        self.generation_s = 0.0
        self.eta: _Eta | None = None
        """Set once the batches are packed."""
        self.prompt_chars: dict[str, int] = {}
        """key -> characters of its prompt, set once the batches are packed."""
        self.work_done = 0.0
        """The largest ``Progress.work_done`` reported so far."""
        self.registered = False
        """Whether the run's manifest is in the store yet."""
        self.source = source
        self.failure_logged = False
        """Whether a traceback was logged yet (only the first one is)."""

    # ---- steps ------------------------------------------------------------
    def prepare(self, plan: RunPlan | None) -> None:
        config = self.config
        if plan is None or plan.run_id != config.run_id or plan.n_rows != len(self.texts):
            plan = plan_run(config, self.texts, self.store)
        self.plan = plan
        stats = self.stats
        # Cleaned again on the way out: outputs imported from version 1 were
        # stored raw (think blocks and all).
        cached = {k: labels.clean_output(v) for k, v in plan.cached.items()}
        # Iterating (not indexing) the texts: a Series may have any index.
        for i, (key, value) in enumerate(zip(plan.keys, self.texts, strict=True)):
            if key is None:
                self.statuses[i] = "blank"
                stats.blank += 1
                continue
            if len(plan.snippets[key]) < len(value if isinstance(value, str) else str(value)):
                self.input_truncated[i] = True
                stats.truncated_inputs += 1
            if key in cached:
                self.outputs[i] = cached[key]
                self.statuses[i] = "cached"
                stats.cached += 1
                if key in plan.cached_hit_limit:
                    self.hit_limit[i] = True
                    stats.hit_token_limit += 1
            else:
                self.todo.setdefault(key, []).append(i)
        stats.pending = sum(len(rows) for rows in self.todo.values())
        if self.todo:
            message = (
                f"{_count(stats.pending, 'row')} to generate "
                f"({_count(len(self.todo), 'model call')})."
            )
        else:
            message = "Everything is already saved - no model needed."
        self._emit("planning", message)

    def log_start(self, order: str = DEFAULT_ORDER) -> None:
        config, stats = self.config, self.stats
        details = (
            f"{_count(stats.total, 'row')}, {stats.pending:,} to generate "
            f"({_count(len(self.todo), 'model call')})"
        )
        if stats.truncated_inputs:
            details += (
                f", {_count(stats.truncated_inputs, 'text')} cut to "
                f"{config.max_input_chars:,} characters"
            )
        if self.todo:
            details += f", {ORDER_LABELS.get(order, order).lower()}"
        model_id = config.model_id
        if config.model_revision:
            model_id += f" @ {config.model_revision}"
        log.info(
            "Run %s started (%s): %s [%s, %s], %s.",
            config.run_id,
            self.source,
            config.model_label,
            model_id,
            config.precision,
            details,
        )

    def generate(
        self,
        manager: ModelManager,
        *,
        batch_size: int,
        max_batch_chars: int,
        order: str,
        cancel: threading.Event | None,
        max_consecutive_failures: int,
    ) -> None:
        assert self.plan is not None
        config, plan = self.config, self.plan
        if cancel is not None and cancel.is_set():
            self.cancelled = True
            return
        # sorted() is stable, also with reverse=True (ties keep their order),
        # and self.todo is in first-seen order.
        longest_first = order == "longest"
        ordered = sorted(self.todo, key=lambda k: len(plan.snippets[k]), reverse=longest_first)
        lengths = {k: len(plan.snippets[k]) for k in ordered}
        if api_provider(config.model_id):
            # One request per text: the character budget guards GPU memory,
            # which a cloud model doesn't use, so the batch size alone counts.
            max_batch_chars = sys.maxsize
        batches = pack_batches(ordered, lengths, batch_size, max_batch_chars)
        # Prompt characters per key and per batch (render_prompt is a plain
        # replace); the former also prices a failed batch's single retries.
        n_text = config.template.count(prompting.TEXT)
        overhead = len(config.template) - n_text * len(prompting.TEXT)
        self.prompt_chars = {k: overhead + n_text * lengths[k] for k in ordered}
        batch_chars = [sum(self.prompt_chars[k] for k in batch) for batch in batches]
        eta = self.eta = _Eta(batch_chars, longest_first=longest_first)

        target = (config.model_id, config.precision)
        already_loaded = _is_loaded(manager, target)
        verb = "Preparing" if already_loaded else "Loading"
        self._emit("loading", f"{verb} {config.model_label}...")

        with contextlib.ExitStack() as stack:
            load_started = time.perf_counter()
            try:
                backend = stack.enter_context(manager.acquire(*target))
            except Exception as exc:
                self.stats.load_s = time.perf_counter() - load_started
                detail = _describe_error(exc, _LOAD_ERROR_LIMIT, plain_backend_errors=True)
                self.aborted_reason = f"Couldn't load {config.model_label}: {detail}"
                return
            if not already_loaded:
                self.stats.load_s = time.perf_counter() - load_started
            # The model is ready: say so now, not after the first batch. A
            # first batch of long prompts can take minutes on a CPU, and all
            # of it would otherwise read as "loading".
            self._emit("generating", f"Starting batch 1 of {len(batches):,}...")

            failures_in_a_row = 0
            for number, batch in enumerate(batches, start=1):
                if cancel is not None and cancel.is_set():
                    self.cancelled = True
                    return
                label = f"Batch {number:,} of {len(batches):,}"
                eta.start_batch(batch_chars[number - 1])
                error = self._run_batch(backend, batch, label, cancel)
                failures_in_a_row = 0 if error is None else failures_in_a_row + 1
                self._emit("generating", label)
                # The store failed, or cancel arrived between single retries.
                if self.aborted_reason is not None or self.cancelled:
                    return
                if (
                    error is not None
                    and max_consecutive_failures > 0
                    and failures_in_a_row >= max_consecutive_failures
                ):
                    batches_word = "batch" if failures_in_a_row == 1 else "batches"
                    self.aborted_reason = (
                        f"{error} (stopped after {failures_in_a_row} failed "
                        f"{batches_word} in a row)"
                    )
                    return

    def finish(self) -> RunResult:
        stats = self.stats
        stats.elapsed_s = time.perf_counter() - self.started
        stats.generation_s = self.generation_s
        if self.cancelled:
            message = "Cancelled."
        elif self.aborted_reason is not None:
            message = f"Stopped early: {self.aborted_reason}"
        elif stats.errors:
            message = f"Finished with {_count(stats.errors, 'failed row')}."
        else:
            message = "Finished."
        self._emit("finished", message, eta=False)
        self._log_end()
        return RunResult(
            run_id=self.config.run_id,
            config=self.config,
            outputs=self.outputs,
            statuses=self.statuses,
            errors=self.errors,
            stats=stats,
            cancelled=self.cancelled,
            aborted_reason=self.aborted_reason,
            input_truncated=self.input_truncated,
            hit_token_limit=self.hit_limit,
        )

    # ---- batches ----------------------------------------------------------
    def _run_batch(
        self, backend: Any, batch: list[str], label: str, cancel: threading.Event | None
    ) -> str | None:
        """Generate one batch. Returns the error message if no row of it
        succeeded, else None."""
        self.stats.model_calls += len(batch)
        failure: tuple[type[BaseException], str] | None = None
        message = ""
        try:
            raw, flags, row_errors = self._generate(backend, batch)
        except Exception as exc:
            message = _describe_error(exc, ERROR_MESSAGE_LIMIT)
            self._log_failure(f"{label} ({_count(len(batch), 'prompt')}) failed", message)
            if getattr(exc, "fatal", False) is True:
                # Retrying the rows one by one can't help (a rejected API key).
                plain = _describe_error(exc, ERROR_MESSAGE_LIMIT, plain_backend_errors=True)
                self._fail(batch, plain)
                self.aborted_reason = plain
                return plain
            # Only the type and text are kept: the exception's traceback can
            # hold the backend's frames (and their GPU tensors) alive.
            failure = (type(exc), str(exc))
        if failure is not None:
            if len(batch) > 1:
                return self._isolate(backend, batch, failure, message, label, cancel)
            self._fail(batch, message)
            return message
        try:
            cleaned = _checked_outputs(raw, len(batch))
        except Exception as exc:
            message = _describe_error(exc, ERROR_MESSAGE_LIMIT)
            self._log_failure(f"{label} failed", message)
            self._fail(batch, message)
            return message
        if any(row_errors):
            return self._commit_some(batch, cleaned, flags, row_errors, label)
        self._commit(batch, cleaned, flags)
        return None

    def _commit_some(
        self,
        batch: list[str],
        outputs: list[str],
        flags: list[bool],
        row_errors: list[str | None],
        label: str,
    ) -> str | None:
        """Keep the rows of a batch that succeeded and fail the rows the
        backend reported an error for (``last_row_errors``) - a cloud API
        answers each prompt separately, so one refused text must not cost
        the other answers of its batch. Returns the first error when no row
        succeeded, else None."""
        good = [i for i, error in enumerate(row_errors) if not error]
        bad = [i for i, error in enumerate(row_errors) if error]
        if good:
            self._commit(
                [batch[i] for i in good], [outputs[i] for i in good], [flags[i] for i in good]
            )
        first = _clip(str(row_errors[bad[0]]), ERROR_MESSAGE_LIMIT)
        # No traceback: the backend reported these, nothing was raised.
        log.warning(
            "Run %s: %s: %s of %s failed: %s",
            self.config.run_id,
            label,
            _count(len(bad), "row"),
            f"{len(batch):,}",
            first,
        )
        for i in bad:
            self._fail([batch[i]], _clip(str(row_errors[i]), ERROR_MESSAGE_LIMIT))
        return None if good else first

    def _isolate(
        self,
        backend: Any,
        batch: list[str],
        batch_failure: tuple[type[BaseException], str],
        batch_message: str,
        label: str,
        cancel: threading.Event | None,
    ) -> str | None:
        """Retry a failed batch one prompt at a time (see execute_run)."""
        succeeded = failed = 0
        same_as_batch = 0
        eta = self.eta
        # Its prompts, one call each, are back in the work left (the batch
        # left it when it started), so the time left doesn't reach 0 - and
        # the bar 100% - while they are retried.
        if eta is not None:
            eta.requeue([self.prompt_chars.get(key, 0) for key in batch])
        for position, key in enumerate(batch):
            if position == 2 and same_as_batch == 2:
                rest = batch[position:]
                if eta is not None:
                    for skipped in rest:
                        eta.start_batch(self.prompt_chars.get(skipped, 0))
                log.warning(
                    "Run %s: %s - the first two rows failed on their own with the same "
                    "error, so it isn't a single bad row; %s marked failed without retrying.",
                    self.config.run_id,
                    label,
                    _count(len(rest), "more row"),
                )
                self._fail(rest, batch_message)
                failed += len(rest)
                break
            if cancel is not None and cancel.is_set():
                self.cancelled = True
                break
            if eta is not None:
                eta.start_batch(self.prompt_chars.get(key, 0))
            try:
                raw, flags, row_errors = self._generate(backend, [key])
                cleaned = _checked_outputs(raw, 1)
            except Exception as exc:
                if position < 2 and (type(exc), str(exc)) == batch_failure:
                    same_as_batch += 1
                message = _describe_error(exc, ERROR_MESSAGE_LIMIT)
                log.debug(
                    "Run %s: %s, row retried alone failed: %s", self.config.run_id, label, message
                )
                self._fail([key], message)
                failed += 1
            else:
                if row_errors[0]:  # the backend reported this row failed
                    self._fail([key], _clip(row_errors[0], ERROR_MESSAGE_LIMIT))
                    failed += 1
                else:
                    succeeded += 1
                    self._commit([key], cleaned, flags)
                    if self.aborted_reason is not None:  # the store failed
                        break
            self._emit(
                "generating",
                f"{label}: retrying rows one at a time ({position + 1:,} of {len(batch):,})",
            )
        else:
            log.info(
                "Run %s: %s retried one row at a time: %s succeeded, %s failed.",
                self.config.run_id,
                label,
                f"{succeeded:,}",
                f"{failed:,}",
            )
        return None if succeeded else batch_message

    def _generate(
        self, backend: Any, keys: list[str]
    ) -> tuple[list[Any], list[bool], list[str | None]]:
        """(raw outputs, hit-limit flags, per-row errors) for ``keys``;
        raises what the backend raises."""
        assert self.plan is not None
        config, plan = self.config, self.plan
        prompts = [prompting.render_prompt(config.template, plan.snippets[k]) for k in keys]
        started = time.perf_counter()
        try:
            raw = list(backend.generate(prompts, config.max_new_tokens))
        finally:
            seconds = time.perf_counter() - started
            self.generation_s += seconds
        if self.eta is not None:
            self.eta.observe(sum(len(p) for p in prompts), seconds)
        usage = _usage(backend)
        if usage is not None:
            self.stats.input_tokens += usage[0]
            self.stats.output_tokens += usage[1]
        return raw, _hit_limit_flags(backend, len(prompts)), _row_errors(backend, len(prompts))

    def _commit(self, keys: list[str], outputs: list[str], flags: list[bool]) -> None:
        """Save finished outputs, then mark their rows generated."""
        items = dict(zip(keys, outputs, strict=True))
        # Saved before any row is marked done or progress reported, so an
        # exception from here on can never lose a finished batch.
        if self.store is not None:
            try:
                self._save(self.store, items, [k for k, f in zip(keys, flags, strict=True) if f])
            except Exception as exc:
                # The outputs are real, so the rows still count as generated;
                # the run stops and a re-run regenerates what wasn't saved.
                where = getattr(self.store, "path", None) or "the results database"
                self.aborted_reason = (
                    f"Couldn't save outputs to {where}: {_describe_error(exc)}. "
                    "Rows finished so far are kept; run again to resume."
                )
                log.warning("Run %s: %s", self.config.run_id, self.aborted_reason)
        assert self.plan is not None
        stats = self.stats
        for key, flag in zip(keys, flags, strict=True):
            output = items[key]
            rows = self.todo.pop(key)
            stats.duplicates += len(rows) - 1
            for i in rows:
                self.outputs[i] = output
                self.statuses[i] = "generated"
                stats.generated += 1
                stats.pending -= 1
                if flag:
                    self.hit_limit[i] = True
                    stats.hit_token_limit += 1
            self.recent.append(
                (
                    _clip(" ".join(self.plan.snippets[key].split()), RECENT_TEXT_CHARS),
                    _clip(output, RECENT_OUTPUT_CHARS),
                )
            )

    def _fail(self, keys: list[str], message: str) -> None:
        stats = self.stats
        for key in keys:
            for i in self.todo.pop(key):
                self.statuses[i] = "error"
                self.errors[i] = message
                stats.errors += 1
                stats.pending -= 1

    def _save(self, store: Store, items: dict[str, str], hit_limit: list[str]) -> None:
        """Store one batch, registering the run's manifest first.
        ``hit_limit``: the keys whose output reached ``max_new_tokens``."""
        config = self.config

        def put() -> None:
            # Only passed when there is something to flag, so a store
            # without the parameter (a minimal stand-in) still works.
            if hit_limit:
                store.put_outputs(config.run_id, items, hit_limit=hit_limit)
            else:
                store.put_outputs(config.run_id, items)

        if not self.registered:
            store.upsert_run(config, source=self.source)
            self.registered = True
        try:
            put()
        except KeyError:
            # The run was deleted (e.g. from another window) while running.
            store.upsert_run(config, source=self.source)
            put()

    # ---- reporting --------------------------------------------------------
    def _emit(self, phase: Phase, message: str, eta: bool = True) -> None:
        if self.progress is None:
            return
        stats = self.stats
        now = time.perf_counter()
        rate: float | None = None
        if stats.generated:
            rate = stats.generated / max(self.generation_s, 1e-6)
        eta_s = self.eta.report(now) if eta and self.eta is not None else None
        done_rows = stats.blank + stats.cached + stats.generated + stats.errors
        self.progress(
            Progress(
                phase=phase,
                total_rows=stats.total,
                done_rows=done_rows,
                generated_rows=stats.generated,
                cached_rows=stats.cached,
                blank_rows=stats.blank,
                error_rows=stats.errors,
                elapsed_s=now - self.started,
                rows_per_s=rate,
                eta_s=eta_s,
                message=message,
                recent=tuple(self.recent),
                generation_s=self.generation_s,
                work_done=self._measure_work(phase, eta_s, done_rows),
            )
        )

    def _measure_work(self, phase: Phase, eta_s: float | None, done_rows: int) -> float:
        """``Progress.work_done`` for a snapshot (see execute_run)."""
        stats = self.stats
        total = stats.total
        complete = not self.cancelled and self.aborted_reason is None and stats.pending == 0
        if phase == "finished" and complete:
            work = 1.0
        elif (
            total and phase == "generating" and eta_s is not None and self.generation_s + eta_s > 0
        ):
            # Rows that needed no model were done before generation started.
            before = (stats.blank + stats.cached) / total
            share = self.generation_s / (self.generation_s + eta_s)
            work = before + (1.0 - before) * share
        else:
            work = done_rows / total if total else 1.0
        self.work_done = max(self.work_done, min(work, 1.0))
        return self.work_done

    def _log_failure(self, what: str, message: str) -> None:
        # A traceback for the first failure only: a run of failing batches
        # would otherwise fill the log file with copies of one stack.
        log.warning(
            "Run %s: %s: %s", self.config.run_id, what, message, exc_info=not self.failure_logged
        )
        self.failure_logged = True

    def _log_end(self) -> None:
        stats = self.stats
        parts = [
            f"{count:,} {what}"
            for count, what in (
                (stats.generated, "generated"),
                (stats.cached, "cached"),
                (stats.blank, "blank"),
                (stats.errors, "failed"),
                (stats.pending, "not run"),
            )
            if count
        ]
        if stats.hit_token_limit:
            parts.append(f"{stats.hit_token_limit:,} at the token limit")
        timing = f"{stats.elapsed_s:.1f}s"
        if stats.load_s:
            timing += f" ({stats.load_s:.1f}s loading)"
        if stats.generated and stats.generation_s > 0:
            timing += f", {stats.generated / stats.generation_s:.2f} rows/s"
        if self.cancelled:
            outcome = "cancelled"
        elif self.aborted_reason is not None:
            outcome = f"stopped early ({self.aborted_reason})"
        else:
            outcome = "finished"
        log.info(
            "Run %s %s: %s - %s; %s.",
            self.config.run_id,
            outcome,
            _count(stats.total, "row"),
            ", ".join(parts) or "nothing to do",
            timing,
        )


class _Eta:
    """Generation time left, estimated from the work left rather than rows.

    Every successful generate call is one observation (prompt characters,
    seconds). A call is taken to cost ``a + b * chars``: a part per call
    (decoding up to ``max_new_tokens``) and a part per character (reading
    the prompts), fitted by least squares and applied to the batches not
    started yet. Rows never enter it, so a text shared by a thousand
    duplicate rows counts as the one prompt it is.

    Pricing what is left per call (the mean seconds per call so far) and
    per character (the seconds per character so far) brackets the time
    left, in either processing order. Shortest first, the batches left are
    longer than those seen, so per-call pricing is the low end and
    per-character pricing the high end; longest first, they are shorter and
    the roles swap. The fit is clamped to that range (:meth:`seconds_left`),
    which also steadies it while the lengths seen are too alike to tell
    ``a`` from ``b``. Then longest first (``longest_first``) takes the high
    end, per call: no batch left is longer than those so far, so the
    estimate starts high and only falls once the lengths differ - the
    middle of the range would start it low wherever decoding dominates
    (long outputs), as with a run of texts all cut to the max input
    characters. Shortest first takes the middle of the range, since per
    character after very short texts can be many times too high.

    The prompts of a failed batch retried one at a time (see
    :func:`execute_run`) rejoin the work left as one call each
    (:meth:`requeue`), and each call leaves it as it starts
    (:meth:`start_batch`) or is marked failed without being retried.

    The estimate reported (:meth:`report`) is smoothed: ``s * fit + (1 - s)
    * max(0, previous - dt)``, with ``s`` = :data:`ETA_SMOOTHING` and ``dt``
    the seconds since the previous report. A fit that was right stays right
    (the previous report counted down is the same number), while a batch
    that ran unusually fast or slow moves the figure only part of the way.
    The first report is the fit itself, and once every call has started it
    is 0.
    """

    def __init__(
        self,
        batch_chars: Sequence[int],
        smoothing: float = ETA_SMOOTHING,
        *,
        longest_first: bool = False,
    ) -> None:
        self.batches_left = len(batch_chars)
        """Calls still to start (batches, and prompts to retry alone)."""
        self.chars_left = float(sum(batch_chars))
        self.smoothing = smoothing
        self.longest_first = longest_first
        self._n = 0
        self._c = self._cc = self._t = self._ct = 0.0
        self._reported: float | None = None
        self._reported_at = 0.0

    def start_batch(self, chars: int) -> None:
        """A call of ``chars`` prompt characters leaves the work left."""
        self.batches_left -= 1
        self.chars_left -= chars

    def requeue(self, prompt_chars: Sequence[int]) -> None:
        """These prompts (of a batch that failed) will be retried one call
        each."""
        self.batches_left += len(prompt_chars)
        self.chars_left += sum(prompt_chars)

    def observe(self, chars: int, seconds: float) -> None:
        self._n += 1
        self._c += chars
        self._cc += chars * chars
        self._t += seconds
        self._ct += chars * seconds

    def seconds_left(self) -> float | None:
        """The fit, unsmoothed. None before the first observation; 0 once
        every call started."""
        n = self._n
        if not n:
            return None
        if self.batches_left <= 0:
            return 0.0
        mean_c, mean_t = self._c / n, self._t / n
        per_call = mean_t * self.batches_left
        per_char = self._t / self._c * self.chars_left if self._c > 0 else per_call
        low, high = min(per_call, per_char), max(per_call, per_char)
        variance = self._cc / n - mean_c * mean_c
        if variance <= 1e-3 * mean_c * mean_c:
            return high if self.longest_first else math.sqrt(low * high)
        b = (self._ct / n - mean_c * mean_t) / variance
        a = mean_t - b * mean_c
        if b < 0:
            a, b = mean_t, 0.0
        elif a < 0:
            a, b = 0.0, self._ct / self._cc
        estimate = a * self.batches_left + b * self.chars_left
        return min(max(estimate, low), high)

    def report(self, now: float) -> float | None:
        """The smoothed estimate at ``now`` (seconds on a clock that only
        counts up, such as ``time.perf_counter()``). None before the first
        observation; 0 once every call started."""
        fit = self.seconds_left()
        if fit is None:
            return None
        if self.batches_left <= 0:
            estimate = 0.0
        elif self._reported is None:
            estimate = fit
        else:
            counted_down = max(0.0, self._reported - max(0.0, now - self._reported_at))
            estimate = self.smoothing * fit + (1.0 - self.smoothing) * counted_down
        self._reported, self._reported_at = estimate, now
        return estimate


def _is_loaded(manager: Any, target: tuple[str, str]) -> bool:
    """Whether ``manager`` already holds this (model_id, precision) - for a
    local folder, unchanged since it was loaded. False if it can't say."""
    try:
        is_loaded = getattr(manager, "is_loaded", None)
        if callable(is_loaded):
            return bool(is_loaded(*target))
        return bool(manager.loaded == target)
    except Exception:
        return False


def _describe_error(
    exc: BaseException, limit: int = ERROR_MESSAGE_LIMIT, plain_backend_errors: bool = False
) -> str:
    """One-line ``"ExcType: message"``, at most ``limit`` characters.

    With ``plain_backend_errors``, a :class:`BackendError` (whose message is
    already written for users) is given without its type name.
    """
    text = " ".join(str(exc).split())
    name = type(exc).__name__
    if plain_backend_errors and isinstance(exc, BackendError) and text:
        message = text
    else:
        message = f"{name}: {text}" if text else name
    if len(message) > limit:
        message = message[: limit - 1].rstrip() + "…"
    return message


def _count(n: int, noun: str) -> str:
    return f"{n:,} {noun}" if n == 1 else f"{n:,} {noun}s"


def _checked_outputs(raw: Sequence[Any], n_prompts: int) -> list[str]:
    """Cleaned outputs; RuntimeError unless there is one per prompt."""
    if len(raw) != n_prompts:
        raise RuntimeError(f"the model returned {len(raw)} outputs for {n_prompts} prompts")
    return [labels.clean_output(text) for text in raw]


def _stored_hit_limits(store: Any, run_id: str, keys: list[str]) -> set[str]:
    """``store.get_hit_limit_keys``, or nothing for a store without it."""
    get = getattr(store, "get_hit_limit_keys", None)
    return set(get(run_id, keys)) if callable(get) else set()


def _hit_limit_flags(backend: Any, n_prompts: int) -> list[bool]:
    """The backend's ``last_hit_limit`` for the call just made, or all False
    when it doesn't report one (or reports the wrong number)."""
    try:
        flags = getattr(backend, "last_hit_limit", None)
        if isinstance(flags, Sequence) and not isinstance(flags, (str, bytes)):
            values = [bool(flag) for flag in flags]
            if len(values) == n_prompts:
                return values
    except Exception:  # a property that raises, a flag that can't be a bool...
        pass
    return [False] * n_prompts


def _row_errors(backend: Any, n_prompts: int) -> list[str | None]:
    """The backend's ``last_row_errors`` for the call just made: an error
    message for each prompt that failed on its own, None for the others.
    All None when the backend doesn't report them (or reports the wrong
    number)."""
    try:
        errors = getattr(backend, "last_row_errors", None)
        if isinstance(errors, Sequence) and not isinstance(errors, (str, bytes)):
            values = [str(error) if error else None for error in errors]
            if len(values) == n_prompts:
                return values
    except Exception:
        pass
    return [None] * n_prompts


def _usage(backend: Any) -> tuple[int, int] | None:
    """The backend's ``last_usage`` (input tokens, output tokens) for the
    call just made, or None when it doesn't report one."""
    try:
        usage = getattr(backend, "last_usage", None)
        if usage is None:
            return None
        sent, generated = usage
        return max(0, int(sent)), max(0, int(generated))
    except Exception:
        return None


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"
