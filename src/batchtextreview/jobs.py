"""Background run jobs.

Streamlit re-executes the page script on every widget interaction, which
would interrupt a long run executing in the script thread. Runs therefore
execute on a worker thread owned by a process-wide :class:`JobManager`; the
UI polls :meth:`Job.snapshot` (e.g. from an ``st.fragment(run_every=...)``)
and can cancel. Only one job runs at a time (one GPU, one model).

Worker threads never call Streamlit APIs.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from batchtextreview import engine
from batchtextreview.engine import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_BATCH_CHARS,
    Progress,
    RunConfig,
    RunResult,
)

if TYPE_CHECKING:
    from batchtextreview.backends.manager import ModelManager
    from batchtextreview.store import Store

log = logging.getLogger(__name__)

JobState = Literal["queued", "running", "succeeded", "failed", "cancelled"]
TERMINAL_STATES: frozenset[str] = frozenset({"succeeded", "failed", "cancelled"})


class JobBusyError(RuntimeError):
    """A job is already running."""


@dataclass
class JobRequest:
    config: RunConfig
    texts: list[Any]
    batch_size: int = DEFAULT_BATCH_SIZE
    max_batch_chars: int = DEFAULT_MAX_BATCH_CHARS
    context: dict[str, Any] = field(default_factory=dict)
    """Opaque data the UI needs to rebuild the results view later (the
    working DataFrame, column choices, preset key, variable, label set,
    dataset name...). The job never inspects it."""
    on_finish: Callable[[Job], None] | None = None
    """Called on the worker thread when the run has ended - after
    ``job.result`` is attached (it may be None if the run failed outright)
    and *before* the job reports a terminal state, so once
    ``snapshot().done`` is True (or :meth:`Job.wait` returns) it has run.
    Exceptions it raises are swallowed. Typical use: log the evaluation even
    if no browser session is watching."""


@dataclass(frozen=True)
class JobSnapshot:
    """Consistent, immutable view of a job for rendering."""

    id: str
    state: JobState
    progress: Progress
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    error: str | None
    """Unexpected exception text (state "failed") or the run's
    aborted_reason."""
    has_result: bool

    @property
    def done(self) -> bool:
        return self.state in TERMINAL_STATES


class Job:
    """One run. Mutated only by its worker thread, under ``_lock``."""

    def __init__(self, request: JobRequest) -> None:
        self.id = uuid.uuid4().hex
        self.request = request
        self.result = None
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._done = threading.Event()
        self._state: JobState = "queued"
        self._progress = Progress(
            phase="planning", total_rows=len(request.texts), message="Waiting to start..."
        )
        self._created_at = datetime.now()
        self._started_at: datetime | None = None
        self._finished_at: datetime | None = None
        self._error: str | None = None

    id: str
    request: JobRequest
    result: RunResult | None

    def __repr__(self) -> str:
        return f"Job({self.id[:8]}, {self._state})"

    def snapshot(self) -> JobSnapshot:
        with self._lock:
            return JobSnapshot(
                id=self.id,
                state=self._state,
                progress=self._progress,
                created_at=self._created_at,
                started_at=self._started_at,
                finished_at=self._finished_at,
                error=self._error,
                has_result=self.result is not None,
            )

    def cancel(self) -> None:
        """Ask the worker to stop after the current batch."""
        self._cancel.set()

    @property
    def cancel_requested(self) -> bool:
        """True once :meth:`cancel` has been called."""
        return self._cancel.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the job reaches a terminal state. True if it did."""
        return self._done.wait(timeout)

    # ---- worker side ------------------------------------------------------
    def _mark_running(self) -> None:
        with self._lock:
            self._state = "running"
            self._started_at = datetime.now()
            self._progress = dataclasses.replace(
                self._progress, message="Checking saved outputs..."
            )

    def _set_progress(self, progress: Progress) -> None:
        with self._lock:
            self._progress = progress

    def _attach(self, result: RunResult | None) -> None:
        with self._lock:
            self.result = result

    def _finish(self, state: JobState, error: str | None) -> None:
        with self._lock:
            self._state = state
            self._error = error
            self._finished_at = datetime.now()
            if self._progress.phase != "finished":
                message = error or ("Cancelled." if state == "cancelled" else "Finished.")
                self._progress = dataclasses.replace(
                    self._progress, phase="finished", eta_s=None, message=message
                )

    def _signal_done(self) -> None:
        self._done.set()


class JobManager:
    """Owns the worker thread. Share one instance per process."""

    def __init__(self, store: Store | None, manager: ModelManager, history: int = 5) -> None:
        """``history`` = how many finished jobs to keep for :meth:`get`."""
        self._store = store
        self._manager = manager
        self._history = max(0, int(history))
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        """Held jobs in submission order (the active one plus finished ones)."""
        self._active: Job | None = None
        self._latest: Job | None = None

    def submit(self, request: JobRequest) -> Job:
        """Start a job on a new daemon thread. Raises JobBusyError if one is
        queued/running. The worker calls ``engine.execute_run`` with the
        job's cancel event and a progress callback that stores the latest
        Progress. Final state: "cancelled" if the result says cancelled;
        "failed" if execute_run raised (error = "ExcType: msg", no result) or
        the result has an aborted_reason (error = that reason; the partial
        result is still attached); else "succeeded"."""
        with self._lock:
            if self._active is not None and not self._active.snapshot().done:
                raise JobBusyError("A run is already in progress.")
            job = Job(request)
            self._jobs[job.id] = job
            self._active = job
            self._latest = job
            self._prune()
            worker = threading.Thread(
                target=self._work, args=(job,), name=f"btr-job-{job.id[:8]}", daemon=True
            )
            try:
                worker.start()
            except BaseException as exc:
                # Never leave a "queued" job behind that blocks every later submit.
                job._finish("failed", _describe(exc))
                job._signal_done()
                self._active = None
                raise
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def active(self) -> Job | None:
        """The queued/running job, if any."""
        with self._lock:
            job = self._active
        if job is not None and job.snapshot().done:
            return None
        return job

    def latest(self) -> Job | None:
        """Most recently submitted job (active or finished)."""
        with self._lock:
            return self._latest

    def jobs(self) -> list[Job]:
        """Every job still held (the active one and up to ``history``
        finished ones), newest first."""
        with self._lock:
            return list(reversed(self._jobs.values()))

    def cancel(self, job_id: str) -> bool:
        """Ask job ``job_id`` to stop. False if unknown or already finished."""
        job = self.get(job_id)
        if job is None or job.snapshot().done:
            return False
        job.cancel()
        return True

    # ---- worker -----------------------------------------------------------
    def _work(self, job: Job) -> None:
        job._mark_running()
        request = job.request
        result: RunResult | None = None
        state: JobState = "failed"
        error: str | None = "The worker stopped unexpectedly."
        try:
            result = engine.execute_run(
                request.config,
                request.texts,
                store=self._store,
                manager=self._manager,
                batch_size=request.batch_size,
                max_batch_chars=request.max_batch_chars,
                progress=job._set_progress,
                cancel=job._cancel,
            )
            if result.cancelled:
                state, error = "cancelled", None
            elif result.aborted_reason:
                state, error = "failed", result.aborted_reason
            else:
                state, error = "succeeded", None
        except Exception as exc:
            log.exception("Run %s failed", job.id)
            result, state, error = None, "failed", _describe(exc)
        finally:
            job._attach(result)
            if request.on_finish is not None:
                # A failing hook must never leave the job unfinished.
                try:
                    request.on_finish(job)
                except Exception:
                    log.exception("on_finish hook of run %s failed", job.id)
            job._finish(state, error)
            with self._lock:
                if self._active is job:
                    self._active = None
                self._prune()
            # Last, so whoever wait()s sees the manager's bookkeeping done too.
            job._signal_done()

    def _prune(self) -> None:
        """Drop the oldest finished jobs beyond ``history`` (lock held)."""
        finished = [j for j in self._jobs.values() if j is not self._active and j.snapshot().done]
        for job in finished[: max(0, len(finished) - self._history)]:
            del self._jobs[job.id]


def _describe(exc: BaseException, limit: int = 1000) -> str:
    """One-line "ExcType: message", at most ``limit`` characters."""
    text = " ".join(str(exc).split())
    name = type(exc).__name__
    message = f"{name}: {text}" if text else name
    if len(message) > limit:
        message = message[: limit - 1].rstrip() + "…"
    return message
