"""Tests for the background job manager."""

from __future__ import annotations

import contextlib
import dataclasses
import sqlite3
import threading
from collections.abc import Sequence

import pytest

from batchtextreview import engine
from batchtextreview.backends.base import ModelLoadError
from batchtextreview.engine import RunConfig
from batchtextreview.jobs import (
    TERMINAL_STATES,
    Job,
    JobBusyError,
    JobManager,
    JobRequest,
    JobSnapshot,
)

CONFIG = RunConfig(
    model_id="demo://fake",
    model_label="Fake",
    precision="none",
    template="Q: {text}",
    max_new_tokens=8,
)
TIMEOUT = 5.0
"""Generous upper bound for waits that normally finish in milliseconds."""


class InstantBackend:
    model_id = "demo://fake"
    precision = "none"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def generate(self, prompts: Sequence[str], max_new_tokens: int) -> list[str]:
        self.calls.append(list(prompts))
        return [p.upper() for p in prompts]

    def close(self) -> None:
        pass


class GatedBackend(InstantBackend):
    """Each generate() call announces itself, then waits to be released."""

    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Semaphore(0)
        self.proceed = threading.Semaphore(0)

    def generate(self, prompts: Sequence[str], max_new_tokens: int) -> list[str]:
        self.calls.append(list(prompts))
        self.started.release()
        if not self.proceed.acquire(timeout=TIMEOUT):
            raise RuntimeError("test never released the batch")
        return [p.upper() for p in prompts]

    def wait_started(self) -> None:
        assert self.started.acquire(timeout=TIMEOUT), "batch never started"

    def release(self, n: int = 1) -> None:
        for _ in range(n):
            self.proceed.release()


class FakeManager:
    def __init__(self, backend=None, load_error: Exception | None = None) -> None:
        self.backend = backend if backend is not None else InstantBackend()
        self.load_error = load_error
        self.loaded = None

    @contextlib.contextmanager
    def acquire(self, model_id: str, precision: str):
        if self.load_error is not None:
            raise self.load_error
        self.loaded = (model_id, precision)
        yield self.backend


def request(texts=("a", "b", "c"), **kwargs) -> JobRequest:
    kwargs.setdefault("batch_size", 1)
    return JobRequest(config=CONFIG, texts=list(texts), **kwargs)


@pytest.fixture
def gated():
    backend = GatedBackend()
    yield backend
    backend.release(50)  # never leave a worker blocked


def finished(job: Job) -> JobSnapshot:
    assert job.wait(TIMEOUT), "job did not finish"
    snap = job.snapshot()
    assert snap.done
    return snap


# --------------------------------------------------------------------------
# Lifecycle
# --------------------------------------------------------------------------


def test_new_job_is_queued():
    job = Job(request())
    snap = job.snapshot()
    assert snap.state == "queued"
    assert not snap.done
    assert snap.progress.total_rows == 3
    assert snap.progress.done_rows == 0
    assert snap.started_at is None and snap.finished_at is None
    assert snap.error is None and not snap.has_result
    assert job.result is None
    assert not job.wait(0.01)


def test_job_ids_are_unique():
    assert Job(request()).id != Job(request()).id


def test_submit_runs_to_success():
    jobs = JobManager(None, FakeManager())
    job = jobs.submit(request(["a", None, "b"]))
    snap = finished(job)
    assert snap.state == "succeeded"
    assert snap.error is None
    assert snap.has_result
    assert snap.started_at is not None and snap.finished_at is not None
    assert snap.created_at <= snap.started_at <= snap.finished_at
    assert snap.progress.phase == "finished"
    assert snap.progress.fraction == 1.0
    assert job.result is not None
    assert job.result.outputs == ["Q: A", "", "Q: B"]
    assert job.result.complete
    assert jobs.active() is None
    assert jobs.latest() is job
    assert jobs.get(job.id) is job


def test_submit_passes_store_and_batch_settings(monkeypatch):
    seen = {}
    real = engine.execute_run

    def spy(config, texts, **kwargs):
        seen.update(kwargs)
        return real(config, texts, **kwargs)

    monkeypatch.setattr(engine, "execute_run", spy)
    store = object()
    manager = FakeManager()
    job = JobManager(store, manager).submit(  # type: ignore[arg-type]
        request(["a"], batch_size=3, max_batch_chars=123)
    )
    # The fake store would fail upsert_run, so don't care how the run ends.
    finished(job)
    assert seen["store"] is store
    assert seen["manager"] is manager
    assert seen["batch_size"] == 3
    assert seen["max_batch_chars"] == 123
    assert isinstance(seen["cancel"], threading.Event)
    assert callable(seen["progress"])


def test_progress_visible_while_running(gated):
    jobs = JobManager(None, FakeManager(gated))
    job = jobs.submit(request(["a", "b", "c"]))
    gated.wait_started()
    snap = job.snapshot()
    assert snap.state == "running"
    # Batch 1 is running: the model is ready, so not "loading" any more.
    assert snap.progress.phase == "generating"
    assert snap.progress.generated_rows == 0
    assert jobs.active() is job
    gated.release()
    gated.wait_started()  # batch 2 started, so batch 1's progress is in
    snap = job.snapshot()
    assert snap.progress.phase == "generating"
    assert snap.progress.done_rows == 1
    assert snap.progress.generated_rows == 1
    gated.release(2)
    assert finished(job).state == "succeeded"


def test_recent_outputs_visible_while_running(gated):
    jobs = JobManager(None, FakeManager(gated))
    job = jobs.submit(request(["a", "b", "c"]))
    gated.wait_started()
    assert job.snapshot().progress.recent == ()
    gated.release()
    gated.wait_started()  # batch 2 started, so batch 1's output is in
    assert job.snapshot().progress.recent == (("a", "Q: A"),)
    gated.release(2)
    snap = finished(job)
    assert snap.progress.recent == (("a", "Q: A"), ("b", "Q: B"), ("c", "Q: C"))


def test_cancel_while_a_failed_batch_is_retried_row_by_row():
    class FailsTogether(GatedBackend):
        def generate(self, prompts, max_new_tokens):
            if len(prompts) > 1:
                self.calls.append(list(prompts))
                raise RuntimeError("batch failed")
            return super().generate(prompts, max_new_tokens)

    backend = FailsTogether()
    jobs = JobManager(None, FakeManager(backend))
    job = jobs.submit(request(["a", "b", "c", "d"], batch_size=4))
    try:
        backend.wait_started()  # the first row, retried on its own
        job.cancel()
        backend.release()
        snap = finished(job)
    finally:
        backend.release(10)
    assert snap.state == "cancelled"
    assert job.result is not None
    assert job.result.statuses == ["generated", "pending", "pending", "pending"]
    assert [len(call) for call in backend.calls] == [4, 1]
    assert snap.progress.recent == (("a", "Q: A"),)


def test_busy_error_while_running(gated):
    jobs = JobManager(None, FakeManager(gated))
    first = jobs.submit(request(["a"]))
    gated.wait_started()
    with pytest.raises(JobBusyError):
        jobs.submit(request(["b"]))
    assert jobs.active() is first
    gated.release()
    finished(first)
    second = jobs.submit(request(["c"]))
    gated.wait_started()
    gated.release()
    assert finished(second).state == "succeeded"
    assert jobs.latest() is second


def test_cancel_mid_run(gated):
    jobs = JobManager(None, FakeManager(gated))
    job = jobs.submit(request(["a", "b", "c", "d"]))
    gated.wait_started()
    assert jobs.cancel(job.id) is True
    assert job.cancel_requested
    gated.release()  # finish the batch in flight; the next one never starts
    snap = finished(job)
    assert snap.state == "cancelled"
    assert snap.error is None
    assert snap.has_result
    result = job.result
    assert result is not None and result.cancelled
    assert result.statuses == ["generated", "pending", "pending", "pending"]
    assert len(gated.calls) == 1
    assert jobs.cancel(job.id) is False  # already finished


def test_cancel_unknown_job():
    assert JobManager(None, FakeManager()).cancel("nope") is False


def test_failed_when_execute_run_raises(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("kaboom\nwith detail")

    monkeypatch.setattr(engine, "execute_run", boom)
    jobs = JobManager(None, FakeManager())
    job = jobs.submit(request())
    snap = finished(job)
    assert snap.state == "failed"
    assert snap.error == "RuntimeError: kaboom with detail"
    assert not snap.has_result
    assert job.result is None
    assert snap.progress.phase == "finished"
    assert jobs.active() is None
    # The manager is free again.
    monkeypatch.undo()
    assert finished(jobs.submit(request())).state == "succeeded"


def test_aborted_run_is_failed_with_partial_result():
    manager = FakeManager(load_error=ModelLoadError("No such model.", hint="Check the name."))
    job = JobManager(None, manager).submit(request(["a", None]))
    snap = finished(job)
    assert snap.state == "failed"
    assert snap.has_result
    result = job.result
    assert result is not None
    assert result.aborted_reason
    assert snap.error == result.aborted_reason
    assert "No such model." in snap.error
    assert result.statuses == ["pending", "blank"]


def test_store_failure_is_failed_with_partial_result():
    class LockedStore:
        path = "/data/btr.db"

        def get_outputs(self, run_id, keys=None):
            return {}

        def upsert_run(self, config, source="app"):
            pass

        def put_outputs(self, run_id, items):
            if "Q: B" in items.values():
                raise sqlite3.OperationalError("database is locked")

    job = JobManager(LockedStore(), FakeManager()).submit(request(["a", "b", "c"]))
    snap = finished(job)
    assert snap.state == "failed"
    assert snap.has_result
    assert job.result is not None
    assert job.result.outputs == ["Q: A", "Q: B", ""]
    assert snap.error == job.result.aborted_reason
    assert snap.error.startswith("Couldn't save outputs to /data/btr.db")


def test_run_with_failed_rows_still_succeeds():
    class Flaky(InstantBackend):
        def generate(self, prompts, max_new_tokens):
            if "Q: b" in prompts:
                raise RuntimeError("bad batch")
            return super().generate(prompts, max_new_tokens)

    job = JobManager(None, FakeManager(Flaky())).submit(request(["a", "b"]))
    snap = finished(job)
    assert snap.state == "succeeded"
    assert job.result.statuses == ["generated", "error"]


# --------------------------------------------------------------------------
# Snapshots
# --------------------------------------------------------------------------


def test_snapshot_is_immutable():
    job = Job(request())
    snap = job.snapshot()
    with pytest.raises(dataclasses.FrozenInstanceError):
        snap.state = "running"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        snap.progress.done_rows = 5  # type: ignore[misc]


def test_snapshot_does_not_change_after_the_fact(gated):
    jobs = JobManager(None, FakeManager(gated))
    job = jobs.submit(request(["a"]))
    gated.wait_started()
    during = job.snapshot()
    gated.release()
    after = finished(job)
    assert during.state == "running" and not during.done
    assert during.finished_at is None and not during.has_result
    assert after.state == "succeeded" and after.finished_at is not None


def test_terminal_snapshots_are_consistent():
    jobs = JobManager(None, FakeManager())
    for texts in (["a"], [None], []):
        job = jobs.submit(request(texts))
        snap = finished(job)
        assert snap.state in TERMINAL_STATES
        assert snap.finished_at is not None
        assert snap.has_result == (job.result is not None)
        assert snap.progress.phase == "finished"
        assert snap.progress.total_rows == len(texts)


# --------------------------------------------------------------------------
# on_finish hook
# --------------------------------------------------------------------------


def test_on_finish_runs_before_job_reports_done():
    seen = {}

    def hook(job: Job) -> None:
        seen["done"] = job.snapshot().done
        seen["result"] = job.result
        job.request.context["logged"] = True

    job = JobManager(None, FakeManager()).submit(request(["a"], on_finish=hook))
    finished(job)
    assert seen["done"] is False
    assert seen["result"] is job.result is not None
    assert job.request.context["logged"] is True


def test_on_finish_errors_are_swallowed():
    def hook(job: Job) -> None:
        raise ValueError("hook broke")

    job = JobManager(None, FakeManager()).submit(request(["a"], on_finish=hook))
    assert finished(job).state == "succeeded"


def test_on_finish_called_when_run_fails(monkeypatch):
    monkeypatch.setattr(engine, "execute_run", lambda *a, **k: 1 / 0)
    calls = []
    job = JobManager(None, FakeManager()).submit(request(on_finish=calls.append))
    assert finished(job).state == "failed"
    assert calls == [job]
    assert job.snapshot().error.startswith("ZeroDivisionError")


def test_context_is_kept_untouched():
    context = {"frame": object(), "preset": "Sentiment"}
    job = JobManager(None, FakeManager()).submit(request(["a"], context=context))
    finished(job)
    assert job.request.context is context
    assert set(context) == {"frame", "preset"}


# --------------------------------------------------------------------------
# History and lookup
# --------------------------------------------------------------------------


def test_history_cap():
    jobs = JobManager(None, FakeManager(), history=2)
    submitted = []
    for _ in range(4):
        job = jobs.submit(request(["a"]))
        finished(job)
        submitted.append(job)
    assert jobs.get(submitted[0].id) is None
    assert jobs.get(submitted[1].id) is None
    assert jobs.get(submitted[2].id) is submitted[2]
    assert jobs.get(submitted[3].id) is submitted[3]
    assert jobs.latest() is submitted[3]
    assert jobs.jobs() == [submitted[3], submitted[2]]


def test_active_job_is_kept_beyond_history(gated):
    jobs = JobManager(None, FakeManager(gated), history=0)
    job = jobs.submit(request(["a"]))
    gated.wait_started()
    assert jobs.get(job.id) is job
    assert jobs.jobs() == [job]
    gated.release()
    finished(job)
    assert jobs.latest() is job


def test_lookups_on_empty_manager():
    jobs = JobManager(None, FakeManager())
    assert jobs.active() is None
    assert jobs.latest() is None
    assert jobs.get("x") is None
    assert jobs.jobs() == []


def test_wait_times_out_then_succeeds(gated):
    job = JobManager(None, FakeManager(gated)).submit(request(["a"]))
    gated.wait_started()
    assert job.wait(0.01) is False
    gated.release()
    assert job.wait(TIMEOUT) is True
    assert job.wait(0) is True


def test_cancel_queued_job_never_loads_model():
    manager = FakeManager()
    manager.acquire = None  # type: ignore[assignment]  # would raise if used
    job = Job(request(["a"]))
    job.cancel()
    jobs = JobManager(None, manager)
    # Drive the worker directly with the pre-cancelled job.
    jobs._work(job)
    snap = job.snapshot()
    assert snap.state == "cancelled"
    assert job.result is not None and job.result.statuses == ["pending"]


# --------------------------------------------------------------------------
# Real Store / ModelManager / FakeBackend (skipped while those are stubs)
# --------------------------------------------------------------------------


def test_real_stack(tmp_path):
    from batchtextreview.backends.fake import FakeBackend
    from batchtextreview.backends.manager import ModelManager
    from batchtextreview.store import Store

    try:
        store = Store(tmp_path / "btr.db")
        backend = FakeBackend(responder=lambda p: "ok")
        backend.generate(["probe"], 1)
        manager = ModelManager(factory=lambda model_id, precision: backend)
        with manager.acquire("demo://fake", "none"):
            pass
    except NotImplementedError:
        pytest.skip("Store / ModelManager / FakeBackend not implemented yet")

    jobs = JobManager(store, manager)
    job = jobs.submit(request(["a", "b", "a"]))
    assert finished(job).state == "succeeded"
    assert job.result.outputs == ["ok", "ok", "ok"]
    assert store.count_outputs(CONFIG.run_id) == 2

    again = jobs.submit(request(["a", "b"]))
    assert finished(again).state == "succeeded"
    assert again.result.statuses == ["cached", "cached"]
