"""Tests for the run engine (planning, batching, execution, checkpointing)."""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import math
import os
import sqlite3
import threading
from collections.abc import Callable, Sequence

import pandas as pd
import pytest

from batchtextreview import engine, labels
from batchtextreview.backends.base import ModelLoadError
from batchtextreview.engine import (
    Progress,
    RunConfig,
    compute_run_id,
    content_key,
    execute_run,
    is_blank,
    pack_batches,
    plan_run,
)

TEMPLATE = "Q: {text}"


def make_config(**changes) -> RunConfig:
    base = RunConfig(
        model_id="demo://fake",
        model_label="Fake model",
        precision="none",
        template=TEMPLATE,
        max_new_tokens=8,
    )
    return dataclasses.replace(base, **changes)


def answer(prompt: str) -> str:
    """Default scripted answer: the row text, upper-cased."""
    return "A:" + prompt[len("Q: ") :].upper()


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


class FakeStore:
    """Dict-backed stand-in for batchtextreview.store.Store."""

    def __init__(self) -> None:
        self.runs: dict[str, RunConfig] = {}
        self.outputs: dict[str, dict[str, str]] = {}
        self.hit_limit: dict[str, set[str]] = {}
        self.put_calls: list[dict[str, str]] = []
        self.get_calls = 0
        self.upserts = 0
        self.sources: list[str] = []

    def upsert_run(self, config: RunConfig, source: str = "app") -> None:
        self.upserts += 1
        self.sources.append(source)
        self.runs[config.run_id] = config

    def get_outputs(self, run_id: str, keys=None) -> dict[str, str]:
        self.get_calls += 1
        data = self.outputs.get(run_id, {})
        if keys is None:
            return dict(data)
        return {k: data[k] for k in keys if k in data}

    def put_outputs(self, run_id: str, items, hit_limit=()) -> None:
        if run_id not in self.runs:
            raise KeyError(run_id)
        self.put_calls.append(dict(items))
        self.outputs.setdefault(run_id, {}).update(items)
        flagged = self.hit_limit.setdefault(run_id, set())
        flagged.difference_update(items)
        flagged.update(hit_limit)

    def get_hit_limit_keys(self, run_id: str, keys) -> set[str]:
        return self.hit_limit.get(run_id, set()) & set(keys)

    def seed(self, config: RunConfig, text: str, output: str, hit_limit: bool = False) -> None:
        self.runs[config.run_id] = config
        key = content_key(text[: config.max_input_chars])
        self.outputs.setdefault(config.run_id, {})[key] = output
        if hit_limit:
            self.hit_limit.setdefault(config.run_id, set()).add(key)


class ScriptedBackend:
    model_id = "demo://fake"
    precision = "none"

    def __init__(
        self,
        responder: Callable[[str], str] = answer,
        fail_when: Callable[[int, Sequence[str]], bool] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.responder = responder
        self.fail_when = fail_when
        self.error = error or RuntimeError("boom")
        self.calls: list[list[str]] = []

    def generate(self, prompts: Sequence[str], max_new_tokens: int) -> list[str]:
        self.calls.append(list(prompts))
        if self.fail_when is not None and self.fail_when(len(self.calls), prompts):
            raise self.error
        return [self.responder(p) for p in prompts]

    def close(self) -> None:
        pass

    @property
    def prompts(self) -> list[str]:
        return [p for call in self.calls for p in call]


class FakeManager:
    def __init__(self, backend=None, load_error: Exception | None = None) -> None:
        self.backend = backend if backend is not None else ScriptedBackend()
        self.load_error = load_error
        self.acquired = 0
        self.loaded: tuple[str, str] | None = None

    @contextlib.contextmanager
    def acquire(self, model_id: str, precision: str):
        self.acquired += 1
        if self.load_error is not None:
            raise self.load_error
        self.loaded = (model_id, precision)
        yield self.backend


def run(texts, *, config=None, store=None, manager=None, **kwargs):
    config = config or make_config()
    manager = manager or FakeManager()
    result = execute_run(config, texts, store=store, manager=manager, **kwargs)
    return result, manager


def check_stats(result) -> None:
    s = result.stats
    assert s.total == len(result.outputs) == len(result.statuses) == len(result.errors)
    assert s.total == s.blank + s.cached + s.generated + s.errors + s.pending
    for status, count in (
        ("blank", s.blank),
        ("cached", s.cached),
        ("generated", s.generated),
        ("error", s.errors),
        ("pending", s.pending),
    ):
        assert result.statuses.count(status) == count, status


# --------------------------------------------------------------------------
# Ids and keys
# --------------------------------------------------------------------------


def test_run_id_is_stable_hex():
    a = make_config().run_id
    assert a == make_config().run_id
    assert len(a) == 16
    int(a, 16)
    assert a == compute_run_id("demo://fake", "none", TEMPLATE, 8, engine.DEFAULT_MAX_INPUT_CHARS)


@pytest.mark.parametrize(
    "change",
    [
        {"model_id": "other/model"},
        {"precision": "fp16"},
        {"template": "Other: {text}"},
        {"max_new_tokens": 9},
        {"max_input_chars": 100},
        {"model_revision": "0123456789ab"},
    ],
)
def test_run_id_changes_with_each_output_field(change):
    assert make_config(**change).run_id != make_config().run_id


def test_run_id_without_a_model_revision_is_unchanged():
    """Hub repos and the demo model keep the ids (and saved outputs) they
    had before the model revision existed."""
    import hashlib
    import json

    old = json.dumps(["v2", "demo://fake", "none", TEMPLATE, 8, engine.DEFAULT_MAX_INPUT_CHARS])
    assert make_config().run_id == hashlib.sha256(old.encode("utf-8")).hexdigest()[:16]
    assert make_config(model_revision="").run_id == make_config().run_id


def make_model_folder(folder, mtime_ns: int = 1_700_000_000_000_000_000):
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "model.safetensors", "tokenizer.json", "README.md"):
        (folder / name).write_text(f"{name} v1", encoding="utf-8")
        os.utime(folder / name, ns=(mtime_ns, mtime_ns))
    return folder


def test_local_model_folder_is_part_of_the_run_id(tmp_path):
    """Regression: a fine-tune saved again into the same folder was served
    the old model's cached outputs, because the run id only had the path."""
    from batchtextreview.models import custom_model

    folder = make_model_folder(tmp_path / "my-classifier")
    spec = custom_model(str(folder))
    first = RunConfig.build(spec, TEMPLATE, 8, precision="fp32")
    assert len(first.model_revision) == 12
    assert RunConfig.build(spec, TEMPLATE, 8, precision="fp32") == first

    store = FakeStore()
    run(["a", "b"], config=first, store=store)
    assert plan_run(first, ["a", "b"], store).fully_cached

    # Retrained: same files, new weights (same size, newer time).
    weights = folder / "model.safetensors"
    weights.write_text("model.safetensors v2", encoding="utf-8")
    os.utime(weights, ns=(1_800_000_000_000_000_000,) * 2)
    second = RunConfig.build(spec, TEMPLATE, 8, precision="fp32")
    assert second.model_revision != first.model_revision
    assert second.run_id != first.run_id
    assert plan_run(second, ["a", "b"], store).n_todo_unique == 2


def test_model_revision_is_empty_for_hub_and_demo_models():
    from batchtextreview.models import DEMO_MODEL, find_model

    assert RunConfig.build(DEMO_MODEL, TEMPLATE, 8).model_revision == ""
    hub = find_model("Qwen/Qwen2.5-3B-Instruct")
    assert hub is not None
    assert RunConfig.build(hub, TEMPLATE, 8, precision="fp32").model_revision == ""


def test_changed_model_folder_is_announced_as_a_load(tmp_path):
    """The engine asks the manager whether the model (unchanged on disk)
    is loaded, not just whether its id is."""

    class Manager(FakeManager):
        def is_loaded(self, model_id: str, precision: str) -> bool:
            return False

    manager = Manager()
    manager.loaded = ("demo://fake", "none")
    events: list[Progress] = []
    run(["a"], manager=manager, progress=events.append)
    assert next(e for e in events if e.phase == "loading").message.startswith("Loading")


def test_run_id_ignores_display_label():
    assert make_config(model_label="Renamed").run_id == make_config().run_id


def test_run_id_ignores_batch_size():
    store = FakeStore()
    texts = ["one", "two", "three"]
    first, _ = run(texts, store=store, batch_size=1)
    second, manager = run(texts, store=store, batch_size=50, max_batch_chars=5)
    assert first.run_id == second.run_id
    assert second.statuses == ["cached"] * 3
    assert manager.acquired == 0


def test_content_key_is_sha256_of_snippet():
    import hashlib

    assert content_key("héllo") == hashlib.sha256("héllo".encode()).hexdigest()


@pytest.mark.parametrize(
    ("value", "blank"),
    [
        (None, True),
        (float("nan"), True),
        (pd.NA, True),
        (pd.NaT, True),
        ("", True),
        ("  \n\t", True),
        ("x", False),
        (0, False),
        (0.0, False),
        ("nan", False),
    ],
)
def test_is_blank(value, blank):
    assert is_blank(value) is blank


def test_build_uses_none_precision_for_demo_and_keeps_explicit_precision():
    from batchtextreview.models import DEMO_MODEL, ModelSpec

    demo = RunConfig.build(DEMO_MODEL, TEMPLATE, 16, precision="4bit")
    assert demo.precision == "none"
    assert demo.model_id == DEMO_MODEL.repo
    real = RunConfig.build(ModelSpec("M", "org/m"), TEMPLATE, 16.0, precision="fp16")  # type: ignore[arg-type]
    assert real.precision == "fp16"
    assert real.max_new_tokens == 16


# --------------------------------------------------------------------------
# plan_run
# --------------------------------------------------------------------------


def test_plan_counts_blank_duplicate_and_unique_rows():
    texts = ["a", None, "b", "a", "  ", float("nan"), "a", 5]
    plan = plan_run(make_config(), texts, None)
    assert plan.n_rows == 8
    assert plan.n_blank == 3
    assert plan.n_unique == 3  # "a", "b", "5"
    assert plan.n_duplicates == 2
    assert plan.n_cached_rows == plan.n_cached_unique == 0
    assert plan.n_todo_rows == 5
    assert plan.n_todo_unique == 3
    assert plan.keys[1] is None and plan.keys[4] is None and plan.keys[5] is None
    assert plan.keys[0] == plan.keys[3] == plan.keys[6] == content_key("a")
    assert plan.keys[7] == content_key("5")
    assert plan.snippets[content_key("5")] == "5"
    assert not plan.fully_cached


def test_plan_uses_store_for_cached_counts():
    config = make_config()
    store = FakeStore()
    store.seed(config, "a", "cached-a")
    plan = plan_run(config, ["a", "b", "a", ""], store)
    assert plan.n_cached_rows == 2
    assert plan.n_cached_unique == 1
    assert plan.n_todo_rows == 1
    assert plan.n_todo_unique == 1
    assert plan.cached == {content_key("a"): "cached-a"}


def test_plan_ignores_outputs_of_other_runs():
    config = make_config()
    store = FakeStore()
    store.seed(make_config(max_new_tokens=99), "a", "other")
    plan = plan_run(config, ["a"], store)
    assert plan.cached == {}
    assert plan.n_todo_rows == 1


def test_plan_with_all_rows_cached_or_blank_is_fully_cached():
    config = make_config()
    store = FakeStore()
    store.seed(config, "a", "x")
    assert plan_run(config, ["a", None, "a"], store).fully_cached
    assert plan_run(config, [], store).fully_cached


def test_truncation_affects_keys_and_prompts():
    config = make_config(max_input_chars=5)
    plan = plan_run(config, ["abcdefg", "abcdeXYZ", "abc"], None)
    assert plan.keys[0] == plan.keys[1] == content_key("abcde")
    assert plan.n_unique == 2
    assert plan.n_duplicates == 1

    result, manager = run(["abcdefg", "abcdeXYZ"], config=config)
    assert manager.backend.prompts == ["Q: abcde"]
    assert result.outputs == ["A:ABCDE", "A:ABCDE"]

    longer = plan_run(make_config(max_input_chars=6), ["abcdefg", "abcdeXYZ"], None)
    assert longer.keys[0] != longer.keys[1]
    assert longer.run_id != plan.run_id


# --------------------------------------------------------------------------
# pack_batches
# --------------------------------------------------------------------------


def test_pack_batches_row_cap():
    keys = list("abcde")
    lengths = dict.fromkeys(keys, 1)
    assert pack_batches(keys, lengths, 2, 1000) == [["a", "b"], ["c", "d"], ["e"]]


def test_pack_batches_char_cap():
    keys = ["a", "b", "c", "d"]
    lengths = {"a": 3, "b": 3, "c": 4, "d": 10}
    # 3+3 = 6 fits a budget of 7; adding 4 would not; 4 then 10 exceeds too.
    assert pack_batches(keys, lengths, 10, 7) == [["a", "b"], ["c"], ["d"]]


def test_pack_batches_exact_budget_fits():
    lengths = {"a": 5, "b": 5}
    assert pack_batches(["a", "b"], lengths, 10, 10) == [["a", "b"]]


def test_pack_batches_oversize_single_row_gets_own_batch():
    keys = ["s", "huge", "t"]
    lengths = {"s": 1, "huge": 500, "t": 1}
    assert pack_batches(keys, lengths, 10, 100) == [["s"], ["huge"], ["t"]]


@pytest.mark.parametrize("size", [0, -3])
def test_pack_batches_size_below_one_is_one(size):
    keys = ["a", "b"]
    assert pack_batches(keys, {"a": 1, "b": 1}, size, 100) == [["a"], ["b"]]


def test_pack_batches_empty():
    assert pack_batches([], {}, 8, 100) == []


# --------------------------------------------------------------------------
# execute_run: basics
# --------------------------------------------------------------------------


def test_generates_each_row_in_order():
    result, manager = run(["alpha", "beta"])
    assert result.outputs == ["A:ALPHA", "A:BETA"]
    assert result.statuses == ["generated", "generated"]
    assert result.errors == [None, None]
    assert result.complete
    assert result.config == make_config()
    assert result.run_id == make_config().run_id
    assert manager.acquired == 1
    check_stats(result)


def test_blank_rows_get_empty_output_without_model_call():
    texts = [None, "", "  ", float("nan"), pd.NA]
    store = FakeStore()
    result, manager = run(texts, store=store)
    assert result.statuses == ["blank"] * 5
    assert result.outputs == [""] * 5
    assert manager.acquired == 0
    assert store.runs == {}  # nothing to do -> no manifest
    assert result.complete
    check_stats(result)


def test_blank_rows_mixed_with_text():
    result, manager = run(["x", None, "y"])
    assert result.statuses == ["generated", "blank", "generated"]
    assert result.outputs == ["A:X", "", "A:Y"]
    assert manager.backend.prompts == ["Q: x", "Q: y"]


def test_duplicate_rows_generated_once_and_shared():
    store = FakeStore()
    result, manager = run(["a", "b", "a", "a"], store=store)
    assert sorted(manager.backend.prompts) == ["Q: a", "Q: b"]
    assert result.outputs == ["A:A", "A:B", "A:A", "A:A"]
    assert result.statuses == ["generated"] * 4
    assert result.stats.duplicates == 2
    assert result.stats.generated == 4
    assert result.stats.model_calls == 2
    assert len(store.outputs[result.run_id]) == 2
    check_stats(result)


def test_duplicates_count_only_rows_this_run_generated():
    """Regression: the summary said "98 duplicate rows were computed once"
    for runs that computed nothing (cancelled first, or fully cached)."""
    texts = ["great"] * 50 + ["awful"] * 50
    cancel = threading.Event()
    cancel.set()
    store = FakeStore()
    cancelled, _ = run(texts, store=store, cancel=cancel)
    assert cancelled.stats.pending == 100 and cancelled.stats.duplicates == 0

    first, _ = run(texts, store=store)
    assert first.stats.duplicates == 98

    again, _ = run(texts, store=store)
    assert again.stats.cached == 100 and again.stats.duplicates == 0

    # Repeats of a text that failed aren't "computed once" either.
    backend = ScriptedBackend(fail_when=lambda n, p: "Q: bad" in p)
    mixed, _ = run(["ok", "bad", "ok", "bad", "bad"], manager=FakeManager(backend), batch_size=1)
    assert mixed.stats.errors == 3 and mixed.stats.duplicates == 1


def test_cached_rows_are_not_regenerated():
    config = make_config()
    store = FakeStore()
    store.seed(config, "a", "stored-a")
    result, manager = run(["a", "b", "a"], store=store)
    assert manager.backend.prompts == ["Q: b"]
    assert result.statuses == ["cached", "generated", "cached"]
    assert result.outputs == ["stored-a", "A:B", "stored-a"]
    assert result.stats.cached == 2
    assert result.stats.generated == 1
    check_stats(result)


def test_fully_cached_run_never_loads_the_model():
    store = FakeStore()
    first, _ = run(["a", "b", "a", None], store=store)
    puts = len(store.put_calls)
    second, manager = run(["a", "b", "a", None], store=store)
    assert manager.acquired == 0
    assert manager.backend.calls == []
    assert second.outputs == first.outputs
    assert second.statuses == ["cached", "cached", "cached", "blank"]
    assert second.stats.model_calls == 0
    assert second.stats.load_s == 0.0
    assert len(store.put_calls) == puts
    assert second.complete


def test_cached_outputs_are_cleaned_on_read():
    # Version-1 checkpoints were stored raw, think blocks included.
    config = make_config()
    store = FakeStore()
    store.seed(config, "a", "<think>hmm</think> positive ")
    result, _ = run(["a"], store=store)
    assert result.outputs == ["positive"]


def test_store_none_works():
    result, _ = run(["a", "b", "a"], store=None)
    assert result.outputs == ["A:A", "A:B", "A:A"]
    assert result.complete
    _, manager2 = run(["a"], store=None)
    assert manager2.acquired == 1  # nothing cached without a store


def test_empty_input():
    events: list[Progress] = []
    result, manager = run([], progress=events.append)
    assert result.outputs == [] and result.statuses == [] and result.errors == []
    assert result.complete
    assert manager.acquired == 0
    assert [e.phase for e in events] == ["planning", "finished"]
    assert events[-1].fraction == 1.0


def test_accepts_a_pandas_series():
    texts = pd.Series(["a", None, "b"], index=[10, 20, 30])
    result, _ = run(texts)
    assert result.statuses == ["generated", "blank", "generated"]


def test_template_without_text_placeholder_is_rejected():
    with pytest.raises(ValueError, match="text"):
        run(["a"], config=make_config(template="no placeholder"))


def test_row_text_with_placeholders_is_inserted_verbatim():
    result, manager = run(["{text} and {var}"])
    assert manager.backend.prompts == ["Q: {text} and {var}"]
    assert result.statuses == ["generated"]


# --------------------------------------------------------------------------
# Batching inside execute_run
# --------------------------------------------------------------------------


def test_batches_sorted_by_length_with_stable_ties():
    texts = ["cccc", "a", "bb", "x", "dddddd", "y"]
    _, manager = run(texts, batch_size=2)
    assert manager.backend.calls == [
        ["Q: a", "Q: x"],
        ["Q: y", "Q: bb"],
        ["Q: cccc", "Q: dddddd"],
    ]


def test_both_caps_apply_in_execute_run():
    texts = ["aa", "bb", "cc", "dd", "eeeeeeeeee"]
    _, manager = run(texts, batch_size=3, max_batch_chars=5)
    sizes = [len(call) for call in manager.backend.calls]
    # 2+2 = 4 fits 5 chars, a third row would not; the 10-char row is alone.
    assert sizes == [2, 2, 1]
    assert manager.backend.calls[-1] == ["Q: eeeeeeeeee"]


def test_oversize_row_still_runs():
    long_text = "z" * 50
    result, manager = run(["a", long_text], max_batch_chars=10)
    assert manager.backend.calls == [["Q: a"], [f"Q: {long_text}"]]
    assert result.statuses == ["generated", "generated"]


def test_outputs_saved_after_every_batch():
    store = FakeStore()
    result, _ = run(["a", "b", "c"], store=store, batch_size=1)
    assert len(store.put_calls) == 3
    assert all(len(call) == 1 for call in store.put_calls)
    assert result.run_id in store.runs
    assert store.upserts == 1


def test_no_manifest_without_saved_outputs():
    store = FakeStore()
    run(["a"], store=store, manager=FakeManager(load_error=OSError("no")))
    cancel = threading.Event()
    cancel.set()
    run(["a"], store=store, cancel=cancel)
    backend = ScriptedBackend(fail_when=lambda n, p: True)
    run(["a"], store=store, manager=FakeManager(backend))
    assert store.runs == {}
    assert store.upserts == 0


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------


def test_outputs_cleaned_before_storing():
    store = FakeStore()
    backend = ScriptedBackend(responder=lambda p: "<think>let me see</think>\n  positive \n")
    result, _ = run(["a"], store=store, manager=FakeManager(backend))
    assert result.outputs == ["positive"]
    assert list(store.outputs[result.run_id].values()) == ["positive"]


def test_cleaning_goes_through_labels_clean_output(monkeypatch):
    monkeypatch.setattr(labels, "clean_output", lambda raw: f"[{raw}]")
    store = FakeStore()
    result, _ = run(["a"], store=store)
    assert result.outputs == ["[A:A]"]
    assert list(store.outputs[result.run_id].values()) == ["[A:A]"]


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


def test_error_batch_marks_rows_and_run_continues():
    store = FakeStore()
    backend = ScriptedBackend(fail_when=lambda n, prompts: "Q: bb" in prompts)
    texts = ["a", "bb", "ccc", "bb"]
    result, _ = run(texts, store=store, manager=FakeManager(backend), batch_size=1)
    assert result.statuses == ["generated", "error", "generated", "error"]
    assert result.outputs == ["A:A", "", "A:CCC", ""]
    assert result.errors[1] == result.errors[3] == "RuntimeError: boom"
    assert result.errors[0] is None and result.errors[2] is None
    assert result.aborted_reason is None
    assert result.complete  # every row was attempted
    assert content_key("bb") not in store.outputs[result.run_id]
    assert result.stats.errors == 2
    assert result.stats.model_calls == 3
    check_stats(result)

    # A re-run retries exactly the failed rows.
    retry, manager = run(texts, store=store, batch_size=1)
    assert manager.backend.prompts == ["Q: bb"]
    assert retry.statuses == ["cached", "generated", "cached", "generated"]


def test_error_message_is_short_single_line():
    backend = ScriptedBackend(
        fail_when=lambda n, p: True, error=ValueError("line one\nline two " + "x" * 600)
    )
    result, _ = run(["a"], manager=FakeManager(backend), max_consecutive_failures=0)
    message = result.errors[0]
    assert message.startswith("ValueError: line one line two")
    assert "\n" not in message
    assert len(message) <= 300


def test_wrong_number_of_outputs_is_a_batch_error():
    class Short(ScriptedBackend):
        def generate(self, prompts, max_new_tokens):
            self.calls.append(list(prompts))
            return ["only one"]

    result, _ = run(["a", "b"], manager=FakeManager(Short()), max_consecutive_failures=0)
    assert result.statuses == ["error", "error"]
    assert "2 prompts" in result.errors[0]


def test_consecutive_failures_abort_leaves_rest_pending():
    store = FakeStore()
    backend = ScriptedBackend(fail_when=lambda n, p: True)
    texts = ["a", "b", "c", "d", "e"]
    result, _ = run(
        texts, store=store, manager=FakeManager(backend), batch_size=1, max_consecutive_failures=2
    )
    assert len(backend.calls) == 2
    assert result.statuses == ["error", "error", "pending", "pending", "pending"]
    assert result.outputs == [""] * 5
    assert result.aborted_reason is not None
    assert "RuntimeError: boom" in result.aborted_reason
    assert not result.complete
    assert result.stats.pending == 3
    assert store.outputs == {}
    check_stats(result)


def test_zero_disables_the_failure_limit():
    backend = ScriptedBackend(fail_when=lambda n, p: True)
    result, _ = run(
        list("abcde"), manager=FakeManager(backend), batch_size=1, max_consecutive_failures=0
    )
    assert len(backend.calls) == 5
    assert result.statuses == ["error"] * 5
    assert result.aborted_reason is None


def test_failure_counter_resets_after_a_success():
    backend = ScriptedBackend(fail_when=lambda n, p: n % 2 == 1)  # fail, ok, fail, ok...
    result, _ = run(
        list("abcdef"), manager=FakeManager(backend), batch_size=1, max_consecutive_failures=2
    )
    assert result.aborted_reason is None
    assert result.statuses.count("error") == 3
    assert result.statuses.count("generated") == 3


def test_model_load_failure_aborts_cleanly():
    config = make_config()
    store = FakeStore()
    store.seed(config, "cached", "C")
    error = ModelLoadError("Could not download org/m.", hint="Check HF_TOKEN.")
    manager = FakeManager(load_error=error)
    events: list[Progress] = []
    result, _ = run(
        ["new", None, "cached", "other", "new"],
        store=store,
        manager=manager,
        progress=events.append,
    )
    assert result.statuses == ["pending", "blank", "cached", "pending", "pending"]
    assert result.outputs == ["", "", "C", "", ""]
    assert result.aborted_reason is not None
    assert "Could not download org/m." in result.aborted_reason
    assert "Check HF_TOKEN." in result.aborted_reason
    assert "Fake model" in result.aborted_reason
    assert not result.complete
    assert result.stats.pending == 3
    assert result.stats.model_calls == 0
    assert [e.phase for e in events] == ["planning", "loading", "finished"]
    check_stats(result)


def test_unexpected_load_exception_names_its_type():
    manager = FakeManager(load_error=OSError("disk on fire"))
    result, _ = run(["a"], manager=manager)
    assert "OSError: disk on fire" in result.aborted_reason
    assert result.statuses == ["pending"]


# --------------------------------------------------------------------------
# Cancellation
# --------------------------------------------------------------------------


def test_cancel_between_batches():
    store = FakeStore()
    cancel = threading.Event()

    def on_progress(p: Progress) -> None:
        if p.phase == "generating" and p.generated_rows:
            cancel.set()

    result, manager = run(
        ["a", "b", "c"],
        store=store,
        batch_size=1,
        cancel=cancel,
        progress=on_progress,
    )
    assert len(manager.backend.calls) == 1
    assert result.cancelled
    assert not result.complete
    assert result.statuses == ["generated", "pending", "pending"]
    assert result.outputs == ["A:A", "", ""]
    assert list(store.outputs[result.run_id].values()) == ["A:A"]
    check_stats(result)


def test_cancel_before_start_never_loads_model():
    cancel = threading.Event()
    cancel.set()
    result, manager = run(["a", None], cancel=cancel)
    assert manager.acquired == 0
    assert result.cancelled
    assert result.statuses == ["pending", "blank"]


def test_cancel_ignored_when_nothing_left_to_do():
    store = FakeStore()
    run(["a"], store=store)
    cancel = threading.Event()
    cancel.set()
    result, _ = run(["a"], store=store, cancel=cancel)
    assert not result.cancelled
    assert result.complete


# --------------------------------------------------------------------------
# Progress
# --------------------------------------------------------------------------


def test_progress_phase_sequence_and_monotonic_rows():
    events: list[Progress] = []
    result, _ = run(["a", None, "bb", "a", "ccc"], batch_size=1, progress=events.append)
    phases = [e.phase for e in events]
    assert phases == [
        "planning",
        "loading",
        "generating",  # the model is ready, batch 1 starts
        "generating",
        "generating",
        "generating",
        "finished",
    ]
    assert events[2].message == "Starting batch 1 of 3..."
    assert events[2].generated_rows == 0
    done = [e.done_rows for e in events]
    assert done == sorted(done)
    assert events[0].done_rows == 1  # the blank row
    assert events[0].blank_rows == 1
    assert all(e.total_rows == 5 for e in events)
    final = events[-1]
    assert final.fraction == 1.0
    assert final.done_rows == 5
    assert final.generated_rows == 4
    assert result.complete


def test_rate_and_eta_after_first_batch():
    events: list[Progress] = []
    run(["a", "bb", "ccc"], batch_size=1, progress=events.append)
    planning = events[0]
    assert planning.rows_per_s is None and planning.eta_s is None
    start = next(e for e in events if e.phase == "generating")
    assert start.rows_per_s is None and start.eta_s is None  # nothing timed yet
    first = next(e for e in events if e.phase == "generating" and e.generated_rows)
    assert first.rows_per_s is not None and first.rows_per_s > 0
    assert first.eta_s is not None and first.eta_s >= 0
    last_generating = [e for e in events if e.phase == "generating"][-1]
    assert last_generating.eta_s == 0
    assert all(e.elapsed_s >= 0 for e in events)
    elapsed = [e.elapsed_s for e in events]
    assert elapsed == sorted(elapsed)


class FakeClock:
    """Stands in for the engine's ``time`` module; the backend advances it."""

    def __init__(self) -> None:
        self.now = 0.0

    def perf_counter(self) -> float:
        return self.now


class TimedBackend(ScriptedBackend):
    """Each generate call costs ``per_call + per_char * prompt chars``."""

    def __init__(self, clock: FakeClock, per_call: float, per_char: float = 0.0) -> None:
        super().__init__()
        self.clock, self.per_call, self.per_char = clock, per_call, per_char

    def generate(self, prompts, max_new_tokens):
        self.clock.now += self.per_call + self.per_char * sum(len(p) for p in prompts)
        return super().generate(prompts, max_new_tokens)


def timed_run(monkeypatch, texts, backend, clock, **kwargs):
    monkeypatch.setattr(engine, "time", clock)
    seen: list[tuple[float, Progress]] = []
    result, _ = run(
        texts,
        manager=FakeManager(backend),
        progress=lambda p: seen.append((clock.now, p)),
        **kwargs,
    )
    # After each batch (the first "generating" event is batch 1 starting).
    start, *generating = [(t, p) for t, p in seen if p.phase == "generating"]
    assert start[1].generated_rows == 0 and start[1].eta_s is None
    return result, clock.now, generating


def test_eta_is_not_fooled_by_duplicate_rows(monkeypatch):
    """Regression: the ETA divided the rows left by a rate that counted
    duplicate rows, so "~0 s left" showed for most of a survey run."""
    clock = FakeClock()
    texts = ["Good"] * 1000 + [f"x{i:03d}" for i in range(79)]  # 80 texts of 4 chars
    _, end, generating = timed_run(
        monkeypatch, texts, TimedBackend(clock, per_call=1.0), clock, batch_size=8
    )
    assert len(generating) == 10
    first_t, first = generating[0]
    assert first.done_rows == 1007
    assert first.eta_s == pytest.approx(end - first_t) == pytest.approx(9.0)
    for t, progress in generating:
        assert progress.eta_s == pytest.approx(end - t)


def test_eta_prices_the_longer_batches_still_to_come(monkeypatch):
    """Regression: batches run shortest first, so a rate from the first
    batches promised far too little time for the long ones left."""
    clock = FakeClock()
    texts = ["w" * (5 * i + 1) for i in range(200)]
    backend = TimedBackend(clock, per_call=0.6, per_char=0.002)
    _, end, generating = timed_run(
        monkeypatch, texts, backend, clock, batch_size=8, max_batch_chars=2000
    )
    # From the second batch on (two lengths seen), the cost model is exact.
    for t, progress in generating[1:]:
        assert progress.eta_s == pytest.approx(end - t, rel=1e-6, abs=1e-9)
    # The old estimate (rows left / rows per second) was over 3x too low here.
    t, tenth = next((t, p) for t, p in generating if p.fraction >= 0.1)
    assert (200 - tenth.done_rows) / tenth.rows_per_s < (end - t) / 3


def test_eta_estimate_stays_between_per_call_and_per_char_pricing():
    eta = engine._Eta([100, 100, 400, 800])
    assert eta.seconds_left() is None
    eta.start_batch(100)
    eta.observe(100, 1.0)
    # One length seen: a + b can't be told apart -> the middle of the range
    # (3 calls at 1 s, or 1,300 chars at 0.01 s).
    assert eta.seconds_left() == pytest.approx(math.sqrt(3.0 * 13.0))
    eta.start_batch(100)
    eta.observe(110, 5.0)  # noisy: the slope alone would predict ~500 s
    per_call, per_char = 3.0 * 2, 6.0 / 210 * 1200
    assert per_call <= eta.seconds_left() <= per_char
    eta.start_batch(400)
    eta.start_batch(800)
    assert eta.seconds_left() == 0.0


def test_no_rate_while_every_batch_fails():
    events: list[Progress] = []
    backend = ScriptedBackend(fail_when=lambda n, p: True)
    run(
        ["a", "b"],
        manager=FakeManager(backend),
        batch_size=1,
        max_consecutive_failures=0,
        progress=events.append,
    )
    generating = [e for e in events if e.phase == "generating"]
    assert [e.error_rows for e in generating] == [0, 1, 2]
    assert all(e.rows_per_s is None for e in generating)


def test_fully_cached_progress_is_planning_then_finished():
    store = FakeStore()
    run(["a"], store=store)
    events: list[Progress] = []
    run(["a", None], store=store, progress=events.append)
    assert [e.phase for e in events] == ["planning", "finished"]
    assert events[0].cached_rows == 1
    assert events[0].fraction == 1.0


def test_fraction_below_one_when_cancelled():
    cancel = threading.Event()
    events: list[Progress] = []

    def on_progress(p: Progress) -> None:
        events.append(p)
        if p.phase == "generating" and p.generated_rows:
            cancel.set()

    run(["a", "b"], batch_size=1, cancel=cancel, progress=on_progress)
    assert events[-1].phase == "finished"
    assert events[-1].fraction == 0.5


class StopRun(Exception):
    """What a UI might raise from its progress callback to stop a run."""


def test_progress_exception_propagates_after_batch_is_saved():
    store = FakeStore()

    def explode(p: Progress) -> None:
        if p.phase == "generating" and p.generated_rows:
            raise StopRun

    with pytest.raises(StopRun):
        run(["a", "b"], store=store, batch_size=1, progress=explode)
    assert len(store.outputs[make_config().run_id]) == 1


class LockedStore(FakeStore):
    """put_outputs fails from call ``fail_from`` on, like a locked database."""

    def __init__(
        self, fail_from: int = 2, upsert_fails: bool = False, path: str | None = "/data/btr.db"
    ) -> None:
        super().__init__()
        self.path = path
        self.fail_from = fail_from
        self.upsert_fails = upsert_fails
        self.attempts = 0

    def upsert_run(self, config: RunConfig, source: str = "app") -> None:
        if self.upsert_fails:
            raise sqlite3.OperationalError("disk I/O error")
        super().upsert_run(config, source)

    def put_outputs(self, run_id: str, items) -> None:
        self.attempts += 1
        if self.attempts >= self.fail_from:
            raise sqlite3.OperationalError("database is locked")
        super().put_outputs(run_id, items)


def test_store_failure_stops_run_but_keeps_finished_rows():
    store = LockedStore(fail_from=2)
    events: list[Progress] = []
    result, manager = run(["a", "b", "c", "a"], store=store, batch_size=1, progress=events.append)
    assert result.outputs == ["A:A", "A:B", "", "A:A"]
    assert result.statuses == ["generated", "generated", "pending", "generated"]
    assert len(manager.backend.calls) == 2  # stopped right after the failed save
    assert not result.complete and not result.cancelled
    reason = result.aborted_reason
    assert reason is not None
    assert reason.startswith("Couldn't save outputs to /data/btr.db: ")
    assert "OperationalError: database is locked" in reason
    assert "run again" in reason
    assert list(store.outputs[result.run_id]) == [content_key("a")]
    assert events[-1].phase == "finished"
    check_stats(result)


def test_store_failure_registering_the_run_is_an_abort():
    store = LockedStore(upsert_fails=True)
    result, _ = run(["a", "b"], store=store, batch_size=1)
    assert result.statuses == ["generated", "pending"]
    assert result.aborted_reason is not None
    assert "disk I/O error" in result.aborted_reason
    assert store.runs == {} and store.outputs == {}


def test_store_without_path_is_named_generically():
    result, _ = run(["a"], store=LockedStore(fail_from=1, path=None))
    assert result.aborted_reason is not None
    assert result.aborted_reason.startswith("Couldn't save outputs to the results database: ")


def test_run_deleted_mid_run_is_registered_again():
    store = FakeStore()
    run_id = make_config().run_id

    def delete_after_first_batch(p: Progress) -> None:
        if p.phase == "generating" and p.generated_rows == 1:
            store.runs.pop(run_id)
            store.outputs.pop(run_id)

    result, _ = run(["a", "b"], store=store, batch_size=1, progress=delete_after_first_batch)
    assert result.complete
    assert result.statuses == ["generated", "generated"]
    assert store.upserts == 2
    assert list(store.outputs[run_id]) == [content_key("b")]


def test_source_is_recorded_with_the_manifest():
    store = FakeStore()
    run(["a"], store=store)
    run(["b"], store=store, source="cli")
    assert store.sources == ["app", "cli"]


def test_load_time_zero_when_model_already_loaded():
    manager = FakeManager()
    manager.loaded = ("demo://fake", "none")
    result, _ = run(["a"], manager=manager)
    assert result.stats.load_s == 0.0
    assert result.stats.elapsed_s > 0


@pytest.mark.parametrize("preloaded", [True, False])
def test_first_batch_runs_under_generating_not_loading(preloaded):
    """Regression: nothing was emitted between acquiring the model and the
    end of batch 1, so a long first batch (minutes on a CPU) showed as
    "Loading <model>" - even when the model was already in memory."""
    events: list[Progress] = []
    seen_during_batches: list[tuple[str, str]] = []

    class Watching(ScriptedBackend):
        def generate(self, prompts, max_new_tokens):
            seen_during_batches.append((events[-1].phase, events[-1].message))
            return super().generate(prompts, max_new_tokens)

    manager = FakeManager(Watching())
    if preloaded:
        manager.loaded = ("demo://fake", "none")
    run(["a", "bb"], manager=manager, batch_size=1, progress=events.append)
    loading = next(e for e in events if e.phase == "loading")
    assert loading.message.startswith("Preparing" if preloaded else "Loading")
    assert seen_during_batches[0] == ("generating", "Starting batch 1 of 2...")
    assert seen_during_batches[1] == ("generating", "Batch 1 of 2")


# --------------------------------------------------------------------------
# Stats
# --------------------------------------------------------------------------


def test_stats_arithmetic_with_every_status():
    config = make_config()
    store = FakeStore()
    store.seed(config, "c", "cached")
    # Batches (batch_size 1, length order): "c" is cached; then "dd" ok,
    # "ee" fails, "ff" fails -> abort; "gggg" never runs.
    backend = ScriptedBackend(fail_when=lambda n, p: n >= 2)
    texts = [None, "c", "dd", "ee", "ff", "gggg", "dd", "c", ""]
    result, _ = run(
        texts,
        store=store,
        manager=FakeManager(backend),
        batch_size=1,
        max_consecutive_failures=2,
    )
    s = result.stats
    assert (s.total, s.blank, s.cached, s.generated, s.errors, s.pending) == (9, 2, 2, 2, 2, 1)
    # Only "dd" shared a model call; the repeated "c" came from the cache.
    assert s.duplicates == 1
    assert s.model_calls == 3
    assert result.aborted_reason is not None
    check_stats(result)


# --------------------------------------------------------------------------
# Plan reuse
# --------------------------------------------------------------------------


def test_plan_reused_when_run_id_matches(monkeypatch):
    config = make_config()
    texts = ["a", "b"]
    plan = plan_run(config, texts, None)

    def fail(*args, **kwargs):
        raise AssertionError("plan_run should not be called")

    monkeypatch.setattr(engine, "plan_run", fail)
    result = execute_run(config, texts, store=None, manager=FakeManager(), plan=plan)
    assert result.outputs == ["A:A", "A:B"]


def test_plan_recomputed_when_run_id_differs(monkeypatch):
    texts = ["a", "b"]
    stale = plan_run(make_config(max_new_tokens=99), texts, None)
    calls = []
    real = engine.plan_run

    def spy(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(engine, "plan_run", spy)
    result = execute_run(make_config(), texts, store=None, manager=FakeManager(), plan=stale)
    assert len(calls) == 1
    assert result.run_id == make_config().run_id


def test_plan_recomputed_when_row_count_differs():
    config = make_config()
    stale = plan_run(config, ["a"], None)
    result = execute_run(config, ["a", "b"], store=None, manager=FakeManager(), plan=stale)
    assert result.outputs == ["A:A", "A:B"]


def test_cached_values_from_given_plan_are_used():
    config = make_config()
    store = FakeStore()
    store.seed(config, "a", "from-plan")
    plan = plan_run(config, ["a"], store)
    store.outputs.clear()  # gone since planning: the plan's copy still counts
    result, manager = run(["a"], config=config, store=store, plan=plan)
    assert result.outputs == ["from-plan"]
    assert manager.acquired == 0


# --------------------------------------------------------------------------
# Real Store / ModelManager / FakeBackend (skipped while those are stubs)
# --------------------------------------------------------------------------


def _real_store(tmp_path):
    from batchtextreview.store import Store

    try:
        return Store(tmp_path / "btr.db")
    except NotImplementedError:
        pytest.skip("Store not implemented yet")


def _real_manager(responder=None):
    from batchtextreview.backends.fake import FakeBackend
    from batchtextreview.backends.manager import ModelManager

    try:
        backend = FakeBackend(responder=responder or answer)
        backend.generate(["Q: probe"], 4)
        backend.calls.clear()
        manager = ModelManager(factory=lambda model_id, precision: backend)
        with manager.acquire("demo://fake", "none"):
            pass
    except NotImplementedError:
        pytest.skip("ModelManager / FakeBackend not implemented yet")
    return manager, backend


def test_real_store_round_trip(tmp_path):
    store = _real_store(tmp_path)
    texts = ["a", "b", None, "a"]
    first, _ = run(texts, store=store)
    assert first.statuses == ["generated", "generated", "blank", "generated"]
    assert store.count_outputs(first.run_id) == 2
    second, manager = run(texts, store=store)
    assert manager.acquired == 0
    assert second.outputs == first.outputs
    assert second.statuses == ["cached", "cached", "blank", "cached"]


def test_real_store_failed_rows_not_saved(tmp_path):
    store = _real_store(tmp_path)
    backend = ScriptedBackend(fail_when=lambda n, prompts: "Q: b" in prompts)
    result, _ = run(["a", "b"], store=store, manager=FakeManager(backend), batch_size=1)
    assert result.statuses == ["generated", "error"]
    assert set(store.get_outputs(result.run_id)) == {content_key("a")}


def test_real_manager_and_fake_backend(tmp_path):
    store = _real_store(tmp_path)
    manager, backend = _real_manager()
    events: list[Progress] = []
    result = execute_run(
        make_config(),
        ["x", "yy", "x"],
        store=store,
        manager=manager,
        batch_size=1,
        progress=events.append,
    )
    assert result.outputs == ["A:X", "A:YY", "A:X"]
    assert result.complete
    assert len(backend.calls) == 2
    assert math.isclose(events[-1].fraction, 1.0)


def test_real_store_run_deleted_mid_run(tmp_path):
    store = _real_store(tmp_path)
    run_id = make_config().run_id

    def delete_after_first_batch(p: Progress) -> None:
        if p.phase == "generating" and p.generated_rows == 1:
            store.delete_run(run_id)

    result, _ = run(["a", "b"], store=store, batch_size=1, progress=delete_after_first_batch)
    assert result.complete
    assert set(store.get_outputs(run_id)) == {content_key("b")}


def test_real_store_records_cli_source(tmp_path):
    store = _real_store(tmp_path)
    run(["a"], store=store, source="cli")
    run(["a", "b"], store=store)  # a later app run never rewrites the source
    assert store.list_runs()["source"].tolist() == ["cli"]


# --------------------------------------------------------------------------
# Live outputs (Progress.recent)
# --------------------------------------------------------------------------


def test_progress_recent_holds_the_latest_pairs_newest_last():
    events: list[Progress] = []
    texts = ["a", "bb", "ccc", "dddd", "eeeee", "ffffff", "ggggggg"]
    run(texts, batch_size=2, progress=events.append)
    assert events[0].phase == "planning" and events[0].recent == ()
    first = next(e for e in events if e.phase == "generating" and e.generated_rows)
    assert first.recent == (("a", "A:A"), ("bb", "A:BB"))
    final = events[-1]
    assert final.phase == "finished"
    assert len(final.recent) == engine.RECENT_LIMIT == 5
    assert final.recent[-1] == ("ggggggg", "A:GGGGGGG")
    assert [text for text, _ in final.recent] == texts[2:]


def test_progress_recent_clips_and_collapses_text():
    long_text = "word\n\n  " * 60
    backend = ScriptedBackend(responder=lambda p: "x" * 500)
    events: list[Progress] = []
    run([long_text, long_text], manager=FakeManager(backend), progress=events.append)
    # Duplicates are one model call, so one pair.
    ((text, output),) = events[-1].recent
    assert len(text) <= engine.RECENT_TEXT_CHARS
    assert text.startswith("word word word") and text.endswith("…")
    assert "\n" not in text and "  " not in text
    assert len(output) <= engine.RECENT_OUTPUT_CHARS and output.endswith("…")


def test_progress_recent_skips_cached_and_failed_rows():
    config = make_config()
    store = FakeStore()
    store.seed(config, "cached", "C")
    backend = ScriptedBackend(fail_when=lambda n, prompts: "Q: bad" in prompts)
    events: list[Progress] = []
    run(
        ["cached", "bad", "good"],
        store=store,
        manager=FakeManager(backend),
        batch_size=1,
        progress=events.append,
    )
    assert events[-1].recent == (("good", "A:GOOD"),)


def test_progress_recent_defaults_to_empty():
    assert Progress(phase="planning", total_rows=3).recent == ()


# --------------------------------------------------------------------------
# Failed-batch isolation
# --------------------------------------------------------------------------


def fake_backend(**kwargs):
    from batchtextreview.backends.fake import FakeBackend

    return FakeBackend(responder=answer, **kwargs)


def test_poisoned_row_is_isolated_and_the_rest_are_kept():
    store = FakeStore()
    backend = fake_backend(fail_when=lambda batch: any("poison" in p for p in batch))
    texts = ["r1", "r2", "r3", "poison", "r5", "r6", "r7", "r8"]
    events: list[Progress] = []
    result, _ = run(
        texts, store=store, manager=FakeManager(backend), batch_size=8, progress=events.append
    )
    assert len(backend.calls) == 1 + 8  # the batch, then each row alone
    assert [len(call) for call in backend.calls] == [8] + [1] * 8
    assert result.statuses == ["generated"] * 3 + ["error"] + ["generated"] * 4
    assert result.errors[3] == "RuntimeError: fake failure"
    assert result.outputs[0] == "A:R1" and result.outputs[3] == ""
    assert result.complete and result.aborted_reason is None
    assert result.stats.model_calls == 8  # unique prompts, not attempts
    stored = store.outputs[result.run_id]
    assert len(stored) == 7 and content_key("poison") not in stored
    assert stored[content_key("r8")] == "A:R8"
    check_stats(result)
    retries = [e for e in events if "one at a time" in e.message]
    assert len(retries) == 8
    assert retries[-1].message == "Batch 1 of 1: retrying rows one at a time (8 of 8)"
    done = [e.done_rows for e in events]
    assert done == sorted(done)


def test_isolated_rows_share_outputs_with_their_duplicates():
    backend = fake_backend(fail_when=lambda batch: len(batch) > 1)
    result, _ = run(["a", "b", "a"], manager=FakeManager(backend), batch_size=8)
    assert result.statuses == ["generated"] * 3
    assert result.outputs == ["A:A", "A:B", "A:A"]


def test_systemic_failure_stops_retrying_after_two_rows():
    store = FakeStore()
    backend = fake_backend(fail_when=lambda batch: True)
    texts = [f"row {i}" for i in range(8)]
    result, _ = run(texts, store=store, manager=FakeManager(backend), batch_size=8)
    assert [len(call) for call in backend.calls] == [8, 1, 1]
    assert result.statuses == ["error"] * 8
    assert set(result.errors) == {"RuntimeError: fake failure"}
    assert result.complete
    assert store.outputs == {} and store.runs == {}
    check_stats(result)


def test_different_single_errors_are_not_systemic():
    class BatchTooBig(ScriptedBackend):
        def generate(self, prompts, max_new_tokens):
            self.calls.append(list(prompts))
            if len(prompts) > 1:
                raise RuntimeError("batch too big")
            raise RuntimeError(f"bad row {prompts[0]}")

    backend = BatchTooBig()
    result, _ = run(list("abcd"), manager=FakeManager(backend), batch_size=4)
    assert [len(call) for call in backend.calls] == [4, 1, 1, 1, 1]
    assert result.errors == [f"RuntimeError: bad row Q: {c}" for c in "abcd"]


def test_a_success_among_the_first_two_retries_keeps_retrying():
    # Batch fails; alone, "a" works and every other row fails like the batch.
    backend = fake_backend(fail_when=lambda batch: batch != ["Q: a"])
    result, _ = run(list("abcde"), manager=FakeManager(backend), batch_size=5)
    assert [len(call) for call in backend.calls] == [5, 1, 1, 1, 1, 1]
    assert result.statuses == ["generated"] + ["error"] * 4


def test_single_prompt_batch_is_not_retried():
    backend = fake_backend(fail_when=lambda batch: True)
    result, _ = run(["a"], manager=FakeManager(backend), batch_size=8)
    assert len(backend.calls) == 1
    assert result.statuses == ["error"]


def test_wrong_output_count_is_not_retried_row_by_row():
    class Short(ScriptedBackend):
        def generate(self, prompts, max_new_tokens):
            self.calls.append(list(prompts))
            return ["only one"]

    backend = Short()
    result, _ = run(["a", "b"], manager=FakeManager(backend), max_consecutive_failures=0)
    assert len(backend.calls) == 1
    assert result.statuses == ["error", "error"]


def test_consecutive_failure_abort_still_works_with_isolation():
    backend = fake_backend(fail_when=lambda batch: True)
    texts = [f"row {i:02d}" for i in range(12)]
    result, _ = run(texts, manager=FakeManager(backend), batch_size=4, max_consecutive_failures=2)
    assert [len(call) for call in backend.calls] == [4, 1, 1, 4, 1, 1]
    assert result.statuses == ["error"] * 8 + ["pending"] * 4
    assert result.aborted_reason is not None
    assert "fake failure" in result.aborted_reason
    assert "stopped after 2 failed batches in a row" in result.aborted_reason
    check_stats(result)


def test_batch_with_any_success_does_not_count_as_failed():
    # Every batch of two holds one poisoned row: no batch fails outright.
    backend = fake_backend(fail_when=lambda batch: any("bad" in p for p in batch))
    texts = ["ok_1", "bad1", "ok_2", "bad2", "ok_3", "bad3"]  # one length: kept in order
    result, _ = run(texts, manager=FakeManager(backend), batch_size=2, max_consecutive_failures=1)
    assert result.aborted_reason is None
    assert result.statuses == ["generated", "error"] * 3


def test_cancel_is_checked_between_single_retries():
    cancel = threading.Event()
    store = FakeStore()

    def on_progress(p: Progress) -> None:
        if "one at a time" in p.message:
            cancel.set()

    backend = fake_backend(fail_when=lambda batch: len(batch) > 1)
    result, _ = run(
        list("abcd"),
        store=store,
        manager=FakeManager(backend),
        batch_size=4,
        cancel=cancel,
        progress=on_progress,
    )
    assert [len(call) for call in backend.calls] == [4, 1]
    assert result.cancelled and not result.complete
    assert result.statuses == ["generated", "pending", "pending", "pending"]
    assert list(store.outputs[result.run_id].values()) == ["A:A"]
    check_stats(result)


def test_store_failure_during_single_retries_stops_the_run():
    store = LockedStore(fail_from=2)
    backend = fake_backend(fail_when=lambda batch: len(batch) > 1)
    result, _ = run(list("abcd"), store=store, manager=FakeManager(backend), batch_size=4)
    assert [len(call) for call in backend.calls] == [4, 1, 1]
    assert result.statuses == ["generated", "generated", "pending", "pending"]
    assert result.aborted_reason is not None
    assert result.aborted_reason.startswith("Couldn't save outputs")
    check_stats(result)


# --------------------------------------------------------------------------
# Truncation visibility
# --------------------------------------------------------------------------


class LimitBackend(ScriptedBackend):
    """Reports ``last_hit_limit`` like the transformers backend does."""

    def __init__(self, flags=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.flags = flags or (lambda prompts: ["long" in p for p in prompts])
        self.last_hit_limit = None

    def generate(self, prompts, max_new_tokens):
        outputs = super().generate(prompts, max_new_tokens)
        self.last_hit_limit = self.flags(list(prompts))
        return outputs


def test_plan_counts_truncated_rows():
    config = make_config(max_input_chars=5)
    plan = plan_run(config, ["abcdefg", "abc", None, 123456, "abcde", "  "], None)
    assert plan.n_truncated_rows == 2
    assert plan_run(make_config(), ["abcdefg"], None).n_truncated_rows == 0


def test_plan_without_a_limit_truncates_nothing():
    plan = plan_run(make_config(max_input_chars=0), ["abcdefg"], None)
    assert plan.n_truncated_rows == 0


def test_result_flags_truncated_inputs_including_cached_rows():
    config = make_config(max_input_chars=5)
    store = FakeStore()
    store.seed(config, "cachedtext", "C")
    texts = pd.Series(["abcdefg", "abc", None, 123456, "cachedtext"], index=[5, 4, 3, 2, 1])
    result, _ = run(texts, config=config, store=store)
    assert result.input_truncated == [True, False, False, True, True]
    assert result.stats.truncated_inputs == 3
    assert result.statuses[-1] == "cached"


def test_hit_token_limit_read_from_the_backend():
    config = make_config()
    store = FakeStore()
    store.seed(config, "long cached", "C")
    backend = LimitBackend()
    texts = ["long one", "short", "long one", None, "long cached"]
    result, _ = run(texts, config=config, store=store, manager=FakeManager(backend))
    assert result.hit_token_limit == [True, False, True, False, False]
    assert result.stats.hit_token_limit == 2
    assert result.input_truncated == [False] * 5


def test_hit_token_limit_from_single_retries():
    backend = LimitBackend(fail_when=lambda n, prompts: len(prompts) > 1)
    result, _ = run(["long a", "b"], manager=FakeManager(backend), batch_size=2)
    assert result.statuses == ["generated", "generated"]
    assert result.hit_token_limit == [True, False]


@pytest.mark.parametrize(
    "flags",
    [
        lambda prompts: [True],  # wrong length
        lambda prompts: "yes" * len(prompts),  # not a list
        lambda prompts: None,
    ],
)
def test_unusable_hit_limit_reports_are_ignored(flags):
    result, _ = run(["a", "b"], manager=FakeManager(LimitBackend(flags=flags)))
    assert result.hit_token_limit == [False, False]
    assert result.stats.hit_token_limit == 0
    assert result.statuses == ["generated", "generated"]


def test_hit_limit_property_that_raises_is_ignored():
    class Raising(ScriptedBackend):
        @property
        def last_hit_limit(self):
            raise RuntimeError("not available")

    result, _ = run(["a"], manager=FakeManager(Raising()))
    assert result.hit_token_limit == [False]
    assert result.statuses == ["generated"]


def test_backend_without_hit_limit_reports_nothing():
    result, _ = run(["a", None])
    assert result.hit_token_limit == [False, False]
    assert result.input_truncated == [False, False]
    assert result.stats.generation_s > 0


def test_hit_token_limit_is_stored_and_kept_for_cached_rows():
    """Regression: a resumed or re-run run lost the "may be cut off" flag
    of every cached row."""
    store = FakeStore()
    texts = ["long one", "short", "long one", None]
    first, _ = run(texts, store=store, manager=FakeManager(LimitBackend()))
    assert first.hit_token_limit == [True, False, True, False]
    assert store.hit_limit[first.run_id] == {content_key("long one")}

    again, manager = run(texts, store=store, manager=FakeManager(LimitBackend()))
    assert manager.acquired == 0
    assert again.statuses == ["cached", "cached", "cached", "blank"]
    assert again.hit_token_limit == [True, False, True, False]
    assert again.stats.hit_token_limit == 2
    plan = plan_run(make_config(), texts, store)
    assert plan.cached_hit_limit == {content_key("long one")}


def test_hit_token_limit_survives_a_cancel_and_resume():
    store = FakeStore()
    texts = ["ax", "b", "cccx", "dddd"]  # batches (by length): [b, ax], [cccx, dddd]
    cancel = threading.Event()

    def stop_after_first_batch(progress: Progress) -> None:
        if progress.phase == "generating" and progress.generated_rows:
            cancel.set()

    def flagged() -> LimitBackend:
        return LimitBackend(flags=lambda prompts: ["x" in p for p in prompts])

    first, _ = run(
        texts,
        store=store,
        manager=FakeManager(flagged()),
        batch_size=2,
        cancel=cancel,
        progress=stop_after_first_batch,
    )
    assert first.cancelled
    assert first.statuses == ["generated", "generated", "pending", "pending"]
    assert first.hit_token_limit == [True, False, False, False]
    resumed, _ = run(texts, store=store, manager=FakeManager(flagged()), batch_size=2)
    assert resumed.statuses == ["cached", "cached", "generated", "generated"]
    assert resumed.hit_token_limit == [True, False, True, False]
    assert resumed.stats.hit_token_limit == 2


def test_store_without_hit_limit_support_still_works():
    """A minimal store (get_outputs / put_outputs(run_id, items) only) works
    as long as no output needs flagging."""

    class MinimalStore:
        def __init__(self) -> None:
            self.outputs: dict[str, str] = {}

        def upsert_run(self, config, source="app") -> None:
            pass

        def get_outputs(self, run_id, keys=None):
            return {k: self.outputs[k] for k in keys or () if k in self.outputs}

        def put_outputs(self, run_id, items) -> None:
            self.outputs.update(items)

    store = MinimalStore()
    first, _ = run(["long a", "b"], store=store)
    assert first.statuses == ["generated", "generated"] and first.complete
    again, _ = run(["long a", "b"], store=store)
    assert again.statuses == ["cached", "cached"]
    assert again.hit_token_limit == [False, False]


def test_result_lists_default_to_empty_for_old_style_construction():
    result = engine.RunResult(run_id="r", config=make_config(), outputs=[], statuses=[], errors=[])
    assert result.input_truncated == [] and result.hit_token_limit == []
    assert result.stats.truncated_inputs == result.stats.hit_token_limit == 0


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------


class Records(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def at(self, level: int) -> list[logging.LogRecord]:
        return [r for r in self.records if r.levelno == level]


@pytest.fixture
def engine_log():
    """Records from the engine's logger, whatever handlers or propagation
    other code configured."""
    logger = logging.getLogger("batchtextreview.engine")
    handler = Records()
    previous = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    yield handler
    logger.removeHandler(handler)
    logger.setLevel(previous)


def test_logs_one_line_at_start_and_end(engine_log):
    result, _ = run(["a", "bb", "a"], config=make_config(max_input_chars=1))
    infos = [r.getMessage() for r in engine_log.at(logging.INFO)]
    assert len(infos) == 2
    start, end = infos
    assert start.startswith(f"Run {result.run_id} started (app): Fake model [demo://fake, none]")
    assert "3 rows, 3 to generate (2 model calls)" in start
    assert "1 text cut to 1 characters" in start
    assert end.startswith(f"Run {result.run_id} finished: 3 rows - 3 generated;")
    assert "rows/s" in end
    assert not engine_log.at(logging.WARNING)


def test_logs_end_of_a_cached_or_stopped_run(engine_log):
    store = FakeStore()
    run(["a"], store=store)
    run(["a"], store=store)
    cached = engine_log.records[-1].getMessage()
    assert "finished: 1 row - 1 cached; " in cached
    assert "rows/s" not in cached and "loading" not in cached
    run(["b"], manager=FakeManager(load_error=OSError("disk on fire")))
    last = engine_log.records[-1]
    assert last.levelno == logging.INFO
    assert "stopped early (Couldn't load Fake model: OSError: disk on fire)" in last.getMessage()


def test_failed_batches_warn_with_one_traceback(engine_log):
    backend = ScriptedBackend(fail_when=lambda n, p: True)
    run(list("abc"), manager=FakeManager(backend), batch_size=1, max_consecutive_failures=0)
    warnings = engine_log.at(logging.WARNING)
    assert len(warnings) == 3
    assert "Batch 1 of 3 (1 prompt) failed: RuntimeError: boom" in warnings[0].getMessage()
    assert warnings[0].exc_info is not None and warnings[0].exc_info[0] is RuntimeError
    assert all(not w.exc_info for w in warnings[1:])


def test_isolation_is_logged(engine_log):
    backend = fake_backend(fail_when=lambda batch: any("poison" in p for p in batch))
    run(["a", "poison", "c"], manager=FakeManager(backend), batch_size=3)
    messages = [r.getMessage() for r in engine_log.records]
    assert any("Batch 1 of 1 (3 prompts) failed: RuntimeError: fake failure" in m for m in messages)
    assert any("retried one row at a time: 2 succeeded, 1 failed" in m for m in messages)
    assert any(
        r.levelno == logging.DEBUG and "row retried alone failed" in r.getMessage()
        for r in engine_log.records
    )

    engine_log.records.clear()
    run(list("wxyz"), manager=FakeManager(fake_backend(fail_when=lambda b: True)), batch_size=4)
    systemic = [r.getMessage() for r in engine_log.at(logging.WARNING)]
    assert any("2 more rows marked failed without retrying" in m for m in systemic)


# --------------------------------------------------------------------------
# A local fine-tune saved again into the same folder (real tiny model)
# --------------------------------------------------------------------------


@pytest.mark.llm
def test_retrained_local_model_is_neither_served_from_cache_nor_kept_loaded(tmp_path, monkeypatch):
    """Regression: re-saving a fine-tune into its folder served the old
    model's cached outputs, and - with the cache deleted - a long-lived
    manager kept generating with the old weights."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import shutil

    from batchtextreview.backends.manager import ModelManager
    from batchtextreview.models import custom_model
    from batchtextreview.store import Store
    from tiny_model import build_tiny_model

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    folder = build_tiny_model(tmp_path / "my-classifier", seed=0)
    spec = custom_model(str(folder))
    template = "classify sentiment of the text below : {text}"
    texts = ["the product is great", "it was bad"]
    store = Store(tmp_path / "results.db")
    manager = ModelManager()  # one manager, like the Streamlit server's

    def config() -> RunConfig:
        return RunConfig.build(spec, template, 4, precision="fp32")

    old_config = config()
    old = execute_run(old_config, texts, store=store, manager=manager)
    assert old.statuses == ["generated"] * 2

    shutil.rmtree(folder)
    build_tiny_model(folder, seed=7)
    for path in folder.iterdir():  # coarse file clocks: make the save visibly newer
        os.utime(path, ns=(path.stat().st_mtime_ns + 10**9,) * 2)
    fresh = execute_run(config(), texts, store=None, manager=ModelManager())
    assert fresh.outputs != old.outputs  # the two seeds really differ

    new_config = config()
    assert new_config.run_id != old_config.run_id
    rerun = execute_run(new_config, texts, store=store, manager=manager)
    assert rerun.statuses == ["generated"] * 2
    assert rerun.outputs == fresh.outputs
    assert rerun.stats.load_s > 0

    # The old configuration, its saved outputs deleted, re-run in the same
    # manager: the folder's current weights, not the ones in memory.
    store.delete_run(old_config.run_id)
    assert execute_run(old_config, texts, store=store, manager=manager).outputs == fresh.outputs
    manager.unload()
