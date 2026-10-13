"""Tests for batchtextreview.store."""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from batchtextreview import store as store_mod
from batchtextreview.engine import RunConfig, compute_run_id, content_key
from batchtextreview.models import LEGACY_LABEL_TO_REPO
from batchtextreview.presets import Preset, make_custom_preset
from batchtextreview.store import (
    EVALUATION_COLUMNS,
    RUN_COLUMNS,
    SCHEMA_VERSION,
    LegacyImportReport,
    Store,
    import_legacy,
    scan_legacy,
)

SENTIMENT_TEMPLATE = "Classify the sentiment.\n\nTEXT:\n{text}"


@pytest.fixture
def store(tmp_path) -> Store:
    return Store(tmp_path / "results" / "batchtextreview.db")


def make_config(template: str = SENTIMENT_TEMPLATE, **overrides) -> RunConfig:
    values = {
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "model_label": "Qwen2.5-3B-Instruct",
        "precision": "4bit",
        "template": template,
        "max_new_tokens": 8,
        "max_input_chars": 8000,
    }
    values.update(overrides)
    return RunConfig(**values)


def clock(monkeypatch, *stamps: str) -> None:
    """Make store timestamps come from ``stamps`` in order (last one repeats)."""
    queue = list(stamps)

    def fake_now() -> str:
        return queue.pop(0) if len(queue) > 1 else queue[0]

    monkeypatch.setattr(store_mod, "_now", fake_now)


@contextmanager
def raw(store: Store) -> Iterator[sqlite3.Connection]:
    """A plain connection to the store's file: committed and *closed* on exit
    (``with sqlite3.connect(...)`` only commits - an open handle keeps the
    database files locked on Windows)."""
    conn = sqlite3.connect(store.path)
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def eval_record(**overrides) -> dict:
    record = {
        "run_id": "run1",
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "model_label": "Qwen2.5-3B-Instruct",
        "precision": "4bit",
        "preset": "Sentiment",
        "var": None,
        "labels": ["positive", "negative", "neutral"],
        "dataset": "reviews.csv",
        "dataset_fingerprint": "fp1",
        "text_column": "TEXT",
        "gold_column": "GOLD",
        "n_rows": 10,
        "n_scored": 9,
        "n_correct": 6,
        "accuracy": 0.6667,
        "macro_f1": 0.61,
        "kappa": 0.5,
        "n_invalid": 1,
        "n_excluded": 0,
        "n_gold_unscorable": 1,
    }
    record.update(overrides)
    return record


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------


def test_creates_parent_dirs_schema_and_version(tmp_path):
    path = tmp_path / "a" / "b" / "store.db"
    store = Store(path)
    assert store.path == path
    assert path.is_file()
    with raw(store) as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert {"meta", "runs", "outputs", "evaluations", "presets", "settings"} <= tables
    assert mode == "wal"
    assert store.get_meta("schema_version") == str(SCHEMA_VERSION)
    assert store.get_meta("nope") is None


def test_raw_helper_commits_and_closes(store):
    """Regression: the helper leaked one connection per use."""
    with raw(store) as conn:
        conn.execute("INSERT INTO settings (key, value) VALUES ('k', '1')")
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        conn.execute("SELECT 1")
    assert store.get_setting("k") == 1


def test_reopen_keeps_data_and_version(tmp_path):
    path = tmp_path / "store.db"
    first = Store(path)
    first.upsert_run(make_config())
    first.put_outputs(make_config().run_id, {"k": "v"})
    first.set_setting("theme", "dark")

    second = Store(path)
    assert second.get_outputs(make_config().run_id) == {"k": "v"}
    assert second.get_setting("theme") == "dark"
    with raw(second) as conn:
        rows = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchall()
    assert rows == [(str(SCHEMA_VERSION),)]


def test_newer_schema_version_is_refused(tmp_path):
    path = tmp_path / "store.db"
    store = Store(path)
    with raw(store) as conn:
        conn.execute(
            "UPDATE meta SET value = ? WHERE key = 'schema_version'", (SCHEMA_VERSION + 1,)
        )
    with pytest.raises(RuntimeError, match="newer version"):
        Store(path)


def test_evaluation_columns_match_table(store):
    with raw(store) as conn:
        cols = tuple(r[1] for r in conn.execute("PRAGMA table_info(evaluations)"))
    assert cols == EVALUATION_COLUMNS


# The schema exactly as the first release (schema_version 1) created it.
V1_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    model_id        TEXT NOT NULL,
    model_label     TEXT NOT NULL,
    precision       TEXT NOT NULL,
    template        TEXT NOT NULL,
    max_new_tokens  INTEGER NOT NULL,
    max_input_chars INTEGER NOT NULL,
    source          TEXT NOT NULL DEFAULT 'app',   -- 'app' | 'cli' | 'legacy'
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS outputs (
    run_id     TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    key        TEXT NOT NULL,
    output     TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, key)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS evaluations (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at          TEXT NOT NULL,
    run_id              TEXT,
    model_id            TEXT,
    model_label         TEXT,
    precision           TEXT,
    preset              TEXT,
    var                 TEXT,
    labels              TEXT,      -- JSON list, or NULL for exact-match
    dataset             TEXT,      -- file name
    dataset_fingerprint TEXT,
    text_column         TEXT,
    gold_column         TEXT,
    n_rows              INTEGER,
    n_scored            INTEGER,
    n_correct           INTEGER,
    accuracy            REAL,
    macro_f1            REAL,
    kappa               REAL,
    n_invalid           INTEGER,
    n_excluded          INTEGER,
    n_gold_unscorable   INTEGER,
    source              TEXT NOT NULL DEFAULT 'app',
    note                TEXT
);
CREATE INDEX IF NOT EXISTS evaluations_identity
    ON evaluations(run_id, dataset_fingerprint, text_column, gold_column);
CREATE TABLE IF NOT EXISTS presets (
    key            TEXT PRIMARY KEY,
    title          TEXT NOT NULL,
    template       TEXT NOT NULL,
    max_new_tokens INTEGER NOT NULL,
    labels         TEXT,           -- JSON list or NULL
    var_hint       TEXT,
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT                       -- JSON
);
"""
PROVENANCE = (
    "selection",
    "rows_per_s",
    "elapsed_s",
    "app_version",
    "template_edited",
    "preset_title",
)


def make_v1_database(path: Path, version: str = "1") -> None:
    """A database file as the first release left it, with one run, one
    output and one history line."""
    conn = sqlite3.connect(path)
    try:
        conn.executescript(V1_SCHEMA)
        with conn:
            conn.execute("INSERT INTO meta (key, value) VALUES ('schema_version', ?)", (version,))
            conn.execute(
                "INSERT INTO runs VALUES ('run1', 'org/m', 'M', '4bit', 'Q: {text}', 8, 8000, "
                "'app', '2026-01-01T00:00:00', '2026-01-01T00:00:00')"
            )
            conn.execute(
                "INSERT INTO outputs VALUES ('run1', 'k', 'positive', '2026-01-01T00:00:00')"
            )
            conn.execute(
                "INSERT INTO evaluations (created_at, run_id, dataset_fingerprint, text_column, "
                "gold_column, labels, accuracy, note) VALUES ('2026-01-02T00:00:00', 'run1', "
                "'fp1', 'TEXT', 'GOLD', '[\"positive\", \"negative\", \"neutral\"]', 0.75, 'old note')"
            )
    finally:
        conn.close()


def table_columns(path: Path) -> tuple[str, ...]:
    conn = sqlite3.connect(path)
    try:
        return tuple(r[1] for r in conn.execute("PRAGMA table_info(evaluations)"))
    finally:
        conn.close()


def output_columns(path: Path) -> tuple[str, ...]:
    conn = sqlite3.connect(path)
    try:
        return tuple(r[1] for r in conn.execute("PRAGMA table_info(outputs)"))
    finally:
        conn.close()


def test_schema_version_is_three():
    assert SCHEMA_VERSION == 3
    assert EVALUATION_COLUMNS[-len(PROVENANCE) :] == PROVENANCE


def test_v1_database_is_migrated_on_open(tmp_path):
    path = tmp_path / "old.db"
    make_v1_database(path)
    assert "selection" not in table_columns(path)

    store = Store(path)
    assert table_columns(path) == EVALUATION_COLUMNS
    assert output_columns(path)[-1] == "hit_limit"
    assert store.get_meta("schema_version") == "3"
    # Everything already there survives; an old output's flag is unknown.
    assert store.get_outputs("run1") == {"k": "positive"}
    assert store.get_hit_limit_keys("run1", ["k"]) == set()
    history = store.list_evaluations()
    assert list(history.columns) == list(EVALUATION_COLUMNS)
    old = history.iloc[0]
    assert old["accuracy"] == 0.75 and old["note"] == "old note"
    assert old["labels"] == ["positive", "negative", "neutral"]
    assert old["selection"] is None and old["template_edited"] is None
    assert pd.isna(old["rows_per_s"]) and pd.isna(old["elapsed_s"])

    # The new columns are writable, and re-scoring the old line updates it.
    eval_id = store.log_evaluation(
        eval_record(selection="All 10 rows", template_edited=True, rows_per_s=2.5)
    )
    assert eval_id == int(old["id"])
    line = store.list_evaluations().iloc[0]
    assert line["selection"] == "All 10 rows"
    assert line["template_edited"] is True
    assert line["rows_per_s"] == 2.5
    assert line["note"] == "old note"

    # Opening again is a no-op.
    Store(path)
    assert table_columns(path) == EVALUATION_COLUMNS
    assert len(Store(path).list_evaluations()) == 1


def test_migration_logs_once(tmp_path):
    path = tmp_path / "old.db"
    make_v1_database(path)
    logger = logging.getLogger("batchtextreview.store")
    records: list[logging.LogRecord] = []
    handler = logging.Handler(logging.DEBUG)
    handler.emit = records.append  # type: ignore[method-assign]
    logger.addHandler(handler)
    previous = logger.level
    logger.setLevel(logging.INFO)
    try:
        Store(path)
        Store(path)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)
    messages = [r.getMessage() for r in records]
    assert len(messages) == 1
    assert "from schema 1 to 3" in messages[0] and "selection" in messages[0]
    assert "outputs.hit_limit" in messages[0]


def test_concurrent_opens_of_a_v1_database_migrate_once(tmp_path):
    path = tmp_path / "old.db"
    make_v1_database(path)
    errors: list[BaseException] = []
    start = threading.Barrier(6)

    def open_it() -> None:
        try:
            start.wait()
            Store(path).list_evaluations()
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=open_it) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert errors == []
    assert table_columns(path) == EVALUATION_COLUMNS


def test_concurrent_first_opens_of_a_new_database(tmp_path):
    """Regression: switching a new file to WAL needs an exclusive lock and
    SQLite doesn't wait for it - simultaneous first opens failed with
    "database is locked"."""
    path = tmp_path / "new.db"
    errors: list[BaseException] = []
    start = threading.Barrier(8)

    def open_it() -> None:
        try:
            start.wait()
            Store(path).log_evaluation(eval_record(run_id=None))
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=open_it) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert errors == []
    assert len(Store(path).list_evaluations()) == 8


class FlakyPragma:
    """Stands in for a connection whose WAL switch is busy ``busy`` times."""

    def __init__(self, busy: int, error: str = "database is locked") -> None:
        self.busy, self.error, self.calls = busy, error, 0

    def execute(self, sql: str):
        self.calls += 1
        if self.calls <= self.busy:
            raise sqlite3.OperationalError(self.error)


def test_wal_switch_waits_out_a_busy_lock():
    conn = FlakyPragma(busy=3)
    store_mod._enable_wal(conn)  # type: ignore[arg-type]
    assert conn.calls == 4


def test_wal_switch_other_errors_raise_at_once():
    conn = FlakyPragma(busy=5, error="disk I/O error")
    with pytest.raises(sqlite3.OperationalError, match="disk I/O"):
        store_mod._enable_wal(conn)  # type: ignore[arg-type]
    assert conn.calls == 1


def test_wal_switch_gives_up_after_the_timeout(monkeypatch):
    monkeypatch.setattr(store_mod, "_TIMEOUT_S", 0.05)
    with pytest.raises(sqlite3.OperationalError, match="locked"):
        store_mod._enable_wal(FlakyPragma(busy=10_000))  # type: ignore[arg-type]


def test_partially_migrated_database_gets_the_rest(tmp_path):
    path = tmp_path / "half.db"
    make_v1_database(path)
    conn = sqlite3.connect(path)
    try:
        conn.execute("ALTER TABLE evaluations ADD COLUMN selection TEXT")
    finally:
        conn.close()
    store = Store(path)
    assert table_columns(path) == EVALUATION_COLUMNS
    assert store.get_meta("schema_version") == "3"


def test_schema_2_database_gets_the_output_flag_column(tmp_path):
    path = tmp_path / "v2.db"
    make_v1_database(path, version="2")
    conn = sqlite3.connect(path)
    try:
        for name, kind in store_mod._EVAL_ADDED_COLUMNS:
            conn.execute(f"ALTER TABLE evaluations ADD COLUMN {name} {kind}")
    finally:
        conn.close()
    assert "hit_limit" not in output_columns(path)
    store = Store(path)
    assert "hit_limit" in output_columns(path)
    assert store.get_meta("schema_version") == "3"
    store.put_outputs("run1", {"k2": "cut"}, hit_limit=["k2"])
    assert store.get_hit_limit_keys("run1", ["k", "k2"]) == {"k2"}


def test_newer_database_is_not_migrated(tmp_path):
    path = tmp_path / "future.db"
    make_v1_database(path, version=str(SCHEMA_VERSION + 1))
    with pytest.raises(RuntimeError, match="newer version"):
        Store(path)
    assert "selection" not in table_columns(path)


def test_provenance_round_trip(store):
    store.log_evaluation(
        eval_record(
            run_id="r1",
            selection="First 5 of 10 rows",
            rows_per_s=np.float64(12.3456),
            elapsed_s=4.5,
            app_version="2.0.0",
            template_edited=False,
            preset_title="Sentiment",
        )
    )
    store.log_evaluation(eval_record(run_id="r2", template_edited=None))
    history = store.list_evaluations().set_index("run_id")
    first = history.loc["r1"]
    assert first["selection"] == "First 5 of 10 rows"
    assert first["rows_per_s"] == pytest.approx(12.3456)
    assert first["elapsed_s"] == 4.5
    assert first["app_version"] == "2.0.0"
    assert first["template_edited"] is False
    assert first["preset_title"] == "Sentiment"
    assert history.loc["r2", "template_edited"] is None
    with raw(store) as conn:
        stored = dict(conn.execute("SELECT run_id, template_edited FROM evaluations"))
    assert stored == {"r1": 0, "r2": None}


def test_connections_enforce_foreign_keys(store):
    with store._connect() as conn:
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 1  # NORMAL
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO outputs (run_id, key, output, created_at) VALUES ('x','k','v','t')"
            )


def test_failed_write_block_rolls_back(store):
    with pytest.raises(ZeroDivisionError), store._connect(write=True) as conn:
        conn.execute("INSERT INTO settings (key, value) VALUES ('a', '1')")
        _ = 1 / 0
    assert store.get_setting("a") is None


# --------------------------------------------------------------------------
# Runs & outputs
# --------------------------------------------------------------------------


def test_upsert_run_is_idempotent_and_bumps_updated_at(store, monkeypatch):
    config = make_config()
    clock(monkeypatch, "2026-01-01T10:00:00")
    store.upsert_run(config, source="cli")
    clock(monkeypatch, "2026-01-02T11:30:00")
    store.upsert_run(config)  # different source on the second call
    store.upsert_run(config)

    run = store.get_run(config.run_id)
    assert run is not None
    assert run["created_at"] == "2026-01-01T10:00:00"
    assert run["updated_at"] == "2026-01-02T11:30:00"
    assert run["source"] == "cli"  # first writer wins
    assert run["model_id"] == config.model_id
    assert run["precision"] == "4bit"
    assert run["template"] == SENTIMENT_TEMPLATE
    assert run["max_new_tokens"] == 8
    assert run["max_input_chars"] == 8000
    assert run["n_outputs"] == 0
    assert len(store.list_runs()) == 1


def test_upsert_run_follows_label_changes(store):
    store.upsert_run(make_config(model_label="Old name"))
    store.upsert_run(make_config(model_label="New name"))
    assert store.get_run(make_config().run_id)["model_label"] == "New name"


def test_get_run_missing(store):
    assert store.get_run("nope") is None


def test_put_and_get_outputs(store):
    config = make_config()
    store.upsert_run(config)
    store.put_outputs(config.run_id, {"a": "positive", "b": "negative", "c": ""})
    assert store.get_outputs(config.run_id) == {"a": "positive", "b": "negative", "c": ""}
    assert store.count_outputs(config.run_id) == 3

    # Limited to keys: unknown keys and None are ignored, duplicates harmless,
    # any iterable works.
    got = store.get_outputs(config.run_id, iter(["a", "zzz", None, "a", "c"]))
    assert got == {"a": "positive", "c": ""}
    assert store.get_outputs(config.run_id, []) == {}
    assert store.get_outputs("other-run", ["a"]) == {}

    # Insert-or-replace.
    store.put_outputs(config.run_id, {"a": "mixed"})
    assert store.get_outputs(config.run_id, ["a"]) == {"a": "mixed"}
    assert store.count_outputs(config.run_id) == 3


def test_outputs_are_scoped_by_run(store):
    one, two = make_config(), make_config(max_new_tokens=16)
    assert one.run_id != two.run_id
    store.upsert_run(one)
    store.upsert_run(two)
    store.put_outputs(one.run_id, {"k": "from one"})
    store.put_outputs(two.run_id, {"k": "from two"})
    assert store.get_outputs(one.run_id, ["k"]) == {"k": "from one"}
    assert store.get_outputs(two.run_id) == {"k": "from two"}


def test_unicode_and_long_outputs_round_trip(store):
    config = make_config()
    store.upsert_run(config)
    items = {"ü": "naïve — “quoted” 日本語 🙂", "long": "x" * 200_000, "nl": "a\r\nb\x00c"}
    store.put_outputs(config.run_id, items)
    assert store.get_outputs(config.run_id) == items


@pytest.mark.skipif(
    not hasattr(sqlite3.Connection, "setlimit"), reason="Connection.setlimit needs Python 3.11+"
)
def test_get_outputs_many_keys_is_chunked(store, monkeypatch):
    config = make_config()
    store.upsert_run(config)
    items = {hashlib.sha256(str(i).encode()).hexdigest(): f"out {i}" for i in range(2345)}
    store.put_outputs(config.run_id, items)

    # Cap bound parameters at 501 (500 keys + the run id) on every connection
    # the store opens: an unchunked IN (...) would fail with "too many SQL
    # variables".
    real_connect = sqlite3.connect

    def limited(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 501)
        return conn

    monkeypatch.setattr(store_mod.sqlite3, "connect", limited)
    wanted = [*items, "missing-1", "missing-2"]
    got = store.get_outputs(config.run_id, wanted)
    assert got == items
    assert store.count_outputs(config.run_id) == 2345
    assert store.get_hit_limit_keys(config.run_id, wanted) == set()


def test_hit_limit_flags_are_stored_with_outputs(store):
    """Regression: the "may be cut off" flag was lost for cached rows."""
    config = make_config()
    store.upsert_run(config)
    store.put_outputs(config.run_id, {"a": "cut", "b": "whole"}, hit_limit=["a"])
    store.put_outputs(config.run_id, {"c": "whole"})
    assert store.get_outputs(config.run_id) == {"a": "cut", "b": "whole", "c": "whole"}
    assert store.get_hit_limit_keys(config.run_id, ["a", "b", "c", "zzz", None]) == {"a"}
    assert store.get_hit_limit_keys(config.run_id, iter(["b"])) == set()
    assert store.get_hit_limit_keys(config.run_id, []) == set()
    assert store.get_hit_limit_keys("other-run", ["a"]) == set()

    # Replacing an output replaces its flag too.
    store.put_outputs(config.run_id, {"a": "now whole"})
    store.put_outputs(config.run_id, {"b": "now cut"}, hit_limit=iter(["b"]))
    assert store.get_hit_limit_keys(config.run_id, ["a", "b", "c"]) == {"b"}


def test_put_outputs_without_run_raises_key_error(store):
    with pytest.raises(KeyError):
        store.put_outputs("no-such-run", {"k": "v"})
    with pytest.raises(KeyError):
        store.put_outputs("no-such-run", {})
    assert store.stats()["outputs"] == 0


def test_put_outputs_rejects_none_output(store):
    config = make_config()
    store.upsert_run(config)
    with pytest.raises(TypeError):
        store.put_outputs(config.run_id, {"a": "ok", "b": None})  # type: ignore[dict-item]
    assert store.count_outputs(config.run_id) == 0


def test_put_outputs_is_one_transaction(store):
    config = make_config()
    store.upsert_run(config)
    # A lone surrogate can't be encoded to UTF-8: binding the second row fails
    # after the first was inserted, and the whole batch must roll back.
    with pytest.raises(UnicodeEncodeError):
        store.put_outputs(config.run_id, {"a": "fine", "b": "bad \ud800"})
    assert store.count_outputs(config.run_id) == 0


def test_list_runs_columns_counts_and_order(store, monkeypatch):
    empty = store.list_runs()
    assert list(empty.columns) == list(RUN_COLUMNS)
    assert empty.empty

    old, new = make_config(), make_config(template="Summarise:\n{text}")
    clock(monkeypatch, "2026-01-01T09:00:00")
    store.upsert_run(old, source="cli")
    clock(monkeypatch, "2026-01-01T10:00:00")
    store.upsert_run(new)
    store.put_outputs(old.run_id, {"a": "1", "b": "2"})

    runs = store.list_runs()
    assert list(runs.columns) == list(RUN_COLUMNS)
    assert runs["run_id"].tolist() == [new.run_id, old.run_id]
    assert runs["n_outputs"].tolist() == [0, 2]
    assert runs["source"].tolist() == ["app", "cli"]
    assert runs["n_outputs"].dtype == np.int64
    assert runs["max_new_tokens"].dtype == np.int64

    # Touching the old run moves it to the top.
    clock(monkeypatch, "2026-01-01T11:00:00")
    store.upsert_run(old)
    assert store.list_runs()["run_id"].tolist() == [old.run_id, new.run_id]


def test_delete_run_cascades_outputs_and_keeps_evaluations(store):
    keep, drop = make_config(), make_config(max_new_tokens=99)
    for config in (keep, drop):
        store.upsert_run(config)
    store.put_outputs(keep.run_id, {"a": "1"})
    store.put_outputs(drop.run_id, {"a": "1", "b": "2", "c": "3"})
    store.log_evaluation(eval_record(run_id=drop.run_id))

    assert store.delete_run(drop.run_id) == 3
    assert store.get_run(drop.run_id) is None
    assert store.count_outputs(drop.run_id) == 0
    with raw(store) as conn:
        assert conn.execute("SELECT COUNT(*) FROM outputs").fetchone()[0] == 1
    assert store.count_outputs(keep.run_id) == 1
    assert len(store.list_evaluations()) == 1
    assert store.delete_run(drop.run_id) == 0
    assert store.delete_run("never-existed") == 0


# --------------------------------------------------------------------------
# Evaluations
# --------------------------------------------------------------------------


def test_log_evaluation_inserts_and_decodes(store, monkeypatch):
    clock(monkeypatch, "2026-03-04T05:06:07")
    eval_id = store.log_evaluation(eval_record(unknown_key="ignored", selection="all"))
    assert isinstance(eval_id, int)
    history = store.list_evaluations()
    assert list(history.columns) == list(EVALUATION_COLUMNS)
    row = history.iloc[0]
    assert row["id"] == eval_id
    assert row["created_at"] == "2026-03-04T05:06:07"
    assert row["labels"] == ["positive", "negative", "neutral"]
    assert row["source"] == "app"
    assert row["accuracy"] == pytest.approx(0.6667)
    assert row["n_scored"] == 9


def test_list_evaluations_empty_frame_has_columns(store):
    history = store.list_evaluations()
    assert history.empty
    assert list(history.columns) == list(EVALUATION_COLUMNS)


def test_log_evaluation_dedupes_by_identity(store, monkeypatch):
    clock(monkeypatch, "2026-01-01T00:00:00")
    first = store.log_evaluation(eval_record(accuracy=0.5, note=None))
    with raw(store) as conn:
        conn.execute("UPDATE evaluations SET note = 'keep me' WHERE id = ?", (first,))
    clock(monkeypatch, "2026-01-02T00:00:00")
    second = store.log_evaluation(eval_record(accuracy=0.75, n_correct=7))

    assert second == first
    history = store.list_evaluations()
    assert len(history) == 1
    row = history.iloc[0]
    assert row["accuracy"] == pytest.approx(0.75)
    assert row["n_correct"] == 7
    assert row["created_at"] == "2026-01-02T00:00:00"
    assert row["note"] == "keep me"  # not in the new record -> untouched


PROVENANCE_VALUES = {
    "selection": "All 10 rows",
    "rows_per_s": 59.75,
    "elapsed_s": 0.0569,
    "app_version": "2.0.0",
    "template_edited": True,
    "preset_title": "Sentiment",
}


def test_log_evaluation_update_keeps_provenance_given_as_none(store):
    """Re-scoring a fully cached run (no throughput) from a caller that
    doesn't know the selection or preset title keeps what the first run
    logged; the metrics are still replaced."""
    first = store.log_evaluation(eval_record(**PROVENANCE_VALUES))
    blanks = dict.fromkeys(PROVENANCE_VALUES)
    again = store.log_evaluation(eval_record(accuracy=0.9, **blanks))
    assert again == first
    row = store.list_evaluations().iloc[0]
    assert row["accuracy"] == pytest.approx(0.9)
    for key, value in PROVENANCE_VALUES.items():
        assert row[key] == value, key


def test_log_evaluation_update_replaces_provenance_given_a_value(store):
    store.log_evaluation(eval_record(**PROVENANCE_VALUES))
    newer = {
        "selection": "First 5 of 10 rows",
        "rows_per_s": 12.5,
        "elapsed_s": 2.0,
        "app_version": "2.1.0",
        "template_edited": False,  # a real value, not "unknown"
        "preset_title": "Sentiment (3 classes)",
    }
    store.log_evaluation(eval_record(**newer))
    row = store.list_evaluations().iloc[0]
    for key, value in newer.items():
        assert row[key] == value, key


def test_log_evaluation_insert_stores_provenance_none_as_null(store):
    store.log_evaluation(eval_record(**dict.fromkeys(PROVENANCE_VALUES)))
    row = store.list_evaluations().iloc[0]
    assert row["selection"] is None
    assert row["template_edited"] is None
    assert pd.isna(row["rows_per_s"])


def test_log_evaluation_null_labels_identity(store):
    exact_1 = store.log_evaluation(eval_record(labels=None, accuracy=0.1))
    exact_2 = store.log_evaluation(eval_record(labels=None, accuracy=0.2))
    with_labels = store.log_evaluation(eval_record(labels=["a", "b"]))
    assert exact_1 == exact_2
    assert with_labels != exact_1
    history = store.list_evaluations().set_index("id")
    assert history.loc[exact_1, "labels"] is None
    assert history.loc[exact_1, "accuracy"] == pytest.approx(0.2)
    assert history.loc[with_labels, "labels"] == ["a", "b"]
    # An empty label list is exact-match too.
    assert store.log_evaluation(eval_record(labels=[], accuracy=0.3)) == exact_1


def test_log_evaluation_labels_list_order(store):
    ab = store.log_evaluation(eval_record(labels=["b", "a"]))
    assert store.log_evaluation(eval_record(labels=("b", "a"))) == ab  # tuple == same list
    # The identity is the stored JSON list, so a reordered set is its own line,
    # and order survives the round trip.
    ba = store.log_evaluation(eval_record(labels=["a", "b"]))
    assert ba != ab
    history = store.list_evaluations().set_index("id")
    assert history.loc[ab, "labels"] == ["b", "a"]
    assert history.loc[ba, "labels"] == ["a", "b"]


@pytest.mark.parametrize(
    "change",
    [
        {"run_id": "run2"},
        {"dataset_fingerprint": "fp2"},
        {"gold_column": "OTHER"},
        {"gold_column": None},
        {"text_column": "BODY"},
        {"text_column": None},
        {"labels": ["positive", "negative"]},
    ],
)
def test_log_evaluation_identity_parts(store, change):
    first = store.log_evaluation(eval_record())
    other = store.log_evaluation(eval_record(**change))
    assert other != first
    assert len(store.list_evaluations()) == 2
    # ... and each of those dedupes with itself.
    assert store.log_evaluation(eval_record(**change)) == other


def test_log_evaluation_text_columns_are_separate_results(store):
    """Regression: run_id and the dataset fingerprint don't depend on the text
    column, so scoring BODY used to overwrite the TITLE line silently."""
    title = store.log_evaluation(eval_record(text_column="TITLE", accuracy=0.9))
    body = store.log_evaluation(eval_record(text_column="BODY", accuracy=0.4))
    assert body != title
    history = store.list_evaluations().set_index("text_column")
    assert history.loc["TITLE", "accuracy"] == pytest.approx(0.9)
    assert history.loc["BODY", "accuracy"] == pytest.approx(0.4)
    # Re-scoring one of them still updates it in place.
    assert store.log_evaluation(eval_record(text_column="TITLE", accuracy=0.95)) == title
    assert len(store.list_evaluations()) == 2


@pytest.mark.parametrize("missing", ["run_id", "dataset_fingerprint"])
def test_log_evaluation_without_full_identity_always_inserts(store, missing):
    one = store.log_evaluation(eval_record(**{missing: None}))
    two = store.log_evaluation(eval_record(**{missing: None}))
    assert one != two
    assert len(store.list_evaluations()) == 2


def test_log_evaluation_coerces_numpy_and_missing(store):
    eval_id = store.log_evaluation(
        eval_record(
            n_rows=np.int64(12),
            n_scored=np.int32(11),
            accuracy=np.float64(0.25),
            kappa=float("nan"),
            macro_f1=pd.NA,
            var=np.str_("café"),
            created_at=datetime.datetime(2026, 5, 6, 7, 8, 9, 123456),
            source=None,
            labels=np.array(["x", "y"]).tolist(),
        )
    )
    row = store.list_evaluations().set_index("id").loc[eval_id]
    assert row["n_rows"] == 12
    assert row["n_scored"] == 11
    assert row["accuracy"] == pytest.approx(0.25)
    assert pd.isna(row["kappa"])
    assert pd.isna(row["macro_f1"])
    assert row["var"] == "café"
    assert row["created_at"] == "2026-05-06T07:08:09"
    assert row["source"] == "app"
    with raw(store) as conn:
        kappa, f1 = conn.execute(
            "SELECT kappa, macro_f1 FROM evaluations WHERE id = ?", (eval_id,)
        ).fetchone()
    assert kappa is None and f1 is None


def test_log_evaluation_labels_given_as_json_text(store):
    eval_id = store.log_evaluation(eval_record(labels='["yes", "no"]'))
    assert store.log_evaluation(eval_record(labels=["yes", "no"])) == eval_id
    row = store.list_evaluations().set_index("id").loc[eval_id]
    assert row["labels"] == ["yes", "no"]


@pytest.mark.parametrize(
    "labels",
    [
        np.array(["a", "b"]),
        pd.Series(["a", "b"]),
        pd.Index(["a", "b"]),
        pd.Series(["a", "b", "a"]).unique(),
        {"a": 1, "b": 2}.keys(),
        (x for x in ["a", "b"]),
        ["a", np.nan, "b", None],
    ],
    ids=["ndarray", "series", "index", "unique", "dict_keys", "generator", "missing-items"],
)
def test_log_evaluation_labels_from_any_iterable(store, labels):
    """Regression: anything list-like but not a list/tuple/set was stored as
    one label - str() of the whole container."""
    eval_id = store.log_evaluation(eval_record(labels=labels))
    assert store.list_evaluations().set_index("id").loc[eval_id, "labels"] == ["a", "b"]
    assert store.log_evaluation(eval_record(labels=["a", "b"])) == eval_id


@pytest.mark.parametrize("label", ["solo", np.str_("solo"), np.array("solo")])
def test_log_evaluation_single_label_scalar(store, label):
    eval_id = store.log_evaluation(eval_record(labels=label))
    assert store.list_evaluations().set_index("id").loc[eval_id, "labels"] == ["solo"]


def test_list_evaluations_newest_first(store, monkeypatch):
    clock(monkeypatch, "2026-01-01T00:00:00")
    a = store.log_evaluation(eval_record(run_id="a"))
    clock(monkeypatch, "2026-02-01T00:00:00")
    b = store.log_evaluation(eval_record(run_id="b"))
    c = store.log_evaluation(eval_record(run_id="c"))  # same second: id breaks the tie
    clock(monkeypatch, "2025-12-31T00:00:00")
    d = store.log_evaluation(eval_record(run_id="d"))
    assert store.list_evaluations()["id"].tolist() == [c, b, a, d]


def test_delete_evaluation(store):
    eval_id = store.log_evaluation(eval_record())
    assert store.delete_evaluation(eval_id) is True
    assert store.delete_evaluation(eval_id) is False
    assert store.list_evaluations().empty
    assert store.delete_evaluation(np.int64(12345)) is False


# --------------------------------------------------------------------------
# Presets
# --------------------------------------------------------------------------


def test_presets_round_trip(store):
    assert store.list_presets() == []
    labelled = make_custom_preset(
        "ticket triage",
        "Route this ticket to {var}.\n\n{text}",
        16,
        labels=["billing", "tech", "other"],
        var_hint="e.g. a team",
    )
    plain = make_custom_preset("Bullet summary", "Summarise as bullets:\n{text}", 128)
    store.save_preset(labelled)
    store.save_preset(plain)

    loaded = store.list_presets()
    assert loaded == [plain, labelled]  # sorted by title, case-insensitively
    assert loaded[1].labels == ("billing", "tech", "other")
    assert loaded[1].output_kind == "label"
    assert loaded[1].var_hint == "e.g. a team"
    assert loaded[0].labels is None
    assert all(not p.builtin for p in loaded)


def test_save_preset_replaces_by_key(store):
    store.save_preset(make_custom_preset("Mine", "Old {text}", 8))
    with raw(store) as conn:
        created = conn.execute("SELECT created_at FROM presets").fetchone()[0]
    store.save_preset(make_custom_preset("Mine", "New {text}", 32, labels=["a"]))
    (preset,) = store.list_presets()
    assert preset.template == "New {text}"
    assert preset.max_new_tokens == 32
    assert preset.labels == ("a",)
    with raw(store) as conn:
        assert conn.execute("SELECT created_at FROM presets").fetchone()[0] == created


def test_save_preset_validates(store):
    bad = Preset(
        key="Broken", title="Broken", category="Custom", template="no placeholder", max_new_tokens=8
    )
    with pytest.raises(ValueError, match=r"\{text\}"):
        store.save_preset(bad)
    builtin_name = Preset(
        key="Sentiment", title="Sentiment", category="Custom", template="{text}", max_new_tokens=8
    )
    with pytest.raises(ValueError, match="built-in"):
        store.save_preset(builtin_name)
    assert store.stats()["presets"] == 0


def test_list_presets_skips_rows_that_no_longer_validate(store):
    store.save_preset(make_custom_preset("Good", "{text}", 8))
    with raw(store) as conn:
        conn.execute(
            "INSERT INTO presets (key, title, template, max_new_tokens, labels, var_hint, "
            "created_at, updated_at) VALUES ('Sentiment', 'Sentiment', '{text}', 8, NULL, NULL, "
            "'t', 't')"
        )
    assert [p.key for p in store.list_presets()] == ["Good"]


def test_delete_preset(store):
    store.save_preset(make_custom_preset("Gone soon", "{text}", 8))
    assert store.delete_preset("Gone soon") is True
    assert store.delete_preset("Gone soon") is False
    assert store.list_presets() == []


# --------------------------------------------------------------------------
# Settings & stats
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    ["text", "ünï", 3, 2.5, True, False, None, [1, "two", None], {"a": {"b": [1, 2]}}],
)
def test_settings_round_trip(store, value):
    store.set_setting("k", value)
    assert store.get_setting("k", default="DEFAULT") == value


def test_settings_default_overwrite_and_paths(store, tmp_path):
    assert store.get_setting("missing") is None
    assert store.get_setting("missing", default=[1]) == [1]
    store.set_setting("k", 1)
    store.set_setting("k", {"now": "dict"})
    assert store.get_setting("k") == {"now": "dict"}
    store.set_setting("folder", tmp_path)
    assert store.get_setting("folder") == str(tmp_path)
    with raw(store) as conn:
        conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES ('bad', '{not json')")
    assert store.get_setting("bad", default="fallback") == "fallback"


def test_stats(store):
    empty = store.stats()
    assert {k: v for k, v in empty.items() if k != "size_bytes"} == {
        "runs": 0,
        "outputs": 0,
        "evaluations": 0,
        "presets": 0,
    }
    assert empty["size_bytes"] > 0

    config = make_config()
    store.upsert_run(config)
    store.put_outputs(config.run_id, {"a": "1", "b": "2"})
    store.log_evaluation(eval_record())
    store.save_preset(make_custom_preset("P", "{text}", 8))
    stats = store.stats()
    assert stats["runs"] == 1
    assert stats["outputs"] == 2
    assert stats["evaluations"] == 1
    assert stats["presets"] == 1
    wal = store.path.with_name(store.path.name + "-wal")
    expected = store.path.stat().st_size + (wal.stat().st_size if wal.exists() else 0)
    assert stats["size_bytes"] == expected


# --------------------------------------------------------------------------
# Concurrency
# --------------------------------------------------------------------------


def test_concurrent_writers_and_readers(tmp_path):
    path = tmp_path / "store.db"
    shared = Store(path)
    n_threads, n_batches, batch = 8, 25, 20
    errors: list[BaseException] = []
    start = threading.Barrier(n_threads + 2)

    def writer(i: int) -> None:
        try:
            own = shared if i % 2 else Store(path)  # shared instance and separate ones
            config = make_config(max_new_tokens=100 + i)
            start.wait()
            own.upsert_run(config, source="cli")
            for b in range(n_batches):
                own.put_outputs(config.run_id, {f"{i}-{b}-{k}": f"out {k}" for k in range(batch)})
                own.upsert_run(config)
                # Every thread re-logs the same identity: must stay one row.
                own.log_evaluation(eval_record(accuracy=b / n_batches))
                own.set_setting(f"thread-{i}", b)
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    def reader() -> None:
        try:
            start.wait()
            for _ in range(60):
                shared.list_runs()
                shared.list_evaluations()
                shared.stats()
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(n_threads)]
    threads += [threading.Thread(target=reader) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not any(t.is_alive() for t in threads)
    assert errors == []

    stats = shared.stats()
    assert stats["runs"] == n_threads
    assert stats["outputs"] == n_threads * n_batches * batch
    assert stats["evaluations"] == 1
    for i in range(n_threads):
        assert shared.get_setting(f"thread-{i}") == n_batches - 1
        assert shared.count_outputs(make_config(max_new_tokens=100 + i).run_id) == (
            n_batches * batch
        )


# --------------------------------------------------------------------------
# Version-1 import
#
# The helpers below write artefacts exactly the way version 1's llm_batch.py
# did (run_id_for, _save_checkpoint, log_eval), so these tests exercise the
# real on-disk format.
# --------------------------------------------------------------------------

QWEN3B = "Qwen2.5-3B-Instruct (faster)"
GRANITE = "Granite-4.0-350M (tiny)"


def v1_run_id(label, template, max_new_tokens, max_input_chars=8000):
    payload = json.dumps(["v1", label, template, int(max_new_tokens), int(max_input_chars)])
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def v1_save_checkpoint(results: Path, run_id: str, ckpt: dict, meta: dict) -> None:
    results.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"key": list(ckpt.keys()), "output": list(ckpt.values())})
    tmp = results / f"{run_id}.tmp"
    df.to_parquet(tmp, index=False)
    tmp.replace(results / f"{run_id}.parquet")
    manifest = results / f"{run_id}.json"
    if not manifest.exists():
        manifest.write_text(json.dumps(meta, indent=2))


def v1_log_eval(results: Path, entry: dict) -> None:
    results.mkdir(parents=True, exist_ok=True)
    path = results / "evals.csv"
    pd.DataFrame([entry]).to_csv(path, mode="a", index=False, header=not path.exists())


def v1_run(
    results: Path,
    label: str,
    template: str,
    texts: list[str],
    *,
    max_new_tokens: int = 8,
    max_input_chars: int = 8000,
    created: str = "2025-06-01T12:00:00",
) -> tuple[str, dict[str, str]]:
    """A finished v1 run: returns (v1 run id, checkpoint {key: output})."""
    run_id = v1_run_id(label, template, max_new_tokens, max_input_chars)
    ckpt = {content_key(t[:max_input_chars]): f"label for {t}" for t in texts}
    meta = {
        "run_id": run_id,
        "model": label,
        "template": template,
        "max_new_tokens": int(max_new_tokens),
        "max_input_chars": int(max_input_chars),
        "created": created,
    }
    v1_save_checkpoint(results, run_id, ckpt, meta)
    return run_id, ckpt


def v1_eval_entry(run_id: str, *, model: str = QWEN3B, **overrides) -> dict:
    entry = {
        "timestamp": "2025-06-01T12:30:00",
        "model": model,
        "preset": "Sentiment",
        "var": "",
        "rows_scored": 40,
        "accuracy": 0.825,
        "invalid_outputs": 2,
        "run_id": run_id,
    }
    entry.update(overrides)
    return entry


def snapshot(folder: Path) -> dict[str, tuple[bytes, int]]:
    return {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in sorted(folder.iterdir())}


@pytest.fixture
def legacy(tmp_path):
    """A version-1 results folder covering the interesting cases."""
    results = tmp_path / "v1results"
    sentiment = "Classify the sentiment as positive/negative/neutral.\n\nTEXT:\n{text}"
    topics = "Give 1-3 topic labels.\n\nTEXT:\n{text}"
    a_id, a_ckpt = v1_run(results, QWEN3B, sentiment, ["good", "bad", "meh"])
    b_id, b_ckpt = v1_run(
        results,
        GRANITE,
        topics,
        ["first text", "second text"],
        max_new_tokens=24,
        max_input_chars=4000,
        created="2025-05-01T08:15:00",
    )
    unknown_id, _ = v1_run(results, "My-Old-Model (retired)", sentiment, ["x"])
    corrupt_id, _ = v1_run(results, QWEN3B, topics, ["y"])
    (results / f"{corrupt_id}.parquet").write_bytes(b"this is not a parquet file")
    pd.DataFrame({"key": ["k"], "output": ["o"]}).to_parquet(results / "0123456789abcdef.parquet")

    v1_log_eval(results, v1_eval_entry(a_id))
    v1_log_eval(
        results,
        v1_eval_entry(
            b_id,
            model=GRANITE,
            preset="Categorise (set {var})",
            var="billing, tech",
            timestamp="2025-05-01T09:00:00",
            accuracy=0.5,
            rows_scored=10,
            invalid_outputs=0,
        ),
    )
    v1_log_eval(results, v1_eval_entry("feedfacefeedface", timestamp="2025-04-01T00:00:00"))
    v1_log_eval(results, v1_eval_entry(corrupt_id, timestamp="2025-04-02T00:00:00", accuracy=0.1))
    v1_log_eval(
        results,
        v1_eval_entry(unknown_id, model="My-Old-Model (retired)", timestamp="2025-04-03T00:00:00"),
    )
    return {
        "dir": results,
        "sentiment": sentiment,
        "topics": topics,
        "a": (a_id, a_ckpt),
        "b": (b_id, b_ckpt),
        "unknown": unknown_id,
        "corrupt": corrupt_id,
    }


def test_scan_legacy(legacy):
    scan = scan_legacy(legacy["dir"])
    assert scan.found
    assert scan.results_dir == legacy["dir"]
    expected = sorted(
        f"{rid}.json"
        for rid in (legacy["a"][0], legacy["b"][0], legacy["unknown"], legacy["corrupt"])
    )
    assert [p.name for p in scan.manifests] == expected
    assert [p.name for p in scan.orphan_checkpoints] == ["0123456789abcdef.parquet"]
    assert scan.evals_csv == legacy["dir"] / "evals.csv"


def test_scan_legacy_empty_or_missing(tmp_path):
    assert not scan_legacy(tmp_path).found
    missing = scan_legacy(tmp_path / "nope")
    assert not missing.found
    assert missing.manifests == [] and missing.orphan_checkpoints == []
    # A v2 database and stray json files are not v1 artefacts.
    Store(tmp_path / "batchtextreview.db")
    (tmp_path / "settings.json").write_text("{}")
    assert not scan_legacy(tmp_path).found


def test_import_legacy_end_to_end(store, legacy, monkeypatch):
    results = legacy["dir"]
    _a_id, a_ckpt = legacy["a"]
    _b_id, b_ckpt = legacy["b"]
    qwen_repo = LEGACY_LABEL_TO_REPO[QWEN3B]
    granite_repo = LEGACY_LABEL_TO_REPO[GRANITE]
    a_v2 = compute_run_id(qwen_repo, "4bit", legacy["sentiment"], 8, 8000)
    b_v2 = compute_run_id(granite_repo, "4bit", legacy["topics"], 24, 4000)
    corrupt_v2 = compute_run_id(qwen_repo, "4bit", legacy["topics"], 8, 8000)

    # Version 2 already produced one of run A's outputs: import must not
    # overwrite it.
    a_config = RunConfig(qwen_repo, "Qwen2.5-3B-Instruct", "4bit", legacy["sentiment"], 8, 8000)
    assert a_config.run_id == a_v2
    store.upsert_run(a_config)
    first_key = next(iter(a_ckpt))
    store.put_outputs(a_v2, {first_key: "from v2"})

    before = snapshot(results)
    clock(monkeypatch, "2026-10-12T09:00:00")
    report = import_legacy(store, results)

    assert isinstance(report, LegacyImportReport)
    assert report.runs_imported == 2
    assert report.outputs_imported == (len(a_ckpt) - 1) + len(b_ckpt)
    assert report.evaluations_imported == 5
    assert len(report.skipped) == 3
    skipped = "\n".join(report.skipped)
    assert "My-Old-Model (retired)" in skipped and f"{legacy['unknown']}.json" in skipped
    assert f"{legacy['corrupt']}.parquet" in skipped
    assert "0123456789abcdef.parquet" in skipped

    # Runs: v2 ids, 4-bit, repo ids; legacy created time kept.
    run_a = store.get_run(a_v2)
    assert run_a["source"] == "app"  # existed before the import
    run_b = store.get_run(b_v2)
    assert run_b["source"] == "legacy"
    assert run_b["model_id"] == granite_repo
    assert run_b["model_label"] == "Granite-4.0-350M"
    assert run_b["precision"] == "4bit"
    assert run_b["max_new_tokens"] == 24
    assert run_b["max_input_chars"] == 4000
    assert run_b["created_at"] == "2025-05-01T08:15:00"
    assert run_b["updated_at"] >= run_b["created_at"]
    assert store.get_run(corrupt_v2) is None
    assert len(store.list_runs()) == 2

    # Outputs: imported, v2 output untouched.
    assert store.get_outputs(b_v2) == b_ckpt
    expected_a = dict(a_ckpt)
    expected_a[first_key] = "from v2"
    assert store.get_outputs(a_v2) == expected_a

    # Evaluations.
    history = store.list_evaluations()
    assert set(history["source"]) == {"legacy"}
    by_time = history.set_index("created_at")
    row_a = by_time.loc["2025-06-01T12:30:00"]
    assert row_a["run_id"] == a_v2
    assert row_a["model_id"] == qwen_repo
    # The current catalogue label, like the imported run's; the v1 label is
    # kept in the note.
    assert row_a["model_label"] == "Qwen2.5-3B-Instruct" == store.get_run(a_v2)["model_label"]
    assert QWEN3B in row_a["note"]
    assert row_a["preset"] == "Sentiment"
    assert row_a["var"] is None or pd.isna(row_a["var"])
    assert row_a["n_scored"] == 40
    assert row_a["accuracy"] == pytest.approx(0.825)
    assert row_a["n_invalid"] == 2
    assert row_a["precision"] == "4bit"
    assert row_a["labels"] is None
    row_b = by_time.loc["2025-05-01T09:00:00"]
    assert row_b["run_id"] == b_v2
    assert row_b["model_id"] == granite_repo
    assert row_b["var"] == "billing, tech"
    assert row_b["preset"] == "Categorise (set {var})"
    unknown_run = by_time.loc["2025-04-01T00:00:00"]
    assert unknown_run["run_id"] is None or pd.isna(unknown_run["run_id"])
    assert "feedfacefeedface" in unknown_run["note"]
    # The manifest was valid, so the history line links to the v2 id even
    # though the checkpoint itself was unreadable.
    assert by_time.loc["2025-04-02T00:00:00"]["run_id"] == corrupt_v2
    unknown_model = by_time.loc["2025-04-03T00:00:00"]
    assert unknown_model["model_label"] == "My-Old-Model (retired)"
    assert unknown_model["model_id"] is None or pd.isna(unknown_model["model_id"])
    assert unknown_model["run_id"] is None or pd.isna(unknown_model["run_id"])

    assert store.get_meta("legacy_imported_at") == "2026-10-12T09:00:00"
    assert snapshot(results) == before  # never modified or deleted

    # Idempotent.
    stats = store.stats()
    again = import_legacy(store, results)
    assert (again.runs_imported, again.outputs_imported, again.evaluations_imported) == (0, 0, 0)
    assert sorted(again.skipped) == sorted(report.skipped)
    assert {k: v for k, v in store.stats().items() if k != "size_bytes"} == {
        k: v for k, v in stats.items() if k != "size_bytes"
    }
    assert store.get_outputs(a_v2)[first_key] == "from v2"


def test_import_legacy_logs_skips(store, tmp_path):
    folder = tmp_path / "v1"
    folder.mkdir()
    (folder / "0123456789abcdef.parquet").write_bytes(b"orphan")
    logger = logging.getLogger("batchtextreview.store")
    records: list[logging.LogRecord] = []
    handler = logging.Handler(logging.DEBUG)
    handler.emit = records.append  # type: ignore[method-assign]
    logger.addHandler(handler)
    previous = logger.level
    logger.setLevel(logging.INFO)
    try:
        report = import_legacy(store, folder)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)
    warnings = [r.getMessage() for r in records if r.levelno == logging.WARNING]
    assert len(warnings) == len(report.skipped) == 1
    assert "0123456789abcdef.parquet: no 0123456789abcdef.json manifest" in warnings[0]
    infos = [r.getMessage() for r in records if r.levelno == logging.INFO]
    assert any("0 runs, 0 outputs, 0 history entries, 1 skipped" in m for m in infos)


def test_import_legacy_empty_folder(store, tmp_path):
    report = import_legacy(store, tmp_path / "nothing-here")
    assert report == LegacyImportReport()
    assert store.get_meta("legacy_imported_at") is not None


def test_import_legacy_manifest_problems(store, tmp_path):
    results = tmp_path / "r"
    run_id, _ckpt = v1_run(results, QWEN3B, "T {text}", ["a"])
    (results / f"{run_id}.json").write_text("{not json")
    no_tokens_id, _ = v1_run(results, GRANITE, "U {text}", ["b"])
    manifest = results / f"{no_tokens_id}.json"
    meta = json.loads(manifest.read_text())
    del meta["max_new_tokens"]
    manifest.write_text(json.dumps(meta))

    report = import_legacy(store, results)
    assert report.runs_imported == 0
    assert len(report.skipped) == 2
    assert any(f"{run_id}.json" in s and "unreadable" in s for s in report.skipped)
    assert any(f"{no_tokens_id}.json" in s for s in report.skipped)
    assert store.stats()["runs"] == 0


def test_import_legacy_defaults_and_current_labels(store, tmp_path):
    """Old manifests without max_input_chars use v1's default (8000); a
    manifest naming a current display label or repo still maps."""
    results = tmp_path / "r"
    run_id, _ckpt = v1_run(results, "Qwen2.5-3B-Instruct", "T {text}", ["a", "b"], created="x")
    manifest = results / f"{run_id}.json"
    meta = json.loads(manifest.read_text())
    del meta["max_input_chars"]
    manifest.write_text(json.dumps(meta))

    report = import_legacy(store, results)
    assert report.runs_imported == 1 and report.skipped == []
    v2 = compute_run_id("Qwen/Qwen2.5-3B-Instruct", "4bit", "T {text}", 8, 8000)
    run = store.get_run(v2)
    assert run is not None and run["n_outputs"] == 2
    # Unparseable "created" falls back to the checkpoint's modification time.
    datetime.datetime.fromisoformat(run["created_at"])


def test_import_legacy_evals_schema_drift_and_extra_columns(store, tmp_path):
    """Regression: an early v1 build logged without run_id and created the
    header; later builds appended wider rows under it (v1 defect). Those
    rows - usually most of the history - were all skipped."""
    results = tmp_path / "r"
    early = v1_eval_entry("x")
    del early["run_id"]
    v1_log_eval(results, early)
    for day in range(1, 6):
        v1_log_eval(
            results,
            v1_eval_entry(
                f"{day:016x}",
                model="Qwen2.5-7B-Instruct (best quality)",
                var="a, b",  # quoted: the comma is not a field separator
                timestamp=f"2025-07-0{day}T00:00:00",
                accuracy=day / 10,
            ),
        )
    with (results / "evals.csv").open("a", encoding="utf-8", newline="") as fh:
        fh.write("\n,,,,,,\n")  # blank-ish lines are ignored
        # 8 fields that don't fit the final v1 layout are still skipped.
        fh.write("not a time,Q,Sentiment,,40,0.5,0,abcd\n")
        fh.write("2025-08-01T00:00:00,Q,Sentiment,,40,82.5,0,abcd\n")

    report = import_legacy(store, results)
    assert report.evaluations_imported == 6
    assert [s.split(":")[0] for s in report.skipped] == ["evals.csv row 10", "evals.csv row 11"]
    history = store.list_evaluations().set_index("created_at")
    early_row = history.loc["2025-06-01T12:30:00"]
    assert early_row["run_id"] is None or pd.isna(early_row["run_id"])
    late = history.loc["2025-07-03T00:00:00"]
    assert late["accuracy"] == pytest.approx(0.3)
    assert late["var"] == "a, b"
    assert late["n_scored"] == 40 and late["n_invalid"] == 2
    assert late["model_id"] == "Qwen/Qwen2.5-7B-Instruct"
    assert f"{3:016x}" in late["note"]
    assert import_legacy(store, results).evaluations_imported == 0


def test_import_legacy_evals_missing_and_extra_columns(store, tmp_path):
    results = tmp_path / "r"
    results.mkdir()
    frame = pd.DataFrame(
        {
            "timestamp": ["2025-01-01T00:00:00", "2025-01-02T00:00:00", "2025-01-03T00:00:00"],
            "model": [QWEN3B, QWEN3B, QWEN3B],
            "accuracy": [0.9, None, 0.7],
            "comment": ["extra", "column", "here"],
        }
    )
    frame.to_csv(results / "evals.csv", index=False)
    report = import_legacy(store, results)
    assert report.evaluations_imported == 2
    assert len(report.skipped) == 1 and "no accuracy" in report.skipped[0]
    history = store.list_evaluations()
    assert history["accuracy"].tolist() == pytest.approx([0.7, 0.9])
    assert history["n_scored"].isna().all()
    assert history["preset"].isna().all()


def test_import_legacy_evals_cp1252_and_duplicates(store, tmp_path):
    results = tmp_path / "r"
    results.mkdir()
    entry = v1_eval_entry("r1", var="café")
    # The same line logged twice (v1 re-logged on every rerun) plus an Excel
    # re-save in cp1252.
    pd.DataFrame([entry, entry]).to_csv(results / "evals.csv", index=False, encoding="cp1252")
    report = import_legacy(store, results)
    assert report.evaluations_imported == 1
    assert store.list_evaluations().iloc[0]["var"] == "café"


def test_import_legacy_links_history_when_checkpoint_arrives_later(store, tmp_path):
    results = tmp_path / "r"
    run_id = v1_run_id(QWEN3B, "T {text}", 8)
    v1_log_eval(results, v1_eval_entry(run_id))
    first = import_legacy(store, results)
    assert first.evaluations_imported == 1
    assert pd.isna(store.list_evaluations().iloc[0]["run_id"])

    v1_run(results, QWEN3B, "T {text}", ["a"])
    second = import_legacy(store, results)
    assert second.evaluations_imported == 0
    assert second.runs_imported == 1
    v2 = compute_run_id(LEGACY_LABEL_TO_REPO[QWEN3B], "4bit", "T {text}", 8, 8000)
    assert store.list_evaluations().iloc[0]["run_id"] == v2
    assert len(store.list_evaluations()) == 1


def test_import_legacy_evals_skip_reasons_use_file_rows(store, tmp_path):
    results = tmp_path / "r"
    results.mkdir()
    (results / "evals.csv").write_text(
        "timestamp,model,preset,accuracy\n"
        "2025-01-01T00:00:00,Q,Sentiment,0.5\n"
        "\n"
        "2025-01-02T00:00:00,Q,Sentiment,0.5,extra\n"
        "2025-01-03T00:00:00,Q,Sentiment,not a number\n",
        encoding="utf-8",
    )
    report = import_legacy(store, results)
    assert report.evaluations_imported == 1
    assert [s.split(":")[0] for s in report.skipped] == ["evals.csv row 4", "evals.csv row 5"]


def test_import_legacy_evals_use_current_model_labels(store, tmp_path):
    """Regression: imported runs carried the catalogue label but imported
    evaluations the v1 one, so one model showed as two chart series."""
    results = tmp_path / "r"
    v1_log_eval(results, v1_eval_entry("r1"))
    v1_log_eval(results, v1_eval_entry("r2", model=GRANITE, timestamp="2025-06-02T00:00:00"))
    v1_log_eval(
        results,
        v1_eval_entry("r3", model="My-Old-Model (retired)", timestamp="2025-06-03T00:00:00"),
    )
    # A line from an import made before the fix, still under the v1 label.
    with raw(store) as conn:
        conn.execute(
            "INSERT INTO evaluations (created_at, model_id, model_label, preset, accuracy, "
            "source) VALUES ('2025-06-01T12:30:00', ?, ?, 'Sentiment', 0.825, 'legacy')",
            (LEGACY_LABEL_TO_REPO[QWEN3B], QWEN3B),
        )

    report = import_legacy(store, results)
    assert report.evaluations_imported == 2  # the Qwen line was there already
    labels = store.list_evaluations().set_index("created_at")["model_label"].to_dict()
    assert labels == {
        "2025-06-01T12:30:00": QWEN3B,  # untouched
        "2025-06-02T00:00:00": "Granite-4.0-350M",
        "2025-06-03T00:00:00": "My-Old-Model (retired)",  # unknown: v1 label kept
    }
    granite = store.list_evaluations().set_index("created_at").loc["2025-06-02T00:00:00"]
    assert f'v1 model "{GRANITE}"' in granite["note"]
    assert import_legacy(store, results).evaluations_imported == 0


def test_import_legacy_evals_merges_v1_reruns(store, tmp_path):
    """Regression: v1 logged the same evaluation again on every Run click
    (cache hits included); each copy became its own history line."""
    results = tmp_path / "r"
    for stamp in ("2025-01-01T10:00:00", "2025-01-01T10:00:41", "2025-01-01T10:02:13"):
        v1_log_eval(results, v1_eval_entry("r1", timestamp=stamp, accuracy=0.8167))
    # Same run, different numbers (gold edited between runs): a real result.
    v1_log_eval(results, v1_eval_entry("r1", timestamp="2025-01-02T00:00:00", accuracy=0.9))
    # Without a run id the same numbers may be different data: not merged.
    for stamp in ("2025-01-03T00:00:00", "2025-01-04T00:00:00"):
        v1_log_eval(results, v1_eval_entry("", timestamp=stamp))

    report = import_legacy(store, results)
    assert report.evaluations_imported == 4
    assert len(report.skipped) == 1
    assert "2 rows repeat an earlier evaluation" in report.skipped[0]
    history = store.list_evaluations()
    assert sorted(history["created_at"]) == [
        "2025-01-01T10:00:00",  # the first time is kept
        "2025-01-02T00:00:00",
        "2025-01-03T00:00:00",
        "2025-01-04T00:00:00",
    ]

    # v1 still in use, re-logging again; a later import stays idempotent.
    v1_log_eval(results, v1_eval_entry("r1", timestamp="2025-02-01T00:00:00", accuracy=0.8167))
    again = import_legacy(store, results)
    assert again.evaluations_imported == 0
    assert "3 rows repeat" in again.skipped[0]
    assert len(store.list_evaluations()) == 4
