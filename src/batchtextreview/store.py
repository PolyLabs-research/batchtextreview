"""SQLite results store: cached outputs, run manifests, evaluation history,
saved presets and small UI settings - one file, ``results/batchtextreview.db``.

Why SQLite rather than one parquet per run (version 1): writes are
incremental (version 1 rewrote the whole parquet after every batch - O(n^2)
on a large corpus), atomic, and safe with several readers plus one writer
(WAL mode).

Concurrency model: every public method opens its own short-lived connection
(``sqlite3.connect(path, timeout=30)``), so a Store instance can be shared
across threads (Streamlit sessions + the job worker). Each connection sets
``PRAGMA journal_mode=WAL``, ``synchronous=NORMAL``, ``foreign_keys=ON``.
Timestamps are ISO-8601 local time, seconds precision.

Schema changes are additive: opening a database written by an older
version adds the missing columns (see :data:`SCHEMA_VERSION`).
"""

from __future__ import annotations

import csv
import io
import json
import logging
import math
import sqlite3
import time
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from batchtextreview.engine import RunConfig
    from batchtextreview.presets import Preset

log = logging.getLogger(__name__)

SCHEMA_VERSION = 3
"""1: the first release. 2: evaluation provenance (selection, rows_per_s,
elapsed_s, app_version, template_edited, preset_title). 3: outputs.hit_limit
(the output reached max_new_tokens)."""

SCHEMA = """
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
    -- schema 3: 1 / 0 = the output did / didn't reach max_new_tokens;
    -- NULL = unknown (saved by an older version or imported from version 1)
    hit_limit  INTEGER,
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
    note                TEXT,
    -- schema 2: provenance
    selection           TEXT,      -- which rows ran, e.g. "First 100 of 2,000 rows"
    rows_per_s          REAL,      -- generation throughput of the run
    elapsed_s           REAL,
    app_version         TEXT,
    template_edited     INTEGER,   -- 1 / 0, NULL when unknown
    preset_title        TEXT
);
-- One history line per (configuration, data, text column, gold column, label
-- set): re-scoring the same thing updates it instead of duplicating it
-- (matched in code with NULL-safe IS comparisons; this index just makes that
-- fast).
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

_TIMEOUT_S = 30.0
_CHUNK = 500
"""Keys per ``IN (...)`` query: well under SQLite's oldest bound-parameter
limit (999), with room for the run id."""

RUN_COLUMNS: tuple[str, ...] = (
    "run_id",
    "model_label",
    "model_id",
    "precision",
    "max_new_tokens",
    "max_input_chars",
    "n_outputs",
    "source",
    "created_at",
    "updated_at",
    "template",
)
"""Columns of :meth:`Store.list_runs`, in order."""

EVALUATION_COLUMNS: tuple[str, ...] = (
    "id",
    "created_at",
    "run_id",
    "model_id",
    "model_label",
    "precision",
    "preset",
    "var",
    "labels",
    "dataset",
    "dataset_fingerprint",
    "text_column",
    "gold_column",
    "n_rows",
    "n_scored",
    "n_correct",
    "accuracy",
    "macro_f1",
    "kappa",
    "n_invalid",
    "n_excluded",
    "n_gold_unscorable",
    "source",
    "note",
    "selection",
    "rows_per_s",
    "elapsed_s",
    "app_version",
    "template_edited",
    "preset_title",
)
"""Columns of the ``evaluations`` table (and :meth:`Store.list_evaluations`)."""

_EVAL_ADDED_COLUMNS: tuple[tuple[str, str], ...] = (
    ("selection", "TEXT"),
    ("rows_per_s", "REAL"),
    ("elapsed_s", "REAL"),
    ("app_version", "TEXT"),
    ("template_edited", "INTEGER"),
    ("preset_title", "TEXT"),
)
"""Columns added to ``evaluations`` after schema 1, in table order: a
database from an older version gets them on open."""
_OUTPUT_ADDED_COLUMNS: tuple[tuple[str, str], ...] = (("hit_limit", "INTEGER"),)
"""Columns added to ``outputs`` after schema 2 (same rule)."""

_EVAL_WRITABLE = frozenset(EVALUATION_COLUMNS) - {"id"}
_EVAL_KEEP_UNLESS_GIVEN = frozenset(
    {"selection", "rows_per_s", "elapsed_s", "app_version", "template_edited", "preset_title"}
)
"""Provenance columns that updating a line in place only overwrites with a
value: re-scoring a fully cached run has no throughput to report, and not
every caller knows the preset title or the row selection."""
_RUN_INT_COLUMNS = ("max_new_tokens", "max_input_chars", "n_outputs")

LEGACY_PRECISION = "4bit"
"""Version 1 always loaded models in 4-bit."""
LEGACY_MAX_INPUT_CHARS = 8000
"""Version 1's default truncation, for manifests that predate the field."""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _mtime_iso(path: Path) -> str | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
    except OSError:
        return None


def _iso(value: Any) -> str | None:
    """A timestamp normalised to ISO-8601 seconds, or None if unparseable."""
    if _missing(value):
        return None
    try:
        return datetime.fromisoformat(str(value).strip()).isoformat(timespec="seconds")
    except ValueError:
        return None


def _missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):  # list-likes
        return False


def _sql_value(value: Any) -> Any:
    """Coerce a record value into something sqlite3 can bind.

    numpy scalars (``np.int64`` is not an ``int``) become Python scalars,
    NaN / pd.NA become NULL, containers become JSON.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (str, int, bytes)):
        return value
    if isinstance(value, float):
        return None if math.isnan(value) else value
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(list(value) if isinstance(value, tuple) else value, ensure_ascii=False)
    if _missing(value):
        return None
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _sql_value(item())
        except (TypeError, ValueError):
            pass
    return str(value)


def _encode_labels(labels: Any) -> str | None:
    """Label set -> JSON list text (order kept), or None for exact-match."""
    if isinstance(labels, str):
        text = labels.strip()
        if not text:
            return None
        try:
            labels = json.loads(text)
        except ValueError:
            from batchtextreview.presets import parse_label_list

            labels = parse_label_list(text)
    if labels is None:
        return None
    if isinstance(labels, (set, frozenset)):
        labels = sorted(labels, key=str)
    elif (
        isinstance(labels, Iterable)
        and not isinstance(labels, bytes)
        and getattr(labels, "ndim", 1) != 0  # a 0-d numpy array is a scalar
    ):
        # list, tuple, numpy array, Series / Index (``df[col].unique()``),
        # dict keys, generator...: str() of those would be one junk label.
        labels = list(labels)
    else:
        if _missing(labels):
            return None
        labels = [labels]
    items = [str(_sql_value(x)) for x in labels if not _missing(x)]
    return json.dumps(items, ensure_ascii=False) if items else None


def _decode_labels(text: Any) -> list[str] | None:
    if _missing(text) or not str(text).strip():
        return None
    try:
        value = json.loads(text)
    except ValueError:
        from batchtextreview.presets import parse_label_list

        return parse_label_list(str(text)) or None
    if value is None:
        return None
    if isinstance(value, list):
        return [str(x) for x in value]
    return [str(value)]


@dataclass
class LegacyScan:
    """What version-1 artefacts exist in a results folder."""

    results_dir: Path
    manifests: list[Path] = field(default_factory=list)
    """``*.json`` files whose stem has a sibling ``*.parquet``."""
    orphan_checkpoints: list[Path] = field(default_factory=list)
    """``*.parquet`` without a manifest - can't be mapped, reported only."""
    evals_csv: Path | None = None

    @property
    def found(self) -> bool:
        return bool(self.manifests or self.evals_csv)


@dataclass
class LegacyImportReport:
    runs_imported: int = 0
    outputs_imported: int = 0
    evaluations_imported: int = 0
    skipped: list[str] = field(default_factory=list)
    """Human-readable reasons, one per skipped file/row."""


class Store:
    def __init__(self, path: str | Path) -> None:
        """Open (creating parent dirs, schema, and meta.schema_version).

        A database from an older version is upgraded in place (missing
        columns added, schema_version bumped); one from a newer version is
        refused with RuntimeError and left untouched."""
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            # One script, one transaction: concurrent first opens can't
            # interleave half-created schemas.
            conn.executescript(
                "BEGIN IMMEDIATE;\n"
                + SCHEMA
                + "\nINSERT OR IGNORE INTO meta (key, value) "
                + f"VALUES ('schema_version', '{SCHEMA_VERSION}');\nCOMMIT;"
            )
            row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
            missing = _missing_columns(conn)
        version = _version(row)
        if version > SCHEMA_VERSION:
            raise RuntimeError(
                f"{self.path} was written by a newer version of Batch Text Review "
                f"(schema {version}, this version reads {SCHEMA_VERSION}). Upgrade, "
                "or point BTR_RESULTS_DIR at another folder."
            )
        if version < SCHEMA_VERSION or missing:
            self._migrate()

    path: Path

    def __repr__(self) -> str:
        return f"Store({str(self.path)!r})"

    @contextmanager
    def _connect(self, write: bool = False) -> Iterator[sqlite3.Connection]:
        """A fresh connection, closed on exit.

        ``write=True`` wraps the block in ``BEGIN IMMEDIATE`` ... ``COMMIT``
        (rolled back on error). Taking the write lock up front means a
        read-then-write block (log_evaluation's identity check) can neither
        race another writer nor fail with SQLITE_BUSY halfway through - it
        simply waits up to the busy timeout for its turn.
        """
        conn = sqlite3.connect(self.path, timeout=_TIMEOUT_S, isolation_level=None)
        try:
            _enable_wal(conn)
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            if not write:
                yield conn
                return
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")
        finally:
            conn.close()

    def get_meta(self, key: str) -> str | None:
        """A raw value from the ``meta`` table (e.g. ``schema_version``,
        ``legacy_imported_at``), or None."""
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else row[0]

    def _migrate(self) -> None:
        """Bring an older database up to :data:`SCHEMA_VERSION`."""
        with self._connect(write=True) as conn:
            # Checked again under the write lock: another process (a second
            # browser session's worker, the CLI) may have just done it.
            row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
            version = _version(row)
            added = _missing_columns(conn)
            for table, name, kind in added:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {kind}")
            if version < SCHEMA_VERSION:
                conn.execute(
                    "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)",
                    (str(SCHEMA_VERSION),),
                )
        if added or version < SCHEMA_VERSION:
            log.info(
                "Upgraded %s from schema %s to %s (added %s).",
                self.path,
                version,
                max(version, SCHEMA_VERSION),
                ", ".join(f"{table}.{name}" for table, name, _ in added) or "no columns",
            )

    def _set_meta(self, key: str, value: str) -> None:
        with self._connect(write=True) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, str(value))
            )

    # ---- runs & outputs -------------------------------------------------
    def upsert_run(self, config: RunConfig, source: str = "app") -> None:
        """Insert the manifest for ``config.run_id`` or bump updated_at."""
        now = _now()
        with self._connect(write=True) as conn:
            # The display label follows the catalogue (labels may be edited);
            # everything that defines the run is in the id and never changes.
            conn.execute(
                """
                INSERT INTO runs (run_id, model_id, model_label, precision, template,
                                  max_new_tokens, max_input_chars, source,
                                  created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    updated_at  = excluded.updated_at,
                    model_label = excluded.model_label
                """,
                (*_run_values(config), source or "app", now, now),
            )

    def get_outputs(self, run_id: str, keys: Iterable[str] | None = None) -> dict[str, str]:
        """Stored outputs for a run, optionally limited to ``keys``. Queries
        in chunks of <= 500 keys (SQLite parameter limits)."""
        if keys is None:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT key, output FROM outputs WHERE run_id = ?", (run_id,)
                ).fetchall()
            return dict(rows)
        wanted = list(dict.fromkeys(str(k) for k in keys if k is not None))
        if not wanted:
            return {}
        found: dict[str, str] = {}
        with self._connect() as conn:
            conn.execute("BEGIN")  # one snapshot across all chunks
            try:
                for start in range(0, len(wanted), _CHUNK):
                    chunk = wanted[start : start + _CHUNK]
                    marks = ",".join("?" * len(chunk))
                    found.update(
                        conn.execute(
                            f"SELECT key, output FROM outputs WHERE run_id = ? AND key IN ({marks})",
                            (run_id, *chunk),
                        ).fetchall()
                    )
            finally:
                if conn.in_transaction:
                    conn.execute("COMMIT")
        return found

    def put_outputs(
        self, run_id: str, items: Mapping[str, str], hit_limit: Iterable[str] = ()
    ) -> None:
        """Insert-or-replace outputs in one transaction. The run row must
        exist (call upsert_run first) - raise KeyError otherwise.

        ``hit_limit``: the keys whose output reached ``max_new_tokens`` (and
        may be cut off); the other outputs are stored as not reaching it.
        """
        now = _now()
        flagged = {str(k) for k in hit_limit}
        rows = []
        for key, output in items.items():
            if output is None:
                raise TypeError(f"Output for key {key!r} is None - store '' for an empty output.")
            rows.append((run_id, str(key), str(output), now, int(str(key) in flagged)))
        with self._connect(write=True) as conn:
            if conn.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_id,)).fetchone() is None:
                raise KeyError(run_id)
            conn.executemany(
                "INSERT OR REPLACE INTO outputs (run_id, key, output, created_at, hit_limit) "
                "VALUES (?, ?, ?, ?, ?)",
                rows,
            )

    def get_hit_limit_keys(self, run_id: str, keys: Iterable[str]) -> set[str]:
        """Those of ``keys`` whose stored output reached ``max_new_tokens``
        when it was generated. Outputs saved before schema 3 (or imported
        from version 1) don't say, and are left out."""
        wanted = list(dict.fromkeys(str(k) for k in keys if k is not None))
        found: set[str] = set()
        if not wanted:
            return found
        with self._connect() as conn:
            conn.execute("BEGIN")  # one snapshot across all chunks
            try:
                for start in range(0, len(wanted), _CHUNK):
                    chunk = wanted[start : start + _CHUNK]
                    marks = ",".join("?" * len(chunk))
                    found.update(
                        key
                        for (key,) in conn.execute(
                            "SELECT key FROM outputs WHERE run_id = ? AND hit_limit = 1 "
                            f"AND key IN ({marks})",
                            (run_id, *chunk),
                        )
                    )
            finally:
                if conn.in_transaction:
                    conn.execute("COMMIT")
        return found

    def count_outputs(self, run_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM outputs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return int(row[0])

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """The run's manifest row plus ``n_outputs``, or None."""
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT r.*, (SELECT COUNT(*) FROM outputs o WHERE o.run_id = r.run_id) "
                "AS n_outputs FROM runs r WHERE r.run_id = ?",
                (run_id,),
            ).fetchone()
        return None if row is None else dict(row)

    def list_runs(self) -> pd.DataFrame:
        """All runs with an ``n_outputs`` column, most recently updated
        first. Columns: run_id, model_label, model_id, precision,
        max_new_tokens, max_input_chars, n_outputs, source, created_at,
        updated_at, template."""
        cols = ", ".join(f"r.{c}" for c in RUN_COLUMNS if c != "n_outputs")
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT {cols}, COALESCE(o.n, 0) AS n_outputs
                FROM runs r
                LEFT JOIN (SELECT run_id, COUNT(*) AS n FROM outputs GROUP BY run_id) o
                       ON o.run_id = r.run_id
                ORDER BY r.updated_at DESC, r.rowid DESC
                """
            ).fetchall()
        order = [c for c in RUN_COLUMNS if c != "n_outputs"] + ["n_outputs"]
        frame = pd.DataFrame.from_records(rows, columns=order)[list(RUN_COLUMNS)]
        return frame.astype(dict.fromkeys(_RUN_INT_COLUMNS, "int64"))

    def delete_run(self, run_id: str) -> int:
        """Delete a run and its outputs (evaluations are kept). Returns the
        number of outputs deleted."""
        with self._connect(write=True) as conn:
            (n,) = conn.execute(
                "SELECT COUNT(*) FROM outputs WHERE run_id = ?", (run_id,)
            ).fetchone()
            # Outputs go with the run (ON DELETE CASCADE).
            conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
        return int(n)

    # ---- evaluations ----------------------------------------------------
    def log_evaluation(self, record: Mapping[str, Any]) -> int:
        """Insert one history line (keys = ``evaluations`` columns except
        id; ``labels`` may be a list -> stored as JSON; ``created_at``
        defaults to now). If a row with the same identity (run_id,
        dataset_fingerprint, text_column, gold_column, labels - compared
        NULL-safely with ``IS``) exists and run_id and dataset_fingerprint
        are both set, update it in place instead. Returns the row id.
        Unknown keys are ignored.

        An update replaces only the columns the record carries, and a
        provenance column (selection, rows_per_s, elapsed_s, app_version,
        template_edited, preset_title) only with a value: None there keeps
        what the line already has.

        text_column is part of the identity because neither run_id nor the
        dataset fingerprint depends on it: scoring the TITLE and then the
        BODY column of one file with one configuration are two results.
        """
        values: dict[str, Any] = {}
        for key, value in record.items():
            if key not in _EVAL_WRITABLE:
                continue
            values[key] = _encode_labels(value) if key == "labels" else _sql_value(value)
        values["created_at"] = _iso(values.get("created_at")) or values.get("created_at") or _now()
        if values.get("source") is None:
            values["source"] = "app"

        with self._connect(write=True) as conn:
            existing = None
            if values.get("run_id") is not None and values.get("dataset_fingerprint") is not None:
                existing = conn.execute(
                    """
                    SELECT id FROM evaluations
                    WHERE run_id IS ? AND dataset_fingerprint IS ?
                      AND text_column IS ? AND gold_column IS ? AND labels IS ?
                    ORDER BY id DESC LIMIT 1
                    """,
                    (
                        values["run_id"],
                        values["dataset_fingerprint"],
                        values.get("text_column"),
                        values.get("gold_column"),
                        values.get("labels"),
                    ),
                ).fetchone()
            if existing is not None:
                # Only the columns the new record carries are replaced, so
                # anything set on the line separately (e.g. a note) survives;
                # so does provenance the new record doesn't know.
                cols = list(values)
                assignments = ", ".join(
                    f"{c} = COALESCE(?, {c})" if c in _EVAL_KEEP_UNLESS_GIVEN else f"{c} = ?"
                    for c in cols
                )
                conn.execute(
                    f"UPDATE evaluations SET {assignments} WHERE id = ?",
                    (*(values[c] for c in cols), existing[0]),
                )
                return int(existing[0])
            cols = list(values)
            cur = conn.execute(
                f"INSERT INTO evaluations ({', '.join(cols)}) "
                f"VALUES ({', '.join('?' * len(cols))})",
                tuple(values[c] for c in cols),
            )
            return int(cur.lastrowid or 0)

    def list_evaluations(self) -> pd.DataFrame:
        """History, newest first, ``labels`` decoded back to a list (or
        None) and ``template_edited`` to True / False / None. Empty frame
        with the right columns when there is none. Lines logged before
        schema 2 have no provenance (NULL / NaN)."""
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {', '.join(EVALUATION_COLUMNS)} FROM evaluations "
                "ORDER BY created_at DESC, id DESC"
            ).fetchall()
        frame = pd.DataFrame.from_records(rows, columns=list(EVALUATION_COLUMNS))
        if not frame.empty:
            frame["labels"] = pd.Series(
                [_decode_labels(v) for v in frame["labels"]], index=frame.index, dtype=object
            )
            frame["template_edited"] = pd.Series(
                [None if _missing(v) else bool(v) for v in frame["template_edited"]],
                index=frame.index,
                dtype=object,
            )
        return frame

    def delete_evaluation(self, eval_id: int) -> bool:
        with self._connect(write=True) as conn:
            cur = conn.execute("DELETE FROM evaluations WHERE id = ?", (int(eval_id),))
        return cur.rowcount > 0

    # ---- custom presets -------------------------------------------------
    def list_presets(self) -> list[Preset]:
        """Saved custom presets (built via presets.make_custom_preset),
        sorted by title."""
        import dataclasses

        from batchtextreview.presets import make_custom_preset

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT key, title, template, max_new_tokens, labels, var_hint FROM presets"
            ).fetchall()
        out: list[Preset] = []
        for key, title, template, max_new_tokens, labels, var_hint in rows:
            try:
                preset = make_custom_preset(
                    title, template, max_new_tokens, _decode_labels(labels), var_hint
                )
            except (TypeError, ValueError):
                # A row that no longer validates (e.g. its name became a
                # built-in) is left in the table but not offered.
                continue
            if preset.key != key:
                preset = dataclasses.replace(preset, key=key)
            out.append(preset)
        return sorted(out, key=lambda p: (p.title.casefold(), p.title))

    def save_preset(self, preset: Preset) -> None:
        """Insert or replace by key."""
        from batchtextreview.presets import make_custom_preset

        labels = list(preset.labels) if preset.labels else None
        # Validate the same way list_presets rebuilds it, so a saved preset
        # can always be loaded again (raises ValueError with a UI message).
        make_custom_preset(
            preset.title, preset.template, preset.max_new_tokens, labels, preset.var_hint
        )
        now = _now()
        with self._connect(write=True) as conn:
            conn.execute(
                """
                INSERT INTO presets (key, title, template, max_new_tokens, labels, var_hint,
                                     created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    title = excluded.title,
                    template = excluded.template,
                    max_new_tokens = excluded.max_new_tokens,
                    labels = excluded.labels,
                    var_hint = excluded.var_hint,
                    updated_at = excluded.updated_at
                """,
                (
                    preset.key,
                    preset.title.strip(),
                    preset.template,
                    int(preset.max_new_tokens),
                    _encode_labels(labels),
                    preset.var_hint,
                    now,
                    now,
                ),
            )

    def delete_preset(self, key: str) -> bool:
        with self._connect(write=True) as conn:
            cur = conn.execute("DELETE FROM presets WHERE key = ?", (key,))
        return cur.rowcount > 0

    # ---- settings -------------------------------------------------------
    def get_setting(self, key: str, default: Any = None) -> Any:
        """JSON-decoded value, or ``default``."""
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        if row is None or row[0] is None:
            return default
        try:
            return json.loads(row[0])
        except ValueError:
            return default

    def set_setting(self, key: str, value: Any) -> None:
        # default=str: a Path (say, the last folder used) is stored as text
        # rather than failing a UI action.
        encoded = json.dumps(value, ensure_ascii=False, default=str)
        with self._connect(write=True) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, encoded)
            )

    # ---- housekeeping ---------------------------------------------------
    def stats(self) -> dict[str, int]:
        """{"runs": n, "outputs": n, "evaluations": n, "presets": n,
        "size_bytes": file size of the db (+wal if present)}."""
        out: dict[str, int] = {}
        with self._connect() as conn:
            for table in ("runs", "outputs", "evaluations", "presets"):
                out[table] = int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        size = 0
        for path in (self.path, self.path.with_name(self.path.name + "-wal")):
            with suppress(OSError):
                size += path.stat().st_size
        out["size_bytes"] = size
        return out


def _enable_wal(conn: sqlite3.Connection) -> None:
    """``PRAGMA journal_mode=WAL``, waiting out a concurrent switch.

    WAL mode is stored in the file, so this only does work on a file's
    first open (or one written by a plain sqlite3 connection). That switch
    needs an exclusive lock, and SQLite reports "database is locked" at once
    rather than using the busy timeout - several first opens at the same
    moment (two browser sessions and the job worker) would fail. Retry for
    up to the timeout instead.
    """
    deadline = time.monotonic() + _TIMEOUT_S
    while True:
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            return
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc) or time.monotonic() >= deadline:
                raise
            time.sleep(0.02)


def _version(row: Any) -> int:
    """The schema_version meta value (a missing or garbled one is taken as
    current: the schema script just made sure the tables exist)."""
    try:
        return int(row[0])
    except (TypeError, ValueError, IndexError):
        return SCHEMA_VERSION


def _missing_columns(conn: sqlite3.Connection) -> list[tuple[str, str, str]]:
    """(table, column, type) of every added column the database lacks."""
    missing: list[tuple[str, str, str]] = []
    for table, added in (("evaluations", _EVAL_ADDED_COLUMNS), ("outputs", _OUTPUT_ADDED_COLUMNS)):
        present = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        missing += [(table, name, kind) for name, kind in added if name not in present]
    return missing


def _run_values(config: RunConfig) -> tuple[Any, ...]:
    """(run_id, model_id, model_label, precision, template, max_new_tokens,
    max_input_chars) for an INSERT into ``runs``."""
    return (
        config.run_id,
        config.model_id,
        config.model_label,
        config.precision,
        config.template,
        int(config.max_new_tokens),
        int(config.max_input_chars),
    )


# --------------------------------------------------------------------------
# Version-1 import
# --------------------------------------------------------------------------


def scan_legacy(results_dir: str | Path) -> LegacyScan:
    """Find version-1 artefacts: ``{run_id}.parquet`` + ``{run_id}.json``
    manifests and ``evals.csv`` directly in ``results_dir``."""
    root = Path(results_dir).expanduser()
    scan = LegacyScan(results_dir=root)
    if not root.is_dir():
        return scan
    try:
        files = sorted(p for p in root.iterdir() if p.is_file())
    except OSError:
        return scan
    names = {p.name for p in files}
    for path in files:
        suffix = path.suffix.lower()
        if suffix == ".json" and f"{path.stem}.parquet" in names:
            scan.manifests.append(path)
        elif suffix == ".parquet" and f"{path.stem}.json" not in names:
            scan.orphan_checkpoints.append(path)
        elif path.name.lower() == "evals.csv":
            scan.evals_csv = path
    return scan


def import_legacy(store: Store, results_dir: str | Path) -> LegacyImportReport:
    """Import version-1 checkpoints and evaluation history. Idempotent.

    For each manifest (``{"run_id", "model", "template", "max_new_tokens",
    "max_input_chars", "created"}``):

    * map ``model`` (a v1 label) to a repo with
      ``models.LEGACY_LABEL_TO_REPO`` (fall back to ``models.find_model``;
      unknown labels are skipped with a reason);
    * precision is "4bit" (v1 always loaded 4-bit);
    * build the v2 RunConfig and ``upsert_run(..., source="legacy")``
      (keeping the manifest's created time);
    * read the parquet's ``key``/``output`` columns and insert them with
      INSERT OR IGNORE (never overwrite outputs produced by v2);
    * remember v1 run_id -> v2 run_id.

    Then ``evals.csv`` (columns timestamp, model, preset, var, rows_scored,
    accuracy, invalid_outputs, run_id - tolerate missing/extra columns):
    each row becomes an evaluation with source "legacy", ``run_id`` mapped to
    v2 when known (else NULL), model_id and the current model_label from the
    label map (the v1 label is kept for unknown models), created_at from
    timestamp, n_scored from rows_scored. A row wider than the header (a
    later v1 build appending under an older header) is read in the final v1
    column order when it plausibly fits it. Rows repeating an earlier one
    with the same v1 run_id and numbers (v1 re-logged on every rerun) are
    merged into the first. Skip rows already imported (same created_at +
    model - model_id, or model_label if unknown - + preset + accuracy with
    source legacy).

    Unreadable files are reported in ``skipped``, never raised. Files are
    never modified or deleted. Records ``legacy_imported_at`` in meta.
    """
    scan = scan_legacy(results_dir)
    report = LegacyImportReport()
    id_map: dict[str, str] = {}
    for manifest in scan.manifests:
        _import_checkpoint(store, manifest, report, id_map)
    for orphan in scan.orphan_checkpoints:
        report.skipped.append(
            f"{orphan.name}: no {orphan.stem}.json manifest next to it, so the model and "
            "prompt that produced it are unknown - not imported."
        )
    if scan.evals_csv is not None:
        _import_evals(store, scan.evals_csv, report, id_map)
    store._set_meta("legacy_imported_at", _now())
    log.info(
        "Version-1 import from %s: %d runs, %d outputs, %d history entries, %d skipped.",
        scan.results_dir,
        report.runs_imported,
        report.outputs_imported,
        report.evaluations_imported,
        len(report.skipped),
    )
    for reason in report.skipped:
        log.warning("Version-1 import from %s skipped %s", scan.results_dir, reason)
    return report


def _short_error(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".strip()
    return text if len(text) <= 200 else text[:199] + "…"


def _legacy_model(label: str) -> tuple[str, str] | None:
    """(repo, current display label) for a v1 model label, or None."""
    from batchtextreview import models

    repo = models.LEGACY_LABEL_TO_REPO.get(label)
    spec = models.find_model(repo or label) if (repo or label) else None
    if repo is not None:
        return repo, (spec.label if spec is not None else label)
    if spec is not None:
        return spec.repo, spec.label
    return None


def _import_checkpoint(
    store: Store, manifest: Path, report: LegacyImportReport, id_map: dict[str, str]
) -> None:
    from batchtextreview.engine import RunConfig

    try:
        meta = json.loads(manifest.read_text(encoding="utf-8-sig"))
        if not isinstance(meta, dict):
            raise ValueError("not a JSON object")
    except (OSError, ValueError) as exc:
        report.skipped.append(f"{manifest.name}: unreadable manifest ({_short_error(exc)}).")
        return

    label = str(meta.get("model") or "").strip()
    model = _legacy_model(label)
    if model is None:
        report.skipped.append(
            f'{manifest.name}: unknown model "{label}" - no matching model in this version, '
            "not imported."
        )
        return
    repo, model_label = model
    template = meta.get("template")
    try:
        if not isinstance(template, str) or not template:
            raise ValueError("no template")
        config = RunConfig(
            model_id=repo,
            model_label=model_label,
            precision=LEGACY_PRECISION,
            template=template,
            max_new_tokens=int(meta["max_new_tokens"]),
            max_input_chars=int(meta.get("max_input_chars") or LEGACY_MAX_INPUT_CHARS),
        )
    except (KeyError, TypeError, ValueError) as exc:
        report.skipped.append(
            f"{manifest.name}: manifest is missing a field or has an invalid value "
            f"({_short_error(exc)}) - not imported."
        )
        return
    # The v2 id is a fact about the configuration, so history rows can link
    # to it even if the checkpoint itself turns out to be unreadable.
    id_map[manifest.stem] = config.run_id
    if meta.get("run_id"):
        id_map[str(meta["run_id"])] = config.run_id

    parquet = manifest.with_suffix(".parquet")
    try:
        frame = pd.read_parquet(parquet, columns=["key", "output"])
    except Exception as exc:  # pyarrow raises a zoo of types for bad files
        report.skipped.append(f"{parquet.name}: unreadable checkpoint ({_short_error(exc)}).")
        return
    items: dict[str, str] = {}
    for key, output in zip(frame["key"].tolist(), frame["output"].tolist(), strict=True):
        if _missing(key) or _missing(output):
            continue
        items.setdefault(str(key), str(output))

    written = _mtime_iso(parquet) or _now()
    created = _iso(meta.get("created")) or written
    updated = max(created, written)
    with store._connect(write=True) as conn:
        cur = conn.execute(
            """
            INSERT OR IGNORE INTO runs (run_id, model_id, model_label, precision, template,
                                        max_new_tokens, max_input_chars, source,
                                        created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'legacy', ?, ?)
            """,
            (*_run_values(config), created, updated),
        )
        new_run = cur.rowcount > 0
        cur = conn.executemany(
            "INSERT OR IGNORE INTO outputs (run_id, key, output, created_at) VALUES (?, ?, ?, ?)",
            [(config.run_id, k, v, updated) for k, v in items.items()],
        )
        added = max(cur.rowcount, 0)
    if new_run or added:
        report.runs_imported += 1
    report.outputs_imported += added


def _decode_text(raw: bytes) -> str:
    # pandas wrote UTF-8; a round trip through Excel may have left a BOM or
    # cp1252. latin-1 never fails, as a last resort.
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")


def _to_float(value: Any) -> float | None:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _to_int(value: Any) -> int | None:
    number = _to_float(value)
    return int(number) if number is not None else None


_V1_EVAL_FIELDS: tuple[str, ...] = (
    "timestamp",
    "model",
    "preset",
    "var",
    "rows_scored",
    "accuracy",
    "invalid_outputs",
    "run_id",
)
"""An ``evals.csv`` row as the last version-1 build wrote it (log_eval's
dict order)."""


def _v1_drift_row(row: list[str]) -> dict[str, str] | None:
    """A row whose width differs from the header, read in the final
    version-1 layout - or None unless it plausibly is one.

    Version 1 wrote the header once, from whichever build created the file,
    and later builds appended wider rows under it: trusting only the header
    would drop all the newer history.
    """
    if len(row) != len(_V1_EVAL_FIELDS):
        return None
    rec = {k: v.strip() for k, v in zip(_V1_EVAL_FIELDS, row, strict=True)}
    accuracy = _to_float(rec["accuracy"])
    if _iso(rec["timestamp"]) is None or accuracy is None or not 0 <= accuracy <= 1:
        return None
    return rec


def _import_evals(
    store: Store, path: Path, report: LegacyImportReport, id_map: dict[str, str]
) -> None:
    try:
        text = _decode_text(path.read_bytes())
        rows = list(csv.reader(io.StringIO(text, newline="")))
    except (OSError, csv.Error) as exc:
        report.skipped.append(f"{path.name}: unreadable ({_short_error(exc)}).")
        return
    # Numbered before blank lines are dropped, so reasons point at real rows.
    numbered = [(n, r) for n, r in enumerate(rows, start=1) if any(c.strip() for c in r)]
    if not numbered:
        return
    header = [h.strip().lower() for h in numbered[0][1]]
    fallback_time = _mtime_iso(path) or _now()

    records: list[dict[str, Any]] = []
    seen_v1: set[tuple[Any, ...]] = set()
    repeats = 0
    for number, row in numbered[1:]:
        if [c.strip().lower() for c in row] == header:
            continue  # a repeated header from concatenated files
        if len(row) == len(header):
            rec = {k: v.strip() for k, v in zip(header, row, strict=True)}
        else:
            drifted = _v1_drift_row(row)
            if drifted is None:
                # Positions can't be trusted when the shape drifted between appends.
                report.skipped.append(
                    f"{path.name} row {number}: {len(row)} fields but the header has "
                    f"{len(header)} - not imported."
                )
                continue
            rec = drifted
        accuracy = _to_float(rec.get("accuracy"))
        if accuracy is None:
            report.skipped.append(f"{path.name} row {number}: no accuracy value - not imported.")
            continue
        stamp = rec.get("timestamp") or ""
        label = rec.get("model") or None
        v1_run = rec.get("run_id") or None
        n_scored = _to_int(rec.get("rows_scored"))
        n_invalid = _to_int(rec.get("invalid_outputs"))
        if v1_run:
            # Version 1 logged the evaluation again on every rerun, cache hits
            # included: keep one line, at its first time. Without a run id the
            # same numbers may well come from different data, so no merging.
            repeat = (
                v1_run,
                label,
                rec.get("preset"),
                rec.get("var"),
                n_scored,
                accuracy,
                n_invalid,
            )
            if repeat in seen_v1:
                repeats += 1
                continue
            seen_v1.add(repeat)
        # The current catalogue label, as on imported runs, so one model is
        # one series in the history charts; unknown models keep the v1 label.
        mapped = _legacy_model(label) if label else None
        model_id, model_label = mapped if mapped is not None else (None, label)
        note = "Imported from version 1 (evals.csv)"
        if label and label != model_label:
            note += f'; v1 model "{label}"'
        if v1_run:
            note += f"; v1 run id {v1_run}"
        records.append(
            {
                "created_at": _iso(stamp) or stamp or fallback_time,
                "run_id": id_map.get(v1_run) if v1_run else None,
                "model_id": model_id,
                "model_label": model_label,
                "precision": LEGACY_PRECISION,
                "preset": rec.get("preset") or None,
                "var": rec.get("var") or None,
                "n_scored": n_scored,
                "accuracy": accuracy,
                "n_invalid": n_invalid,
                "source": "legacy",
                "note": note,
            }
        )
    if repeats:
        rows_text = "1 row repeats" if repeats == 1 else f"{repeats} rows repeat"
        report.skipped.append(
            f"{path.name}: {rows_text} an earlier evaluation of the same run (version 1 "
            "logged it again on every rerun) - kept once, at its first time."
        )
    if not records:
        return

    cols = list(records[0])
    insert = f"INSERT INTO evaluations ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})"
    with store._connect(write=True) as conn:
        for rec in records:
            # Same model by repo id (stable if a display label is renamed),
            # or by label for models this version doesn't know.
            seen = conn.execute(
                """
                SELECT id, run_id FROM evaluations
                WHERE source = 'legacy' AND created_at IS ?
                  AND COALESCE(model_id, model_label) IS ?
                  AND preset IS ? AND accuracy IS ?
                ORDER BY id LIMIT 1
                """,
                (
                    rec["created_at"],
                    rec["model_id"] or rec["model_label"],
                    rec["preset"],
                    rec["accuracy"],
                ),
            ).fetchone()
            if seen is None:
                conn.execute(insert, tuple(rec[c] for c in cols))
                report.evaluations_imported += 1
            elif seen[1] is None and rec["run_id"] is not None:
                # Imported before its checkpoint was: link it up now.
                conn.execute(
                    "UPDATE evaluations SET run_id = ? WHERE id = ?", (rec["run_id"], seen[0])
                )
