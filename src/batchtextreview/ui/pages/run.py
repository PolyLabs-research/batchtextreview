"""Run page: load a table, choose a task and prompt, check the plan, run.

Layout: two columns. Left: "1 · Data" and "2 · Task" cards. Right: the run
plan card (model, rows, advanced settings, what will happen, the run button
or - while a job runs - live progress) and a "Last run" card.

Sections depend on each other in both directions (the gold-column default
depends on the task's answers; the prompt preview depends on the selected
rows, which are chosen in the plan card), so the page creates its containers
in visual order first and fills them in dependency order:

    data (source, table, text/ID columns)  ->  task (preset, prompt, answers)
    -> plan settings (model, rows, advanced) -> data (gold column, checks)
    -> task (exact match, preview, presets)  -> plan summary + run button

Every widget is keyed and uses ``persist_state="session"`` so the page comes
back exactly as it was after visiting another page. Values a widget starts
with are written to session state before it is created (never passed as
``value=``), so callbacks can change them without Streamlit warnings.
"""

from __future__ import annotations

import contextlib
import hashlib
import math
import os
import re
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

import pandas as pd
import streamlit as st

from batchtextreview import data, engine, models, presets, prompting
from batchtextreview.backends.base import PRECISION_LABELS, resolve_precision
from batchtextreview.labels import is_missing, normalise_label
from batchtextreview.ui import analysis, components, nav, resources, state
from batchtextreview.ui import format as fmt

if TYPE_CHECKING:
    from streamlit.delta_generator import DeltaGenerator

    from batchtextreview.backends.base import DeviceInfo
    from batchtextreview.config import Settings
    from batchtextreview.jobs import Job
    from batchtextreview.store import Store

SAMPLE_FILE = Path("examples") / "sample_feedback.csv"
RANDOM_SEED = 42
LARGE_TABLE_ROWS = 500
"""Above this, the row scope defaults to "First N" so a first run is quick."""
DEFAULT_N = 20
PREVIEW_ROWS = 200
MAX_CATEGORY_SUGGESTIONS = 30
GOLD_MATCH_WARNING = 0.5
NUMERIC_TEXT_SHARE = 0.9
"""A text column whose values are this numeric looks like IDs or scores."""
SHORT_TEXT_CHARS = 5
MAX_SUBFOLDERS = 6
CUSTOM_PRESET_KEY = "Custom"
CUSTOM_SCAFFOLD = (
    "Describe what to do with the text below, and how to answer (for example: "
    "reply with one word).\n\nTEXT:\n{text}"
)
"""Starting prompt of the built-in "Custom prompt" preset (whose own template
is just ``{text}``): a shape to fill in rather than an empty box."""

# ---- widget keys -------------------------------------------------------------
SOURCE_KEY = "run.source"
UPLOAD_KEY = "run.upload"
FOLDER_KEY = "run.folder"
FOLDER_FILE_KEY = "run.folder_file"
CATEGORY_KEY = "run.category"
MODEL_KEY = "run.model"
CUSTOM_MODEL_KEY = "run.custom_model"
BATCH_KEY = "run.batch_size"
ORDER_KEY = "run.order"
PRECISION_KEY = "run.precision"
MAX_CHARS_KEY = "run.max_input_chars"
PRESET_NAME_KEY = "run.preset_name"
RUN_BUTTON_KEY = "run.go"
RUN_SHORTCUT = "Ctrl+Shift+Enter"

# ---- other session keys (page-private) -------------------------------------
_TABLES = "btr.run.tables"
"""source -> _Dataset most recently loaded from that source."""
_MEMO = "btr.run.memo"
"""slot -> (key, value): one-entry memo per slot for per-rerun work."""
_SAVED_FOLDER = "btr.run.saved_folder"
_PRESET_MSG = "btr.run.preset_msg"
_DELETE_MSG = "btr.run.delete_msg"
_CANCELLING = "btr.run.cancelling"
_DELETED = "btr.run.deleted_preset"

NO_COLUMN = "\u2063none"
"""Selectbox value meaning "no column" (a None option would show as an empty
placeholder). Starts with an invisible separator so no real header matches."""
_SAVED_CATEGORY = "Saved"
_CUSTOM_MODEL = "__custom__"
_SOURCE_LABELS = {"upload": "Upload", "path": "From a folder", "sample": "Sample data"}
_SCOPE_LABELS = {"all": "All", "first": "First rows", "random": "Random sample"}
_PRECISION_CHOICES = ("auto", "4bit", "fp16", "fp32")
API_CONSENT_KEY = "run.api_consent"
"""Prefix of the per-provider "send these texts to <provider>" checkbox (one
per browser session: never remembered across sessions)."""
CHARS_PER_TOKEN = 4
"""Rough characters per token for the Run page's cost estimate (English
text; other languages, and code, use more tokens)."""
_FORMAT_NAMES = {
    "csv": "CSV",
    "tsv": "TSV",
    "excel": "Excel",
    "parquet": "Parquet",
    "json": "JSON",
    "jsonl": "JSON Lines",
}
_ENCODING_NAMES = data.ENCODING_LABELS
_DELIMITER_NAMES = {",": "comma", ";": "semicolon", "\t": "tab", "|": "pipe"}
_EXCEL_SUFFIXES = (".xlsx", ".xlsm")
ORDER_SETTING = "run_order"
"""Store setting holding the processing order chosen last (any session)."""
ORDER_HELP = (
    "Longest first: the time estimate starts high and counts down, and a model that runs "
    "out of memory fails in the first batch. Shortest first: the first results appear sooner."
)


# =============================================================================
# Small value objects passed between the steps
# =============================================================================


@dataclass
class _Dataset:
    table: data.LoadedTable
    fingerprint: str
    origin: str
    """"upload", "path" or "sample"."""


@dataclass
class _DataStep:
    ds: _Dataset | None = None
    text_col: str | None = None
    id_col: str | None = None
    gold_col: str | None = None
    problems: list[str] = field(default_factory=list)
    """Block the run (the first one is the run button's reason)."""
    warnings: list[str] = field(default_factory=list)
    """Don't block the run, but deserve a look (e.g. a gold column that
    doesn't fit the task)."""
    waiting: bool = False
    """No table yet and nothing went wrong: the user hasn't chosen one."""
    gold_cell: DeltaGenerator | None = None
    """Where the gold-column selectbox goes (filled in a later step)."""
    checks: DeltaGenerator | None = None
    """Below the column mapping: gold check + preview (later step)."""

    @property
    def frame(self) -> pd.DataFrame | None:
        return self.ds.table.frame if self.ds else None


@dataclass
class _TaskStep:
    preset: presets.Preset
    template_raw: str
    edited: bool
    uses_var: bool
    var: str | None
    labels: list[str] | None
    output_kind: str
    problems: list[str]
    final_template: str | None
    rest: DeltaGenerator
    """Bottom of the task card (filled in a later step)."""


@dataclass
class _Selection:
    frame: pd.DataFrame
    """Selected rows, original index kept."""
    mode: str
    n: int
    description: str


@dataclass
class _PlanSettings:
    model: models.ModelSpec | None
    model_problem: str | None
    max_new_tokens: int
    batch_size: int
    precision: str
    max_input_chars: int
    order: str
    """Processing order: one of ``engine.ORDERS``."""


_NO_TABLE = "Load a table first (step 1)."
_NO_FILE_CHOSEN = "Choose a file in step 1."


class _NoTable(Exception):
    """Raised by a data source that has shown why there is no table (or that
    needs a choice first); ``problem`` becomes the run button's reason."""

    def __init__(self, problem: str = "Fix the file problem in step 1.") -> None:
        super().__init__(problem)
        self.problem = problem


# =============================================================================
# Page
# =============================================================================


def render() -> None:
    """The Run page."""
    settings = resources.get_settings()
    store = resources.get_store()
    device = resources.get_device_info()
    jobs = resources.get_job_manager()
    active = jobs.active()

    components.page_header(
        "Run", "Apply a prompt to every row of a table with a local or cloud model."
    )

    # Every element keeps its place whether or not a run is in progress: when
    # a run ends and the page reruns, an element tree that shifted would make
    # the browser rebuild the widgets below the shift - and throw away text
    # still being typed into the prompt editor (it is only sent to the app
    # when the box loses focus).
    left, right = st.columns([3, 2], gap="large")
    with left:
        notice = st.empty()
        if active is not None:
            notice.caption(
                ":material/info: A run is in progress. Changes here apply to your next run."
            )
        data_box = components.section("Data", step=1)
        task_box = components.section("Task", step=2)
    # Keyed so the right column stays in view while the left one scrolls
    # (ui/styles.py) - not during a run (the progress panel, keyed
    # btr_progress, can be taller than the window).
    with right, st.container(key="btr_plan"):
        with components.section("Run plan"):
            plan_settings_area = st.container()
            plan_summary_area = st.container()
        last_run_area = st.container()

    with data_box:
        data_step = _data_step(settings, store)
    with task_box:
        task_step = _task_step(store, data_step)
    with plan_settings_area:
        plan_settings, selection = _plan_settings(settings, store, device, task_step, data_step)
    if data_step.checks is not None:
        with data_step.checks:
            _data_checks(data_step, task_step, selection)
    components.set_section_badge(
        data_box,
        ("No data yet", "gray")  # nothing to fix yet: no alarm colour
        if data_step.waiting
        else components.ready_badge(data_step.problems + data_step.warnings),
    )
    with task_step.rest:
        exact = _task_extras(
            data_step, task_step, selection, plan_settings, busy=active is not None
        )
    components.set_section_badge(task_box, components.ready_badge(task_step.problems))

    config = _build_config(task_step, plan_settings)
    st.session_state[state.RUN_SETUP] = _setup_signature(
        data_step, task_step, selection, plan_settings, exact
    )
    shown_job: str | None = None
    with plan_summary_area:
        if active is not None:
            with st.container(key="btr_progress"):
                _progress_panel(active.id)
        else:
            shown_job = _plan_summary_and_button(
                store, device, jobs, data_step, task_step, selection, plan_settings, exact, config
            )
    with last_run_area:
        _last_run_card(jobs, active, hide_open=shown_job)


# =============================================================================
# 1 · Data
# =============================================================================


def _data_step(settings: Settings, store: Store) -> _DataStep:
    sample = _sample_path()
    sources = ["upload", "path"] + (["sample"] if sample else [])
    _init(SOURCE_KEY, "upload")
    _ensure_option(SOURCE_KEY, sources, "upload")
    source = st.segmented_control(
        "Source",
        sources,
        format_func=_SOURCE_LABELS.__getitem__,
        key=SOURCE_KEY,
        required=True,
        label_visibility="collapsed",
        persist_state="session",
    )
    source = source or "upload"
    st.session_state[state.DATA_SOURCE] = source

    ds: _Dataset | None = None
    no_table = _NO_TABLE
    try:
        if source == "upload":
            ds = _upload_source()
        elif source == "path":
            ds = _folder_source(settings, store)
        else:
            ds = _sample_source(sample)
    except _NoTable as exc:
        no_table = exc.problem

    tables: dict[str, _Dataset] = st.session_state.setdefault(_TABLES, {})
    step = _DataStep(ds=ds)
    if ds is None:
        if st.session_state.pop(state.LOADED_TABLE, None) is not None:
            components.set_context_dataset(None)
        step.problems.append(no_table)
        step.waiting = no_table in (_NO_TABLE, _NO_FILE_CHOSEN)
        if source == "upload" and no_table == _NO_TABLE:
            _data_empty_state(sample is not None)
        return step
    tables[source] = ds
    if st.session_state.get(state.LOADED_TABLE) is not ds.table:
        st.session_state[state.LOADED_TABLE] = ds.table
        components.set_context_dataset(ds.table)

    frame = ds.table.frame
    st.markdown(_summary_line(ds.table))
    columns = [str(c) for c in frame.columns]
    fp = ds.fingerprint

    # The required column gets a full row; the two optional ones share the next.
    text_cell = st.container()
    id_cell, gold_cell = st.columns(2, gap="small")
    with text_cell:
        key = f"run.textcol::{fp}"
        guess = _safe(lambda: data.guess_text_column(frame), None)
        _init(key, guess if guess in columns else columns[0])
        _ensure_option(key, columns, columns[0])
        text_col = st.selectbox(
            "Text column",
            columns,
            key=key,
            help="The column whose text is inserted into the prompt for each row.",
            persist_state="session",
        )
    with id_cell:
        key = f"run.idcol::{fp}"
        options = [NO_COLUMN, *columns]
        guess = _safe(lambda: data.guess_id_column(frame), None)
        _init(key, guess if guess in columns else NO_COLUMN)
        _ensure_option(key, options, NO_COLUMN)
        id_col = _column_or_none(
            st.selectbox(
                "ID column",
                options,
                key=key,
                format_func=_column_label,
                help="Optional. Identifies rows in the results and exports.",
                persist_state="session",
            )
        )

    step.text_col, step.id_col, step.gold_cell = text_col, id_col, gold_cell
    if text_col is not None:
        stats = _text_stats(fp, frame, text_col)
        if stats.non_blank == 0:
            step.problems.append(
                f'The text column "{text_col}" has no text. Choose another column.'
            )
        numeric = stats.non_blank and (
            stats.numeric_share >= NUMERIC_TEXT_SHARE or stats.median_len < SHORT_TEXT_CHARS
        )
        if numeric:
            step.warnings.append(
                f'Column "{text_col}" looks like numbers or codes, not text. Pick the column '
                "that holds the text in step 1."
            )
        with st.container(horizontal=True, gap="xsmall"):
            if numeric:
                st.badge(
                    "Looks like numbers, not text",
                    icon=":material/warning:",
                    color="orange",
                    help="Most values are numbers or very short. The model would be asked "
                    "about IDs or codes rather than text.",
                )
            if stats.blank:
                st.badge(
                    fmt.plural(stats.blank, "blank text"),
                    icon=":material/info:",
                    color="orange",
                    help="Blank rows are skipped; no model call is made.",
                )
            if stats.duplicates:
                st.badge(
                    fmt.plural(stats.duplicates, "duplicate text"),
                    color="gray",
                    help="Duplicate texts are generated once and share the answer.",
                )
            if stats.non_blank and not (stats.blank or stats.duplicates):
                st.badge("No blank or duplicate texts", icon=":material/check:", color="gray")
            if stats.non_blank:
                st.badge(f"Median length {fmt.plural(stats.median_len, 'character')}", color="gray")
    else:
        step.problems.append("Choose a text column.")
    step.checks = st.container()
    return step


def _upload_source() -> _Dataset | None:
    upload = st.file_uploader(
        "Upload a table",
        type=[ext.lstrip(".") for ext in data.SUPPORTED_EXTENSIONS],
        key=UPLOAD_KEY,
        on_change=_on_upload_change,
        label_visibility="collapsed",
    )
    tables: dict[str, _Dataset] = st.session_state.setdefault(_TABLES, {})
    if upload is None:
        previous = tables.get("upload")
        if previous is not None:
            with st.container(horizontal=True, vertical_alignment="center", gap="small"):
                st.caption(
                    f"Using **{previous.table.name}**, uploaded earlier. "
                    "Drop a file above to replace it."
                )
                st.button(
                    "Remove",
                    key="run.upload_remove",
                    type="tertiary",
                    icon=":material/close:",
                    on_click=lambda: st.session_state.get(_TABLES, {}).pop("upload", None),
                )
        return previous

    name = upload.name
    digest = _memo("upload_digest", upload.file_id, lambda: _digest(upload.getvalue()))
    sheet = None
    if Path(name).suffix.lower() in _EXCEL_SUFFIXES:
        sheets = _sheets_from_bytes(digest, upload.getvalue())
        sheet = _sheet_picker(f"run.sheet::{digest}", sheets)
    try:
        table, fingerprint = _memo(
            "upload_table",
            (digest, sheet),
            lambda: _read_upload(digest, name, sheet, upload.getvalue()),
        )
    except data.DataLoadError as exc:
        _load_error(name, exc)
    except Exception:
        _unexpected_load_error(name)
    return _Dataset(table, fingerprint, "upload")


def _on_upload_change() -> None:
    if st.session_state.get(UPLOAD_KEY) is None:
        st.session_state.get(_TABLES, {}).pop("upload", None)


def _folder_source(settings: Settings, store: Store) -> _Dataset | None:
    default_folder = _safe(lambda: store.get_setting("last_folder"), None) or str(settings.data_dir)
    _init(FOLDER_KEY, default_folder)
    raw = st.text_input(
        "Folder or file path",
        key=FOLDER_KEY,
        placeholder="e.g. C:\\Users\\me\\data  or  C:\\Users\\me\\data\\feedback.csv",
        help="Paste a folder, then pick a file, or paste the full path of a file. "
        "Surrounding quotes (from Explorer's “Copy as path”) are removed.",
        persist_state="session",
    )
    path = clean_path(raw or "")
    if path is None:
        return None
    untouched = (raw or "").strip() == str(default_folder).strip()
    try:
        is_file, is_dir = path.is_file(), path.is_dir()
    except OSError:
        is_file = is_dir = False

    if is_file:
        if path.suffix.lower() not in data.SUPPORTED_EXTENSIONS:
            st.warning(
                f"{path.name} isn't a supported file type. Use {_extension_list()}.",
                icon=":material/warning:",
            )
            raise _NoTable("Choose a supported file in step 1.")
        target, folder = path, path.parent
    elif is_dir:
        files = _safe(lambda: data.list_data_files(path), [])
        if not files:
            nearby = _subfolders_with_data(path)
            if untouched:
                # The starting folder simply has no data: nothing is wrong yet.
                st.caption(
                    ":material/folder_open: No data files in this folder. Paste the folder "
                    "or file you want to use" + (", or open one below." if nearby else ".")
                )
            else:
                st.info(
                    f"No supported files in this folder. Looking for {_extension_list()}.",
                    icon=":material/folder_open:",
                )
            if nearby:
                with st.container(horizontal=True, gap="small"):
                    for folder_path, n_files in nearby:
                        st.button(
                            f"{folder_path.name}/ · {fmt.plural(n_files, 'file')}",
                            key=f"run.subfolder::{folder_path}",
                            icon=":material/folder:",
                            type="tertiary",
                            on_click=_set_state,
                            args=(FOLDER_KEY, str(folder_path)),
                        )
            if untouched:
                raise _NoTable(_NO_TABLE)
            raise _NoTable("Choose a folder that contains a data file (step 1).")
        # Newest first: the most recently saved file is the likeliest choice,
        # and with a single file there is nothing to choose.
        options = [str(p) for p in files]
        _ensure_option(FOLDER_FILE_KEY, options, options[0])
        _init(FOLDER_FILE_KEY, options[0])
        choice = st.selectbox(
            "File",
            options,
            key=FOLDER_FILE_KEY,
            format_func=_file_label,
            help="Newest first.",
            persist_state="session",
        )
        if choice is None:
            raise _NoTable(_NO_FILE_CHOSEN)
        target, folder = Path(choice), path
    else:
        st.warning(
            f"Couldn't find {path}. Check the path, or paste the folder that contains the file.",
            icon=":material/warning:",
        )
        raise _NoTable("Check the folder or file path in step 1.")

    dataset = _load_path(target, "path")
    if dataset is not None and st.session_state.get(_SAVED_FOLDER) != str(folder):
        st.session_state[_SAVED_FOLDER] = str(folder)
        # Remembering the folder is a convenience only.
        with contextlib.suppress(Exception):
            store.set_setting("last_folder", str(folder))
    return dataset


def _subfolders_with_data(folder: Path) -> list[tuple[Path, int]]:
    """Up to :data:`MAX_SUBFOLDERS` direct subfolders of ``folder`` that hold
    supported files, with how many (hidden and private folders skipped)."""
    try:
        subfolders = sorted(
            (p for p in folder.iterdir() if p.is_dir() and not p.name.startswith((".", "_"))),
            key=lambda p: p.name.casefold(),
        )
    except OSError:
        return []
    found: list[tuple[Path, int]] = []
    for sub in subfolders[:60]:
        n_files = _count_data_files(sub)
        if n_files:
            found.append((sub, n_files))
            if len(found) >= MAX_SUBFOLDERS:
                break
    return found


def _count_data_files(folder: Path) -> int:
    try:
        return len(data.list_data_files(folder))
    except Exception:
        return 0


def _sample_source(sample: Path | None) -> _Dataset | None:
    if sample is None:
        return None
    st.caption(
        "A built-in example: short customer messages with gold sentiment and intent "
        "labels: a quick way to try a classification task and its evaluation."
    )
    return _load_path(sample, "sample")


def _load_path(target: Path, origin: str) -> _Dataset | None:
    try:
        stat = target.stat()
    except OSError as exc:
        components.error_with_details(
            f"Couldn't open {target.name}.",
            details=str(exc),
            hint="Check that the file still exists and that you can open it, then try again.",
        )
        raise _NoTable() from None
    sheet = None
    if target.suffix.lower() in _EXCEL_SUFFIXES:
        sheets = _sheets_from_path(str(target), stat.st_mtime_ns, stat.st_size)
        sheet = _sheet_picker(f"run.sheet::{target}", sheets)
    key = (str(target), stat.st_mtime_ns, stat.st_size, sheet)
    try:
        table, fingerprint = _memo(f"{origin}_table", key, lambda: _read_path(*key))
    except data.DataLoadError as exc:
        _load_error(target.name, exc)
    except PermissionError:
        components.error_with_details(
            f"Couldn't open {target.name}: the file is locked.",
            hint="Close it in Excel (or any other program) and try again.",
        )
        raise _NoTable() from None
    except Exception:
        _unexpected_load_error(target.name)
    return _Dataset(table, fingerprint, origin)


def _load_error(name: str, exc: data.DataLoadError) -> NoReturn:
    """Show a DataLoadError (its message is user-facing; a low-level reason
    embedded in it goes to the technical details) and stop the data step."""
    message, details = split_load_error(exc)
    if name not in message:
        message = f"Couldn't read {name}. {message}"
    if Path(name).suffix.lower() in _EXCEL_SUFFIXES:
        hint = "Check that the file opens in Excel, or save it as CSV (UTF-8) and try again."
    else:
        hint = (
            "Check that the file is a table with a header row, or save it as CSV (UTF-8) "
            "and try again."
        )
    components.error_with_details(message, details=details, hint=hint)
    raise _NoTable() from None


def _unexpected_load_error(name: str) -> NoReturn:
    components.error_with_details(
        f"Couldn't read {name}.",
        details=traceback.format_exc(),
        hint="Save the file as CSV (UTF-8) and try again.",
    )
    raise _NoTable() from None


def split_load_error(exc: BaseException) -> tuple[str, str | None]:
    """(plain message, technical details) for a load error. A message that
    embeds its cause's text ("Couldn't read x.xlsx as an Excel workbook: File
    is not a zip file.") loses it to the details; other messages (already
    rewritten for people, often with a hint) are kept whole."""
    message = " ".join(str(exc).split()) or "The file couldn't be read as a table."
    cause = exc.__cause__
    cause_text = " ".join(str(cause).split()).rstrip(".") if cause is not None else ""
    marker = f": {cause_text}"
    at = message.find(marker) if cause_text else -1
    if at < 0:
        return message, None
    rest = message[at + len(marker) :].strip(" .")
    plain = message[:at].rstrip(".") + "." + (f" {rest}." if rest else "")
    return plain, f"{type(cause).__name__}: {cause_text}"


def _sheet_picker(key: str, sheets: list[str]) -> str | None:
    if len(sheets) <= 1:
        return sheets[0] if sheets else None
    _init(key, sheets[0])
    _ensure_option(key, sheets, sheets[0])
    return st.selectbox("Sheet", sheets, key=key, persist_state="session")


def _data_empty_state(has_sample: bool) -> None:
    st.space("small")
    components.empty_state(
        ":material/table_chart:",
        "No table loaded yet",
        "Use one row per text: a column with the text, plus optional ID and gold-label "
        f"columns. Works with {_extension_list()}.",
    )
    if has_sample:
        with st.container(horizontal_alignment="center"):
            st.button(
                "Try the sample data",
                key="run.try_sample",
                icon=":material/science:",
                on_click=_set_state,
                args=(SOURCE_KEY, "sample"),
            )
    st.space("small")


def _data_checks(step: _DataStep, task: _TaskStep, selection: _Selection | None) -> None:
    """Gold column picker, gold/answers compatibility, and the table preview."""
    frame = step.frame
    if frame is None or step.gold_cell is None or step.ds is None:
        return
    columns = [str(c) for c in frame.columns]
    candidates = [c for c in columns if c != step.text_col]
    with step.gold_cell:
        key = f"run.gold::{step.ds.fingerprint}::{task.preset.key}"
        options = [NO_COLUMN, *candidates]
        if key not in st.session_state:
            st.session_state[key] = _default_gold(step, task.preset, task.labels) or NO_COLUMN
        _ensure_option(key, options, NO_COLUMN)
        step.gold_col = _column_or_none(
            st.selectbox(
                "Gold labels",
                options,
                key=key,
                format_func=_column_label,
                help="Optional. The correct answer for each row: choose it to score the run.",
                persist_state="session",
            )
        )

    if step.gold_col and task.labels and selection is not None:
        golds = selection.frame[step.gold_col]
        matched, total = _gold_match(golds, task.labels)
        skipped = total - matched
        line = f"{matched:,} / {total:,} gold values match the allowed answers"
        if skipped:
            line += f"; {skipped:,} will be skipped"
        with st.container(horizontal=True, vertical_alignment="center", gap="small", wrap=False):
            if total and matched / total < GOLD_MATCH_WARNING:
                st.badge(
                    "Gold column may not fit this task",
                    icon=":material/warning:",
                    color="orange",
                )
                step.warnings.append(
                    (
                        f"None of the {total:,} gold values match the allowed answers, so "
                        "no row can be scored."
                        if not matched
                        else f"Only {matched:,} of {total:,} gold values match the allowed "
                        "answers, so most rows can't be scored."
                    )
                    + " Check the gold column in step 1."
                )
            else:
                st.badge("Gold labels fit", icon=":material/check:", color="gray")
            st.caption(line + ".", width="stretch")
    elif step.gold_col and not task.labels:
        if st.session_state.get(_exact_key(task.preset), True):
            st.caption(
                "This task has open-ended answers: each output is compared with the gold "
                "column by exact match (“Score by exact match” in step 2)."
            )
        else:
            st.caption(
                "This task has open-ended answers. Tick “Score by exact match” in step 2 "
                "to compare outputs with the gold column."
            )

    shown = min(PREVIEW_ROWS, len(frame))
    title = (
        f"Preview · all {fmt.plural(shown, 'row')}"
        if shown == len(frame)
        else f"Preview · first {shown:,} of {fmt.plural(len(frame), 'row')}"
    )
    with st.expander(title, icon=":material/visibility:"):
        config = {}
        if step.text_col:
            config[step.text_col] = st.column_config.TextColumn(width="large")
        st.dataframe(
            frame.head(PREVIEW_ROWS),
            hide_index=True,
            column_config=config,
            height=320,
            alt=f"First {shown} rows of {step.ds.table.name}",
        )


# =============================================================================
# 2 · Task
# =============================================================================


def _task_step(store: Store, data_step: _DataStep) -> _TaskStep:
    custom = _safe(store.list_presets, [])
    categories = list(presets.CATEGORIES) + ([_SAVED_CATEGORY] if custom else [])
    _init(CATEGORY_KEY, categories[0])
    _ensure_option(CATEGORY_KEY, categories, categories[0])
    category = (
        st.segmented_control(
            "Task type",
            categories,
            key=CATEGORY_KEY,
            required=True,
            label_visibility="collapsed",
            persist_state="session",
        )
        or categories[0]
    )
    if category == _SAVED_CATEGORY:
        choices = list(custom)
    else:
        choices = [p for p in presets.builtin_presets() if p.category == category]
    by_key = {p.key: p for p in choices}
    if len(choices) == 1 and category != _SAVED_CATEGORY:
        # One built-in preset in the category (Custom): nothing to pick.
        # A saved preset keeps its pill - it shows the name the user gave it.
        preset = choices[0]
    else:
        pick_key = f"run.preset::{category}"
        _init(pick_key, choices[0].key)
        _ensure_option(pick_key, list(by_key), choices[0].key)
        picked = st.pills(
            "Task",
            list(by_key),
            key=pick_key,
            format_func=lambda k: by_key[k].title,
            required=True,
            label_visibility="collapsed",
            persist_state="session",
        )
        preset = by_key.get(picked or "", choices[0])

    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.caption(preset.description or "Saved preset.", width="content")
        scorable_slot = st.empty()  # filled once the allowed answers are known

    start = starting_template(preset)
    tmpl_key = f"tmpl::{preset.key}"
    _init(tmpl_key, start)
    template_raw = str(st.session_state.get(tmpl_key) or "")
    uses_var = prompting.VAR in template_raw

    var: str | None = None
    if uses_var:
        var = _variable_input(preset, data_step)

    edited = template_raw != start
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.markdown("**Prompt**", width="content")
        if edited:
            st.badge("Edited", icon=":material/edit:", color="orange")
        st.space("stretch")
        st.button(
            "Reset to preset",
            key=f"run.reset::{preset.key}",
            type="tertiary",
            icon=":material/undo:",
            disabled=not edited,
            on_click=_set_state,
            args=(tmpl_key, start),
        )
    template_raw = (
        st.text_area(
            "Prompt template",
            key=tmpl_key,
            # Grows with the prompt: short presets don't leave an empty box.
            height="content",
            label_visibility="collapsed",
            persist_state="session",
        )
        or ""
    )
    problems = prompting.template_problems(template_raw, var if uses_var else None)
    worded = [_word_problem(problem, preset) for problem in problems]
    if not worded:
        worded = _instruction_problems(template_raw, start, preset)
    with st.container(horizontal=True, vertical_alignment="center", gap="xsmall"):
        if worded:
            for chip, _reason in worded:
                st.badge(chip, icon=":material/warning:", color="orange")
        else:
            st.badge("Uses {text}", icon=":material/check:", color="gray")
            if uses_var:
                st.badge("{var} filled in", icon=":material/check:", color="gray")
        st.space("xsmall")
        st.caption(
            "`{text}` is replaced by each row's text"
            + (f"; `{{var}}` by the {_var_noun(preset)} above." if uses_var else "."),
            width="content",
        )

    labels = None if preset.output_kind == "json" else _allowed_answers(preset, var)
    if labels:
        output_kind = "label"
        scorable_slot.badge(
            "Scorable",
            icon=":material/rule:",
            color="blue",
            help="Outputs are matched to a fixed set of answers, so a gold column can score them.",
        )
    else:
        output_kind = "text" if preset.output_kind == "label" else preset.output_kind

    reasons = [reason for _chip, reason in worded]
    final_template = None
    if not problems:
        try:
            final_template = prompting.apply_variable(template_raw, var if uses_var else None)
        except prompting.TemplateError as exc:  # pragma: no cover - problems covers it
            reasons = [str(exc)]

    return _TaskStep(
        preset=preset,
        template_raw=template_raw,
        edited=edited,
        uses_var=uses_var,
        var=var if uses_var else None,
        labels=labels,
        output_kind=output_kind,
        problems=reasons,
        final_template=final_template,
        rest=st.container(),
    )


def starting_template(preset: presets.Preset) -> str:
    """The prompt a preset starts from (and "Reset to preset" returns to):
    its template, except the built-in "Custom prompt", whose bare ``{text}``
    gets :data:`CUSTOM_SCAFFOLD` to fill in."""
    if preset.builtin and preset.key == CUSTOM_PRESET_KEY and preset.template.strip() == "{text}":
        return CUSTOM_SCAFFOLD
    return preset.template


def _instruction_problems(
    template: str, start: str, preset: presets.Preset
) -> list[tuple[str, str]]:
    """A prompt that tells the model nothing: only ``{text}`` (and ``{var}``),
    or the custom scaffold left as it is."""
    rest = template.replace(prompting.TEXT, "").replace(prompting.VAR, "")
    if not rest.strip():
        return [
            (
                "Add an instruction: the prompt only contains {text}",
                "Add an instruction to the prompt in step 2: it only contains {text}.",
            )
        ]
    if start == CUSTOM_SCAFFOLD and template.strip() == CUSTOM_SCAFFOLD.strip():
        return [
            (
                "Write your own instruction",
                "Replace the example instruction in step 2 with your own.",
            )
        ]
    return []


def _word_problem(problem: str, preset: presets.Preset) -> tuple[str, str]:
    """(chip text next to the prompt, run-button reason) for a
    ``prompting.template_problems`` message - in the words of the form
    rather than template jargon."""
    noun = _var_noun(preset)
    if problem.startswith("The prompt template is empty"):
        return ("The prompt is empty", "Write a prompt in step 2.")
    if problem.startswith("The prompt template must include"):
        return (
            "Add {text} where each row's text goes",
            "Add {text} to the prompt in step 2: it marks where each row's text goes.",
        )
    if problem.startswith("This task uses {var}"):
        return (f"Enter the {noun} above", f"Enter the {noun} in step 2.")
    if problem.startswith("The variable can't contain"):
        return (f"The {noun} can't contain {{text}}", f"Remove {{text}} from the {noun} in step 2.")
    return (problem.rstrip("."), problem)


def _var_label(preset: presets.Preset) -> str:
    """Label of the variable's input. A chip input needs no "(comma-separated)"."""
    label = preset.var_label or "Variable"
    if preset.labels_from_var:
        label = re.sub(r"\s*\(comma-separated\)\s*$", "", label) or "Categories"
    return label


def _var_noun(preset: presets.Preset) -> str:
    """The variable as a noun for messages: "categories", "question"."""
    noun = re.sub(r"\s*\([^)]*\)\s*$", "", preset.var_label or "") or "value"
    return noun[0].lower() + noun[1:]


def _variable_input(preset: presets.Preset, data_step: _DataStep) -> str:
    key = f"var::{preset.key}"
    label = _var_label(preset)
    if preset.labels_from_var:
        _init(key, [])
        current = [str(v) for v in st.session_state.get(key) or []]
        suggestions = _category_suggestions(data_step, preset)
        options = list(dict.fromkeys([*current, *suggestions]))
        chosen = st.multiselect(
            label,
            options,
            key=key,
            accept_new_options=True,
            placeholder="Type a category and press Enter",
            help="Type a category and press Enter, then the next one. Values from the gold "
            "column are suggested once you pick it.",
            persist_state="session",
        )
        return ", ".join(str(c).strip() for c in chosen if str(c).strip())
    _init(key, "")
    return (
        st.text_input(
            label,
            key=key,
            placeholder=preset.var_hint or "",
            persist_state="session",
        )
        or ""
    ).strip()


def _category_suggestions(data_step: _DataStep, preset: presets.Preset) -> list[str]:
    """Distinct values of the chosen gold column, so "Categorise" can reuse them."""
    if data_step.ds is None:
        return []
    gold = st.session_state.get(f"run.gold::{data_step.ds.fingerprint}::{preset.key}")
    frame = data_step.ds.table.frame
    if gold is None or gold == NO_COLUMN or gold not in frame.columns:
        return []
    values = frame[gold].dropna().astype(str).str.strip()
    distinct = [v for v in values.value_counts().index if v]
    return distinct[:MAX_CATEGORY_SUGGESTIONS]


def _allowed_answers(preset: presets.Preset, var: str | None) -> list[str] | None:
    if preset.labels_from_var:
        labels = preset.resolve_labels(var)
        st.caption(
            "The categories above are the allowed answers: outputs are matched to them "
            "for the LABEL column and scoring."
        )
        return labels

    key = f"labels::{preset.key}"
    _init(key, list(preset.labels or ()))
    current = st.session_state.get(key)
    if isinstance(current, str):  # an older session kept the comma-separated text
        st.session_state[key] = presets.parse_label_list(current)
    chosen_now = [str(v) for v in st.session_state.get(key) or []]
    options = list(dict.fromkeys([*(preset.labels or ()), *chosen_now]))

    def answers_input() -> list[str]:
        chosen = st.multiselect(
            "Allowed answers",
            options,
            key=key,
            accept_new_options=True,
            placeholder="Type an answer and press Enter",
            help="Outputs are matched to one of these answers for the LABEL column and "
            "scoring. Type a new answer and press Enter; remove one with its ×.",
            persist_state="session",
        )
        # "yes, no" typed in one go counts as two answers.
        out: list[str] = []
        for item in chosen:
            out += presets.parse_label_list(str(item))
        return list(dict.fromkeys(out))

    if preset.labels:
        labels = answers_input()
        st.caption("Outputs are matched to these answers for the LABEL column and scoring.")
        return labels or None
    # Open-ended task: the default view is just the prompt.
    with st.expander(
        "Score against a fixed set of answers (optional)",
        icon=":material/rule:",
        expanded=bool(chosen_now),
    ):
        labels = answers_input()
        st.caption(
            "With a list of answers, each output is matched to one of them (LABEL column) "
            "and can be scored against a gold column. Leave empty for open-ended output."
        )
    return labels or None


def _task_extras(
    data_step: _DataStep,
    task: _TaskStep,
    selection: _Selection | None,
    plan: _PlanSettings,
    *,
    busy: bool,
) -> bool:
    """Exact-match opt-in, prompt preview and preset management. Returns
    whether exact-match scoring is on."""
    st.session_state.pop(_DELETED, None)
    exact = False
    if data_step.gold_col and not task.labels:
        # On by default: an open-ended task gets no gold column unless
        # someone picks one, and picking one means "score against it" (as in
        # version 1 and the command line).
        _init(_exact_key(task.preset), True)
        exact = st.checkbox(
            "Score by exact match",
            key=_exact_key(task.preset),
            help="Compare each output with the gold value after normalising case, spacing "
            "and punctuation. Only sensible for short, well-defined answers.",
            persist_state="session",
        )

    with st.expander("Preview prompt", icon=":material/preview:"):
        _prompt_preview(data_step, task, selection, plan)

    with st.container(horizontal=True, gap="small"):
        _save_preset_popover(task, plan)
        if not task.preset.builtin and st.button(
            "Delete preset",
            key="run.delete_preset",
            icon=":material/delete:",
            type="tertiary",
            disabled=busy,
            help="Disabled while a run is in progress." if busy else None,
        ):
            _confirm_delete_preset(task.preset.key, task.preset.title)
    return exact


def _exact_key(preset: presets.Preset) -> str:
    return f"run.exact::{preset.key}"


def _prompt_preview(
    data_step: _DataStep, task: _TaskStep, selection: _Selection | None, plan: _PlanSettings
) -> None:
    if task.final_template is None:
        st.caption("Fix the prompt problems above to see a preview.")
        return
    if selection is None or data_step.text_col is None:
        st.code(task.final_template, language=None, wrap_lines=True)
        st.caption("Load a table to see the prompt for its first row.")
        return
    texts = selection.frame[data_step.text_col].tolist()
    first = next((i for i, t in enumerate(texts) if not engine.is_blank(t)), None)
    if first is None:
        st.caption("All selected rows are blank.")
        return
    text = str(texts[first])
    truncated = len(text) > plan.max_input_chars
    prompt = prompting.render_prompt(task.final_template, text[: plan.max_input_chars])
    st.code(prompt, language=None, wrap_lines=True)
    note = f"Row {first + 1} of the selection."
    if truncated:
        note += f" The text is cut to {plan.max_input_chars:,} characters."
    st.caption(note + " At run time the model's chat template wraps this prompt.")


def _save_preset_popover(task: _TaskStep, plan: _PlanSettings) -> None:
    # A form, so the typed name and the click arrive together (a plain
    # text_input only commits on blur or Enter).
    with (
        st.popover("Save as preset", icon=":material/bookmark_add:", type="tertiary"),
        st.form("run.save_preset_form", border=False),
    ):
        st.text_input(
            "Preset name",
            key=PRESET_NAME_KEY,
            placeholder="e.g. Complaint triage",
            max_chars=60,
        )
        message = st.session_state.pop(_PRESET_MSG, None)
        if message:
            st.error(message, icon=":material/error:")
        st.caption(
            "Saves the prompt, allowed answers and max output tokens. "
            "Saved presets appear under “Saved”."
        )
        st.form_submit_button(
            "Save preset",
            key="run.save_preset",
            type="primary",
            icon=":material/save:",
            on_click=_save_preset,
            # The final template has {var} filled in, so the saved prompt and
            # its fixed label set always agree.
            args=(task.final_template, plan.max_new_tokens, task.labels, task.preset.var_hint),
        )


def _save_preset(
    template: str | None, max_new_tokens: int, labels: list[str] | None, var_hint: str | None
) -> None:
    store = resources.get_store()
    name = str(st.session_state.get(PRESET_NAME_KEY) or "")
    if not name.strip():
        st.session_state[_PRESET_MSG] = "Enter a name for the preset."
        return
    if not template:
        st.session_state[_PRESET_MSG] = "Fix the prompt above before saving it as a preset."
        return
    try:
        preset = presets.make_custom_preset(name, template, max_new_tokens, labels, var_hint)
        store.save_preset(preset)
    except ValueError as exc:
        st.session_state[_PRESET_MSG] = str(exc).replace(" - ", ": ")
        return
    except Exception as exc:
        st.session_state[_PRESET_MSG] = f"Couldn't save the preset: {exc}"
        return
    st.session_state[CATEGORY_KEY] = _SAVED_CATEGORY
    st.session_state[f"run.preset::{_SAVED_CATEGORY}"] = preset.key
    st.session_state[f"tmpl::{preset.key}"] = preset.template
    st.session_state[f"maxtok::{preset.key}"] = preset.max_new_tokens
    st.session_state[f"labels::{preset.key}"] = list(preset.labels or ())
    st.session_state[PRESET_NAME_KEY] = ""
    components.queue_toast(f"Saved preset “{preset.title}”", ":material/bookmark_added:")


@st.dialog("Delete preset?")
def _confirm_delete_preset(key: str, title: str) -> None:
    if st.session_state.get(_DELETED) == key:  # deleted by the callback below
        st.session_state.pop(_DELETED, None)
        st.rerun()
    st.write(f"Delete the saved preset “{title}”? This can't be undone.")
    with st.container(horizontal=True, horizontal_alignment="right", gap="small"):
        if st.button("Cancel", key="run.delete_cancel"):
            st.rerun()
        # A callback (not the button's return value) so the deletion happens
        # whether Streamlit reruns only the dialog or the whole page.
        st.button(
            "Delete preset",
            key="run.delete_confirm",
            type="primary",
            icon=":material/delete:",
            on_click=_delete_preset,
            args=(key, title),
        )
    message = st.session_state.pop(_DELETE_MSG, None)
    if message:
        st.warning(message, icon=":material/warning:")


def _delete_preset(key: str, title: str) -> None:
    try:
        resources.get_store().delete_preset(key)
    except Exception as exc:
        st.session_state[_DELETE_MSG] = f"Couldn't delete the preset: {exc}"
        return
    st.session_state[_DELETED] = key
    st.session_state.pop(f"run.preset::{_SAVED_CATEGORY}", None)
    components.queue_toast(f"Deleted preset “{title}”", ":material/delete:")


# =============================================================================
# Run plan: model, rows, advanced settings
# =============================================================================


def _scope_controls(data_step: _DataStep) -> _Selection | None:
    frame = data_step.frame
    n_rows = len(frame) if frame is not None else 0
    fp = data_step.ds.fingerprint if data_step.ds else "none"
    mode_key, n_key = f"run.scope::{fp}", f"run.n::{fp}"
    _init(mode_key, "first" if n_rows > LARGE_TABLE_ROWS else "all")
    _init(n_key, max(1, min(DEFAULT_N, n_rows)))
    with st.container(horizontal=True, vertical_alignment="bottom", gap="small"):
        mode = (
            st.segmented_control(
                "Rows",
                list(_SCOPE_LABELS),
                format_func=_SCOPE_LABELS.__getitem__,
                key=mode_key,
                required=True,
                disabled=frame is None,
                persist_state="session",
            )
            or "all"
        )
        n = n_rows
        if mode != "all":
            if frame is not None and st.session_state.get(n_key, 1) > n_rows:
                st.session_state[n_key] = n_rows
            n = int(
                st.number_input(
                    "Number of rows",
                    min_value=1,
                    max_value=max(1, n_rows),
                    step=10,
                    key=n_key,
                    disabled=frame is None,
                    width=150,
                    persist_state="session",
                )
                or 1
            )
    if frame is None:
        return None
    return select_rows(frame, mode, n)


def select_rows(frame: pd.DataFrame, mode: str, n: int) -> _Selection:
    """Apply a row scope: "all", "first" N, or "random" N (seeded, original
    order kept)."""
    total = len(frame)
    if mode == "all" or n >= total:
        return _Selection(frame, "all", total, f"All {fmt.plural(total, 'row')}")
    if mode == "first":
        return _Selection(frame.head(n), mode, n, f"First {n:,} of {fmt.plural(total, 'row')}")
    picked = frame.sample(n=n, random_state=RANDOM_SEED).sort_index()
    return _Selection(
        picked, mode, n, f"Random {n:,} of {fmt.plural(total, 'row')} (seed {RANDOM_SEED})"
    )


def _plan_settings(
    settings: Settings,
    store: Store,
    device: DeviceInfo,
    task: _TaskStep,
    data_step: _DataStep,
) -> tuple[_PlanSettings, _Selection | None]:
    """Model, row scope and the Advanced popover, in that order."""
    specs = [*models.available_models(include_demo=settings.demo_mode), *models.api_models()]
    by_repo = {s.repo: s for s in specs}
    options = [*by_repo, _CUSTOM_MODEL]
    last = _safe(lambda: store.get_setting("last_model"), None)
    # The last model run, a local folder or another repo id included (it
    # must still exist / be a valid id): people who work with their own
    # fine-tune shouldn't have to paste its path in every session.
    restore_custom = (
        bool(last)
        and last not in by_repo
        and not str(last).startswith("demo://")
        and _safe(lambda: models.custom_model(str(last)), None) is not None
    )
    _init(MODEL_KEY, last if last in by_repo else _CUSTOM_MODEL if restore_custom else options[0])
    _ensure_option(MODEL_KEY, options, options[0])
    choice = st.selectbox(
        "Model",
        options,
        key=MODEL_KEY,
        format_func=lambda r: "Custom model…" if r == _CUSTOM_MODEL else _model_label(by_repo[r]),
        help="Every built-in local model fits an 8 GB GPU at 4-bit; larger models are slower "
        "but usually more accurate. Cloud models (Anthropic, OpenAI) run on the provider's "
        "servers: they need an API key, and the texts are sent to the provider.",
        persist_state="session",
    )
    spec: models.ModelSpec | None = None
    problem: str | None = None
    if choice == _CUSTOM_MODEL:
        _init(CUSTOM_MODEL_KEY, str(last) if restore_custom else "")
        raw = st.text_input(
            "Hugging Face repo id, local model folder or cloud model",
            key=CUSTOM_MODEL_KEY,
            placeholder="e.g. Qwen/Qwen2.5-0.5B-Instruct or anthropic:claude-opus-4-8",
            help="A cloud model is the provider and its model id: anthropic:… or openai:… "
            "(with OPENAI_BASE_URL set, any model of an OpenAI-compatible server).",
            persist_state="session",
        )
        value = strip_quotes(raw or "")
        if not value:
            problem = (
                "Enter a Hugging Face repo id, a local model folder, or a cloud model such as "
                "anthropic:claude-opus-4-8."
            )
        else:
            try:
                spec = models.custom_model(value)
            except ValueError as exc:
                problem = str(exc)
    else:
        spec = by_repo.get(str(choice))
    if spec is not None:
        components.model_badges(spec, device)

    selection = _scope_controls(data_step)

    preset = task.preset
    tok_key = f"maxtok::{preset.key}"
    _init(tok_key, preset.max_new_tokens)
    _init(BATCH_KEY, engine.DEFAULT_BATCH_SIZE)
    _init(PRECISION_KEY, "auto")
    _ensure_option(PRECISION_KEY, list(_PRECISION_CHOICES), "auto")
    _init(MAX_CHARS_KEY, engine.DEFAULT_MAX_INPUT_CHARS)
    if ORDER_KEY not in st.session_state:  # read the store once per session
        st.session_state[ORDER_KEY] = remembered_order(store)
    _ensure_option(ORDER_KEY, list(engine.ORDERS), engine.DEFAULT_ORDER)
    is_demo = spec is not None and spec.is_demo
    is_api = spec is not None and spec.is_api
    auto_resolves = PRECISION_LABELS[resolve_precision("auto", device)]

    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        with st.popover("Advanced", icon=":material/tune:"):
            max_tokens = int(
                st.number_input(
                    "Max output tokens",
                    min_value=1,
                    max_value=32768,
                    step=8,
                    key=tok_key,
                    help="Generation stops after this many tokens. Short answers need few "
                    "tokens; a smaller limit is much faster.",
                    persist_state="session",
                )
                or preset.max_new_tokens
            )
            with st.container(horizontal=True, vertical_alignment="center", gap="small"):
                st.caption(f"Suggested for {preset.title}: {preset.max_new_tokens:,}")
                if max_tokens < preset.max_new_tokens:
                    st.badge("Answers may be cut off", icon=":material/warning:", color="orange")
            batch_size = int(
                st.slider(
                    "Batch size",
                    min_value=1,
                    max_value=64,
                    key=BATCH_KEY,
                    help="Rows sent to the model together. Bigger batches are faster until "
                    "GPU memory runs out; on an out-of-memory error the batch is split "
                    "automatically. For a cloud model, the number of requests sent at once: "
                    "lower it if the provider reports rate limits.",
                    persist_state="session",
                )
            )
            order = (
                st.segmented_control(
                    "Processing order",
                    list(engine.ORDERS),
                    format_func=engine.ORDER_LABELS.__getitem__,
                    key=ORDER_KEY,
                    required=True,
                    help=ORDER_HELP,
                    on_change=_remember_order,
                    persist_state="session",
                )
                or engine.DEFAULT_ORDER
            )
            precision = (
                st.segmented_control(
                    "Weights",
                    list(_PRECISION_CHOICES),
                    format_func=fmt.precision,
                    key=PRECISION_KEY,
                    required=True,
                    disabled=is_demo or is_api,
                    help="Numeric precision the model weights are loaded in. 4-bit fits the "
                    "largest models in 8 GB of GPU memory.",
                    persist_state="session",
                )
                or "auto"
            )
            if is_demo:
                st.caption("The demo model has no weights to load.")
            elif is_api:
                st.caption("Cloud models run on the provider's servers: nothing loads here.")
            else:
                st.caption(f"Auto = {auto_resolves} on this machine.")
                if precision == "4bit" and not (device.kind == "cuda" and device.bitsandbytes):
                    st.badge(
                        "4-bit needs an NVIDIA GPU with bitsandbytes",
                        icon=":material/warning:",
                        color="orange",
                    )
            max_chars = int(
                st.number_input(
                    "Max input characters",
                    min_value=100,
                    max_value=1_000_000,
                    step=500,
                    key=MAX_CHARS_KEY,
                    help="Longer texts are cut to this length before they go into the prompt.",
                    persist_state="session",
                )
                or engine.DEFAULT_MAX_INPUT_CHARS
            )
            if selection is not None and data_step.text_col:
                over = _count_longer(selection.frame[data_step.text_col], max_chars)
                st.caption(
                    f"{fmt.plural(over, 'selected row')} {'is' if over == 1 else 'are'} longer "
                    "and will be cut."
                    if over
                    else "No selected row is longer than this."
                )
        precision_note = "" if is_demo or is_api else f" · {fmt.precision(precision)} weights"
        if precision == "auto" and not (is_demo or is_api):
            precision_note += f" ({fmt.precision(resolve_precision('auto', device))})"
        batch_note = (
            f"{fmt.plural(batch_size, 'request')} at once" if is_api else f"batch size {batch_size}"
        )
        st.caption(
            f"Up to {fmt.plural(max_tokens, 'output token')} · {batch_note} · "
            f"{engine.ORDER_LABELS[order].lower()}" + precision_note
        )

    chosen = _PlanSettings(
        model=spec,
        model_problem=problem,
        max_new_tokens=max_tokens,
        batch_size=batch_size,
        precision=precision,
        max_input_chars=max_chars,
        order=order,
    )
    return chosen, selection


def remembered_order(store: Store) -> str:
    """The processing order chosen last (in any session), or
    ``engine.DEFAULT_ORDER`` when none was chosen or the stored value isn't
    one of ``engine.ORDERS`` (any more)."""
    stored = _safe(lambda: store.get_setting(ORDER_SETTING), None)
    return stored if stored in engine.ORDERS else engine.DEFAULT_ORDER


def _remember_order() -> None:
    """Make the processing order just chosen the default of later sessions."""
    order = st.session_state.get(ORDER_KEY)
    if order in engine.ORDERS:
        # Remembering the choice is a convenience only.
        with contextlib.suppress(Exception):
            resources.get_store().set_setting(ORDER_SETTING, order)


# =============================================================================
# Run plan: summary, run button, progress
# =============================================================================


def _build_config(task: _TaskStep, plan_settings: _PlanSettings) -> engine.RunConfig | None:
    """The run configuration the page is set up for, or None while the model
    or prompt has a problem (or the settings are out of range)."""
    spec = plan_settings.model
    if spec is None or task.final_template is None:
        return None
    try:
        return engine.RunConfig.build(
            spec,
            task.final_template,
            plan_settings.max_new_tokens,
            precision=plan_settings.precision,
            max_input_chars=plan_settings.max_input_chars,
        )
    except ValueError:
        return None


def _setup_signature(
    data_step: _DataStep,
    task: _TaskStep,
    selection: _Selection | None,
    plan_settings: _PlanSettings,
    exact: bool,
) -> tuple[Any, ...]:
    """Everything a run depends on, as set up on the page right now. Equal
    signatures mean "the same run" (see ``components.setup_changed``)."""
    spec = plan_settings.model
    return (
        spec.repo if spec is not None else None,
        task.preset.key,
        task.template_raw,
        task.var,
        tuple(task.labels or ()),
        plan_settings.max_new_tokens,
        plan_settings.batch_size,
        plan_settings.order,
        plan_settings.precision,
        plan_settings.max_input_chars,
        data_step.ds.fingerprint if data_step.ds else None,
        data_step.text_col,
        data_step.id_col,
        data_step.gold_col,
        selection.mode if selection else None,
        selection.n if selection else None,
        exact,
    )


def _plan_summary_and_button(
    store: Store,
    device: DeviceInfo,
    jobs: Any,
    data_step: _DataStep,
    task: _TaskStep,
    selection: _Selection | None,
    plan_settings: _PlanSettings,
    exact: bool,
    config: engine.RunConfig | None,
) -> str | None:
    """The plan metrics and the run button. Returns the id of the finished
    job the button opens, when it opens one rather than starting a run."""
    spec = plan_settings.model
    plan: engine.RunPlan | None = None
    texts: list[Any] = []
    if config is None and spec is not None and task.final_template is not None:
        try:  # only to show why: _build_config returned None
            engine.RunConfig.build(
                spec,
                task.final_template,
                plan_settings.max_new_tokens,
                precision=plan_settings.precision,
                max_input_chars=plan_settings.max_input_chars,
            )
        except ValueError as exc:
            components.error_with_details(
                "These settings can't be used.",
                details=str(exc),
                hint="Check the max output tokens and max input characters under Advanced.",
            )
    if config is not None and selection is not None and data_step.text_col and data_step.ds:
        texts = selection.frame[data_step.text_col].tolist()
        plan = _compute_plan(config, texts, store, jobs, data_step, selection)

    _plan_metrics(plan, config)
    if plan is not None and plan.n_cached_rows and config is not None and spec is not None:
        _discard_button(store, config, f"{spec.label} · {task.preset.title}")

    problem = _first_problem(data_step, task, plan_settings, plan, device, jobs.active())
    if spec is not None and spec.is_api and plan is not None and config is not None:
        consent = _cloud_notice(spec, plan, config)
        if problem is None and plan.n_todo_rows and not consent:
            problem = f"Tick “Send these texts to {spec.provider_label}” to run a cloud model."
    existing = _same_finished_job(jobs) if plan is not None and plan.fully_cached else None
    if existing is not None and problem is None:
        # The very same run finished in this session: open it rather than
        # adding an identical copy to the Results run list.
        if st.button(
            "Open results",
            key=RUN_BUTTON_KEY,
            type="primary",
            icon=":material/table_view:",
            width="stretch",
            shortcut=RUN_SHORTCUT,
            help="This exact run finished a moment ago: its results are ready.",
        ):
            st.session_state[state.ACTIVE_JOB] = existing.id
            st.switch_page(nav.page("results"))
        return existing.id
    if plan is not None and plan.fully_cached:
        label, icon = "Show saved results", ":material/bolt:"
    elif plan is not None:
        label, icon = f"Generate {fmt.plural(plan.n_todo_rows, 'row')}", ":material/play_arrow:"
    else:
        label, icon = "Generate", ":material/play_arrow:"
    clicked = st.button(
        label,
        key=RUN_BUTTON_KEY,
        type="primary",
        icon=icon,
        width="stretch",
        # Not Ctrl+Enter: that applies an edit in a text area (the prompt
        # editor says so), and must not start a run on the GPU as well.
        shortcut=RUN_SHORTCUT,
        disabled=problem is not None,
        help=problem,
    )
    if problem:
        st.caption(f":material/info: {problem}")
    else:
        for warning in data_step.warnings:
            st.caption(f":orange[:material/warning:] {warning}")
    if clicked and problem is None and config is not None and selection is not None:
        _submit(store, jobs, config, texts, data_step, task, selection, plan_settings, exact)
    return None


def _cloud_notice(spec: models.ModelSpec, plan: engine.RunPlan, config: engine.RunConfig) -> bool:
    """What a cloud model run sends and roughly costs, and the checkbox
    that agrees to send it. True when ticked (or when nothing is sent)."""
    if not plan.n_todo_rows:
        return True
    provider = spec.provider_label
    calls, sent, answers = api_usage_estimate(plan, config)
    words = (
        f"Sends {fmt.plural(calls, 'text')} to {provider}: about "
        f"{fmt.token_count(sent)} input tokens, and answers of up to "
        f"{fmt.plural(config.max_new_tokens, 'token')} each"
    )
    if spec.price:
        cost = (sent * spec.price[0] + answers * spec.price[1]) / 1e6
        words += (
            f" · at most about {fmt.no_math(fmt.money(cost))} at list prices "
            f"({models.PRICES_AS_OF}), "
            "plus any reasoning"
        )
    st.caption(f":material/cloud_upload: {words}.")
    return bool(
        st.checkbox(
            f"Send these texts to {provider}",
            key=f"{API_CONSENT_KEY}::{spec.provider}",
            help=f"Cloud models run on {provider}'s servers: each text is sent to {provider} "
            "over the internet and handled under your API account's terms. Only send data "
            "you're allowed to share with them. Asked once per browser session.",
        )
    )


def api_usage_estimate(plan: engine.RunPlan, config: engine.RunConfig) -> tuple[int, int, int]:
    """(requests, input tokens, most answer tokens) for the texts a run
    would send - one request per distinct text not saved yet. Tokens are
    estimated at :data:`CHARS_PER_TOKEN` characters each."""
    todo = [key for key in plan.snippets if key not in plan.cached]
    n_text = config.template.count(prompting.TEXT)
    overhead = len(config.template) - n_text * len(prompting.TEXT)
    chars = sum(overhead + n_text * len(plan.snippets[key]) for key in todo)
    return len(todo), math.ceil(chars / CHARS_PER_TOKEN), len(todo) * config.max_new_tokens


def _discard_button(store: Store, config: engine.RunConfig, name: str) -> None:
    """Version 1's "Clear cache": throw away this configuration's saved
    outputs (after a confirmation), so the next run generates every row."""
    from batchtextreview.ui.pages import settings as settings_page

    if st.button(
        "Discard saved outputs",
        key="run.discard_saved",
        icon=":material/delete:",
        type="tertiary",
        help="Deletes the saved outputs of this model + prompt + settings, so the next run "
        "generates every row again. Evaluation history is kept. All saved outputs are listed "
        "under Models & settings.",
    ):
        n = _safe(lambda: store.count_outputs(config.run_id), 0)
        settings_page.confirm_delete_run(config.run_id, n, name)


def _same_finished_job(jobs: Any) -> Job | None:
    """The newest successful job of this session that ran exactly the setup
    on the page now (and is still in memory). A job that was retried later
    is skipped: its retry has the newer results."""
    current = st.session_state.get(state.RUN_SETUP)
    setups = st.session_state.get(state.JOB_SETUPS, {})
    held = [jobs.get(i) for i in st.session_state.get(state.SESSION_JOBS, [])]
    retried = {j.request.context.get(state.RETRY_OF) for j in held if j is not None}
    for job in held:  # newest first
        if job is None or job.id in retried or setups.get(job.id) != current:
            continue
        if job.result is not None and job.snapshot().state == "succeeded":
            return job
    return None


def _compute_plan(
    config: engine.RunConfig,
    texts: list[Any],
    store: Store,
    jobs: Any,
    data_step: _DataStep,
    selection: _Selection,
) -> engine.RunPlan | None:
    latest = jobs.latest()
    marker = (latest.id, latest.snapshot().state) if latest is not None else None
    assert data_step.ds is not None
    key = (
        config.run_id,
        data_step.ds.fingerprint,
        data_step.text_col,
        selection.mode,
        selection.n,
        marker,
        # Saved outputs can be deleted elsewhere (Models & settings, another
        # tab); one COUNT query keeps "Saved" / "Show saved results" honest.
        _safe(lambda: store.count_outputs(config.run_id), None),
    )
    try:
        return _memo("plan", key, lambda: engine.plan_run(config, texts, store))
    except Exception as exc:
        st.warning(
            "Couldn't check for saved outputs, so the counts below are unknown. "
            "Running still works.",
            icon=":material/warning:",
        )
        with st.expander("Technical details", icon=":material/code:"):
            st.code(f"{type(exc).__name__}: {exc}", language=None, wrap_lines=True)
        return None


def _plan_metrics(plan: engine.RunPlan | None, config: engine.RunConfig | None) -> None:
    with st.container(horizontal=True, gap="small"):
        st.metric("Selected", f"{plan.n_rows:,}" if plan else fmt.DASH)
        st.metric(
            "Already saved",
            f"{plan.n_cached_rows:,}" if plan else fmt.DASH,
            help="Rows with a saved output from an earlier run of this model + prompt + "
            "settings: they're instant.",
        )
        st.metric("To generate", f"{plan.n_todo_rows:,}" if plan else fmt.DASH)
    if plan is None:
        return
    notes = []
    if plan.n_blank:
        notes.append(f"{fmt.plural(plan.n_blank, 'blank text')} skipped")
    if plan.n_duplicates:
        notes.append(analysis.duplicates_note(plan.n_duplicates))
    if config is not None and plan.n_todo_rows:
        rates = st.session_state.get(components.RATES, {})
        rate = rates.get((config.model_id, config.precision))
        if rate:
            notes.append(
                f"{fmt.estimate(plan.n_todo_rows / rate)} at {fmt.rate(rate)} "
                "(last speed seen for this model)"
            )
    if notes:
        text = " · ".join(notes)
        st.caption(text[0].upper() + text[1:])


def _first_problem(
    data_step: _DataStep,
    task: _TaskStep,
    plan_settings: _PlanSettings,
    plan: engine.RunPlan | None,
    device: DeviceInfo,
    active: Job | None,
) -> str | None:
    if active is not None:
        mine = active.id in st.session_state.get(state.SESSION_JOBS, [])
        return (
            "A run is in progress. Wait for it to finish or cancel it."
            if mine
            else "Another run is in progress (started in another tab or before the page "
            "was reloaded). Wait for it to finish: one model runs at a time."
        )
    if data_step.problems:
        return data_step.problems[0]
    if task.problems:
        return task.problems[0]
    if plan_settings.model_problem:
        return plan_settings.model_problem
    if plan is not None and plan.n_rows and plan.n_blank == plan.n_rows:
        return "All selected rows are blank. Choose another text column or other rows."
    if plan is not None and plan.fully_cached:
        return None  # every row is saved: no model is loaded at all
    spec = plan_settings.model
    if spec is None or spec.is_demo:
        return None
    return model_problem(spec, plan_settings.precision, device)


def model_problem(spec: models.ModelSpec, precision: str, device: DeviceInfo) -> str | None:
    """Why ``spec`` can't be loaded at ``precision`` on this machine - checked
    before running, so a run that is certain to fail at load never starts -
    or None. Only called when the model has to be loaded."""
    if spec.is_api:
        return api_model_problem(spec)
    if not device.has_torch:
        return (
            "PyTorch isn't installed, so only the demo model can run. "
            "Install it as described in the README, then restart the app."
        )
    if precision == "4bit" and not (device.kind == "cuda" and device.bitsandbytes):
        better = "fp16" if device.kind in ("cuda", "mps") else "fp32"
        why = (
            "the bitsandbytes package, which isn't installed"
            if device.kind == "cuda"
            else "an NVIDIA GPU, and this machine has none"
        )
        return (
            f"4-bit weights need {why}. Choose Auto or {better} under Advanced "
            f"(Auto is {fmt.precision(resolve_precision('auto', device))} here)."
        )
    if precision in ("fp16", "bf16") and device.kind == "cpu":
        return (
            f"{fmt.precision(precision)} weights on the CPU are unusably slow. Choose Auto "
            "(fp32 on this machine) under Advanced."
        )
    if spec.is_local:
        folder = Path(spec.repo)
        if not (folder / "config.json").is_file():
            return (
                f"{spec.label} isn't a model folder: it has no config.json. Choose the folder "
                "that holds config.json, the tokenizer files and the weights."
            )
        return None
    on_disk = resources.downloaded(spec.repo)
    if on_disk is not True and spec.gated and not resources.hf_token_available():
        return (
            f"{spec.label} is gated: accept its licence on huggingface.co, set HF_TOKEN (or "
            "run hf auth login) and restart the app, or pick a Qwen, Phi or Granite model."
        )
    if on_disk is False and resources.offline_mode():
        return (
            f"{spec.label} isn't downloaded, and offline mode is on (HF_HUB_OFFLINE). Pick a "
            "downloaded model or a local folder, or unset HF_HUB_OFFLINE and restart the app."
        )
    return None


def api_model_problem(spec: models.ModelSpec) -> str | None:
    """Why a cloud model can't run here (its SDK isn't installed, or no
    credentials are set), or None."""
    from batchtextreview.backends import api

    return api.setup_problem(spec.provider or "", in_app=True)


def _submit(
    store: Store,
    jobs: Any,
    config: engine.RunConfig,
    texts: list[Any],
    data_step: _DataStep,
    task: _TaskStep,
    selection: _Selection,
    plan_settings: _PlanSettings,
    exact: bool,
) -> None:
    from batchtextreview.jobs import JobBusyError, JobRequest

    assert data_step.ds is not None and data_step.text_col is not None
    spec = plan_settings.model
    assert spec is not None
    frame = selection.frame.reset_index(drop=True)
    exclude = tuple(c for c in (data_step.text_col, data_step.id_col, data_step.gold_col) if c)
    context = state.RunContext(
        name=state.run_name(task.preset.title, data_step.ds.table.name),
        dataset_name=data_step.ds.table.name,
        dataset_fingerprint=_safe(lambda: data.frame_fingerprint(frame), data_step.ds.fingerprint),
        frame=frame,
        text_column=data_step.text_col,
        preset_key=task.preset.key,
        preset_title=task.preset.title,
        template_raw=task.template_raw,
        template_edited=task.edited,
        var=task.var if task.uses_var else None,
        labels=task.labels,
        output_kind=task.output_kind,
        selection=selection.description,
        id_column=data_step.id_col,
        gold_column=data_step.gold_col,
        group_columns=_safe(lambda: data.guess_group_columns(frame, exclude=exclude), []),
        extra={
            "exact_match": bool(exact and not task.labels and data_step.gold_col),
            "batch_size": plan_settings.batch_size,
            "order": plan_settings.order,
            "precision_requested": plan_settings.precision,
            "data_source": data_step.ds.origin,
            "suggested_max_new_tokens": task.preset.max_new_tokens,
        },
    )
    request = JobRequest(
        config=config,
        texts=texts,
        batch_size=plan_settings.batch_size,
        order=plan_settings.order,
        context={state.CONTEXT_KEY: context},
        # Logs the evaluation even if nobody is on the Results page when the
        # run ends - the only place it is logged (analysis.history_entry).
        on_finish=analysis.finish_hook(store),
    )
    try:
        job = jobs.submit(request)
    except JobBusyError:
        st.warning(
            "Another run started a moment ago. Wait for it to finish, then try again.",
            icon=":material/hourglass_top:",
        )
        return
    except Exception:
        components.error_with_details(
            "Couldn't start the run.",
            details=traceback.format_exc(),
            hint="Try again. If it keeps happening, restart the app.",
        )
        return

    session_jobs = [job.id, *st.session_state.get(state.SESSION_JOBS, [])]
    st.session_state[state.SESSION_JOBS] = session_jobs
    setups = st.session_state.setdefault(state.JOB_SETUPS, {})
    setups[job.id] = st.session_state.get(state.RUN_SETUP)
    st.session_state[state.ACTIVE_JOB] = job.id
    with contextlib.suppress(Exception):
        store.set_setting("last_model", spec.repo)
    st.rerun()


@st.fragment(run_every="1s")
def _progress_panel(job_id: str) -> None:
    """Live progress of the running job; replaces the plan metrics.

    The only poller on the Run page (the context strip stays static here).
    When the job ends it just reruns the whole app: that full run shows the
    toast and switches to Results (``components.context_strip``) - never
    from inside this fragment.
    """
    jobs = resources.get_job_manager()
    job = jobs.get(job_id)
    if job is None or job.snapshot().done:
        st.rerun(scope="app")
    assert job is not None
    snap = job.snapshot()
    p = snap.progress
    config = job.request.config
    context = job.request.context.get(state.CONTEXT_KEY)
    mine = job.id in st.session_state.get(state.SESSION_JOBS, [])
    if snap.state == "queued" or p.phase == "planning":
        label = "Checking saved outputs…"
    elif p.phase == "loading" and models.api_provider(config.model_id):
        provider = models.API_PROVIDERS[models.api_provider(config.model_id) or ""]
        label = f"Connecting to {provider} · checking {config.model_label}…"
    elif p.phase == "loading":
        label = f"Loading {config.model_label}…"
        size = _first_download_gb(config.model_id)
        if size:
            label = f"Loading {config.model_label} · the first use downloads about {size} GB"
    elif p.phase == "finished":
        label = "Finishing…"
    else:
        label = "Generating"

    # Not a ``with`` block: leaving st.status's context marks it complete,
    # and this fragment re-renders every second while the job still runs.
    status = st.status(label, state="running", expanded=True)
    if context is not None:
        status.caption(f"{fmt.md(context.name)} · {fmt.md(context.selection)}")
    if not mine:
        status.caption(":material/info: Started in another tab or before this page was reloaded.")
    loading_s = None
    if p.phase == "loading" and snap.started_at is not None:
        loading_s = (datetime.now(snap.started_at.tzinfo) - snap.started_at).total_seconds()
    status.progress(bar_fraction(p), text=progress_text(p, loading_s))
    if p.blank_rows:  # so the counts below add up to the rows done
        status.caption(
            f"Includes {fmt.plural(p.blank_rows, 'blank row')}, skipped without a model call."
        )
    metrics = status.container(horizontal=True, gap="small")
    metrics.metric("Generated", f"{p.generated_rows:,}")
    metrics.metric(
        "Reused",
        f"{p.cached_rows:,}",
        help="Rows answered from saved outputs of an earlier identical run.",
    )
    metrics.metric("Errors", f"{p.error_rows:,}")
    recent = list(p.recent)[-4:]
    if recent:
        # One fixed-height line per output: the text on the left, the answer
        # as a chip on the right, so the feed never wraps or scrolls sideways.
        status.caption("Latest outputs")
        for text, output in reversed(recent):
            with status.container(horizontal=True, gap="small", vertical_alignment="center"):
                st.caption(fmt.md(fmt.truncate(text, 48)), width="stretch")
                st.badge(fmt.truncate(output, 24) or fmt.DASH, color="blue")
    cancelling = job_id in st.session_state.get(_CANCELLING, set())
    st.button(
        "Stopping after this batch…" if cancelling else "Cancel run",
        key="run.cancel",
        icon=":material/stop_circle:",
        disabled=cancelling,
        on_click=_cancel_job,
        args=(job_id,),
        width="stretch",
    )


def _first_download_gb(model_id: str) -> int | None:
    """About how many GB loading ``model_id`` downloads first, or None when
    nothing will be downloaded (demo model, local folder, already in the
    cache) or the size isn't known."""
    spec = models.find_model(model_id) or _safe(lambda: models.custom_model(model_id), None)
    if spec is None or spec.is_demo or spec.is_api or spec.is_local:
        return None
    if resources.downloaded(spec.repo) is not False or resources.offline_mode():
        return None
    return resources.download_size_gb(spec.params_b)


def bar_fraction(p: engine.Progress) -> float:
    """How full the progress bar is, 0-1: the share of the work done
    (``Progress.work_fraction``, measured in generation time rather than
    rows, and never going back), so the bar moves with the time left
    whichever order the texts run in - the text beside it counts rows.
    Falls back to the share of rows done (``Progress.fraction``)."""
    value = getattr(p, "work_fraction", None)
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        value = p.fraction
    return min(max(float(value), 0.0), 1.0)


def progress_text(p: engine.Progress, loading_s: float | None = None) -> str:
    """ "312 / 1,000 rows · 4.1 rows/s · about 2 min 50 s left" (while the
    model loads: "0 / 1,000 rows · loading for 1 min 20 s")."""
    parts = [f"{p.done_rows:,} / {fmt.plural(p.total_rows, 'row')}"]
    if p.phase == "loading" and loading_s is not None and loading_s >= 1:
        parts.append(f"loading for {fmt.duration(loading_s)}")
    elif p.phase == "generating" and p.rows_per_s:
        parts.append(fmt.rate(p.rows_per_s))
        if p.eta_s is not None:
            parts.append(f"{fmt.estimate(p.eta_s)} left")
    elif p.phase == "generating":
        parts.append("estimating time left…")
    return " · ".join(parts)


def _cancel_job(job_id: str) -> None:
    resources.get_job_manager().cancel(job_id)
    st.session_state.setdefault(_CANCELLING, set()).add(job_id)


def _last_run_card(jobs: Any, active: Job | None, hide_open: str | None = None) -> None:
    """The most recent run's status, summary and a way to open it.
    ``hide_open``: the job the run button already opens (no second button)."""
    ids = st.session_state.get(state.SESSION_JOBS) or []
    job = next((j for j in (jobs.get(i) for i in ids) if j is not None), None)
    elsewhere = False
    if job is None:
        # Nothing started here - e.g. the page was reloaded during a run.
        latest = jobs.latest()
        if latest is not None and latest.snapshot().done:
            job, elsewhere = latest, True
    if job is None or (active is not None and job.id == active.id):
        return
    snap = job.snapshot()
    context = job.request.context.get(state.CONTEXT_KEY)
    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center", gap="small"):
            st.subheader("Last run", anchor=False, width="content")
            st.space("stretch")
            status, icon, color = components.job_status(snap)
            st.badge(status, icon=icon, color=color)  # type: ignore[arg-type]
        if context is not None:
            st.caption(
                f"{fmt.md(context.name)}  \n{fmt.md(job.request.config.model_label)} · "
                f"{fmt.md(context.selection)}"
            )
        else:
            st.caption(job.request.config.model_label)
        if elsewhere:
            st.caption(":material/info: Started in another tab or before a reload.")
        if snap.state == "failed":
            problem = analysis.describe_failure(
                job.result, snap.error, job.request.config.model_label
            )
            components.error_with_details(
                f"**{problem.headline}** {fmt.md(problem.cause)}",
                details=problem.details,
                hint=problem.hint,
            )
        elif job.result is not None:
            st.caption(analysis.run_summary(job.result))
        if (
            job.result is not None
            and job.id != hide_open
            and st.button(
                "Open results",
                key="run.open_results",
                icon=":material/table_view:",
                width="stretch",
            )
        ):
            st.session_state[state.ACTIVE_JOB] = job.id
            st.switch_page(nav.page("results"))


# =============================================================================
# Helpers
# =============================================================================


def strip_quotes(raw: str) -> str:
    """Whitespace and surrounding quotes removed: Windows Explorer's "Copy as
    path" wraps paths in double quotes."""
    value = raw.strip()
    while len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1].strip()
    return value.strip("\"'").strip()


def clean_path(raw: str) -> Path | None:
    """A user-pasted path: see :func:`strip_quotes`; ``~`` and environment
    variables (``%USERPROFILE%``, ``$HOME``) are expanded. None when empty."""
    value = strip_quotes(raw)
    if not value:
        return None
    return Path(os.path.expandvars(value)).expanduser()


def _sample_path() -> Path | None:
    roots = [Path.cwd()]
    with contextlib.suppress(IndexError):  # src/batchtextreview/ui/pages -> repo root
        roots.insert(0, Path(__file__).resolve().parents[4])
    for root in roots:
        candidate = root / SAMPLE_FILE
        if candidate.is_file():
            return candidate
    return None


@st.cache_data(show_spinner="Reading the file…", max_entries=16)
def _read_path(
    path: str, mtime_ns: int, size: int, sheet: str | None
) -> tuple[data.LoadedTable, str]:
    table = data.read_table(path, sheet=sheet)
    return table, data.frame_fingerprint(table.frame)


@st.cache_data(show_spinner="Reading the file…", max_entries=8)
def _read_upload(
    digest: str, name: str, sheet: str | None, _payload: bytes
) -> tuple[data.LoadedTable, str]:
    table = data.read_table(_payload, name=name, sheet=sheet)
    return table, data.frame_fingerprint(table.frame)


@st.cache_data(show_spinner=False, max_entries=16)
def _sheets_from_path(path: str, mtime_ns: int, size: int) -> list[str]:
    return data.excel_sheets(path)


@st.cache_data(show_spinner=False, max_entries=8)
def _sheets_from_bytes(digest: str, _payload: bytes) -> list[str]:
    return data.excel_sheets(_payload)


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()[:32]


def _memo(slot: str, key: Any, compute: Callable[[], Any]) -> Any:
    """One-entry memo per slot in session state - avoids re-reading cached
    tables (st.cache_data copies on every hit) and re-planning on reruns."""
    memo: dict[str, tuple[Any, Any]] = st.session_state.setdefault(_MEMO, {})
    hit = memo.get(slot)
    if hit is not None and hit[0] == key:
        return hit[1]
    value = compute()
    memo[slot] = (key, value)
    return value


def _init(key: str, value: Any) -> None:
    if key not in st.session_state:
        st.session_state[key] = value


def _ensure_option(key: str, options: list[Any], fallback: Any) -> None:
    if key in st.session_state and st.session_state[key] not in options:
        st.session_state[key] = fallback


def _set_state(key: str, value: Any) -> None:
    st.session_state[key] = value


def _safe(fn: Callable[[], Any], default: Any) -> Any:
    try:
        return fn()
    except Exception:
        return default


def _column_label(value: str) -> str:
    return "None" if value == NO_COLUMN else value


def _column_or_none(value: str | None) -> str | None:
    return None if value in (None, NO_COLUMN) else value


@dataclass(frozen=True)
class _TextStats:
    non_blank: int
    blank: int
    duplicates: int
    median_len: int
    numeric_share: float
    """Share of the non-blank values that parse as numbers."""


def _text_stats(fp: str, frame: pd.DataFrame, column: str) -> _TextStats:
    """Counts the Data card shows for a text column (memoised per column)."""

    def compute() -> _TextStats:
        series = frame[column]
        present = series[~series.isna()].astype(str)
        stripped = present.str.strip()
        texts = present[stripped != ""]
        blank = len(series) - len(texts)
        duplicates = int(texts.duplicated().sum())
        median = int(texts.str.len().median()) if len(texts) else 0
        sample = stripped[stripped != ""].head(5000)
        numbers = pd.to_numeric(sample.str.replace(",", "", regex=False), errors="coerce")
        numeric = float(numbers.notna().mean()) if len(sample) else 0.0
        return _TextStats(len(texts), blank, duplicates, median, numeric)

    return _memo("text_stats", (fp, column), compute)


def _count_longer(series: pd.Series, limit: int) -> int:
    present = series[~series.isna()].astype(str)
    return int((present.str.len() > limit).sum())


def _gold_match(golds: pd.Series, labels: list[str]) -> tuple[int, int]:
    """(rows whose gold value resolves to one of ``labels``, rows)."""
    matched = 0
    for value, count in golds.value_counts(dropna=False).items():
        if not is_missing(value) and normalise_label(value, labels) is not None:
            matched += int(count)
    return matched, len(golds)


_GOLD_NAME = re.compile(r"(gold|label|target|truth|expected|y_)", re.IGNORECASE)
"""Column-name starts that say "the correct answers" (as data.guess_gold_columns)."""
_EARLIER_RUN_LABEL = re.compile(r"LABEL(?:_\d+)?")
_EARLIER_RUN_OUTPUT = re.compile(r"OUTPUT(?:_\d+)?")
_GENERIC_WORDS = frozenset({"custom", "label", "labels", "text", "with", "from"})


def _named_gold(column: Any, frame: pd.DataFrame, preset: presets.Preset) -> bool:
    """Whether ``column``'s name says it holds this task's correct answers:
    it starts with gold / label / target / truth / expected / y_, or names
    the task ("SENTIMENT" for Sentiment). An earlier run's LABEL column in
    an exported results file is that run's predictions, not gold."""
    name = str(column).strip()
    if _EARLIER_RUN_LABEL.fullmatch(name) and any(
        _EARLIER_RUN_OUTPUT.fullmatch(str(c)) for c in frame.columns
    ):
        return False
    if _GOLD_NAME.match(name):
        return True
    words = {w.casefold() for w in re.findall(r"[A-Za-z]{4,}", preset.title)} - _GENERIC_WORDS
    lowered = name.casefold()
    return any(word in lowered for word in words)


def _default_gold(step: _DataStep, preset: presets.Preset, labels: list[str] | None) -> str | None:
    """The first column named like this task's gold labels (:func:`_named_gold`)
    whose values mostly match the task's answers - else None, as in version
    1. A column that merely has few values (a 1-3 TIER column fits the 1-5
    of Urgency) is never picked: the run would be scored against it and the
    score logged to history without anyone choosing it."""
    frame = step.frame
    if frame is None or not labels:
        return None
    exclude = tuple(c for c in (step.text_col, step.id_col) if c)
    candidates = _safe(lambda: data.guess_gold_columns(frame, exclude=exclude), [])
    sample = frame.head(2000)
    for column in candidates:
        if column not in sample.columns or not _named_gold(column, frame, preset):
            continue
        values = sample[column][~sample[column].isna()]
        if not len(values):
            continue
        matched, total = _gold_match(values, labels)
        if matched / total >= GOLD_MATCH_WARNING:
            return str(column)
    return None


def _summary_line(table: data.LoadedTable) -> str:
    rows, cols = table.frame.shape
    kind = _FORMAT_NAMES.get(table.fmt, table.fmt.upper())
    details = []
    if table.sheet:
        details.append(f"sheet “{table.sheet}”")
    if table.encoding:
        details.append(_ENCODING_NAMES.get(table.encoding.lower(), table.encoding))
    if table.delimiter and table.fmt == "csv" and table.delimiter != ",":
        details.append(f"{_DELIMITER_NAMES.get(table.delimiter, repr(table.delimiter))}-separated")
    fmt_text = f"{kind} ({', '.join(details)})" if details else kind
    name = table.name.replace("*", "\\*").replace("_", "\\_")
    return (
        f":material/description: **{name}** · {fmt.plural(rows, 'row')} · "
        f"{fmt.plural(cols, 'column')} · {fmt_text}"
    )


def _file_label(path_str: str) -> str:
    path = Path(path_str)
    try:
        stat = path.stat()
    except OSError:
        return path.name
    modified = fmt.timestamp(datetime.fromtimestamp(stat.st_mtime))
    return f"{path.name} · {fmt.file_size(stat.st_size)} · {modified}"


def _extension_list() -> str:
    names = ["CSV", "TSV", "TXT", "Excel", "Parquet", "JSON"]
    return ", ".join(names[:-1]) + " or " + names[-1]


def _model_label(spec: models.ModelSpec) -> str:
    """ "Qwen2.5-3B-Instruct · 3B · downloaded" for the model picker."""
    if spec.is_api:
        return f"{spec.label} · {spec.provider_label} API"
    label = f"{spec.label} · {spec.size_label}" if spec.size_label else spec.label
    if not spec.is_demo and resources.downloaded(spec.repo):
        label += " · downloaded"
    return label
