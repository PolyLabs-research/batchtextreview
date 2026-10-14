"""Command-line interface (``batchtextreview`` / ``python -m batchtextreview``).

Subcommands
-----------
ui               Launch the Streamlit app (``streamlit run`` on the
                 repository's app.py in a source checkout, else the entry point
                 bundled in the package; extra args after ``--`` - or any
                 option ``ui`` doesn't know - are passed through).
run              Headless run over a file, writing a results CSV/XLSX.
models           List built-in models (label, repo, size, flags).
presets          List presets (key, category, max tokens, labels).
history          Print evaluation history (task with its {var} value,
                 scores, invalid outputs), or save all of it with
                 --output PATH (.csv, or .xlsx for Excel).
import-legacy    Import version-1 results/*.parquet checkpoints and evals.csv.
device           Print device / precision / library versions.

Global options: ``-v`` / ``--verbose`` (before the command) also logs debug
details to stderr (``ui`` passes it on to the app as ``BTR_VERBOSE=1``,
which works for any command too). Commands that use the results folder log
to ``<results>/logs/batchtextreview.log``.

``run`` options
---------------
INPUT                    file to read (any format data.read_table supports)
--sheet NAME             Excel sheet (default: the first visible one)
--text-column COL        default: data.guess_text_column
--preset KEY             preset key or title (default "Custom" when
                         --template/--template-file given, else required)
--template TEXT | --template-file PATH   override the preset's template
--var VALUE              value for {var}
--labels "a,b,c"         override the label set
--model NAME             label, repo id, v1 label, or local path (required)
--precision {auto,4bit,fp16,bf16,fp32}   default auto
--max-new-tokens N       default: the preset's
--max-input-chars N      default 8000
--batch-size N           default 8
--order {longest,shortest}
                         which texts go first (default longest: the ETA
                         counts down and an out-of-memory model fails in
                         the first batch; shortest: first results sooner)
--limit N                first N rows only (0 = all, default)
--gold COL               score against this column and log to history
--output PATH            default: <input stem>_<preset slug>_<model slug>_results.csv
                         (.xlsx extension -> Excel); expands JSON outputs
                         into columns for JSON presets. Checked before the
                         model loads; an existing file is replaced (with a
                         note), and if it can't be written at the end (open
                         in Excel), the results go to a free name next to it
                         or in the results folder instead (exit code 1)
--no-cache               don't reuse or save outputs, and don't log to history
                         (saved custom presets are still found)
--results-dir DIR        overrides BTR_RESULTS_DIR

Progress goes to stderr as a single updating line (share of the work done,
rows done/total, rate, ETA); Ctrl+C cancels cleanly after the current
batch, writes the partial results, and exits 130. Exit code 1 on validation
or load errors (message to stderr, no traceback), when no row could be
generated and when the results had to be saved somewhere other than
--output, 0 otherwise.
"""

from __future__ import annotations

import argparse
import contextlib
import difflib
import importlib.metadata
import importlib.util
import logging
import math
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

import pandas as pd

from batchtextreview import APP_NAME, __version__, data, engine, evaluation, presets, prompting
from batchtextreview import models as model_catalogue
from batchtextreview import results as results_mod
from batchtextreview.backends import base as backend_base
from batchtextreview.backends.manager import ModelManager, default_factory
from batchtextreview.config import ENV_DEMO, ENV_RESULTS_DIR, Settings, hf_token_present
from batchtextreview.labels import is_missing, normalise_label
from batchtextreview.logs import ENV_VERBOSE, configure_logging, verbose_from_env
from batchtextreview.store import Store, import_legacy, scan_legacy

if TYPE_CHECKING:
    from batchtextreview.evaluation import EvalReport
    from batchtextreview.models import ModelSpec
    from batchtextreview.presets import Preset

log = logging.getLogger(__name__)

PROG = "batchtextreview"

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_INTERRUPTED = 130
"""128 + SIGINT, what shells report for a Ctrl+C."""

CORRECT_COL = "CORRECT"
"""Per-row True/False column added when scoring (same name as the app's export)."""

_UNWRITABLE_SUFFIXES = frozenset({".xls", ".xlsm", ".xlsb", ".ods", ".parquet", ".json", ".jsonl"})
_EXCEL_SUFFIXES = (".xlsx", ".xlsm")
_MAX_SPARE_NAMES = 20
"""Free names tried next to an output that can't be written ("x (1).csv"...)."""
_MAX_LISTED_COLUMNS = 30
_HISTORY_DEFAULT_LIMIT = 20


class CliError(Exception):
    """A problem the user can fix: printed as one message, exit code 1."""


# ==========================================================================
# Parser
# ==========================================================================


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a whole number, got {value!r}") from None
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {number}")
    return number


def _non_negative_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a whole number, got {value!r}") from None
    if number < 0:
        raise argparse.ArgumentTypeError(f"can't be negative, got {number}")
    return number


_RUN_EXAMPLES = f"""\
examples:
  {PROG} run examples/sample_feedback.csv --preset Sentiment \\
      --model Qwen2.5-1.5B-Instruct --gold GOLD_SENTIMENT
  {PROG} run tickets.xlsx --text-column Body --preset Categorise \\
      --var "billing, delivery, account, other" --model Qwen/Qwen2.5-3B-Instruct \\
      --output tickets_categorised.xlsx
  {PROG} run notes.csv --template-file prompt.txt --model ./models/my-model --limit 100
  {PROG} run examples/sample_feedback.csv --preset Intent --model demo://fake

Finished batches are saved as they complete, so an interrupted run resumes
where it stopped and an identical re-run is instant. Ctrl+C stops after the
current batch and writes the partial results (exit code 130).
"""


def build_parser() -> argparse.ArgumentParser:
    """The ``batchtextreview`` argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            f"{APP_NAME}: run prompts over a text corpus with local LLMs, then score "
            "the results against gold labels."
        ),
        epilog=f'Run "{PROG} <command> --help" for the options of a command.',
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "-v",
        "--verbose",
        dest="log_debug",  # not "verbose": "presets -v" has its own meaning
        action="store_true",
        help="also log debug details to stderr (put it before the command)",
    )
    commands = parser.add_subparsers(dest="command", metavar="<command>", title="commands")

    # ---- ui ---------------------------------------------------------------
    ui = commands.add_parser(
        "ui",
        help="launch the web app",
        description=(
            "Launch the Streamlit app in your browser. Only this computer can open "
            "it. Arguments after -- go to "
            f'"streamlit run", e.g. "{PROG} ui -- --server.port 8502", or '
            '"-- --server.address 0.0.0.0" to share the app with your network.'
        ),
    )
    ui.add_argument("--results-dir", metavar="DIR", help=f"results folder (sets {ENV_RESULTS_DIR})")
    ui.add_argument(
        "--demo",
        action="store_true",
        help=f"offer the Demo model, which answers instantly without a GPU (sets {ENV_DEMO}=1)",
    )

    # ---- run --------------------------------------------------------------
    run = commands.add_parser(
        "run",
        help="run a preset or prompt over a file, without the browser",
        description="Run a preset or your own prompt over every row of a file and save the results.",
        epilog=_RUN_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    run.add_argument(
        "input", metavar="INPUT", help="data file: .csv .tsv .txt .xlsx .parquet .json .jsonl"
    )
    run.add_argument(
        "--sheet", metavar="NAME", help="Excel sheet to read (default: the first visible one)"
    )
    run.add_argument(
        "--text-column", metavar="COL", help="column holding the text (default: guessed)"
    )
    run.add_argument(
        "--preset",
        metavar="KEY",
        help=f'task preset, by name or key (see "{PROG} presets"); '
        "optional with --template / --template-file",
    )
    prompt = run.add_mutually_exclusive_group()
    prompt.add_argument("--template", metavar="TEXT", help="your own prompt; must contain {text}")
    prompt.add_argument("--template-file", metavar="PATH", help="read the prompt from a text file")
    run.add_argument("--var", metavar="VALUE", help="value for the {var} placeholder")
    run.add_argument(
        "--labels", metavar='"a,b,c"', help="allowed answers, overriding the preset's label set"
    )
    run.add_argument(
        "--model",
        metavar="NAME",
        required=True,
        help=f'model name, Hugging Face repo id or local folder (see "{PROG} models")',
    )
    run.add_argument(
        "--precision",
        choices=backend_base.PRECISIONS,
        default=backend_base.AUTO,
        help="weights precision (default: auto - 4bit on an NVIDIA GPU with bitsandbytes, "
        "fp16 on an NVIDIA GPU without it or on Apple silicon, fp32 on CPU; "
        f'"{PROG} device" shows the choice)',
    )
    run.add_argument(
        "--max-new-tokens",
        type=_positive_int,
        metavar="N",
        help="max output tokens (default: the preset's)",
    )
    run.add_argument(
        "--max-input-chars",
        type=_positive_int,
        default=engine.DEFAULT_MAX_INPUT_CHARS,
        metavar="N",
        help=f"truncate longer texts (default: {engine.DEFAULT_MAX_INPUT_CHARS})",
    )
    run.add_argument(
        "--batch-size",
        type=_positive_int,
        default=engine.DEFAULT_BATCH_SIZE,
        metavar="N",
        help=f"rows generated together (default: {engine.DEFAULT_BATCH_SIZE})",
    )
    run.add_argument(
        "--order",
        choices=engine.ORDERS,
        default=engine.DEFAULT_ORDER,
        help=f"which texts go first (default: {engine.DEFAULT_ORDER}) - longest: the ETA counts "
        "down and a model that runs out of memory fails in the first batch; shortest: the "
        "first results appear sooner",
    )
    run.add_argument(
        "--limit",
        type=_non_negative_int,
        default=0,
        metavar="N",
        help="only the first N rows (default: 0 = all)",
    )
    run.add_argument("--gold", metavar="COL", help="score against this column and log to history")
    run.add_argument(
        "--output",
        metavar="PATH",
        help="results file; .xlsx writes Excel (default: <input>_<preset>_<model>_results.csv)",
    )
    run.add_argument(
        "--no-cache",
        action="store_true",
        help="don't reuse or save outputs (and don't log to history)",
    )
    run.add_argument(
        "--results-dir", metavar="DIR", help=f"results folder (overrides {ENV_RESULTS_DIR})"
    )

    # ---- listings ---------------------------------------------------------
    commands.add_parser(
        "models", help="list the built-in models", description="List the built-in models."
    )

    presets_cmd = commands.add_parser(
        "presets", help="list the task presets", description="List the task presets."
    )
    presets_cmd.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="show each preset's key, description and prompt",
    )
    presets_cmd.add_argument("--results-dir", metavar="DIR", help="where saved custom presets live")

    history = commands.add_parser(
        "history",
        help="print or save the evaluation history",
        description="Print the evaluation history, or save it to a file with --output.",
    )
    history.add_argument(
        "--limit",
        type=_non_negative_int,
        metavar="N",
        help=f"newest N entries (default: {_HISTORY_DEFAULT_LIMIT} on screen, all with "
        "--output; 0 = all)",
    )
    history.add_argument(
        "--output",
        metavar="PATH",
        help="save the history, every column, to this file (.csv, or .xlsx for Excel) "
        "instead of printing it",
    )
    history.add_argument(
        "--results-dir", metavar="DIR", help=f"results folder (overrides {ENV_RESULTS_DIR})"
    )

    legacy = commands.add_parser(
        "import-legacy",
        help="import version-1 results",
        description=(
            "Import version-1 checkpoints (*.parquet + *.json) and evals.csv into the "
            "results database. Safe to run more than once; files are never changed."
        ),
    )
    legacy.add_argument(
        "folder",
        nargs="?",
        metavar="FOLDER",
        help="folder holding the version-1 files (default: the results folder)",
    )
    legacy.add_argument(
        "--results-dir", metavar="DIR", help=f"results folder (overrides {ENV_RESULTS_DIR})"
    )

    commands.add_parser(
        "device",
        help="show the GPU, precision and library versions",
        description="Show the compute device, the precision 'auto' picks, and library versions.",
    )
    return parser


# ==========================================================================
# Entry point
# ==========================================================================


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI; returns the process exit code."""
    args_list = list(sys.argv[1:] if argv is None else argv)
    passthrough: list[str] = []
    # "ui -- --server.port 8502": everything after "--" belongs to streamlit.
    lead = 0
    while args_list[lead : lead + 1] in (["-v"], ["--verbose"]):
        lead += 1
    if args_list[lead : lead + 1] == ["ui"] and "--" in args_list:
        cut = args_list.index("--")
        args_list, passthrough = args_list[:cut], args_list[cut + 1 :]

    parser = build_parser()
    args, unknown = parser.parse_known_args(args_list)
    if unknown:
        if args.command == "ui":
            passthrough = unknown + passthrough
        else:
            parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    if args.command is None:
        parser.print_help()
        return EXIT_OK
    if args.log_debug or verbose_from_env():
        _start_logging(args, None)
        log.debug("%s %s, arguments: %s", PROG, __version__, args_list)

    handlers: dict[str, Callable[[argparse.Namespace], int]] = {
        "run": _cmd_run,
        "models": _cmd_models,
        "presets": _cmd_presets,
        "history": _cmd_history,
        "import-legacy": _cmd_import_legacy,
        "device": _cmd_device,
    }
    try:
        if args.command == "ui":
            return _cmd_ui(args, passthrough)
        return handlers[args.command](args)
    except CliError as exc:
        _err(f"error: {exc}")
        return EXIT_ERROR
    except KeyboardInterrupt:
        _err("\nStopped. Finished batches are saved - run the same command again to resume.")
        return EXIT_INTERRUPTED
    except BrokenPipeError:
        # Output piped into something that stopped reading ("... | head").
        # Point stdout at devnull so the interpreter's final flush is quiet.
        with contextlib.suppress(OSError, ValueError, AttributeError):
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, sys.stdout.fileno())
        return EXIT_OK


# ==========================================================================
# ui
# ==========================================================================


def _ui_script() -> Path:
    """The Streamlit entry script.

    In a source checkout (editable install) that is the repository's
    ``app.py``: Streamlit then also reads the ``.streamlit/config.toml`` next
    to it (theme, no file watcher) whatever folder you launch from. Otherwise
    the entry bundled inside the package.
    """
    package_dir = Path(__file__).resolve().parent
    project_root = package_dir.parent.parent
    checkout_app = project_root / "app.py"
    if checkout_app.is_file() and (project_root / "pyproject.toml").is_file():
        return checkout_app
    return package_dir / "ui" / "streamlit_app.py"


def _ui_command(passthrough: Sequence[str] = ()) -> list[str]:
    """The ``streamlit run`` command line for :func:`_cmd_ui`."""
    script = _ui_script()
    command = [sys.executable, "-m", "streamlit", "run", str(script)]
    if not (script.parent / ".streamlit" / "config.toml").is_file():
        # Without the project's config: no source watching (it walks every
        # imported module, transformers included) and no usage statistics.
        command += ["--server.fileWatcherType", "none", "--browser.gatherUsageStats", "false"]
    if not os.environ.get("STREAMLIT_SERVER_ADDRESS"):
        # This computer only, whichever config applies: Streamlit's default
        # listens on every interface, and the app reads files from this disk.
        # Streamlit takes the last occurrence of an option, so
        # "ui -- --server.address 0.0.0.0" still shares it on purpose.
        command += ["--server.address", "localhost"]
    return command + list(passthrough)


def _cmd_ui(args: argparse.Namespace, passthrough: Sequence[str]) -> int:
    if importlib.util.find_spec("streamlit") is None:
        raise CliError(
            "Streamlit isn't installed in this Python environment. Install the app with "
            '"pip install -e ." (or "pip install streamlit").'
        )
    env = dict(os.environ)
    if args.results_dir:
        env[ENV_RESULTS_DIR] = str(_abspath(args.results_dir))
    if args.demo:
        env[ENV_DEMO] = "1"
    if args.log_debug:
        # Streamlit runs in its own process, which can't see -v: the app's
        # configure_logging() reads this instead.
        env[ENV_VERBOSE] = "1"
    command = _ui_command(passthrough)
    _err(f"Starting {APP_NAME} - press Ctrl+C here to stop it.")
    try:
        return subprocess.run(command, env=env, check=False).returncode
    except KeyboardInterrupt:  # the usual way to stop the server
        return EXIT_OK


# ==========================================================================
# run
# ==========================================================================


def _cmd_run(args: argparse.Namespace) -> int:
    settings = _settings(args.results_dir)
    if not args.no_cache and settings.db_path.is_file():
        # Before the first open (saved presets), which may upgrade the database.
        _start_logging(args, settings.results_dir)
    input_path = _input_path(args.input)

    # Validate everything that doesn't need the data first, so a typo fails
    # in a second rather than after reading a large file.
    preset, template = _resolve_task(args, _saved_presets(settings))
    var = args.var if prompting.VAR in template else None
    if args.var is not None and var is None:
        _err("note: --var is ignored - the prompt has no {var} placeholder.")
    final_template = _final_template(template, args.var, preset)
    if args.labels is not None:
        # An explicitly empty --labels "" means open output (exact-match scoring).
        labels = presets.parse_label_list(args.labels) or None
    else:
        labels = preset.resolve_labels(var)
    spec = _resolve_model(args.model)
    output_path, fmt = _output_target(args.output, input_path, preset, spec)
    max_new_tokens = args.max_new_tokens or preset.max_new_tokens

    table = _read_input(input_path, args.sheet)
    frame = table.frame
    text_col = _text_column(frame, args.text_column, table.name)
    gold_col = _pick_column(frame, args.gold, table.name, "gold") if args.gold else None
    working = (frame.head(args.limit) if args.limit else frame).reset_index(drop=True)
    texts = working[text_col].tolist()
    if all(engine.is_blank(text) for text in texts):
        scope = f"the first {len(texts):,} rows of" if args.limit else "every row of"
        raise CliError(
            f'Column "{text_col}" is empty in {scope} {table.name}. Choose another with --text-column.'
        )
    if gold_col is not None and labels:
        _warn_unmatched_gold(working[gold_col].tolist(), labels, gold_col)
    # Before the model loads: a locked or unwritable output fails in a
    # second, not after the whole run.
    if _check_output(output_path, args.output):
        _err(f"note: {output_path} exists - it will be replaced.")

    # Without the store, nothing is written to the results folder - not
    # even a log (records go to stderr with -v, else nowhere).
    _start_logging(args, None if args.no_cache else settings.results_dir)
    store = None if args.no_cache else _open_store(settings)
    try:
        config = engine.RunConfig.build(
            spec, final_template, max_new_tokens, args.precision, args.max_input_chars
        )
    except ValueError as exc:  # pragma: no cover - argparse restricts --precision
        raise CliError(str(exc)) from None

    plan = engine.plan_run(config, texts, store)
    model_text = spec.label
    if not spec.is_demo:
        model_text += f" ({backend_base.PRECISION_LABELS.get(config.precision, config.precision)})"
    header = (
        f"{preset.title} | {model_text} | {_count(len(texts), 'row')} of {_source_text(table)}, "
        f"column {_shown(text_col)}"
    )
    if plan.n_todo_unique:
        header += f" | {args.order} texts first"
    _err(header)
    if plan.n_truncated_rows:
        _err(
            f"note: {_count(plan.n_truncated_rows, 'text')} longer than "
            f"{config.max_input_chars:,} characters will be cut before sending "
            "(see --max-input-chars)."
        )
    manager = ModelManager(default_factory)
    cancel = threading.Event()
    progress = ProgressPrinter(sys.stderr)
    try:
        with _cancel_on_sigint(cancel, progress.note):
            result = engine.execute_run(
                config,
                texts,
                store=store,
                manager=manager,
                batch_size=args.batch_size,
                order=args.order,
                progress=progress,
                cancel=cancel,
                plan=plan,
                source="cli",
            )
    finally:
        progress.close()
        manager.unload()

    produced = result.stats.generated + result.stats.cached
    if result.aborted_reason and not produced:
        # Typically the model didn't load: there is nothing worth writing.
        raise CliError(result.aborted_reason)

    report = None
    if gold_col is not None:
        report = evaluation.evaluate(
            result.outputs,
            working[gold_col].tolist(),
            allowed=labels,
            include=results_mod.scorable_mask(result),
        )
    table_out = results_mod.build_results_frame(
        working, result, labels=labels, expand_json=preset.output_kind == "json"
    )
    if report is not None:
        table_out[results_mod.free_column_name(table_out, CORRECT_COL)] = _correct_column(
            report, len(table_out)
        )
    written = _write_output(
        table_out, output_path, fmt, kept=store is not None, spare_dir=settings.results_dir
    )

    _out(results_mod.summarise_stats(result, token_setting="--max-new-tokens"))
    _out(_timing(result))
    if result.stats.errors and not result.aborted_reason:
        _out(
            f"{_count(result.stats.errors, 'row')} failed (see the ERROR column) - "
            "run the same command again to retry just those."
        )
    _out(f"Results: {written}")

    if report is not None and gold_col is not None:
        partial = not result.complete
        _print_report(report, gold_col, partial=partial)
        if partial:
            _out("Not logged to history: the run didn't finish.")
        elif report.n_scored == 0:
            _out("Not logged to history: nothing could be scored.")
        elif store is None:
            _out("Not logged to history (--no-cache).")
        else:
            record = results_mod.history_record(
                report,
                result,
                preset_key=preset.key,
                var=var,
                dataset=table.name,
                dataset_fingerprint=data.frame_fingerprint(working),
                text_column=text_col,
                gold_column=gold_col,
                n_rows=len(working),
                source="cli",
                selection=_selection(len(working), len(frame)),
                template_edited=template != preset.template,
                preset_title=preset.title,
            )
            try:
                entry = store.log_evaluation(record)
            except Exception as exc:
                _err(f"warning: couldn't log the evaluation to history: {_describe(exc)}")
            else:
                _out(f"Logged to history as entry #{entry} ({store.path}).")

    if result.cancelled:
        _err("Cancelled - partial results written. Run the same command again to finish the rest.")
        return EXIT_INTERRUPTED
    if result.aborted_reason:
        _err(f"error: stopped early: {result.aborted_reason}")
        return EXIT_ERROR
    if result.stats.errors and not produced:
        first = next((e for e in result.errors if e), "unknown error")
        _err(f"error: no row could be generated ({first}).")
        return EXIT_ERROR
    if written != output_path:
        _err(f"error: the results are in {written}, not {output_path}.")
        return EXIT_ERROR
    return EXIT_OK


# ---- run: validation -------------------------------------------------------


def _selection(n_used: int, n_total: int) -> str:
    """Which rows ran, for the history line ("First 100 of 2,000 rows")."""
    if n_used < n_total:
        return f"First {n_used:,} of {n_total:,} rows"
    return f"All {_count(n_total, 'row')}"


def _start_logging(args: argparse.Namespace, results_dir: Path | None) -> Path | None:
    """Log to ``results_dir/logs`` (None: no file) and, with -v, to stderr.
    Without -v, ``BTR_VERBOSE`` decides (see ``logs.configure_logging``)."""
    if getattr(args, "log_debug", False):
        return configure_logging(results_dir, level=logging.DEBUG, console=True)
    return configure_logging(results_dir)


def _settings(results_dir: str | None) -> Settings:
    settings = Settings.from_env()
    if results_dir:
        settings = Settings(
            results_dir=_abspath(results_dir),
            data_dir=settings.data_dir,
            demo_mode=settings.demo_mode,
        )
    return settings


def _abspath(value: str) -> Path:
    path = Path(value).expanduser()
    try:
        return path.resolve()
    except OSError:  # pragma: no cover - unusual on every platform we support
        return path.absolute()


def _input_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.exists():
        raise CliError(f"File not found: {value}")
    if path.is_dir():
        raise CliError(f"{value} is a folder - give a data file such as a .csv or .xlsx.")
    return path


def _read_input(path: Path, sheet: str | None = None) -> data.LoadedTable:
    sheets: list[str] = []
    if path.suffix.lower() in _EXCEL_SUFFIXES:
        sheets = data.excel_sheets(path)
        if sheet and sheet not in sheets:
            # "--sheet tickets" for "Tickets": take a unique loose match.
            loose = [name for name in sheets if _loose_column(name) == _loose_column(sheet)]
            sheet = loose[0] if len(loose) == 1 else sheet
    elif sheet is not None:
        _err(f"note: --sheet is ignored - {path.name} isn't an Excel workbook.")
        sheet = None
    try:
        table = data.read_table(path, sheet=sheet)
    except data.DataLoadError as exc:
        raise CliError(str(exc)) from None
    except OSError as exc:
        raise CliError(f"Couldn't read {path}: {_describe(exc)}") from None
    if not sheet and len(sheets) > 1:
        _err(
            f'note: reading sheet "{table.sheet}" - the workbook has {len(sheets)}: '
            f"{', '.join(sheets)} (choose one with --sheet)."
        )
    return table


def _source_text(table: data.LoadedTable) -> str:
    """The file name for the header line, with the sheet, or the encoding
    when it was guessed rather than UTF-8 or marked by a byte-order mark
    (so a mis-detection shows)."""
    text = table.name
    if table.sheet:
        text += f', sheet "{table.sheet}"'
    elif table.encoding in ("cp1252", "latin-1", data.MIXED_ENCODING):
        text += f" ({data.ENCODING_LABELS.get(table.encoding, table.encoding)})"
    return text


def _resolve_task(args: argparse.Namespace, saved: Sequence[Preset]) -> tuple[Preset, str]:
    """(preset, raw template) from --preset / --template / --template-file."""
    if args.template is not None or args.template_file is not None:
        template = (
            args.template if args.template is not None else _read_template_file(args.template_file)
        )
        preset = find_preset(args.preset, saved) if args.preset else presets.get_preset("Custom")
        assert preset is not None
        return preset, template
    if not args.preset:
        raise CliError(
            f'Choose a task with --preset (see "{PROG} presets"), or give your own prompt '
            "with --template or --template-file."
        )
    preset = find_preset(args.preset, saved)
    return preset, preset.template


def _read_template_file(value: str) -> str:
    path = Path(value).expanduser()
    if not path.is_file():
        raise CliError(f"Template file not found: {value}")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise CliError(f"Couldn't read {value}: {_describe(exc)}") from None
    text = raw.decode("latin-1")  # never fails; the last resort
    # Notepad's "Unicode" is UTF-16 with a BOM.
    utf16 = ("utf-16",) if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else ()
    for encoding in (*utf16, *data.ENCODINGS):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    # Windows line endings and the editor's final newline would otherwise
    # make the "same" prompt a different run (and a cache miss).
    return text.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n")


def _final_template(template: str, var: str | None, preset: Preset) -> str:
    if prompting.VAR in template and not (var or "").strip():
        own_prompt = template != preset.template
        subject = "The prompt" if own_prompt else f'"{preset.title}"'
        what = f' for "{preset.var_label}"' if preset.var_label and not own_prompt else ""
        example = re.sub(r"^e\.g\.\s*", "", (preset.var_hint or "").strip())
        example_text = f', e.g. --var "{example}"' if example and not own_prompt else ""
        raise CliError(f"{subject} uses {{var}}{what} - pass it with --var{example_text}.")
    try:
        return prompting.apply_variable(template, var)
    except prompting.TemplateError as exc:
        message = str(exc)
        if prompting.TEXT not in template and template.strip():
            message += " Put {text} where each row's text should go."
        raise CliError(message) from None


def _loose(value: str) -> str:
    """Comparison form for preset names: case-, punctuation- and
    "(set {var})"-insensitive."""
    text = value.casefold().replace("(set {var})", "")
    return "".join(ch for ch in text if ch.isalnum())


def find_preset(name: str, saved: Sequence[Preset] = ()) -> Preset:
    """A preset by key or title (exact, then case-insensitive, then ignoring
    punctuation). Raises :class:`CliError` with close matches otherwise."""
    wanted = name.strip()
    candidates = [*presets.builtin_presets(), *saved]
    tests: list[Callable[[Preset], bool]] = [
        lambda p: wanted == p.key,
        lambda p: wanted == p.title,
        lambda p: wanted.casefold() in (p.key.casefold(), p.title.casefold()),
    ]
    loose = _loose(wanted)
    if loose:
        tests.append(lambda p: loose in (_loose(p.key), _loose(p.title)))
    for test in tests:
        for preset in candidates:
            if test(preset):
                return preset
    names: dict[str, str] = {}
    for preset in candidates:
        names.setdefault(preset.title.casefold(), preset.title)
        names.setdefault(preset.key.casefold(), preset.title)
    close = difflib.get_close_matches(wanted.casefold(), list(names), n=5, cutoff=0.6)
    suggestions = list(dict.fromkeys(names[c] for c in close))[:3]
    message = f'Unknown preset "{wanted}".'
    if suggestions:
        message += f" Did you mean {_quoted_list(suggestions, 'or')}?"
    raise CliError(f'{message} Run "{PROG} presets" to list them.')


def _resolve_model(name: str) -> ModelSpec:
    value = name.strip()
    if not value:
        raise CliError(f'Give a model with --model (see "{PROG} models").')
    spec = model_catalogue.find_model(value)
    if spec is not None:
        return spec
    known = model_catalogue.available_models(include_demo=True)
    folded = value.casefold()
    for spec in known:
        if folded in (spec.label.casefold(), spec.repo.casefold()):
            return spec
    for label, repo in model_catalogue.LEGACY_LABEL_TO_REPO.items():
        if folded == label.casefold():
            found = model_catalogue.find_model(repo)
            if found is not None:
                return found
    looks_like_path_or_repo = "/" in value or "\\" in value or value.startswith(("demo://", "."))
    if not looks_like_path_or_repo and not _is_dir(value):
        # A bare name is a valid (namespace-less) repo id, but far more often a
        # typo of a built-in model; loading would fail only after a download.
        names = {s.label.casefold(): s.label for s in known}
        # Names that start with what was typed ("Qwen2.5-1.5B") first.
        close = [c for c in names if c.startswith(folded)]
        close += [
            c
            for c in difflib.get_close_matches(folded, list(names), n=3, cutoff=0.6)
            if c not in close
        ]
        close = close[:3]
        if close:
            raise CliError(
                f'Unknown model "{value}". Did you mean '
                f'{_quoted_list([names[c] for c in close], "or")}? Run "{PROG} models" for '
                "the list, or give a Hugging Face repo id (namespace/name) or a local model folder."
            )
    try:
        return model_catalogue.resolve_model(value)
    except ValueError as exc:
        raise CliError(str(exc)) from None


def _is_dir(value: str) -> bool:
    try:
        return Path(value).expanduser().is_dir()
    except (OSError, RuntimeError, ValueError):  # unknown ~user, invalid name...
        return False


def _slug(value: str, keep: str = "", limit: int = 40) -> str:
    """Lower-case ASCII letters, digits and ``keep``, other runs as "-"."""
    ascii_text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    text = re.sub(rf"[^A-Za-z0-9{re.escape(keep)}]+", "-", ascii_text).strip("-" + keep)
    return text.lower()[:limit].strip("-" + keep) or "results"


def default_output_path(input_path: Path, preset: Preset, model: ModelSpec | None = None) -> Path:
    """``<input stem>_<preset slug>_<model slug>_results.csv`` next to the
    input file (no model part without ``model``), so runs of different
    models on the same file don't replace each other's results."""
    parts = [input_path.stem, _slug(preset.title)]
    if model is not None:
        parts.append(_slug(model.label, keep=".", limit=30))
    return input_path.with_name("_".join(parts) + "_results.csv")


def _output_target(
    value: str | None, input_path: Path, preset: Preset, model: ModelSpec | None = None
) -> tuple[Path, results_mod.ExportFormat]:
    if value and '"' in value:
        # cmd.exe/PowerShell read "D:\Results\" as D:\Results" - the backslash
        # escapes the closing quote.
        raise CliError(
            f"--output {value} contains a double quote. If you ended a folder with a "
            'backslash inside quotes ("D:\\Results\\"), leave that backslash out.'
        )
    if value:
        path = Path(value).expanduser()
        if path.is_dir() or value.endswith(("/", "\\")):
            path = path / default_output_path(input_path, preset, model).name
    else:
        path = default_output_path(input_path, preset, model)
    suffix = path.suffix.lower()
    if suffix in _UNWRITABLE_SUFFIXES:
        raise CliError(f"Can't write {suffix} files - use a .csv or .xlsx --output.")
    fmt: results_mod.ExportFormat = "xlsx" if suffix == ".xlsx" else "csv"
    if _same_file(path, input_path):
        raise CliError(f"--output {value} would overwrite the input file. Choose another name.")
    return path, fmt


def _check_output(path: Path, value: str | None) -> bool:
    """Make sure ``path`` can be written (its folder is created), so a
    locked file (open in Excel), a read-only folder or an invalid name
    fails before the model loads. True if the file already exists."""
    what = f"--output {value}" if value else str(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except (FileExistsError, NotADirectoryError):
        folders = [*reversed(path.parent.parents), path.parent]
        blocker = next((p for p in folders if p.exists() and not p.is_dir()), path.parent)
        raise CliError(f"Can't write {what}: {blocker} is a file, not a folder.") from None
    except OSError as exc:
        raise CliError(f"Can't create the folder for {what}: {_describe(exc)}.") from None
    try:
        if path.exists():
            if path.is_dir():
                raise CliError(f"Can't write {what}: it is a folder.")
            with open(path, "ab"):  # Excel's lock makes this fail on Windows
                pass
            return True
        with open(path, "xb"):
            pass
        with contextlib.suppress(OSError):
            path.unlink()
        return False
    except OSError as exc:
        hint = " If it is open in Excel, close it and try again." if _is_locked(exc) else ""
        raise CliError(f"Can't write {what}: {_describe(exc)}.{hint}") from None


def _is_locked(exc: OSError) -> bool:
    """Excel's lock on an open file shows as a permission error on Windows."""
    return isinstance(exc, PermissionError)


def _same_file(a: Path, b: Path) -> bool:
    try:
        if a.exists() and b.exists():
            return os.path.samefile(a, b)
        return a.resolve() == b.resolve()
    except OSError:
        return False


def _columns_hint(frame: pd.DataFrame) -> str:
    columns = [_shown(c) for c in frame.columns]
    shown = ", ".join(columns[:_MAX_LISTED_COLUMNS])
    if len(columns) > _MAX_LISTED_COLUMNS:
        shown += f" and {len(columns) - _MAX_LISTED_COLUMNS:,} more"
    return f"Available columns: {shown}."


def _shown(name: Any) -> str:
    """A column name on one line: a wrapped Excel header ("Gold<Alt+Enter>
    Sentiment") or a non-breaking space shows as a plain space."""
    return " ".join(str(name).split())


def _loose_column(name: str) -> str:
    return _shown(name).casefold()


def _pick_column(frame: pd.DataFrame, wanted: str, file_name: str, role: str) -> str:
    """The column named ``wanted``: exact, else the unique match ignoring
    case, else ignoring case and how the words are spaced (line breaks in
    a wrapped Excel header, non-breaking spaces)."""
    columns = [str(c) for c in frame.columns]
    if wanted in columns:
        return wanted
    folded = [c for c in columns if c.casefold() == wanted.strip().casefold()]
    if len(folded) == 1:
        return folded[0]
    loose = [c for c in columns if _loose_column(c) == _loose_column(wanted)]
    if len(loose) == 1:
        return loose[0]
    message = f'No {role} column "{wanted}" in {file_name}.'
    shown = list(dict.fromkeys(_shown(c) for c in columns))
    close = difflib.get_close_matches(_shown(wanted), shown, n=1, cutoff=0.6)
    if close:
        message += f' Did you mean "{close[0]}"?'
    raise CliError(f"{message} {_columns_hint(frame)}")


def _text_column(frame: pd.DataFrame, wanted: str | None, file_name: str) -> str:
    if wanted:
        return _pick_column(frame, wanted, file_name, "text")
    guessed = data.guess_text_column(frame)
    if guessed is None:  # pragma: no cover - read_table never returns an empty table
        raise CliError(f"Couldn't find a text column in {file_name}. {_columns_hint(frame)}")
    if len(frame.columns) > 1:
        _err(f'note: using text column "{_shown(guessed)}" (choose another with --text-column).')
    return str(guessed)


def _warn_unmatched_gold(golds: Sequence[Any], labels: Sequence[str], gold_col: str) -> None:
    """Warn up front when no gold value is one of the labels (nothing would score)."""
    present = [g for g in golds if not is_missing(g)]
    if present and all(normalise_label(g, labels) is None for g in present):
        _err(
            f'warning: none of the values in "{_shown(gold_col)}" is one of the labels '
            f"({', '.join(labels)}), so nothing can be scored. Check --gold and --labels."
        )


# ---- run: store -------------------------------------------------------------


def _open_store(settings: Settings) -> Store:
    try:
        return Store(settings.db_path)
    except Exception as exc:
        raise CliError(
            f"Couldn't open the results database {settings.db_path}: {_describe(exc)}"
        ) from None


def _existing_store(settings: Settings) -> Store | None:
    """The store if its database already exists (never creates one)."""
    if not settings.db_path.is_file():
        return None
    return _open_store(settings)


def _saved_presets(settings: Settings) -> list[Preset]:
    """Custom presets saved from the app; [] when there is no database.

    Read even with --no-cache: that flag is about cached outputs."""
    try:
        store = _existing_store(settings)
        return store.list_presets() if store is not None else []
    except Exception as exc:
        _err(f"warning: couldn't read saved presets: {_describe(exc)}")
        return []


# ---- run: output ------------------------------------------------------------


def _correct_column(report: EvalReport, n_rows: int) -> pd.Series:
    correct: list[bool | None] = [None] * n_rows
    for row, ok in zip(report.rows["row"], report.rows["correct"], strict=True):
        correct[int(row)] = bool(ok)
    return pd.Series(correct, dtype=object)


def _write_output(
    frame: pd.DataFrame,
    path: Path,
    fmt: results_mod.ExportFormat,
    *,
    kept: bool,
    spare_dir: Path | None = None,
) -> Path:
    """Write the results to ``path`` - or, if that fails (the file was
    opened in Excel during the run), to a free name next to it, in
    ``spare_dir`` or in the temp folder, with a warning: the run's outputs
    may exist nowhere else (``kept`` False: --no-cache). Returns where they
    went."""
    payload, _mime = results_mod.export_bytes(frame, fmt)
    try:
        _replace_file(path, payload)
        return path
    except OSError as exc:
        error = exc
    reason = _describe(error) + (" - is it open in Excel?" if _is_locked(error) else "")
    for spare in _spare_paths(path, spare_dir):
        try:
            _replace_file(spare, payload)
        except OSError:
            continue
        _err(f"warning: couldn't write {path} ({reason}). Saved the results as {spare} instead.")
        return spare
    rerun = (
        "finished rows are saved, so the re-run is quick."
        if kept
        else "nothing was saved (--no-cache), so the run has to be repeated."
    )
    raise CliError(f"Couldn't write {path} ({reason}) or anywhere else - {rerun}")


def _replace_file(path: Path, payload: bytes) -> None:
    """Write via a temporary file in the same folder, so a failed write
    never leaves the previous results cut short."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".batchtextreview-{os.getpid()}.tmp")  # short: MAX_PATH
    try:
        temp.write_bytes(payload)
        os.replace(temp, path)
    finally:
        with contextlib.suppress(OSError):
            temp.unlink()


def _spare_paths(path: Path, spare_dir: Path | None) -> Iterator[Path]:
    """Free names for results that can't go to ``path``: "name (1).csv" ...
    next to it, then the same name in ``spare_dir`` and the temp folder."""
    for number in range(1, _MAX_SPARE_NAMES + 1):
        candidate = path.with_name(f"{path.stem} ({number}){path.suffix}")
        if not candidate.exists():
            yield candidate
            break
    for folder in (spare_dir, Path(tempfile.gettempdir())):
        if folder is None:
            continue
        for number in range(_MAX_SPARE_NAMES + 1):
            name = path.name if not number else f"{path.stem} ({number}){path.suffix}"
            if not (folder / name).exists():
                yield folder / name
                break


def _timing(result: engine.RunResult) -> str:
    stats = result.stats
    text = f"Took {_duration(stats.elapsed_s)}"
    if stats.load_s >= 0.5:
        text += f" ({_duration(stats.load_s)} loading the model)"
    generating = stats.elapsed_s - stats.load_s
    if stats.generated and generating > 0:
        text += f", {_rate(stats.generated / generating)}"
    return text + "."


def _print_report(report: EvalReport, gold_col: str, partial: bool = False) -> None:
    how = f"labels: {', '.join(report.labels)}" if report.mode == "labels" else "exact match"
    _out("")
    _out(f"Evaluation against {gold_col} ({how}){' - partial run' if partial else ''}")
    if report.n_scored == 0:
        _out("  Nothing could be scored.")
    else:
        _out(
            f"  Accuracy   {_percent(report.accuracy)}  "
            f"({report.n_correct:,} of {report.n_scored:,} correct)"
        )
        if report.macro_f1 is not None:
            _out(f"  Macro-F1   {_number(report.macro_f1)}")
        if report.kappa is not None:
            _out(f"  Kappa      {_number(report.kappa)}")
    if report.n_pred_invalid:
        _out(
            f"  {_count(report.n_pred_invalid, 'invalid output')} (not one of the labels; counted as wrong)"
        )
    if report.mode == "exact" and len(report.rows):
        # No label set, so no "invalid": the only unresolved prediction is
        # an empty one (a think-only or blank generation).
        n_empty = int((report.rows["predicted"] == evaluation.INVALID).sum())
        if n_empty:
            _out(f"  {_count(n_empty, 'empty output')} (counted as wrong)")
    if report.n_excluded:
        _out(f"  {_count(report.n_excluded, 'row')} excluded (blank text, failed or not run)")
    if report.n_gold_unscorable:
        reason = "missing or not one of the labels" if report.mode == "labels" else "missing"
        _out(f"  {_count(report.n_gold_unscorable, 'gold value')} unscorable ({reason})")
    if report.mode == "labels" and report.n_scored and report.per_class:
        rows = [
            [m.label, _number(m.precision), _number(m.recall), _number(m.f1), f"{m.support:,}"]
            for m in report.per_class
        ]
        _out("")
        _out(
            _format_table(
                ["LABEL", "PRECISION", "RECALL", "F1", "SUPPORT"], rows, "lrrrr", indent=2
            )
        )


# ==========================================================================
# models / presets / history / import-legacy / device
# ==========================================================================


def _cmd_models(args: argparse.Namespace) -> int:
    rows = []
    for spec in model_catalogue.available_models(include_demo=True):
        notes = [spec.tier] if spec.tier else []
        if spec.gated:
            notes.append("gated: needs HF_TOKEN")
        if spec.experimental:
            notes.append("experimental")
        rows.append([spec.label, spec.size_label or "-", spec.repo, ", ".join(notes)])
    _out(_format_table(["MODEL", "SIZE", "REPO", "NOTES"], rows, "lrll"))
    _out("")
    _out(
        "--model takes the name, the repo id, a version-1 name, any other Hugging Face "
        "repo id, or a local model folder. Every built-in model fits an 8 GB GPU at 4-bit."
    )
    return EXIT_OK


def _answers(preset: Preset) -> str:
    if preset.labels_from_var:
        return "one of your --var list"
    if preset.labels:
        return ", ".join(preset.labels)
    if preset.output_kind == "json":
        return "JSON"
    return "free text"


def _cmd_presets(args: argparse.Namespace) -> int:
    settings = _settings(args.results_dir)
    if settings.db_path.is_file():  # listing never creates the results folder
        _start_logging(args, settings.results_dir)
    saved = _saved_presets(settings)
    all_presets = [*presets.builtin_presets(), *saved]
    if args.verbose:
        for preset in all_presets:
            _out(f"{preset.title}  [{preset.category}]")
            if preset.key != preset.title:
                _out(f"  key:         {preset.key}")
            if preset.description:
                _out(f"  description: {preset.description}")
            _out(f"  answers:     {_answers(preset)}")
            _out(f"  max tokens:  {preset.max_new_tokens}")
            if preset.needs_var:
                _out(f"  --var:       {preset.var_label or 'value for {var}'}")
            _out("  template:")
            for line in preset.template.splitlines() or [""]:
                _out(f"    {line}")
            _out("")
    else:
        rows = [
            [
                preset.title,
                preset.category if preset.builtin else "Saved",
                str(preset.max_new_tokens),
                _clip(_answers(preset), 56),
                "--var" if preset.needs_var else "",
            ]
            for preset in all_presets
        ]
        _out(_format_table(["PRESET", "CATEGORY", "TOKENS", "ANSWERS", "NEEDS"], rows, "llrll"))
        _out("")
    _out(
        '--preset takes the name or the key (e.g. "Categorise" or "Categorise (set {var})"). '
        f'"{PROG} presets -v" shows the prompts.'
    )
    return EXIT_OK


def _cmd_history(args: argparse.Namespace) -> int:
    settings = _settings(args.results_dir)
    output = _history_output(args.output) if args.output else None
    if not settings.db_path.is_file():  # never creates a database
        _out(f"No evaluation history yet ({settings.db_path} doesn't exist).")
        return EXIT_OK
    _start_logging(args, settings.results_dir)
    store = _open_store(settings)
    frame = store.list_evaluations()
    if frame.empty:
        _out("No evaluations logged yet. Score a run with --gold, or in the app.")
        return EXIT_OK
    if output is not None:
        path, fmt = output
        chosen = frame if not args.limit else frame.head(args.limit)
        payload, _mime = results_mod.export_bytes(_history_export(chosen), fmt)
        try:
            _replace_file(path, payload)
        except OSError as exc:
            reason = _describe(exc) + (" - is it open in Excel?" if _is_locked(exc) else "")
            raise CliError(f"Couldn't write {path} ({reason}).") from exc
        _out(f"Saved {_count(len(chosen), 'evaluation')} to {path}")
        return EXIT_OK
    limit = _HISTORY_DEFAULT_LIMIT if args.limit is None else args.limit
    shown = frame if limit == 0 else frame.head(limit)
    saved = _saved_presets(settings)
    rows = []
    for record in shown.to_dict("records"):
        rows.append(
            [
                _when(record.get("created_at")),
                _clip(_text(record.get("model_label")) or _text(record.get("model_id")), 28),
                _clip(_task_text(record, saved), 40),
                _clip(_text(record.get("dataset")), 28),
                _clip(_text(record.get("gold_column")), 18),
                _integer(record.get("n_scored")),
                _integer(record.get("n_invalid")),
                _percent(record.get("accuracy")),
                _number(record.get("macro_f1")),
                _number(record.get("kappa")),
                _text(record.get("source")),
            ]
        )
    headers = [
        "WHEN",
        "MODEL",
        "TASK",
        "DATASET",
        "GOLD",
        "SCORED",
        "INVALID",
        "ACCURACY",
        "MACRO-F1",
        "KAPPA",
        "SOURCE",
    ]
    _out(_format_table(headers, rows, "lllllrrrrrl"))
    if len(shown) < len(frame):
        _out(f"\nShowing the newest {len(shown):,} of {len(frame):,} - use --limit 0 for all.")
    return EXIT_OK


def _history_output(value: str) -> tuple[Path, results_mod.ExportFormat]:
    """Where ``history --output`` writes, checked before anything is read."""
    path = Path(value).expanduser()
    if path.is_dir() or value.endswith(("/", "\\")):
        raise CliError(f"--output {value} is a folder - give a file name such as history.csv.")
    suffix = path.suffix.lower()
    if suffix in _UNWRITABLE_SUFFIXES:
        raise CliError(f"Can't write {suffix} files - use a .csv or .xlsx --output.")
    return path, "xlsx" if suffix == ".xlsx" else "csv"


def _history_export(frame: pd.DataFrame) -> pd.DataFrame:
    """The history as stored, one column per field (labels as "a, b, c")."""
    out = frame.copy()
    out["labels"] = [
        ", ".join(str(label) for label in value) if isinstance(value, list) else ""
        for value in out["labels"]
    ]
    return out


def _task_text(record: dict[str, Any], saved: list[Preset]) -> str:
    """A history line's task: the preset title with the {var} value it ran
    with ("Categorise · praise, complaint, other"), so runs of one preset
    with different categories or questions can be told apart."""
    key = _text(record.get("preset"))
    if not key:
        return "-"
    preset = presets.get_preset(key, saved)
    title = preset.title if preset is not None else _text(record.get("preset_title")) or key
    var = " ".join(_text(record.get("var")).split())
    return f"{title} · {var}" if var else title


def _cmd_import_legacy(args: argparse.Namespace) -> int:
    settings = _settings(args.results_dir)
    folder = _abspath(args.folder) if args.folder else settings.results_dir
    if not folder.is_dir():
        raise CliError(f"Folder not found: {folder}")
    scan = scan_legacy(folder)
    if not scan.found:
        _out(
            f"No version-1 results found in {folder} (looked for <run id>.parquet + "
            "<run id>.json checkpoints and evals.csv)."
        )
        for orphan in scan.orphan_checkpoints:
            _out(f"  skipped {orphan.name}: no {orphan.stem}.json manifest next to it.")
        return EXIT_OK
    _start_logging(args, settings.results_dir)
    store = _open_store(settings)
    report = import_legacy(store, folder)
    _out(
        f"Imported {_count(report.runs_imported, 'run')} "
        f"({_count(report.outputs_imported, 'output')}) and "
        f"{_count(report.evaluations_imported, 'history entry', 'history entries')} "
        f"from {folder} into {store.path}."
    )
    if report.skipped:
        _out(f"Skipped {len(report.skipped):,}:")
        for reason in report.skipped:
            _out(f"  - {reason}")
    if not (report.runs_imported or report.outputs_imported or report.evaluations_imported):
        _out("Nothing new to import - it may have been imported already.")
    return EXIT_OK


def _cmd_device(args: argparse.Namespace) -> int:
    info = backend_base.device_info()
    auto = backend_base.resolve_precision(backend_base.AUTO, info)
    if info.torch_version is None:
        torch_text = "not installed" if info.has_torch else info.name
    elif info.cuda_version:
        torch_text = f"{info.torch_version} (built for CUDA {info.cuda_version})"
    elif info.kind == "cpu":
        torch_text = f"{info.torch_version} (CPU-only build)"
    else:
        torch_text = info.torch_version
    rows = [
        [APP_NAME, __version__],
        ["Python", f"{platform.python_version()} on {platform.platform(terse=True)}"],
        [
            "Device",
            f"{info.summary} [{info.kind}]" if info.has_torch else "none usable - demo model only",
        ],
        ["Auto precision", f"{auto} - {backend_base.PRECISION_LABELS.get(auto, auto)}"],
        ["PyTorch", torch_text],
        ["Transformers", info.transformers_version or _package_version("transformers")],
        ["bitsandbytes", "available" if info.bitsandbytes else "not available (no 4-bit)"],
        ["Streamlit", _package_version("streamlit")],
        ["pandas", _package_version("pandas")],
        [
            "Hugging Face token",
            "set" if hf_token_present() else "not set (needed for gated models)",
        ],
        ["Results database", str(Settings.from_env().db_path)],
    ]
    _out(_format_table(["", ""], rows, "ll", header=False))
    return EXIT_OK


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


# ==========================================================================
# Progress and Ctrl+C
# ==========================================================================


class ProgressPrinter:
    """Engine progress callback that reports on a text stream (stderr).

    On a terminal: one line, redrawn in place (at most ~10 times a second).
    Otherwise (redirected to a file or a CI log): plain lines - one per
    phase, then one per 10% of the work (at most one every 2 seconds, so a
    quick run stays quiet) and at least one every 30 seconds. The
    percentage (and the bar) is ``Progress.work_fraction``: the work done,
    which weighs a long text more than a short one, not the share of rows.

    Never raises: an exception from a progress callback would stop the run.
    """

    def __init__(
        self,
        stream: IO[str],
        *,
        interactive: bool | None = None,
        clock: Callable[[], float] = time.monotonic,
        min_redraw_s: float = 0.1,
        plain_min_gap_s: float = 2.0,
        plain_every_s: float = 30.0,
    ) -> None:
        self.stream = stream
        self.interactive = _isatty(stream) if interactive is None else interactive
        self._clock = clock
        self._min_redraw_s = min_redraw_s
        self._plain_min_gap_s = plain_min_gap_s
        self._plain_every_s = plain_every_s
        self._shown = 0
        """Width of the transient line currently on screen (interactive)."""
        self._last_draw = -math.inf
        self._last_plain = clock()
        self._next_step = 0.1
        self._last: engine.Progress | None = None

    def __call__(self, progress: engine.Progress) -> None:
        # Never stop a run over a display glitch.
        with contextlib.suppress(Exception):
            self._update(progress)

    def note(self, message: str) -> None:
        """Print a permanent line without breaking the progress line."""
        try:
            self._clear()
            _write(self.stream, message + "\n")
            if self._last is not None and self._last.phase == "generating":
                self._draw(_progress_text(self._last, bar=True))
        except Exception:  # pragma: no cover
            pass

    def close(self) -> None:
        """Remove the transient line (call before printing the summary)."""
        with contextlib.suppress(Exception):
            self._clear()

    # ---- internals ---------------------------------------------------------
    def _update(self, progress: engine.Progress) -> None:
        previous, self._last = self._last, progress
        if progress.phase == "finished":
            self._clear()
            return
        if progress.phase in ("planning", "loading"):
            if progress.message and (previous is None or previous.message != progress.message):
                if self.interactive and progress.phase == "loading":
                    self._draw(progress.message)
                else:
                    self.note(progress.message)
            return
        now = self._clock()
        if self.interactive:
            done = progress.done_rows >= progress.total_rows
            if done or now - self._last_draw >= self._min_redraw_s:
                self._draw(_progress_text(progress, bar=True))
                self._last_draw = now
            return
        fraction = _work_fraction(progress)
        since = now - self._last_plain
        crossed = fraction >= self._next_step and since >= self._plain_min_gap_s
        if crossed or since >= self._plain_every_s:
            _write(self.stream, _progress_text(progress, bar=False) + "\n")
            self._last_plain = now
            while self._next_step <= fraction:
                self._next_step += 0.1

    def _draw(self, text: str) -> None:
        if not self.interactive:
            return
        width = max(20, shutil.get_terminal_size((100, 24)).columns - 1)
        line = text if len(text) <= width else text[: width - 1] + "…"
        padding = " " * max(0, self._shown - len(line))
        _write(self.stream, "\r" + line + padding)
        self._shown = len(line)

    def _clear(self) -> None:
        if self.interactive and self._shown:
            _write(self.stream, "\r" + " " * self._shown + "\r")
            self._shown = 0


def _work_fraction(progress: engine.Progress) -> float:
    """``progress.work_fraction``, kept within 0-1."""
    return min(max(progress.work_fraction, 0.0), 1.0)


def _progress_text(progress: engine.Progress, bar: bool) -> str:
    total = progress.total_rows
    fraction = _work_fraction(progress)
    percent = int(fraction * 100)
    parts = [f"{progress.done_rows:,}/{total:,} rows"]
    if progress.rows_per_s:
        parts.append(_rate(progress.rows_per_s))
    if progress.eta_s is not None and progress.done_rows < total:
        parts.append(f"ETA {_duration(progress.eta_s)}")
    if progress.error_rows:
        parts.append(f"{progress.error_rows:,} failed")
    if bar:
        filled = int(fraction * 20)
        meter = "█" * filled + "░" * (20 - filled)
        return f"Generating {meter} {percent:3d}%  " + " · ".join(parts)
    return f"Generating: {percent}% - " + ", ".join(parts)


@contextlib.contextmanager
def _cancel_on_sigint(cancel: threading.Event, notify: Callable[[str], None]) -> Iterator[None]:
    """While active, the first Ctrl+C sets ``cancel`` (the engine stops after
    the current batch); a second one raises KeyboardInterrupt at once."""

    def handler(signum: int, frame: Any) -> None:
        if cancel.is_set():
            raise KeyboardInterrupt
        cancel.set()
        notify("Stopping after the current batch - press Ctrl+C again to quit now.")

    previous: Any = None
    installed = False
    if threading.current_thread() is threading.main_thread():
        try:
            previous = signal.signal(signal.SIGINT, handler)
            installed = True
        except (ValueError, OSError):  # pragma: no cover - embedded interpreters
            pass
    try:
        yield
    finally:
        if installed:
            signal.signal(
                signal.SIGINT, previous if previous is not None else signal.default_int_handler
            )


# ==========================================================================
# Output helpers
# ==========================================================================


def _write(stream: IO[str], text: str) -> None:
    """Write, replacing what the stream's encoding can't hold (a cp1252
    console or a redirected Windows pipe)."""
    try:
        stream.write(text)
    except UnicodeEncodeError:
        encoding = getattr(stream, "encoding", None) or "ascii"
        stream.write(text.encode(encoding, errors="replace").decode(encoding, errors="replace"))
    with contextlib.suppress(Exception):
        stream.flush()


def _out(text: str = "") -> None:
    _write(sys.stdout, text + "\n")


def _err(text: str) -> None:
    _write(sys.stderr, text + "\n")


def _isatty(stream: Any) -> bool:
    try:
        return bool(stream.isatty())
    except Exception:
        return False


def _format_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    align: str = "",
    indent: int = 0,
    header: bool = True,
) -> str:
    """Plain aligned columns; ``align`` has one "l"/"r" per column."""
    table = [list(headers), *[list(r) for r in rows]] if header else [list(r) for r in rows]
    widths = [max(len(str(row[i])) for row in table) for i in range(len(headers))]
    lines = []
    for row in table:
        cells = []
        for i, cell in enumerate(row):
            text = str(cell)
            right = i < len(align) and align[i] == "r"
            cells.append(text.rjust(widths[i]) if right else text.ljust(widths[i]))
        lines.append((" " * indent + "  ".join(cells)).rstrip())
    return "\n".join(lines)


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _text(value: Any) -> str:
    return "" if is_missing(value) else str(value)


def _when(value: Any) -> str:
    text = _text(value)
    return text[:16].replace("T", " ")


def _integer(value: Any) -> str:
    if is_missing(value):
        return "-"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def _number(value: Any) -> str:
    if is_missing(value):
        return "-"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def _percent(value: Any) -> str:
    if is_missing(value):
        return "-"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return str(value)


def _count(n: int, noun: str, plural: str | None = None) -> str:
    return f"{n:,} {noun}" if n == 1 else f"{n:,} {plural or noun + 's'}"


def _quoted_list(items: Sequence[str], conjunction: str) -> str:
    quoted = [f'"{item}"' for item in items]
    if len(quoted) <= 1:
        return "".join(quoted)
    return f"{', '.join(quoted[:-1])} {conjunction} {quoted[-1]}"


def _duration(seconds: float) -> str:
    total = max(0, round(seconds))
    if seconds < 9.95:
        return f"{max(seconds, 0):.1f}s"
    if total < 60:
        return f"{total}s"
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def _rate(rows_per_s: float) -> str:
    if rows_per_s >= 10:
        return f"{rows_per_s:,.0f} rows/s"
    if rows_per_s >= 1:
        return f"{rows_per_s:.1f} rows/s"
    if rows_per_s > 0:
        return f"{1 / rows_per_s:.1f}s per row"
    return "0 rows/s"


def _describe(exc: BaseException) -> str:
    text = " ".join(str(exc).split())
    return text or type(exc).__name__


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
