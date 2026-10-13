"""Where the package's log records go.

Everything Batch Text Review logs goes through the ``batchtextreview``
logger (module loggers such as ``batchtextreview.engine`` are its
children). :func:`configure_logging` points it at a rotating file in the
results folder - ``<results>/logs/batchtextreview.log``, 1 MB x 3 - and
optionally at stderr, so a run that failed overnight can be diagnosed
without a terminal.

The Streamlit app calls it on every rerun and the CLI at the start of each
command that uses the results folder: repeated calls never stack handlers,
and a call with another folder moves the file. It never raises - logging is
never a reason for the app to fail.

``BTR_VERBOSE=1`` in the environment does what the CLI's ``-v`` does (debug
details, to stderr too) for callers that leave ``level`` / ``console``
unset - which is how ``batchtextreview -v ui`` reaches the Streamlit
process it starts.
"""

from __future__ import annotations

import contextlib
import logging
import logging.handlers
import os
import sys
import threading
import time
from pathlib import Path

LOGGER_NAME = "batchtextreview"
LOG_DIRNAME = "logs"
LOG_FILENAME = "batchtextreview.log"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
MAX_BYTES = 1024 * 1024
BACKUP_COUNT = 3
ENV_VERBOSE = "BTR_VERBOSE"
"""``1`` makes :func:`configure_logging` default to debug level and stderr."""
ROLLOVER_RETRY_S = 60.0
"""After a rollover failed (another process has the file open), how long the
file just keeps growing before the next attempt."""

# Our handlers are recognised by name rather than kept in module globals, so
# a reloaded module (Streamlit's file watcher) still finds the ones already
# attached instead of adding a second set.
_FILE_HANDLER = "batchtextreview-file"
_CONSOLE_HANDLER = "batchtextreview-console"
_QUIET_HANDLER = "batchtextreview-quiet"
_lock = threading.Lock()


def log_path(results_dir: str | os.PathLike[str]) -> Path:
    """``<results_dir>/logs/batchtextreview.log`` (absolute)."""
    return Path(os.path.abspath(Path(results_dir).expanduser() / LOG_DIRNAME / LOG_FILENAME))


def configure_logging(
    results_dir: Path | None = None,
    *,
    level: int | None = None,
    console: bool | None = None,
) -> Path | None:
    """Send the ``batchtextreview`` logger's records to a log file in
    ``results_dir`` (and to stderr with ``console``). Returns the log file's
    path, or None when there is no file.

    * ``level`` / ``console`` left as None: INFO and no stderr - or DEBUG
      and stderr when ``BTR_VERBOSE`` is set (see :func:`verbose_from_env`).
    * Only the ``batchtextreview`` logger is touched (level set to
      ``level``, ``propagate`` off); the root logger and other libraries'
      loggers are left alone.
    * File: a ``RotatingFileHandler`` at ``results_dir/logs/batchtextreview.log``
      (1 MB, 3 backups, UTF-8, ``"%(asctime)s %(levelname)s %(name)s:
      %(message)s"``). The folder is created if needed; if it can't be
      written, there is simply no file handler. ``results_dir=None`` removes
      the file handler. The app and the CLI may share the file: when it
      can't be rotated because another process has it open (Windows), it
      keeps growing and rotation is retried a minute later, and a write
      error is reported on stderr once rather than for every record.
    * Idempotent: safe to call on every Streamlit rerun. The same folder
      keeps the open handler; another folder replaces it; ``console=False``
      removes a stderr handler added earlier.
    * With neither a file nor stderr, records are dropped rather than left
      to Python's last-resort handler (which would print warnings and
      tracebacks into the CLI's own output).

    Never raises.
    """
    try:
        verbose = verbose_from_env()
        if level is None:
            level = logging.DEBUG if verbose else logging.INFO
        if console is None:
            console = verbose
        with _lock:
            logger = logging.getLogger(LOGGER_NAME)
            logger.setLevel(level)
            logger.propagate = False
            formatter = logging.Formatter(LOG_FORMAT)
            path = _configure_file(logger, results_dir, formatter)
            has_console = _configure_console(logger, console, formatter)
            _configure_quiet(logger, path is None and not has_console)
            return path
    except Exception:  # pragma: no cover - defensive: logging must never break the app
        return None


def current_log_file() -> Path | None:
    """The file :func:`configure_logging` is writing to, or None."""
    handler = _find(logging.getLogger(LOGGER_NAME), _FILE_HANDLER)
    if isinstance(handler, logging.FileHandler):
        return Path(handler.baseFilename)
    return None


def verbose_from_env() -> bool:
    """True when ``BTR_VERBOSE`` is "1" / "true" / "yes" / "on"."""
    value = os.environ.get(ENV_VERBOSE, "").strip().strip("\"'").strip()
    return value.lower() in {"1", "true", "yes", "on"}


class _SharedRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """A ``RotatingFileHandler`` for a file other processes may have open.

    The app and the CLI log to the same file. Windows can't rename a file
    another process has open, so there the stock rollover fails - and then
    fails again on every later record, each time printing "--- Logging
    error ---" and a traceback to stderr, dropping the record and shifting
    one more backup out of existence. Here the live file is moved first
    (nothing has changed if that fails), a failed rollover just keeps
    appending and is retried after :data:`ROLLOVER_RETRY_S`, and a write
    error is reported once rather than for every record.
    """

    def __init__(
        self, filename: str | os.PathLike[str], *, max_bytes: int, backup_count: int
    ) -> None:
        super().__init__(filename, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
        self._retry_at = 0.0
        self._reported = False

    def shouldRollover(self, record: logging.LogRecord) -> int:
        if time.monotonic() < self._retry_at:
            return False
        return super().shouldRollover(record)

    def doRollover(self) -> None:
        if self.stream:
            self.stream.close()
            self.stream = None
        try:
            self._rotate()
        except OSError:
            self._retry_at = time.monotonic() + ROLLOVER_RETRY_S
        self.stream = self._open()

    def _rotate(self) -> None:
        if self.backupCount <= 0:
            return
        base = self.baseFilename
        aside = f"{base}.rotating"
        os.replace(base, aside)  # the step that fails while the file is open elsewhere
        for index in range(self.backupCount - 1, 0, -1):
            older = f"{base}.{index}"
            if os.path.exists(older):
                os.replace(older, f"{base}.{index + 1}")
        os.replace(aside, f"{base}.1")

    def handleError(self, record: logging.LogRecord) -> None:
        if isinstance(sys.exc_info()[1], OSError):
            if self._reported:
                return
            self._reported = True
        super().handleError(record)


# --------------------------------------------------------------------------
# Helpers (called with _lock held)
# --------------------------------------------------------------------------


def _find(logger: logging.Logger, name: str) -> logging.Handler | None:
    return next((h for h in logger.handlers if h.get_name() == name), None)


def _remove(logger: logging.Logger, handler: logging.Handler | None) -> None:
    if handler is None:
        return
    logger.removeHandler(handler)
    with contextlib.suppress(Exception):  # a file already gone, say
        handler.close()


def _configure_file(
    logger: logging.Logger, results_dir: Path | None, formatter: logging.Formatter
) -> Path | None:
    current = _find(logger, _FILE_HANDLER)
    if results_dir is None:
        _remove(logger, current)
        return None
    try:
        target = log_path(results_dir)
    except Exception:  # "~nobody/results": no such home folder
        _remove(logger, current)
        return None
    if isinstance(current, logging.FileHandler) and Path(current.baseFilename) == target:
        current.setFormatter(formatter)
        return target
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        handler = _SharedRotatingFileHandler(target, max_bytes=MAX_BYTES, backup_count=BACKUP_COUNT)
    except Exception:
        # Read-only folder, a file where the folder should be, a bad path...
        _remove(logger, current)
        return None
    handler.set_name(_FILE_HANDLER)
    handler.setFormatter(formatter)
    _remove(logger, current)
    logger.addHandler(handler)
    return target


def _configure_console(logger: logging.Logger, console: bool, formatter: logging.Formatter) -> bool:
    current = _find(logger, _CONSOLE_HANDLER)
    stream = sys.stderr
    if not console or stream is None:  # pythonw.exe has no stderr
        _remove(logger, current)
        return False
    if isinstance(current, logging.StreamHandler) and current.stream is stream:
        current.setFormatter(formatter)
        return True
    # sys.stderr was replaced since (a test's capture, a redirect): follow it.
    _remove(logger, current)
    handler = logging.StreamHandler(stream)
    handler.set_name(_CONSOLE_HANDLER)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return True


def _configure_quiet(logger: logging.Logger, wanted: bool) -> None:
    current = _find(logger, _QUIET_HANDLER)
    if not wanted:
        _remove(logger, current)
    elif current is None:
        handler = logging.NullHandler()
        handler.set_name(_QUIET_HANDLER)
        logger.addHandler(handler)
