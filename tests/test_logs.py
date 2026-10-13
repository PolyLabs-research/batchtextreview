"""Tests for batchtextreview.logs."""

from __future__ import annotations

import logging
import logging.handlers
import os
import re
from pathlib import Path

import pytest

from batchtextreview import logs
from batchtextreview.logs import LOGGER_NAME, configure_logging, current_log_file, log_path

PACKAGE = logging.getLogger(LOGGER_NAME)


@pytest.fixture(autouse=True)
def restore_package_logger(monkeypatch):
    """Leave the package logger as each test found it (handlers closed, so
    tmp_path can be removed on Windows too). BTR_VERBOSE is cleared so a
    developer's own setting can't change the defaults under test."""
    monkeypatch.delenv(logs.ENV_VERBOSE, raising=False)
    handlers, level, propagate = list(PACKAGE.handlers), PACKAGE.level, PACKAGE.propagate
    root_handlers = list(logging.getLogger().handlers)
    yield
    for handler in PACKAGE.handlers:
        if handler not in handlers:
            PACKAGE.removeHandler(handler)
            handler.close()
    for handler in handlers:
        if handler not in PACKAGE.handlers:
            PACKAGE.addHandler(handler)
    PACKAGE.setLevel(level)
    PACKAGE.propagate = propagate
    assert logging.getLogger().handlers == root_handlers  # never touched


def ours(kind: type[logging.Handler] = logging.Handler) -> list[logging.Handler]:
    return [
        h
        for h in PACKAGE.handlers
        if (h.get_name() or "").startswith("batchtextreview-") and isinstance(h, kind)
    ]


def file_handlers() -> list[logging.handlers.RotatingFileHandler]:
    return ours(logging.handlers.RotatingFileHandler)  # type: ignore[return-value]


def flush() -> None:
    for handler in PACKAGE.handlers:
        handler.flush()


def test_writes_to_the_results_folder(tmp_path):
    results = tmp_path / "results"
    path = configure_logging(results)
    assert path == results / "logs" / "batchtextreview.log"
    assert path == log_path(results) == current_log_file()
    assert path.is_file()
    assert PACKAGE.propagate is False
    assert PACKAGE.level == logging.INFO

    logging.getLogger("batchtextreview.engine").info("Run %s started - café", "abc")
    logging.getLogger("batchtextreview.engine").debug("not at INFO")
    logging.getLogger("someone.else").warning("other libraries are left alone")
    flush()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert re.fullmatch(
        r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3} INFO batchtextreview\.engine: "
        r"Run abc started - café",
        lines[0],
    )


def test_rotation_settings(tmp_path):
    configure_logging(tmp_path)
    (handler,) = file_handlers()
    assert handler.maxBytes == 1024 * 1024
    assert handler.backupCount == 3
    assert handler.encoding == "utf-8"


def test_repeated_calls_never_add_handlers(tmp_path):
    for _ in range(5):
        configure_logging(tmp_path / "results")
    configure_logging(str(tmp_path / "results"))  # type: ignore[arg-type]
    assert len(file_handlers()) == 1
    assert len(ours()) == 1
    first = file_handlers()[0]
    configure_logging(tmp_path / "results")
    assert file_handlers()[0] is first


def test_new_results_folder_moves_the_file(tmp_path):
    old = configure_logging(tmp_path / "a")
    old_handler = file_handlers()[0]
    new = configure_logging(tmp_path / "b")
    assert new == tmp_path / "b" / "logs" / "batchtextreview.log"
    assert len(file_handlers()) == 1
    assert old_handler.stream is None or old_handler.stream.closed  # closed
    logging.getLogger("batchtextreview.store").warning("moved")
    flush()
    assert old is not None and "moved" not in old.read_text(encoding="utf-8")
    assert new is not None and "WARNING batchtextreview.store: moved" in new.read_text(
        encoding="utf-8"
    )


def test_none_removes_the_file_handler(tmp_path):
    configure_logging(tmp_path)
    assert configure_logging(None) is None
    assert file_handlers() == []
    assert current_log_file() is None


def test_unwritable_folder_falls_back_to_no_file(tmp_path):
    configure_logging(tmp_path / "good")
    blocker = tmp_path / "not-a-folder"
    blocker.write_text("a file where the results folder should be")
    assert configure_logging(blocker) is None
    assert file_handlers() == []
    # Logging still works (and doesn't raise).
    logging.getLogger("batchtextreview.engine").warning("still fine")


def test_log_file_path_taken_by_a_folder(tmp_path):
    (tmp_path / "logs" / "batchtextreview.log").mkdir(parents=True)
    assert configure_logging(tmp_path) is None
    assert file_handlers() == []


def test_console_handler_is_optional_and_single(tmp_path, capsys):
    configure_logging(tmp_path, console=True, level=logging.DEBUG)
    configure_logging(tmp_path, console=True, level=logging.DEBUG)
    consoles = [h for h in ours(logging.StreamHandler) if h not in file_handlers()]
    assert len(consoles) == 1
    logging.getLogger("batchtextreview.cli").debug("debug detail")
    err = capsys.readouterr().err
    assert "DEBUG batchtextreview.cli: debug detail" in err

    configure_logging(tmp_path, console=False)
    assert [h for h in ours(logging.StreamHandler) if h not in file_handlers()] == []
    logging.getLogger("batchtextreview.cli").warning("file only")
    assert capsys.readouterr().err == ""


def test_console_only(capsys):
    assert configure_logging(None, console=True) is None
    logging.getLogger("batchtextreview.jobs").info("hello")
    assert "INFO batchtextreview.jobs: hello" in capsys.readouterr().err


def test_debug_level_reaches_the_file(tmp_path):
    path = configure_logging(tmp_path, level=logging.DEBUG)
    logging.getLogger("batchtextreview.engine").debug("row retried alone failed")
    flush()
    assert path is not None and "DEBUG" in path.read_text(encoding="utf-8")


def test_nowhere_to_log_is_silent(capsys):
    """Configured with neither a file nor stderr: no last-resort output
    (warnings with tracebacks would land in the CLI's own messages)."""
    configure_logging(None)
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        logging.getLogger("batchtextreview.engine").warning("failed", exc_info=True)
    assert capsys.readouterr().err == ""
    # A file or console later replaces the quiet handler.
    configure_logging(None, console=True)
    assert [h for h in ours() if isinstance(h, logging.NullHandler)] == []


def test_foreign_handlers_are_kept(tmp_path):
    mine = logging.NullHandler()
    PACKAGE.addHandler(mine)
    configure_logging(tmp_path)
    configure_logging(tmp_path / "other", console=True)
    configure_logging(None)
    assert mine in PACKAGE.handlers


def test_never_raises(monkeypatch, tmp_path):
    def broken(*args, **kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(logs, "_SharedRotatingFileHandler", broken)
    assert configure_logging(tmp_path) is None
    monkeypatch.setattr(Path, "mkdir", broken)
    assert configure_logging(tmp_path / "x") is None


# --------------------------------------------------------------------------
# BTR_VERBOSE (how "batchtextreview -v ui" reaches the app)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["1", "true", " Yes ", "ON", '"1"'])
def test_verbose_from_env_true(monkeypatch, value):
    monkeypatch.setenv(logs.ENV_VERBOSE, value)
    assert logs.verbose_from_env() is True


@pytest.mark.parametrize("value", ["", "0", "no", "false", "debug"])
def test_verbose_from_env_false(monkeypatch, value):
    monkeypatch.setenv(logs.ENV_VERBOSE, value)
    assert logs.verbose_from_env() is False


def test_env_verbose_means_debug_and_stderr(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(logs.ENV_VERBOSE, "1")
    path = configure_logging(tmp_path)
    assert PACKAGE.level == logging.DEBUG
    logging.getLogger("batchtextreview.engine").debug("detail")
    flush()
    assert "DEBUG batchtextreview.engine: detail" in capsys.readouterr().err
    assert path is not None and "detail" in path.read_text(encoding="utf-8")


def test_explicit_level_and_console_win_over_env(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(logs.ENV_VERBOSE, "1")
    configure_logging(tmp_path, level=logging.WARNING, console=False)
    assert PACKAGE.level == logging.WARNING
    logging.getLogger("batchtextreview.engine").warning("file only")
    assert capsys.readouterr().err == ""


def test_defaults_without_env(tmp_path, capsys):
    configure_logging(tmp_path)
    assert PACKAGE.level == logging.INFO
    logging.getLogger("batchtextreview.engine").info("file only")
    assert capsys.readouterr().err == ""


# --------------------------------------------------------------------------
# A log file shared with another process
# --------------------------------------------------------------------------


@pytest.fixture
def small_log(tmp_path):
    """Logging to tmp_path with a 300-byte rotation threshold and two old
    backups; returns the log file's path."""
    path = configure_logging(tmp_path)
    assert path is not None
    (handler,) = file_handlers()
    handler.maxBytes = 300
    Path(f"{path}.1").write_text("backup one\n", encoding="utf-8")
    Path(f"{path}.2").write_text("backup two\n", encoding="utf-8")
    return path


def held_open_elsewhere(monkeypatch, path: Path) -> list[str]:
    """Make renaming the live log file fail the way Windows does while
    another process has it open; returns the attempted sources."""
    real_replace = os.replace
    attempts: list[str] = []

    def replace(src, dst, *args, **kwargs):
        attempts.append(os.fspath(src))
        if os.fspath(src) == os.fspath(path):
            raise PermissionError(32, "The process cannot access the file", os.fspath(src))
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(logs.os, "replace", replace)
    return attempts


def write_records(n: int, tag: str = "record") -> None:
    for i in range(n):
        logging.getLogger("batchtextreview.engine").info("%s %02d %s", tag, i, "x" * 40)
    flush()


def test_rotation_moves_backups_along(small_log):
    path = small_log
    write_records(4)  # about 105 bytes each: one rollover, before record 02
    first = Path(f"{path}.1").read_text(encoding="utf-8")
    assert "record 00" in first and "record 01" in first and "record 02" not in first
    assert Path(f"{path}.2").read_text(encoding="utf-8") == "backup one\n"
    assert Path(f"{path}.3").read_text(encoding="utf-8") == "backup two\n"
    assert not Path(f"{path}.rotating").exists()
    text = path.read_text(encoding="utf-8")
    assert "record 02" in text and "record 03" in text and "record 01" not in text
    write_records(1, "more")  # one more rollover: the oldest backup drops off
    assert Path(f"{path}.3").read_text(encoding="utf-8") == "backup one\n"


def test_rollover_while_held_open_keeps_appending_quietly(small_log, monkeypatch, capsys):
    """Windows: the stock handler would print "--- Logging error ---" for
    every record, drop it, and shift one more backup away each time."""
    path = small_log
    attempts = held_open_elsewhere(monkeypatch, path)
    write_records(20)
    assert capsys.readouterr().err == ""
    text = path.read_text(encoding="utf-8")
    assert all(f"record {i:02d} " in text for i in range(20))  # nothing dropped
    assert Path(f"{path}.1").read_text(encoding="utf-8") == "backup one\n"  # untouched
    assert Path(f"{path}.2").read_text(encoding="utf-8") == "backup two\n"
    assert not Path(f"{path}.3").exists()
    assert attempts == [str(path)]  # tried once, not on every record


def test_rollover_is_retried_once_the_file_is_free(small_log, monkeypatch):
    path = small_log
    held_open_elsewhere(monkeypatch, path)
    write_records(10, "early")
    monkeypatch.undo()  # the other process closed the file...
    write_records(1, "still waiting")
    assert not Path(f"{path}.3").exists()  # ...but the retry isn't due yet
    clock = logs.time.monotonic() + logs.ROLLOVER_RETRY_S + 1
    monkeypatch.setattr(logs.time, "monotonic", lambda: clock)
    write_records(1, "late")
    first = Path(f"{path}.1").read_text(encoding="utf-8")
    assert "early 00" in first and "early 09" in first and "still waiting" in first
    assert Path(f"{path}.2").read_text(encoding="utf-8") == "backup one\n"
    assert Path(f"{path}.3").read_text(encoding="utf-8") == "backup two\n"
    assert "late 00" in path.read_text(encoding="utf-8")


def test_write_errors_are_reported_once(small_log, capsys):
    (handler,) = file_handlers()

    class Broken:
        closed = False

        def write(self, text):
            raise OSError(28, "No space left on device")

        def flush(self):
            pass

        def tell(self):
            return 0

        def seek(self, *args):
            return 0

        def close(self):
            pass

    handler.stream = Broken()
    for _ in range(5):
        logging.getLogger("batchtextreview.engine").warning("lost")
    err = capsys.readouterr().err
    assert err.count("--- Logging error ---") == 1
    assert "No space left on device" in err
