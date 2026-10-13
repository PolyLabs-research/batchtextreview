"""Tests for batchtextreview.config and the environment bootstrap (_env)."""

from __future__ import annotations

import dataclasses
import ntpath
import os
import sys
import types
from pathlib import Path

import pytest

from batchtextreview import _env
from batchtextreview.config import (
    DB_FILENAME,
    ENV_DATA_DIR,
    ENV_DEMO,
    ENV_RESULTS_DIR,
    Settings,
    hf_token_present,
)

# --------------------------------------------------------------------------
# Settings.from_env
# --------------------------------------------------------------------------


@pytest.fixture
def cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path.resolve()


def test_defaults_resolve_against_launch_directory(cwd: Path) -> None:
    settings = Settings.from_env({})
    assert settings.results_dir == cwd / "results"
    assert settings.data_dir == cwd
    assert settings.demo_mode is False
    assert settings.results_dir.is_absolute()
    assert settings.data_dir.is_absolute()


def test_db_path_lives_in_results_dir(cwd: Path) -> None:
    settings = Settings.from_env({})
    assert settings.db_path == cwd / "results" / DB_FILENAME
    assert DB_FILENAME.endswith(".db")


def test_from_env_does_not_create_directories(cwd: Path) -> None:
    Settings.from_env({ENV_RESULTS_DIR: "out/deep"})
    assert not (cwd / "out").exists()


def test_absolute_overrides(tmp_path: Path, cwd: Path) -> None:
    results = tmp_path / "elsewhere" / "res"
    data = tmp_path / "corpus"
    settings = Settings.from_env({ENV_RESULTS_DIR: str(results), ENV_DATA_DIR: str(data)})
    assert settings.results_dir == results.resolve()
    assert settings.data_dir == data.resolve()


def test_relative_overrides_resolve_against_cwd(cwd: Path) -> None:
    settings = Settings.from_env({ENV_RESULTS_DIR: "runs/v2", ENV_DATA_DIR: "../data"})
    assert settings.results_dir == cwd / "runs" / "v2"
    assert settings.data_dir == (cwd / ".." / "data").resolve()


def test_home_directory_is_expanded(
    tmp_path: Path, cwd: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))  # Windows
    settings = Settings.from_env({ENV_RESULTS_DIR: "~/btr", ENV_DATA_DIR: "~"})
    assert settings.results_dir == (home / "btr").resolve()
    assert settings.data_dir == home.resolve()


@pytest.mark.parametrize("blank", ["", "   ", "\t"])
def test_blank_values_fall_back_to_defaults(cwd: Path, blank: str) -> None:
    settings = Settings.from_env({ENV_RESULTS_DIR: blank, ENV_DATA_DIR: blank})
    assert settings.results_dir == cwd / "results"
    assert settings.data_dir == cwd


def test_surrounding_whitespace_is_ignored(cwd: Path) -> None:
    settings = Settings.from_env({ENV_RESULTS_DIR: " store \n"})
    assert settings.results_dir == cwd / "store"


@pytest.mark.parametrize(
    ("value", "folder"),
    [
        ('"My results"', "My results"),  # Windows cmd: set X="My results"
        ("'My results'", "My results"),
        (' " My results " ', "My results"),
        ('"store', '"store'),  # unmatched quote: left alone
        ('"', '"'),
    ],
)
def test_surrounding_quotes_are_dropped(cwd: Path, value: str, folder: str) -> None:
    settings = Settings.from_env({ENV_RESULTS_DIR: value, ENV_DATA_DIR: value})
    assert settings.results_dir == cwd / folder
    assert settings.data_dir == cwd / folder


@pytest.mark.parametrize("quoted", ['""', "''", '" "'])
def test_quoted_blank_falls_back_to_default(cwd: Path, quoted: str) -> None:
    assert Settings.from_env({ENV_RESULTS_DIR: quoted}).results_dir == cwd / "results"


def test_quoted_absolute_path(tmp_path: Path, cwd: Path) -> None:
    target = tmp_path / "with space" / "results"
    settings = Settings.from_env({ENV_RESULTS_DIR: f'"{target}"'})
    assert settings.results_dir == target.resolve()


def test_unknown_home_user_does_not_crash(
    cwd: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Path.expanduser raises RuntimeError for an unknown ~user; keep it
    # literal (as a shell does) instead of crashing every page.
    #
    # On Windows, ntpath.expanduser guesses "~other" as a sibling of
    # USERPROFILE whenever that folder is named after USERNAME (it is on
    # GitHub's runners and on most PCs: C:\Users\<name>), so the user only
    # stays unknown with a profile folder named otherwise.
    monkeypatch.setenv("USERNAME", "btr-test-user")
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "profile"))
    assert ntpath.expanduser("~nosuchuser-btr-test/res") == "~nosuchuser-btr-test/res"

    settings = Settings.from_env({ENV_RESULTS_DIR: "~nosuchuser-btr-test/res"})
    assert settings.results_dir == cwd / "~nosuchuser-btr-test" / "res"


@pytest.mark.parametrize("value", ['"1"', "'true'", '" yes "'])
def test_demo_flag_accepts_quotes(cwd: Path, value: str) -> None:
    assert Settings.from_env({ENV_DEMO: value}).demo_mode is True


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "True", " yes ", "on", "ON\n"])
def test_demo_flag_truthy(cwd: Path, value: str) -> None:
    assert Settings.from_env({ENV_DEMO: value}).demo_mode is True


@pytest.mark.parametrize("value", ["", " ", "0", "false", "no", "off", "2", "y", "enabled"])
def test_demo_flag_falsy(cwd: Path, value: str) -> None:
    assert Settings.from_env({ENV_DEMO: value}).demo_mode is False


def test_reads_process_environment_by_default(
    tmp_path: Path, cwd: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV_RESULTS_DIR, str(tmp_path / "r"))
    monkeypatch.setenv(ENV_DATA_DIR, str(tmp_path / "d"))
    monkeypatch.setenv(ENV_DEMO, "1")
    settings = Settings.from_env()
    assert settings.results_dir == (tmp_path / "r").resolve()
    assert settings.data_dir == (tmp_path / "d").resolve()
    assert settings.demo_mode is True


def test_explicit_mapping_ignores_process_environment(
    cwd: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV_RESULTS_DIR, "/should/not/be/used")
    monkeypatch.setenv(ENV_DEMO, "1")
    settings = Settings.from_env({})
    assert settings.results_dir == cwd / "results"
    assert settings.demo_mode is False


def test_results_dir_fixture_is_honoured(results_dir: Path, cwd: Path) -> None:
    assert Settings.from_env().results_dir == results_dir.resolve()


def test_settings_is_frozen(cwd: Path) -> None:
    settings = Settings.from_env({})
    with pytest.raises(dataclasses.FrozenInstanceError):
        settings.demo_mode = True  # type: ignore[misc]


# --------------------------------------------------------------------------
# hf_token_present
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("environ", "expected"),
    [
        ({}, False),
        ({"HF_TOKEN": ""}, False),
        ({"HF_TOKEN": "hf_abc"}, True),
        ({"HUGGING_FACE_HUB_TOKEN": "hf_abc"}, True),
        ({"HF_TOKEN": "", "HUGGING_FACE_HUB_TOKEN": "hf_abc"}, True),
    ],
)
def test_hf_token_present(environ: dict[str, str], expected: bool) -> None:
    assert hf_token_present(environ) is expected


def test_hf_token_present_reads_process_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    assert hf_token_present() is False
    monkeypatch.setenv("HF_TOKEN", "hf_x")
    assert hf_token_present() is True


# --------------------------------------------------------------------------
# _env.bootstrap
# --------------------------------------------------------------------------

_DEFAULTED = ("KMP_DUPLICATE_LIB_OK", "HF_HUB_DISABLE_SYMLINKS_WARNING", "TOKENIZERS_PARALLELISM")


@pytest.fixture
def fresh_env(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Reset bootstrap's run-once flag, clear the variables it sets, and
    install a fake ``truststore`` that records ``inject_into_ssl`` calls."""
    monkeypatch.setattr(_env, "_DONE", False)
    for name in (*_DEFAULTED, "BTR_NO_TRUSTSTORE"):
        monkeypatch.delenv(name, raising=False)
    calls: list[str] = []
    fake = types.ModuleType("truststore")
    fake.inject_into_ssl = lambda: calls.append("inject")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "truststore", fake)
    return calls


def test_bootstrap_sets_defaults(fresh_env: list[str]) -> None:
    _env.bootstrap()
    assert os.environ["KMP_DUPLICATE_LIB_OK"] == "TRUE"
    assert os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] == "1"
    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"


def test_bootstrap_never_overrides_explicit_values(
    fresh_env: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KMP_DUPLICATE_LIB_OK", "FALSE")
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "true")
    _env.bootstrap()
    assert os.environ["KMP_DUPLICATE_LIB_OK"] == "FALSE"
    assert os.environ["TOKENIZERS_PARALLELISM"] == "true"
    assert os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] == "1"


def test_bootstrap_injects_truststore_once(fresh_env: list[str]) -> None:
    _env.bootstrap()
    _env.bootstrap()
    _env.bootstrap()
    assert fresh_env == ["inject"]


def test_bootstrap_is_idempotent_for_environment(
    fresh_env: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _env.bootstrap()
    # A second call must not re-apply anything, even if the user changed or
    # removed a value in between.
    monkeypatch.delenv("KMP_DUPLICATE_LIB_OK")
    _env.bootstrap()
    assert "KMP_DUPLICATE_LIB_OK" not in os.environ


@pytest.mark.parametrize("value", ["1", "true", "TRUE", " yes ", "on"])
def test_bootstrap_respects_no_truststore(
    fresh_env: list[str], monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("BTR_NO_TRUSTSTORE", value)
    _env.bootstrap()
    assert fresh_env == []
    # The other defaults still apply.
    assert os.environ["KMP_DUPLICATE_LIB_OK"] == "TRUE"


@pytest.mark.parametrize("value", ["", "0", "false", "no"])
def test_bootstrap_falsy_no_truststore_still_injects(
    fresh_env: list[str], monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("BTR_NO_TRUSTSTORE", value)
    _env.bootstrap()
    assert fresh_env == ["inject"]


def test_bootstrap_survives_missing_truststore(
    fresh_env: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "truststore", None)  # import raises ImportError
    _env.bootstrap()
    assert _env._DONE is True


def test_bootstrap_survives_truststore_failure(
    fresh_env: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom() -> None:
        raise RuntimeError("unsupported platform")

    monkeypatch.setattr(sys.modules["truststore"], "inject_into_ssl", boom)
    _env.bootstrap()
    assert _env._DONE is True


def test_package_import_runs_bootstrap() -> None:
    import batchtextreview

    assert _env._DONE is True
    assert batchtextreview.__version__
