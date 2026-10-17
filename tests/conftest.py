"""Shared pytest configuration."""

from __future__ import annotations

import importlib.util
import os

import pytest

# Keep tests hermetic: no OS trust-store patching, no demo flag leaking in
# from the developer's shell.
os.environ.setdefault("BTR_NO_TRUSTSTORE", "1")
os.environ.pop("BTR_DEMO", None)

from batchtextreview import api_keys as _api_keys  # after the environment is set

_REAL_KEYRING_BACKEND = _api_keys._keyring_backend

HAS_LLM_STACK = all(
    importlib.util.find_spec(name) is not None for name in ("torch", "transformers")
)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if HAS_LLM_STACK:
        return
    skip = pytest.mark.skip(reason="torch/transformers not installed")
    for item in items:
        if "llm" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def results_dir(tmp_path, monkeypatch):
    """An isolated results directory, also exported as BTR_RESULTS_DIR."""
    path = tmp_path / "results"
    monkeypatch.setenv("BTR_RESULTS_DIR", str(path))
    return path


_PROVIDER_VARIABLES = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_PROFILE",
)


@pytest.fixture(autouse=True)
def _no_provider_credentials(monkeypatch):
    """No cloud API key, server or sign-in leaks in from the developer's
    shell, keyring or SDK profile: tests that need one set it."""
    from batchtextreview import api_keys

    for name in _PROVIDER_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(api_keys, "_keyring_backend", lambda: None)
    monkeypatch.setattr(api_keys, "_anthropic_sdk_signed_in", lambda: False)
    api_keys._reset_for_tests()
    yield
    api_keys._reset_for_tests()


@pytest.fixture
def real_keyring_lookup(monkeypatch):
    """Undo the keyring stub of :func:`_no_provider_credentials` (for tests
    that install a keyring backend of their own)."""
    from batchtextreview import api_keys

    monkeypatch.setattr(api_keys, "_keyring_backend", _REAL_KEYRING_BACKEND)
