"""Shared pytest configuration."""

from __future__ import annotations

import importlib.util
import os

import pytest

# Keep tests hermetic: no OS trust-store patching, no demo flag leaking in
# from the developer's shell.
os.environ.setdefault("BTR_NO_TRUSTSTORE", "1")
os.environ.pop("BTR_DEMO", None)

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
