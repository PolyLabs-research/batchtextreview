"""Runtime settings, read from environment variables.

=====================  ======================================================
Variable               Meaning
=====================  ======================================================
``BTR_RESULTS_DIR``    Where the results database lives (default
                       ``./results``, resolved against the launch directory -
                       the same folder the previous version used).
``BTR_DATA_DIR``       Default folder offered when loading a file from disk
                       (default: the launch directory).
``BTR_DEMO``           ``1`` adds a "Demo" model that returns instant fake
                       outputs - for trying the UI without a GPU.
``BTR_NO_TRUSTSTORE``  ``1`` disables OS trust-store TLS validation.
``HF_TOKEN``           Hugging Face token, needed for gated models.
=====================  ======================================================
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

ENV_RESULTS_DIR = "BTR_RESULTS_DIR"
ENV_DATA_DIR = "BTR_DATA_DIR"
ENV_DEMO = "BTR_DEMO"

DB_FILENAME = "batchtextreview.db"

_TRUTHY = {"1", "true", "yes", "on"}


def _clean(value: str | None) -> str:
    """Trim whitespace and one pair of matching surrounding quotes.

    Windows cmd keeps the quotes in ``set X="D:\\My results"``.
    """
    text = (value or "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return text


def _flag(value: str | None) -> bool:
    return _clean(value).lower() in _TRUTHY


def _env_path(value: str | None, default: str) -> Path:
    """A folder from an environment variable.

    Blank / whitespace-only means "use the default", not a folder literally
    named " " (an easy slip with Windows ``set X= ``).
    """
    path = Path(_clean(value) or default)
    try:
        return path.expanduser()
    except RuntimeError:
        # Unknown ``~user``: keep it literal, as a shell would, rather than
        # crash every page that reads the settings.
        return path


@dataclass(frozen=True)
class Settings:
    results_dir: Path
    data_dir: Path
    demo_mode: bool = False

    @property
    def db_path(self) -> Path:
        return self.results_dir / DB_FILENAME

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if environ is None else environ
        results = _env_path(env.get(ENV_RESULTS_DIR), "results")
        data = _env_path(env.get(ENV_DATA_DIR), ".")
        return cls(
            results_dir=results.resolve(),
            data_dir=data.resolve(),
            demo_mode=_flag(env.get(ENV_DEMO)),
        )


def hf_token_present(environ: Mapping[str, str] | None = None) -> bool:
    """True when a Hugging Face token is available via the environment."""
    env = os.environ if environ is None else environ
    return bool(env.get("HF_TOKEN") or env.get("HUGGING_FACE_HUB_TOKEN"))
