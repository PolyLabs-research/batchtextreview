"""Process-wide singletons for the Streamlit app.

``st.cache_resource`` makes each of these one object per server process,
shared by every browser session - which is what we want: one results
database, one model in memory, one job queue.
"""

from __future__ import annotations

import os

import streamlit as st

from batchtextreview.backends.base import DeviceInfo, device_info
from batchtextreview.backends.manager import ModelManager, default_manager
from batchtextreview.config import Settings, hf_token_present
from batchtextreview.jobs import JobManager
from batchtextreview.logs import configure_logging
from batchtextreview.store import Store

JOB_HISTORY = 10
"""Finished jobs kept in memory (their results stay viewable)."""


@st.cache_resource(show_spinner=False)
def get_settings() -> Settings:
    settings = Settings.from_env()
    # Once per server process: results/logs/batchtextreview.log (+ stderr when
    # BTR_VERBOSE is set, e.g. by "batchtextreview -v ui").
    configure_logging(settings.results_dir)
    return settings


@st.cache_resource(show_spinner=False)
def get_store() -> Store:
    return Store(get_settings().db_path)


@st.cache_resource(show_spinner=False)
def get_model_manager() -> ModelManager:
    return default_manager()


@st.cache_resource(show_spinner=False)
def get_job_manager() -> JobManager:
    return JobManager(get_store(), get_model_manager(), history=JOB_HISTORY)


@st.cache_resource(show_spinner="Checking for a GPU…")
def get_device_info() -> DeviceInfo:
    return device_info()


@st.cache_data(ttl=60, show_spinner=False)
def downloaded(repo: str) -> bool | None:
    """Whether a Hugging Face model is already in the local cache (an offline
    check of its config file: no network). None when it can't be told: no
    huggingface_hub, the demo model, or a local folder."""
    if repo.startswith("demo://"):
        return None
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:
        return None
    try:
        return isinstance(try_to_load_from_cache(repo, "config.json"), str)
    except Exception:  # e.g. a local path, which isn't a valid repo id
        return None


def download_size_gb(params_b: float | None) -> int | None:
    """Rough first-download size of a model in GB (16-bit weights: about two
    bytes per parameter), or None when the size isn't known."""
    if not params_b:
        return None
    return max(1, round(params_b * 2))


@st.cache_data(ttl=30, show_spinner=False)
def hf_token_available() -> bool:
    """Whether downloads can authenticate to Hugging Face: HF_TOKEN (or
    HUGGING_FACE_HUB_TOKEN) is set, or a token was saved by ``hf auth login``
    / ``huggingface-cli login`` (read from disk; no network)."""
    if hf_token_present():
        return True
    try:
        from huggingface_hub import get_token
    except Exception:
        return False
    try:
        return bool(get_token())
    except Exception:
        return False


def offline_mode() -> bool:
    """True when Hugging Face downloads are switched off (HF_HUB_OFFLINE or
    TRANSFORMERS_OFFLINE), as transformers will see it."""
    try:
        from huggingface_hub import constants

        if constants.HF_HUB_OFFLINE:
            return True
    except Exception:
        pass
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if (os.environ.get(name) or "").strip().lower() in {"1", "true", "yes", "on"}:
            return True
    return False
