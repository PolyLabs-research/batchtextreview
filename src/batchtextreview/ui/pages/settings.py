"""Models & settings page: this computer, the loaded model, the model
catalogue, cloud model keys, saved outputs and where everything is stored.

Read-only except for unloading the model and deleting one configuration's
saved outputs (behind a confirm dialog) - both disabled while a run is in
progress - and managing the cloud providers' API keys. Nothing here imports torch: the GPU memory readout
only appears when torch is already loaded (the device check imports it).
"""

from __future__ import annotations

import os
import platform
import re
import sys
import zlib
from dataclasses import dataclass
from importlib import metadata
from typing import Any

import pandas as pd
import streamlit as st

from batchtextreview import __version__, api_keys, models, presets
from batchtextreview import store as store_mod
from batchtextreview.backends.base import PRECISION_LABELS, DeviceInfo, resolve_precision
from batchtextreview.config import ENV_DATA_DIR, ENV_DEMO, ENV_RESULTS_DIR, hf_token_present
from batchtextreview.logs import ENV_VERBOSE, log_path
from batchtextreview.ui import components, resources
from batchtextreview.ui import format as fmt

# ---- session-state keys (page-local) --------------------------------------
K_RUNS_TABLE = "settings.runs"
"""Prefix of the saved-outputs table's key (see :func:`runs_table_key`)."""
K_DELETE_RUN = "settings.delete_run"
"""Prefix of the per-configuration delete button's key."""
_K_SELECTED_RUN = "settings.selected_run"
"""run_id selected in the saved-outputs table (or None)."""
_K_RUN_DELETED = "settings.run_deleted"
_K_RUN_MSG = "settings.run_msg"
K_KEY_INPUT = "settings.api_key"
"""Prefix of each provider's key input (``settings.api_key::anthropic``)."""
_K_KEY_MSG = "settings.api_key_msg"
"""Prefix of each provider's last key action / connection test result."""

_GB = 1024**3
_DATE_FORMAT = "D MMM YYYY, HH:mm"

_ENV_VARS: tuple[tuple[str, str], ...] = (
    (ENV_RESULTS_DIR, "Folder for the results database (saved outputs, history, presets)."),
    (ENV_DATA_DIR, "Folder the Run page offers first when loading a file from disk."),
    (ENV_DEMO, "1 adds the Demo model: instant fake outputs, no GPU needed."),
    ("HF_TOKEN", "Hugging Face token, needed for gated models such as Llama."),
    ("HF_HUB_OFFLINE", "1 never contacts Hugging Face: only downloaded models can load."),
    ("ANTHROPIC_API_KEY", "Anthropic API key, for Claude models (see Cloud models above)."),
    ("OPENAI_API_KEY", "OpenAI API key, for GPT models (see Cloud models above)."),
    (
        "OPENAI_BASE_URL",
        "Address of an OpenAI-compatible server (Azure, a local server) to use instead of OpenAI.",
    ),
    (
        "BTR_NO_TRUSTSTORE",
        "1 stops using the operating system's certificates for HTTPS (they are used by "
        "default so downloads work behind corporate proxies).",
    ),
    (
        ENV_VERBOSE,
        "1 adds debug details to the log file and the terminal (the same as "
        "batchtextreview -v ui).",
    ),
)


# ==========================================================================
# Entry point
# ==========================================================================


def render() -> None:
    components.page_header(
        "Models & settings",
        "This computer, the model catalogue, cloud model keys, saved outputs and where "
        "everything is stored.",
    )
    settings = resources.get_settings()
    store = resources.get_store()
    device = resources.get_device_info()
    busy = resources.get_job_manager().active() is not None
    saved = _read_store(store)
    if saved.error:
        components.error_with_details(
            "Couldn't read the results database.",
            details=saved.error,
            hint=f"Check that {store.path} is readable and not open in another program, "
            "then reload the page. Saved outputs and storage figures are unavailable until then.",
        )

    left, right = st.columns([3, 2], gap="medium")
    with left:
        _device_card(device, settings.demo_mode)
    with right:
        _loaded_model_card(busy, device)

    st.space("small")
    _catalogue(device)
    st.space("small")
    _cloud_models()
    st.space("small")
    _saved_outputs(saved, busy)
    st.space("small")
    _storage(settings, store, saved.stats)


@dataclass
class _Saved:
    """What the page reads from the results database, read once per run."""

    runs: pd.DataFrame | None = None
    history: pd.DataFrame | None = None
    custom: list[presets.Preset] | None = None
    stats: dict[str, int] | None = None
    error: str | None = None
    """First failure ("ExcType: message"); the parts it stopped are None."""


def _read_store(store: store_mod.Store) -> _Saved:
    saved = _Saved()
    try:
        saved.runs = store.list_runs()
        saved.history = store.list_evaluations()
        saved.custom = store.list_presets()
        saved.stats = store.stats()
    except Exception as exc:
        saved.error = f"{type(exc).__name__}: {exc}"
    return saved


# ==========================================================================
# This computer
# ==========================================================================


def _device_card(device: DeviceInfo, demo: bool) -> None:
    auto = resolve_precision("auto", device)
    with st.container(border=True, height="stretch"):
        with st.container(horizontal=True, vertical_alignment="center", gap="small"):
            st.subheader("This computer", anchor=False, width="content")
            st.space("stretch")
            text, icon, color = device_badge(device)
            st.badge(text, icon=icon, color=color)  # type: ignore[arg-type]

        live = device.kind == "cuda" and "torch" in sys.modules
        facts: list[tuple[str, str]] = [("Device", fmt.md(device.name))]
        if device.kind == "cuda" and not live:
            facts.append(("GPU memory", _gb(device.total_memory_gb)))
        facts.append(("Auto weights", _auto_text(auto, device)))
        components.facts(facts)
        if live:
            _live_gpu_memory()
        components.facts(
            [
                ("PyTorch", _torch_text(device)),
                ("Transformers", _version("transformers") or "Not installed"),
                ("bitsandbytes", _version("bitsandbytes") or "Not installed"),
                ("Hugging Face token", _token_text()),
            ]
        )
        note = device_note(device, demo)
        if note is not None:
            title, body, level = note
            box = st.warning if level == "warning" else st.info
            box(body, title=title, icon=":material/memory:")
        hint = driver_hint(device)
        if hint:
            with st.expander("Have an NVIDIA GPU?", icon=":material/help:"):
                st.markdown(hint)


def _token_text() -> str:
    if hf_token_present():
        return "Set"
    if resources.hf_token_available():
        return "Set :gray[· saved by hf auth login]"
    return "Not set"


def driver_hint(device: DeviceInfo) -> str | None:
    """What to check when a CPU-only machine may have a GPU after all
    (Markdown), or None."""
    if device.kind != "cpu":
        return None
    if device.cuda_version:
        return (
            f"PyTorch was built for CUDA {device.cuda_version} but can't use a GPU: check the "
            "NVIDIA driver (`nvidia-smi` should list the card), then restart the app."
        )
    return (
        "This PyTorch build is for the CPU only. Install the CUDA build of PyTorch as described "
        "in the README, then restart the app."
    )


def device_badge(device: DeviceInfo) -> tuple[str, str, str]:
    """(text, icon, colour) summarising what this machine can run."""
    if device.kind == "cuda":
        if device.bitsandbytes:
            return ("GPU · 4-bit ready", ":material/check_circle:", "green")
        return ("GPU · no 4-bit", ":material/error:", "orange")
    if device.kind == "mps":
        return ("Apple GPU · 16-bit", ":material/check_circle:", "blue")
    if device.kind == "cpu":
        return ("CPU only · slow", ":material/error:", "orange")
    return ("PyTorch not installed", ":material/cancel:", "red")


def device_note(device: DeviceInfo, demo: bool) -> tuple[str, str, str] | None:
    """(title, message, "warning" | "info") explaining a limited device, or None."""
    demo_hint = (
        "Demo mode is on: the Demo model answers instantly without a GPU."
        if demo
        else f"To try the app without a GPU, start it with {ENV_DEMO}=1 for instant fake outputs."
    )
    if device.kind == "none":
        return (
            "PyTorch isn't installed",
            "Local models need PyTorch and transformers (see the README). Cloud models "
            "(Anthropic, OpenAI) run without them. " + demo_hint,
            "info" if demo else "warning",
        )
    if device.kind == "cpu":
        return (
            "No NVIDIA GPU detected",
            "Runs use 32-bit weights on the CPU and are slow; 4-bit models need an NVIDIA GPU. "
            + demo_hint,
            "info" if demo else "warning",
        )
    if device.kind == "mps":
        return (
            "Apple GPU",
            "Models run in 16-bit. 4-bit loading needs an NVIDIA GPU, so the larger models "
            "may not fit in memory.",
            "info",
        )
    if not device.bitsandbytes:
        return (
            "4-bit loading unavailable",
            "bitsandbytes isn't installed, so Auto loads models in 16-bit and 7B+ models "
            "won't fit in 8 GB. Install bitsandbytes to load them in 4-bit.",
            "warning",
        )
    return None


@st.fragment(run_every="5s")
def _live_gpu_memory() -> None:
    """Free / total VRAM, refreshed every few seconds (CUDA only)."""
    torch = sys.modules.get("torch")
    try:
        free, total = torch.cuda.mem_get_info() if torch is not None else (0, 0)
    except Exception:
        free, total = 0, 0
    if not total:
        return
    used = max(0.0, min(1.0, (total - free) / total))
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.markdown(":gray[GPU memory]", width=160)
        st.progress(
            used,
            text=f"{free / _GB:.1f} GB free of {total / _GB:.1f} GB",
            width="stretch",
        )


def _auto_text(auto: str, device: DeviceInfo) -> str:
    label = PRECISION_LABELS.get(auto, auto)
    why = {
        "4bit": "fits 7-9B models in 8 GB",
        "fp16": "no 4-bit on this machine" if device.kind == "cuda" else "Apple GPU",
        "fp32": "CPU",
    }.get(auto)
    return f"{label} :gray[· {why}]" if why else label


def _torch_text(device: DeviceInfo) -> str:
    version = device.torch_version or _version("torch")
    if not version:
        return "Not installed"
    if device.cuda_version:
        return f"{version} :gray[· built for CUDA {device.cuda_version}]"
    return f"{version} :gray[· CPU build]"


def _version(package: str) -> str | None:
    """Installed version of ``package`` without importing it."""
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None
    except Exception:
        return None


# ==========================================================================
# Loaded model
# ==========================================================================


def _loaded_model_card(busy: bool, device: DeviceInfo) -> None:
    manager = resources.get_model_manager()
    loaded = manager.loaded
    in_use = manager.busy
    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center", gap="small"):
            st.subheader("Loaded model", anchor=False, width="content")
            st.space("stretch")
            if loaded is not None:
                if in_use:
                    st.badge("In use", icon=":material/progress_activity:", color="blue")
                else:
                    st.badge("Idle", icon=":material/pause_circle:", color="gray")
        if loaded is None:
            st.markdown("**No model loaded**")
            st.caption(
                "A model loads on its first run and stays in memory, so later runs start "
                "straight away. Running another model replaces it."
            )
        else:
            model_id, precision = loaded
            spec = models.find_model(model_id)
            st.markdown(f"**{fmt.md(spec.label if spec else fmt.model_name(model_id))}**")
            if spec is not None and spec.is_demo:
                st.badge("Instant fake outputs", icon=":material/science:", color="violet")
                st.caption("The demo model uses no GPU memory.")
            else:
                with st.container(horizontal=True, gap="xsmall"):
                    if precision in PRECISION_LABELS and precision != "none":
                        st.badge(f"{PRECISION_LABELS[precision]} weights", color="gray")
                    if spec is not None and spec.size_label:
                        st.badge(f"{spec.size_label} parameters", color="gray")
                st.caption(fmt.md(model_id))
                memory = "GPU memory" if device.kind in ("cuda", "mps") else "memory"
                st.caption(
                    f"Stays in memory between runs. Unload it to free {memory} for other programs."
                )
        st.space("small")
        st.button(
            "Unload model",
            key="settings.unload",
            icon=":material/eject:",
            on_click=_unload,
            disabled=loaded is None or busy or in_use,
            help="Wait for the current run to finish." if (busy or in_use) else None,
        )


def _unload() -> None:
    manager = resources.get_model_manager()
    loaded = manager.loaded
    if manager.busy or resources.get_job_manager().active() is not None:
        # unload() would wait for the whole run to finish, freezing this page.
        components.queue_toast(
            "A run is using the model. Unload it when the run has finished.", ":material/info:"
        )
        return
    try:
        unloaded = manager.unload()
    except Exception as exc:
        components.queue_toast(f"Couldn't unload the model: {exc}", ":material/error:")
        return
    if unloaded and loaded is not None:
        spec = models.find_model(loaded[0])
        name = spec.label if spec else fmt.model_name(loaded[0])
        components.queue_toast(f"Unloaded {name}. Its memory is free again.", ":material/eject:")


# ==========================================================================
# Model catalogue
# ==========================================================================


def _catalogue(device: DeviceInfo) -> None:
    st.subheader("Model catalogue", anchor=False)
    specs = models.available_models()
    downloaded = {s.repo: resources.downloaded(s.repo) for s in specs}
    loaded = resources.get_model_manager().loaded
    offline = resources.offline_mode()
    frame = catalogue_frame(
        specs,
        downloaded,
        token=resources.hf_token_available(),
        loaded_repo=loaded[0] if loaded else None,
        offline=offline,
    )
    st.dataframe(
        frame,
        hide_index=True,
        placeholder=fmt.DASH,
        height="content",
        column_config={
            "Model": st.column_config.TextColumn(width="medium"),
            "Size": st.column_config.TextColumn(width="small", help="Parameter count"),
            "Repository": st.column_config.LinkColumn(
                width="medium",
                display_text=r"https://huggingface\.co/(.*)",
                help="Opens the model card on huggingface.co",
            ),
            "Notes": st.column_config.ListColumn(width="medium"),
            "On disk": st.column_config.TextColumn(
                help="Downloaded models load without contacting Hugging Face; the others are "
                "downloaded on their first run (16-bit size shown).",
            ),
        },
        key="settings.catalogue",
        alt="Built-in models with size, repository, notes and download status",
    )
    if offline:
        download = (
            "Offline mode is on (HF_HUB_OFFLINE): only models already downloaded can run, "
            "until the app is restarted without it."
        )
    else:
        download = (
            f"The first run with a model downloads it from Hugging Face "
            f"({download_range(specs)}); later runs load it from disk."
        )
    caption = (
        f"Every built-in model fits an 8 GB GPU at 4-bit. {download} Any other Hugging Face "
        "repo id or a local model folder works through “Custom model…” on the Run page."
    )
    if device.kind in ("cpu", "none"):
        caption += " On this computer, prefer the smallest models."
    st.caption(caption)


# ==========================================================================
# Cloud models
# ==========================================================================


def _cloud_models() -> None:
    st.subheader("Cloud models", anchor=False)
    st.caption(
        "Anthropic's Claude and OpenAI's GPT models run on the providers' servers: the texts "
        "you run are sent to them, and the requests are billed to your API account. Each "
        "needs an API key. Keys are never written to the results database or the log, and "
        "only their first and last characters are shown."
    )
    columns = st.columns(len(models.API_PROVIDERS), gap="medium")
    for column, provider in zip(columns, models.API_PROVIDERS, strict=True):
        with column:
            _provider_card(provider)


def keyring_name() -> str:
    """What the system keyring is called on this operating system."""
    if sys.platform == "win32":
        return "Windows Credential Manager"
    if sys.platform == "darwin":
        return "the macOS Keychain"
    return "the system keyring"


def _provider_card(provider: str) -> None:
    from batchtextreview.backends import api

    name = models.API_PROVIDERS[provider]
    env_var = api_keys.ENV_VARS[provider]
    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center", gap="small"):
            st.markdown(f"**{name}**", width="content")
            ok, words = components.key_status(provider)
            st.badge(
                words,
                icon=":material/key:" if ok else ":material/key_off:",
                color="green" if ok else "orange",
            )
        found = api_keys.lookup(provider)
        if found is not None:
            st.caption(f"Key in use: `{api_keys.masked(found[0])}`")
        base_url = api_keys.base_url(provider)
        if base_url:
            st.caption(
                f"Requests go to {fmt.md(base_url)} ({api_keys.BASE_URL_VARS[provider]}), "
                f"not to {name}."
            )
        installed = api.sdk_installed(provider)
        if not installed:
            st.warning(
                f"The {api.SDK_PACKAGES[provider]} package isn't installed. Install the cloud "
                'API extra (pip install -e ".[api]"), then restart the app.',
                icon=":material/extension_off:",
            )

        can_save = api_keys.keyring_available()
        # Buttons with callbacks, not a form: the callback empties the field,
        # so the typed key doesn't linger in the session state.
        st.text_input(
            f"{name} API key",
            type="password",
            key=f"{K_KEY_INPUT}::{provider}",
            placeholder="sk-ant-…" if provider == "anthropic" else "sk-…",
            help=f"“Use for this session” comes before {env_var}; a saved key is used "
            f"only when {env_var} isn't set.",
        )
        with st.container(horizontal=True, gap="small"):
            st.button(
                "Use for this session",
                key=f"settings.api_use::{provider}",
                icon=":material/login:",
                on_click=_use_typed_key,
                args=(provider, False),
                help="Until the app stops, for every browser tab of this app.",
            )
            st.button(
                "Save",
                key=f"settings.api_save::{provider}",
                icon=":material/save:",
                disabled=not can_save,
                on_click=_use_typed_key,
                args=(provider, True),
                help=f"Saves the key in {keyring_name()}, so later sessions use it too."
                if can_save
                else "No system keyring is available here: set "
                f"{env_var} instead, or install the keyring package.",
            )

        source = found[1] if found is not None else None
        with st.container(horizontal=True, gap="small"):
            if st.button(
                "Test connection",
                key=f"settings.api_test::{provider}",
                icon=":material/network_check:",
                disabled=not (installed and ok),
                help=f"Asks {name} for the models this key can use.",
            ):
                _test_connection(provider)
            if source == "session" and st.button(
                "Forget",
                key=f"settings.api_forget::{provider}",
                icon=":material/logout:",
                type="tertiary",
                help="Stop using the key entered for this session.",
            ):
                api_keys.set_session_key(provider, None)
                _set_key_message(provider, "info", "The session key is forgotten.")
                st.rerun()
            if api_keys.saved_key(provider) and st.button(
                "Remove saved key",
                key=f"settings.api_remove::{provider}",
                icon=":material/delete:",
                type="tertiary",
                help=f"Deletes the key saved in {keyring_name()}.",
            ):
                try:
                    api_keys.delete_saved_key(provider)
                except RuntimeError as exc:
                    _set_key_message(provider, "error", str(exc))
                else:
                    _set_key_message(provider, "info", "The saved key is removed.")
                st.rerun()
        message = st.session_state.get(f"{_K_KEY_MSG}::{provider}")
        if message:
            kind, text = message
            {"success": st.success, "error": st.error, "warning": st.warning}.get(kind, st.info)(
                text
            )
        builtin = models.api_models(provider)
        if builtin:
            st.caption(
                "Built in: "
                + " · ".join(
                    f"{s.label} ({fmt.no_math(fmt.price(s.price))})" if s.price else s.label
                    for s in builtin
                )
                + f". Any other model works as {provider}:<model id> through “Custom model…” "
                "on the Run page."
            )


def _use_typed_key(provider: str, save: bool) -> None:
    """Button callback: use or save the key typed in, then empty the field."""
    field = f"{K_KEY_INPUT}::{provider}"
    typed = str(st.session_state.get(field) or "")
    st.session_state[field] = ""
    _store_key(provider, typed, save=save)


def _store_key(provider: str, key: str, *, save: bool) -> None:
    value = (key or "").strip()
    if not value:
        _set_key_message(provider, "error", "Enter a key first.")
        return
    if save:
        try:
            api_keys.save_key(provider, value)
        except (RuntimeError, ValueError) as exc:
            _set_key_message(provider, "error", str(exc))
            return
        # A session key would hide the one just saved.
        api_keys.set_session_key(provider, None)
        env_var = api_keys.ENV_VARS[provider]
        if os.environ.get(env_var, "").strip():
            _set_key_message(
                provider,
                "warning",
                f"Saved in {keyring_name()}, but {env_var} is set and is used first. Unset it "
                "and restart the app to use the saved key.",
            )
        else:
            _set_key_message(provider, "success", f"Saved in {keyring_name()}.")
    else:
        api_keys.set_session_key(provider, value)
        _set_key_message(provider, "success", "Used until the app stops.")


def _test_connection(provider: str) -> None:
    from batchtextreview.backends import api
    from batchtextreview.backends.base import ModelLoadError

    name = models.API_PROVIDERS[provider]
    try:
        with st.spinner(f"Asking {name}…"):
            ids = api.check_connection(provider)
    except ModelLoadError as exc:
        _set_key_message(provider, "error", str(exc))
        return
    except Exception as exc:  # unexpected: show it rather than crash the page
        _set_key_message(provider, "error", f"{type(exc).__name__}: {exc}")
        return
    known = [s.api_model for s in models.api_models(provider) if s.api_model in ids]
    text = f"Connected: this key can use {fmt.plural(len(ids), 'model')}"
    text += f", including {', '.join(known)}." if known else "."
    _set_key_message(provider, "success", text)


def _set_key_message(provider: str, kind: str, text: str) -> None:
    st.session_state[f"{_K_KEY_MSG}::{provider}"] = (kind, text)


def download_range(specs: list[models.ModelSpec]) -> str:
    """ "about 1-18 GB": the smallest to the largest first download among
    ``specs`` (the demo model and unknown sizes aside)."""
    sizes = [s for s in (resources.download_size_gb(spec.params_b) for spec in specs) if s]
    if not sizes:
        return "several GB"
    low, high = min(sizes), max(sizes)
    return f"about {low} GB" if low == high else f"about {low}-{high} GB"


def catalogue_frame(
    specs: list[models.ModelSpec],
    downloaded: dict[str, bool | None] | None,
    *,
    token: bool,
    loaded_repo: str | None = None,
    offline: bool = False,
) -> pd.DataFrame:
    """One row per model: Model, Size, Repository (URL), Notes (list of
    short tags) and On disk ("Downloaded", "≈6 GB download" - "Not
    downloaded" in offline mode - or "–" when it can't be told)."""
    rows = []
    for spec in specs:
        notes: list[str] = []
        if spec.repo == loaded_repo:
            notes.append("Loaded")
        if spec.tier:
            notes.append(spec.tier.capitalize())
        if spec.gated:
            notes.append("Gated" if token else "Gated · needs HF_TOKEN")
        if spec.experimental:
            notes.append("Experimental")
        on_disk = (downloaded or {}).get(spec.repo)
        size = resources.download_size_gb(spec.params_b)
        if on_disk:
            disk = "Downloaded"
        elif on_disk is False and offline:
            disk = "Not downloaded"
        elif on_disk is False and size:
            disk = f"≈{size} GB download"
        else:
            disk = fmt.DASH
        rows.append(
            {
                "Model": spec.label,
                "Size": spec.size_label or fmt.DASH,
                "Repository": spec.hf_url or spec.repo,
                "Notes": notes,
                "On disk": disk,
            }
        )
    return pd.DataFrame(rows, columns=["Model", "Size", "Repository", "Notes", "On disk"])


# ==========================================================================
# Saved outputs
# ==========================================================================


def _saved_outputs(saved: _Saved, busy: bool) -> None:
    st.subheader("Saved outputs", anchor=False)
    st.caption(
        "Saved outputs are reused when you run the same model + prompt + settings again. "
        "Deleting them frees space; evaluation history is kept."
    )
    if saved.runs is None:
        st.caption(":material/block: Unavailable until the results database can be read.")
        return
    runs = saved.runs
    if runs.empty:
        with components.empty_card():
            components.empty_state(
                ":material/inventory_2:",
                "No saved outputs yet",
                "Every row a model generates is saved here, so running the same "
                "configuration again is instant.",
            )
        return

    frame = runs_frame(runs, saved.history, saved.custom)
    show_source = bool((frame["Source"] != "App").any())
    ids = list(frame["run_id"])
    # The selection is remembered by run id: the key changes whenever the
    # list does (a run started in any tab adds or reorders rows), and the
    # new table starts on the same configuration - never on whatever row now
    # sits at the old position.
    remembered = st.session_state.get(_K_SELECTED_RUN)
    event = st.dataframe(
        frame,
        hide_index=True,
        placeholder=fmt.DASH,
        key=runs_table_key(ids),
        on_select="rerun",
        selection_mode="single-row",
        selection_default={"selection": {"rows": [ids.index(remembered)]}}
        if remembered in ids
        else None,
        column_order=[
            "Model",
            "Task",
            "Weights",
            "Saved outputs",
            "Max tokens",
            "Last used",
            *(["Source"] if show_source else []),
            "Prompt",
        ],
        column_config={
            "Model": st.column_config.TextColumn(width="medium"),
            "Weights": st.column_config.TextColumn(
                help="Numeric precision the model weights were loaded in."
            ),
            "Saved outputs": st.column_config.NumberColumn(format="localized"),
            "Max tokens": st.column_config.NumberColumn(
                format="localized", help="Max output tokens"
            ),
            "Last used": st.column_config.DatetimeColumn(format=_DATE_FORMAT),
            "Source": st.column_config.TextColumn(
                help="App, command line (CLI), or imported from version 1 (v1)."
            ),
            "Prompt": st.column_config.TextColumn(width="medium"),
        },
        alt="Saved outputs per model and prompt configuration",
    )
    rows = list(getattr(getattr(event, "selection", None), "rows", None) or [])
    chosen = ids[rows[0]] if rows and 0 <= rows[0] < len(ids) else None
    st.session_state[_K_SELECTED_RUN] = chosen
    if chosen is None:
        st.caption("Select a row to see its full prompt or delete its saved outputs.")
        return
    _run_details(frame.iloc[ids.index(chosen)], busy)


def runs_table_key(run_ids: list[str]) -> str:
    """Key of the saved-outputs table listing ``run_ids`` (in order)."""
    return f"{K_RUNS_TABLE}::{zlib.crc32(','.join(run_ids).encode()):08x}"


def runs_frame(
    runs: pd.DataFrame,
    history: pd.DataFrame | None = None,
    custom: list[presets.Preset] | None = None,
) -> pd.DataFrame:
    """``Store.list_runs()`` as display columns (Model, Task, Weights,
    Saved outputs, Max tokens, Last used, Source, Prompt) plus the raw
    run_id, template, created_at and max_input_chars for the details view.

    The task comes from the newest history line for the run, else from a
    preset whose template matches the run's prompt, else "Custom prompt"."""
    by_run: dict[str, str] = {}
    if history is not None and not history.empty and {"run_id", "preset"} <= set(history.columns):
        for run_id, key in zip(history["run_id"], history["preset"], strict=True):
            if isinstance(run_id, str) and isinstance(key, str) and run_id not in by_run:
                by_run[run_id] = key  # newest first, so the first one wins
    candidates = presets.builtin_presets() + list(custom or [])

    tasks = []
    for run_id, template in zip(runs["run_id"], runs["template"], strict=True):
        key = by_run.get(run_id)
        preset = presets.get_preset(key, custom) if key else None
        if preset is not None:
            tasks.append(preset.title)
        elif key:
            tasks.append(key)
        else:
            tasks.append(guess_task(str(template), candidates) or "Custom prompt")

    return pd.DataFrame(
        {
            "Model": fmt.distinct_model_names(
                [
                    label or fmt.model_name(mid)
                    for label, mid in zip(runs["model_label"], runs["model_id"], strict=True)
                ],
                [str(mid) for mid in runs["model_id"]],
            ),
            "Task": tasks,
            "Weights": [fmt.precision(p) for p in runs["precision"]],
            "Saved outputs": runs["n_outputs"].astype("int64"),
            "Max tokens": runs["max_new_tokens"].astype("int64"),
            "Last used": pd.to_datetime(runs["updated_at"], errors="coerce"),
            "Source": [fmt.source_word(s) for s in runs["source"]],
            "Prompt": [fmt.truncate(str(t), 90) for t in runs["template"]],
            "run_id": runs["run_id"].astype(str),
            "template": runs["template"].astype(str),
            "created_at": pd.to_datetime(runs["created_at"], errors="coerce"),
            "max_input_chars": runs["max_input_chars"].astype("int64"),
            "precision_raw": runs["precision"].astype(str),
        }
    )


def guess_task(template: str, candidates: list[presets.Preset]) -> str | None:
    """Title of the preset whose template produced ``template`` (with any
    ``{var}`` filled in), or None for an edited prompt."""
    for preset in candidates:
        if preset.template == template:
            return preset.title
    for preset in candidates:
        if "{var}" not in preset.template:
            continue
        pattern = "(.*)".join(re.escape(part) for part in preset.template.split("{var}"))
        if re.fullmatch(pattern, template, flags=re.DOTALL):
            return preset.title
    return None


def _run_details(row: pd.Series, busy: bool) -> None:
    run_id = str(row["run_id"])
    name = f"{row['Model']} · {row['Task']}"
    n = int(row["Saved outputs"])
    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center", gap="small"):
            st.markdown(f"**{fmt.md(name)}**", width="content")
            st.space("stretch")
            # Keyed by the configuration it deletes, so a click on a card
            # that has since gone stale can't reach another configuration.
            clicked = st.button(
                "Delete saved outputs",
                key=f"{K_DELETE_RUN}::{run_id}",
                icon=":material/delete:",
                disabled=busy,
                help="Wait for the current run to finish." if busy else None,
            )
        raw_precision = str(row["precision_raw"])
        parts = [fmt.plural(n, "saved output")]
        if raw_precision != "none":  # the demo model has no weights to speak of
            parts.append(f"{PRECISION_LABELS.get(raw_precision, raw_precision)} weights")
        parts += [
            f"up to {int(row['Max tokens']):,} output tokens",
            f"inputs cut at {int(row['max_input_chars']):,} characters",
        ]
        if pd.notna(row["created_at"]):
            parts.append(f"first saved {fmt.timestamp(row['created_at'].to_pydatetime())}")
        st.caption(
            " · ".join(parts),
            help=f"Configuration id `{run_id}`, as the command line and the results "
            "database name it.",
        )
        st.code(str(row["template"]), language=None, wrap_lines=True)
    if clicked and not busy:
        confirm_delete_run(run_id, n, name)


@st.dialog("Delete saved outputs?")
def confirm_delete_run(run_id: str, n: int, name: str) -> None:
    """Confirm, then delete one configuration's ``n`` saved outputs (also
    used by the Run page's "Discard saved outputs")."""
    if st.session_state.pop(_K_RUN_DELETED, None):  # done by the callback below
        st.rerun()
    st.write(f"Delete {fmt.plural(n, 'saved output')} for {fmt.md(name)}? This can't be undone.")
    st.caption(
        "Evaluation history is kept. Running this configuration again generates every row "
        "from scratch."
    )
    with st.container(horizontal=True, horizontal_alignment="right", gap="small"):
        if st.button("Cancel", key="settings.delete_run_cancel"):
            st.rerun()
        st.button(
            f"Delete {fmt.plural(n, 'saved output')}",
            key="settings.delete_run_confirm",
            type="primary",
            icon=":material/delete:",
            on_click=_delete_run,
            args=(run_id, name),
        )
    message = st.session_state.pop(_K_RUN_MSG, None)
    if message:
        st.warning(message, icon=":material/warning:")


def _delete_run(run_id: str, name: str) -> None:
    if resources.get_job_manager().active() is not None:
        st.session_state[_K_RUN_MSG] = "A run is in progress. Try again when it has finished."
        return
    try:
        deleted = resources.get_store().delete_run(run_id)
    except Exception as exc:
        st.session_state[_K_RUN_MSG] = f"Couldn't delete the saved outputs: {exc}"
        return
    st.session_state[_K_RUN_DELETED] = True
    if st.session_state.get(_K_SELECTED_RUN) == run_id:
        st.session_state[_K_SELECTED_RUN] = None
    components.queue_toast(
        f"Deleted {fmt.plural(deleted, 'saved output')} for {name}", ":material/delete:"
    )


# ==========================================================================
# Storage
# ==========================================================================


def _storage(settings: Any, store: store_mod.Store, stats: dict[str, int] | None) -> None:
    st.subheader("Storage", anchor=False)
    stats = stats or {}
    size = stats.get("size_bytes")
    tiles = [
        (
            "Saved output sets",
            stats.get("runs"),
            "One per model + prompt + settings combination; running it again reuses its outputs.",
        ),
        (
            "Saved outputs (unique texts)",
            stats.get("outputs"),
            "One per distinct text a configuration answered: duplicate rows share one.",
        ),
        ("Evaluations", stats.get("evaluations"), None),
        ("Saved presets", stats.get("presets"), None),
    ]
    # Equal columns, like the Results page's KPI tiles.
    columns = st.columns(len(tiles) + 1, gap="small")
    for column, (label, value, help_text) in zip(columns, tiles, strict=False):
        with column:
            st.metric(
                label,
                fmt.DASH if value is None else f"{value:,}",
                border=True,
                help=help_text,
                height="stretch",
            )
    with columns[-1]:
        st.metric(
            "Database size",
            fmt.DASH if size is None else fmt.file_size(size),
            border=True,
            help="The database file plus its write-ahead log.",
            height="stretch",
        )

    left, middle, right = st.columns(3, gap="medium")
    with left:
        components.subheading("Results folder")
        st.code(str(settings.results_dir), language=None, wrap_lines=True)
    with middle:
        components.subheading("Database file")
        st.code(str(store.path), language=None, wrap_lines=True)
    with right:
        components.subheading("Log file")
        st.code(str(log_path(settings.results_dir)), language=None, wrap_lines=True)

    components.subheading("Environment variables")
    # A static table, so long values and descriptions wrap instead of being cut off.
    table = env_table(
        settings, login_token=not hf_token_present() and resources.hf_token_available()
    )
    st.table(
        pd.DataFrame(
            {
                "Variable": [f"`{name}`" for name in table["Variable"]],
                "Current value": [_value_md(v) for v in table["Current value"]],
                "What it does": [fmt.md(v) for v in table["What it does"]],
            }
        ),
        border="horizontal",
        hide_index=True,
        alt="Environment variables the app reads, with their current values",
    )
    st.caption(
        "Set these before starting the app; changes take effect after a restart. "
        f"Batch Text Review {__version__} · Python {platform.python_version()} · "
        f"Streamlit {_version('streamlit') or '?'}"
    )


def env_table(
    settings: Any, environ: dict[str, str] | None = None, *, login_token: bool = False
) -> pd.DataFrame:
    """Variable / Current value / What it does, as plain text. The token's
    value is never shown, only whether one is set (``login_token``: a token
    saved by ``hf auth login`` is used instead); the results folder's path
    is left to the "Results folder" box above the table."""
    env = os.environ if environ is None else environ
    rows = []
    for name, purpose in _ENV_VARS:
        raw = (env.get(name) or "").strip()
        if name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
            value = "Set (hidden)" if raw else "Not set"
        elif name == "HF_TOKEN":
            if hf_token_present(env):
                value = "Set (hidden)"
            elif login_token:
                value = "Not set · the token saved by hf auth login is used"
            else:
                value = "Not set"
        elif name == ENV_RESULTS_DIR:
            value = (
                "Set · see Results folder above"
                if raw
                else "Not set · uses a results folder where the app was started"
            )
        elif name == ENV_DATA_DIR:
            value = raw or "Not set · uses the folder the app was started from"
        else:
            value = raw or "Not set"
        rows.append({"Variable": name, "Current value": value, "What it does": purpose})
    return pd.DataFrame(rows, columns=["Variable", "Current value", "What it does"])


def _value_md(value: str) -> str:
    """A current value for the Markdown table: our own words as text, the
    user's value (a path, a flag) as code."""
    if value.startswith(("Not set", "Set ")):
        return fmt.md(value)
    return f"`{value.replace('`', '')}`"


# ==========================================================================
# Small helpers
# ==========================================================================


def _gb(value: float | None) -> str:
    return f"{value:.1f} GB" if value else fmt.DASH
