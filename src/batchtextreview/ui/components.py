"""Small reusable Streamlit building blocks shared by the pages.

Keep these thin: layout + copy conventions, no business logic.

Besides the widgets, this module owns the small amount of cross-page job
bookkeeping every page needs (:func:`track_jobs`, which the app shell calls
before every page): noticing that a run finished, queueing the "Run
complete" toast so it survives a page switch, and remembering the
generation speed per model for time estimates.
"""

from __future__ import annotations

import contextlib
import html
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, Any

import streamlit as st

from batchtextreview.ui import format as fmt
from batchtextreview.ui import nav, resources, state

if TYPE_CHECKING:
    from streamlit.delta_generator import DeltaGenerator

    from batchtextreview.backends.base import DeviceInfo
    from batchtextreview.data import LoadedTable
    from batchtextreview.jobs import Job, JobSnapshot
    from batchtextreview.models import ModelSpec

# Session-state keys private to this module.
_TOASTS = "btr.toasts"
"""Queued (message, icon) pairs, shown at the top of the next full run."""
_SEEN_DONE = "btr.seen_done"
"""Job ids whose completion this session has already reacted to."""
RATES = "btr.rates"
"""(model_id, precision) -> rows/s seen in this session (for estimates)."""
_BADGE_SLOT = "_btr_badge_slot"
_DATASET_SLOT = "btr.strip_dataset_slot"

_BADGE_ICONS = {
    "gray": ":material/radio_button_unchecked:",
    "green": ":material/check_circle:",
    "orange": ":material/error:",
    "red": ":material/cancel:",
}


# Help texts for the scores, shared by the Results and History pages.
HELP_ACCURACY = "Share of scored rows where the label matches the gold value."
HELP_MACRO_F1 = (
    "F1 averaged over labels with equal weight, so rare labels count as much as common "
    "ones. Labels that never occur in gold or predictions are left out."
)
HELP_KAPPA = (
    "Agreement with the gold labels beyond chance: 1 is perfect, 0 is no better than chance."
)

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def page_header(title: str, caption: str | None = None) -> None:
    """Page title (st.title is too large next to top navigation: use a
    level-2 heading) with the context strip (:func:`context_strip`) on the
    same row, right-aligned, plus an optional one-line muted caption.

    The strip only draws; the app shell has already done the job
    bookkeeping (:func:`track_jobs`) before the page started."""
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.header(title, anchor=False, width="content")
        st.space("stretch")
        context_strip()
    if caption:
        st.caption(caption)


def subheading(text: str, width: int | str = "stretch") -> None:
    """A subsection title inside a section or tab (level-5 heading, no
    anchor link). Heading levels across the app: page title
    (:func:`page_header`, h2), section or card title (``st.subheader`` /
    :func:`section`, h3), subsection (this, h5)."""
    st.markdown(f"##### {text}", width=width, anchors=False)  # type: ignore[arg-type]


def page_button(
    name: str,
    label: str,
    icon: str | None = None,
    *,
    type: str = "primary",
    key: str | None = None,
) -> None:
    """A button that opens page ``name`` (e.g. the call to action of an
    empty page: "Go to the Run page")."""
    if st.button(label, key=key, icon=icon, type=type):  # type: ignore[arg-type]
        st.switch_page(nav.page(name))


def section(
    title: str, step: int | None = None, badge: tuple[str, str] | None = None
) -> DeltaGenerator:
    """A bordered container whose first row is "{step} · {title}" as a
    subheader with an optional right-aligned ``st.badge(text, color=...)``
    (badge = (text, color)). Returns the container for the caller to fill:

        with section("Data", step=1, badge=("Ready", "green")):
            ...
    """
    box = st.container(border=True)
    heading = f"{step} · {title}" if step is not None else title
    with box, st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.subheader(heading, anchor=False, width="content")
        st.space("stretch")
        slot = st.empty()
    # Remember the badge slot on the container so the badge can be set after
    # the section's content has decided whether it is ready.
    setattr(box, _BADGE_SLOT, slot)
    if badge:
        set_section_badge(box, badge)
    return box


def set_section_badge(box: DeltaGenerator, badge: tuple[str, str] | None) -> None:
    """Set (or clear, with None) the header badge of a :func:`section` after
    its content has rendered:

        box = section("Data", step=1)
        with box:
            problems = ...
        set_section_badge(box, ready_badge(problems))
    """
    slot = getattr(box, _BADGE_SLOT, None)
    if slot is None:
        return
    if badge is None:
        slot.empty()
        return
    text, color = badge
    slot.badge(text, icon=_BADGE_ICONS.get(color), color=color)


def empty_state(icon: str, title: str, body: str) -> None:
    """Centered muted placeholder: a Material icon (":material/...:"), a bold
    title and one or two lines of body text that tell the user what to do.
    The text block is capped at a readable measure (about 70 characters a
    line) however wide the page is."""
    with (
        st.container(horizontal_alignment="center"),
        st.container(width=560, horizontal_alignment="center", gap="xsmall"),
    ):
        decorative_icon(icon)
        st.markdown(f"**{title}**", text_alignment="center")
        st.caption(body, text_alignment="center")


def facts(rows: Iterable[tuple[str, Any]], label_width: int = 160) -> None:
    """Label / value lines with the labels in a fixed-width muted column, so
    values line up within and across cards. Values are Markdown; None and ""
    show as a dash."""
    with st.container(gap="xsmall"):
        for label, value in rows:
            text = fmt.DASH if value is None or value == "" else str(value)
            with st.container(horizontal=True, vertical_alignment="center", gap="small"):
                st.markdown(f":gray[{label}]", width=label_width)
                st.markdown(text, width="stretch")


def color_key(colors: dict[str, str], label: str = "Colour key") -> None:
    """One colour key (dot + name per entry) drawn above a group of charts
    that share it, in place of a legend inside each chart."""
    items = "".join(
        f'<span><i style="background:{html.escape(color)}"></i>{html.escape(name)}</span>'
        for name, color in colors.items()
    )
    st.html(f'<div class="btr-key" role="list" aria-label="{html.escape(label)}">{items}</div>')


def decorative_icon(icon: str, width: int | str = "stretch") -> None:
    """A large, muted Material icon (":material/name:") that is decoration
    only: plain HTML hidden from screen readers (a markdown heading would
    announce a heading named after the icon)."""
    name = html.escape(icon.removeprefix(":material/").removesuffix(":"))
    st.html(
        f'<div class="btr-empty-icon" aria-hidden="true" translate="no">{name}</div>',
        width=width,  # type: ignore[arg-type]
    )


@contextlib.contextmanager
def empty_card() -> Iterator[DeltaGenerator]:
    """A bordered card for an empty area, padded the same on every page.
    Wrap :func:`empty_state` (and an optional centred action) in it:

        with components.empty_card():
            components.empty_state(...)
    """
    box = st.container(border=True)
    with box:
        st.space("small")
        yield box
        st.space("small")


def ready_badge(problems: Iterable[str]) -> tuple[str, str]:
    """("Ready", "green") when there are no problems, else
    ("Needs attention", "orange") - for :func:`section`."""
    return ("Needs attention", "orange") if list(problems) else ("Ready", "green")


def error_with_details(message: str, details: str | None = None, hint: str | None = None) -> None:
    """st.error(message) followed by the hint (what to do next) and, if
    given, a collapsed "Technical details" expander with ``details`` in a
    code block."""
    body = f"{message}\n\n{hint}" if hint else message
    st.error(body, icon=":material/error:")
    if details:
        with st.expander("Technical details", icon=":material/code:"):
            st.code(details, language=None, wrap_lines=True)


# ---------------------------------------------------------------------------
# Badges
# ---------------------------------------------------------------------------


KEY_SOURCE_WORDS: dict[str, str] = {
    "session": "entered for this session",
    "env": "from {env}",
    "keyring": "saved in the system keyring",
}


def key_status(provider: str) -> tuple[bool, str]:
    """(credentials available, a few words saying where they come from) for
    a cloud provider. Never shows the key itself."""
    from batchtextreview import api_keys

    source = api_keys.key_source(provider)
    if source is not None:
        words = KEY_SOURCE_WORDS[source].format(env=api_keys.ENV_VARS[provider])
        return True, f"Key {words}"
    if api_keys.credentials_available(provider):
        return True, "No key needed: OPENAI_BASE_URL is set" if provider == "openai" else (
            "Signed in without a key"
        )
    return False, "No API key"


def _api_badges(spec: ModelSpec) -> None:
    from batchtextreview import api_keys
    from batchtextreview.models import PRICES_AS_OF

    provider = spec.provider_label
    who = api_keys.recipient(spec.provider or "")
    if who == provider:
        st.badge(
            f"{provider} API",
            icon=":material/cloud:",
            color="blue",
            help=f"Runs on {provider}'s servers: the texts you run are sent to {provider}, "
            "and the requests are billed to your API account.",
        )
    else:
        variable = api_keys.BASE_URL_VARS[spec.provider or "openai"]
        st.badge(
            f"{provider} API · {fmt.md(who)}",
            icon=":material/cloud:",
            color="violet",
            help=f"{variable} is set: the texts you run are sent to {who}, not to {provider}.",
        )
    if spec.tier:
        st.badge(spec.tier.capitalize(), icon=":material/speed:", color="blue")
    if spec.price:
        st.badge(
            fmt.no_math(fmt.price(spec.price)),
            color="gray",
            help=f"List price per million input / output tokens ({PRICES_AS_OF}). Check "
            f"{provider}'s pricing page for the current price.",
        )
    ok, words = key_status(spec.provider or "")
    if ok:
        st.badge(words, icon=":material/key:", color="green")
    else:
        st.badge(
            words,
            icon=":material/key_off:",
            color="orange",
            help="Add a key on the Models & settings page, or set "
            f"{'ANTHROPIC_API_KEY' if spec.provider == 'anthropic' else 'OPENAI_API_KEY'} "
            "and restart the app.",
        )


def model_badges(spec: ModelSpec, device: DeviceInfo) -> None:
    """Inline badges for a model: size, tier, "Gated" (orange, with a note
    whether a token is available), "Experimental" (violet), "Local folder",
    and whether the model is downloaded. A cloud model: its provider, tier,
    list price and whether a key is set."""
    with st.container(horizontal=True, gap="xsmall"):
        if spec.is_api:
            _api_badges(spec)
            return
        if spec.is_demo:
            st.badge(
                "Instant fake outputs",
                icon=":material/science:",
                color="violet",
                help="The demo model answers from a keyword list, without a GPU. "
                "Use it to try the app, not to judge results.",
            )
            return
        if spec.size_label:
            st.badge(f"{spec.size_label} parameters", color="gray")
        if spec.tier:
            st.badge(spec.tier.capitalize(), icon=":material/speed:", color="blue")
        if spec.gated:
            if resources.hf_token_available():
                st.badge(
                    "Gated · token set",
                    icon=":material/lock_open:",
                    color="orange",
                    help="Needs the model licence accepted on huggingface.co with the "
                    "account that owns your token.",
                )
            else:
                st.badge(
                    "Gated · needs HF_TOKEN",
                    icon=":material/lock:",
                    color="orange",
                    help="Accept the licence on huggingface.co, set the HF_TOKEN "
                    "environment variable, then restart the app.",
                )
        if spec.experimental:
            st.badge(
                "Experimental",
                icon=":material/science:",
                color="violet",
                help="New architecture: needs an up-to-date transformers release.",
            )
        if spec.is_local:
            st.badge("Local folder", icon=":material/folder:", color="gray")
        else:
            on_disk = resources.downloaded(spec.repo)
            size = resources.download_size_gb(spec.params_b)
            if on_disk:
                st.badge(
                    "Downloaded",
                    icon=":material/download_done:",
                    color="green",
                    help="Already on this computer: it loads without contacting Hugging Face.",
                )
            elif on_disk is False and resources.offline_mode():
                st.badge(
                    "Not downloaded · offline mode",
                    icon=":material/cloud_off:",
                    color="orange",
                    help="HF_HUB_OFFLINE is set, so the app can't download it. Pick a downloaded "
                    "model or a local folder, or unset HF_HUB_OFFLINE and restart the app.",
                )
            elif on_disk is False and size:
                st.badge(
                    f"First run downloads ≈{size} GB",
                    icon=":material/download:",
                    color="gray",
                    help="Downloaded once from Hugging Face, then loaded from disk.",
                )
        if not device.has_torch:
            st.badge(
                "PyTorch not installed",
                icon=":material/cancel:",
                color="red",
                help="Install PyTorch and transformers to run real models (see the README).",
            )
        elif device.kind == "cpu":
            st.badge(
                "CPU only · slow",
                icon=":material/memory:",
                color="orange",
                help="No CUDA GPU detected: the model runs in 32-bit on the CPU.",
            )


# ---------------------------------------------------------------------------
# Context strip and finished-job bookkeeping
# ---------------------------------------------------------------------------


def track_jobs() -> None:
    """Cross-page job bookkeeping, run by the app shell before every page:
    react to a session run that finished between two full runs (a toast)
    and show the queued toasts."""
    _catch_up_finished(resources.get_job_manager())
    _flush_toasts()


def context_strip() -> None:
    """Slim horizontal row of chips in every page's title row
    (:func:`page_header`): loaded dataset (name · rows), device (GPU name /
    CPU), model, and - while a job is running - a compact progress chip.
    Cheap: reads session state and the shared managers only; never loads
    anything."""
    jobs = resources.get_job_manager()
    active = jobs.active()
    if active is None:
        _strip_row(None)
    elif on_page("run"):
        # The Run page's progress panel already polls the job. One poller
        # per page: two fragments noticing the end of a job in the same
        # instant can leave the next page blank after the switch.
        _strip_row(active, live=False)
    else:
        _live_strip(active.id)


@st.fragment(run_every="2s")
def _live_strip(job_id: str) -> None:
    """The strip, refreshed while a job runs; reruns the app when it ends."""
    job = resources.get_job_manager().get(job_id)
    if job is None or job.snapshot().done:
        st.rerun(scope="app")
    _strip_row(job)


def _strip_row(job: Job | None, live: bool = True) -> None:
    settings = resources.get_settings()
    # Content width: the chips sit flush right in the title row.
    with st.container(horizontal=True, gap="xsmall", key="btr_context_strip", width="content"):
        slot = st.empty()
        st.session_state[_DATASET_SLOT] = slot
        _dataset_chip(slot, st.session_state.get(state.LOADED_TABLE))

        device = resources.get_device_info()
        _chip(*_device_chip(device, settings.demo_mode))

        model_chip = _model_chip(job)
        if model_chip:
            _chip(*model_chip)

        if settings.demo_mode:
            st.badge(
                "Demo mode",
                icon=":material/science:",
                color="violet",
                help="BTR_DEMO is set: the demo model is available.",
            )

        if job is not None:
            _chip(*_job_chip(job.snapshot(), live))


def set_context_dataset(table: LoadedTable | None) -> None:
    """Refresh the strip's dataset chip in the current run. The strip renders
    before the page, so a page that loads (or unloads) a table calls this to
    keep the chip in step without another rerun."""
    slot = st.session_state.get(_DATASET_SLOT)
    if slot is None:
        return
    # A stale slot (e.g. from a fragment rerun) is harmless: the next run fixes it.
    with contextlib.suppress(Exception):
        _dataset_chip(slot, table)


def _dataset_chip(slot: DeltaGenerator, table: LoadedTable | None) -> None:
    if table is None:
        slot.badge("No data loaded", icon=":material/table_chart:", color="gray")
        return
    slot.badge(
        f"{fmt.truncate(table.name, 40)} · {fmt.plural(len(table.frame), 'row')}",
        icon=":material/table_chart:",
        color="gray",
    )


def _chip(text: str, icon: str, color: str) -> None:
    st.badge(text, icon=icon, color=color)  # type: ignore[arg-type]


def _device_chip(device: DeviceInfo, demo: bool) -> tuple[str, str, str]:
    if device.kind == "cuda":
        text = device.name
        if device.total_memory_gb:
            text += f" · {device.total_memory_gb:.0f} GB"
        return (text, ":material/memory:", "gray")
    if device.kind == "mps":
        return ("Apple GPU", ":material/memory:", "gray")
    if device.kind == "cpu":
        return ("CPU only", ":material/memory:", "gray" if demo else "orange")
    return ("PyTorch not installed", ":material/memory:", "gray" if demo else "orange")


def _model_chip(job: Job | None = None) -> tuple[str, str, str] | None:
    from batchtextreview.models import find_model

    if job is not None:
        # While a run loads or uses a model, name that model rather than the
        # manager's state ("No model loaded" next to "Loading model" reads
        # as a contradiction).
        return (job.request.config.model_label, ":material/neurology:", "gray")
    try:
        loaded = resources.get_model_manager().loaded
    except Exception:  # never let a status chip break a page
        return None
    if loaded is None:
        return ("No model loaded", ":material/neurology:", "gray")
    model_id, precision = loaded
    spec = find_model(model_id)
    label = spec.label if spec else fmt.model_name(model_id)
    short = fmt.precision(precision)
    if short in (fmt.DASH, fmt.PRECISION_NAMES["none"]):  # the demo model has none
        return (label, ":material/neurology:", "gray")
    return (f"{label} · {short}", ":material/neurology:", "gray")


def _job_chip(snap: JobSnapshot, live: bool = True) -> tuple[str, str, str]:
    p = snap.progress
    if not live:  # not refreshed: no counts that would go stale
        text = "Run in progress"
    elif snap.state == "queued" or p.phase == "planning":
        text = "Starting run"
    elif p.phase == "loading":
        text = "Loading model"
    else:
        text = f"Generating · {p.done_rows:,} / {p.total_rows:,}"
    return (text, ":material/progress_activity:", "blue")


def queue_toast(message: str, icon: str | None = None) -> None:
    """Show ``message`` as a toast at the top of the next full run - the
    only way a toast survives ``st.switch_page`` or ``st.rerun``."""
    st.session_state.setdefault(_TOASTS, []).append((message, icon))


def _flush_toasts() -> None:
    for message, icon in st.session_state.pop(_TOASTS, []):
        st.toast(message, icon=icon)


def on_page(name: str) -> bool:
    """True when the page currently running is ``nav.page(name)``."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        ctx = get_script_run_ctx()
        return ctx is not None and ctx.page_script_hash == nav.page(name)._script_hash
    except Exception:
        return False


def notice_job_finished(job: Job, *, rerun: bool) -> None:
    """React once to a job reaching a terminal state.

    Records its speed for estimates and, for jobs started from this browser
    session, queues a "Run complete" toast. The page never changes by
    itself: someone may be typing the next prompt on the Run page (text in
    a box that still has the focus isn't sent until it loses it, and
    switching pages would throw it away). On the Run page the toast points
    to "Open results" (the run button, or the Last run card's), and the
    Last run card explains a run that was cancelled or failed. Repeated
    calls for the same job do nothing.

    From a polling fragment pass ``rerun=True``: that only reruns the whole
    app, and the full run reacts (:func:`context_strip` catches up on the
    session's newest job).
    """
    if rerun:
        st.rerun(scope="app")
    seen: set[str] = st.session_state.setdefault(_SEEN_DONE, set())
    if job.id in seen:
        return
    seen.add(job.id)
    snap = job.snapshot()
    _record_rate(job, snap)
    mine = job.id in st.session_state.get(state.SESSION_JOBS, [])
    if not mine:
        return
    message, icon = _finish_message(job, snap)
    if on_page("run"):
        if snap.state == "succeeded" and job.result is not None:
            message += " Open results to see them."
        else:
            message += " Details are in the Last run card."
    queue_toast(message, icon)


def setup_changed(job_id: str) -> bool:
    """True when the Run page's setup (model, prompt, settings, data, rows)
    differs from what it was when job ``job_id`` was submitted (a job with
    no recorded setup counts as unchanged)."""
    submitted = st.session_state.get(state.JOB_SETUPS, {}).get(job_id)
    current = st.session_state.get(state.RUN_SETUP)
    return submitted is not None and current is not None and submitted != current


_PAGE = "btr.page"
_PREVIOUS_PAGE = "btr.previous_page"


def note_page(name: str) -> None:
    """Record the page this full run shows (called by the app shell)."""
    st.session_state[_PREVIOUS_PAGE] = st.session_state.get(_PAGE)
    st.session_state[_PAGE] = name


def arrived_on_page() -> bool:
    """True on the first full run of a page after another page was shown
    (or the session started), False on the page's own reruns."""
    return st.session_state.get(_PREVIOUS_PAGE) != st.session_state.get(_PAGE)


def _catch_up_finished(jobs: Any) -> None:
    """A job can finish between two full runs without any fragment seeing it
    (e.g. an instant cached run); react to the newest session job here."""
    ids = st.session_state.get(state.SESSION_JOBS) or []
    if not ids or ids[0] in st.session_state.get(_SEEN_DONE, set()):
        return
    job = jobs.get(ids[0])
    if job is not None and job.snapshot().done:
        notice_job_finished(job, rerun=False)


def _record_rate(job: Job, snap: JobSnapshot) -> None:
    rate = snap.progress.rows_per_s
    if rate and snap.progress.generated_rows:
        cfg = job.request.config
        st.session_state.setdefault(RATES, {})[(cfg.model_id, cfg.precision)] = rate


def _finish_message(job: Job, snap: JobSnapshot) -> tuple[str, str]:
    p = snap.progress
    if snap.started_at and snap.finished_at:
        elapsed = (snap.finished_at - snap.started_at).total_seconds()
    else:
        elapsed = p.elapsed_s
    if snap.state == "succeeded":
        message = f"Run complete: {fmt.plural(p.total_rows, 'row')} in {fmt.duration(elapsed)}"
        failed = job.result.stats.errors if job.result is not None else 0
        if failed:
            # Shown on any page: a plain success would hide them.
            failed_text = f"{failed:,} row failed" if failed == 1 else f"{failed:,} rows failed"
            return (
                f"{message} · {failed_text}: retry them on the Results page.",
                ":material/warning:",
            )
        return (f"{message}.", ":material/check_circle:")
    if snap.state == "cancelled":
        return (
            f"Run cancelled after {p.done_rows:,} of {fmt.plural(p.total_rows, 'row')}. "
            "Finished rows are saved.",
            ":material/cancel:",
        )
    if job_status(snap)[0] == "Failed":
        return ("Run failed before producing any output.", ":material/error:")
    return (
        f"Run stopped early after {p.done_rows:,} of {fmt.plural(p.total_rows, 'row')}. "
        "Finished rows are saved.",
        ":material/error:",
    )


def job_status(snap: JobSnapshot) -> tuple[str, str, str]:
    """(text, icon, colour) describing a job's state for a badge:
    Queued / Running / Complete / Cancelled / Stopped early (failed after
    producing some outputs) / Failed (failed with nothing to show - e.g. the
    model never loaded)."""
    if snap.state == "queued":
        return ("Queued", ":material/schedule:", "gray")
    if snap.state == "running":
        return ("Running", ":material/progress_activity:", "blue")
    if snap.state == "succeeded":
        return ("Complete", ":material/check_circle:", "green")
    if snap.state == "cancelled":
        return ("Cancelled", ":material/cancel:", "gray")
    produced = snap.progress.generated_rows + snap.progress.cached_rows
    if snap.has_result and produced:
        return ("Stopped early", ":material/error:", "orange")
    return ("Failed", ":material/error:", "red")
