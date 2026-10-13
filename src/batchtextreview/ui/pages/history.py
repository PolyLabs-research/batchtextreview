"""History page: compare scored runs across models, prompts and datasets.

Everything renders from the results database (``Store.list_evaluations``) on
every rerun. The filters open on the newest evaluation and are mirrored in
the URL (every filter, every time), so a filtered view can be bookmarked or
shared and opens the same way later. Version-1 artefacts found in the
results folder are offered for a one-click import at the top of the page.
"""

from __future__ import annotations

import re
import zlib
from collections.abc import Callable, Iterable
from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st

from batchtextreview import models, presets, results
from batchtextreview import store as store_mod
from batchtextreview.ui import charts, components, resources
from batchtextreview.ui import format as fmt

# ---- widget keys ------------------------------------------------------------
K_DATASET = "history.dataset"
K_TASK = "history.task"
K_GOLD = "history.gold"
K_SELECTION = "history.rows"
K_VIEW = "view"
"""Bound to the URL by Streamlit; its default ("Best per model") never shows there."""
K_SCATTER = "history.scatter_x"
K_TABLE = "history.table"
"""Prefix of the full-history table's key (see :func:`table_key`)."""
K_DELETE = "history.delete"
"""Prefix of the delete button's key (see :func:`delete_key`)."""
_K_SELECTED = "history.selected"
"""Evaluation ids selected in the full-history table (only ids in view)."""
_K_STARTED = "history.filters_started"
_K_AUTO = "history.filters_auto"
"""True while the filters follow the newest evaluation (until the user picks)."""
_K_FOLLOWED = "history.followed_id"
"""Newest evaluation id the following filters were last set from."""
_K_NEWEST = "history.newest_id"
_K_SEEN = "history.seen_newest_id"
"""Newest evaluation id when the user last chose filters."""
_K_IMPORT_REPORT = "history.import_report"
_K_IMPORT_ERROR = "history.import_error"
_K_LEGACY_HIDDEN = "history.legacy_hidden"
_K_DELETED = "history.deleted"
_K_DELETE_MSG = "history.delete_msg"

URL_PARAMS = {K_DATASET: "dataset", K_TASK: "task", K_GOLD: "gold", K_SELECTION: "rows"}
"""Query-parameter name for each filter."""
URL_ALL = "all"
"""How "All" is spelled in the URL."""
URL_UNKNOWN = "not-recorded"
"""How :data:`UNKNOWN` is spelled in the URL."""
_FP_CHARS = 10
"""Fingerprint characters that name a row selection in the URL."""

ALL = "All"
"""Filter value meaning "don't filter"."""
UNKNOWN = "(not recorded)"
"""Filter value for history lines without a dataset / gold column (version 1)."""

VIEW_BEST = "Best per model"
VIEW_ALL = "All runs"

_SCATTER_QUALITY = "Accuracy vs macro-F1"
_SCATTER_SPEED = "Quality vs speed"

_PERCENT = st.column_config.ProgressColumn
_HELP_ACCURACY = components.HELP_ACCURACY
_HELP_F1 = components.HELP_MACRO_F1
_DATE_FORMAT = "D MMM YYYY, HH:mm"


# ==========================================================================
# Entry point
# ==========================================================================


def render() -> None:
    components.page_header("History", "Compare scored runs across models, prompts and datasets.")
    store = resources.get_store()
    _legacy_import(store)

    try:
        raw = store.list_evaluations()
    except Exception as exc:
        components.error_with_details(
            "Couldn't read the evaluation history.",
            details=f"{type(exc).__name__}: {exc}",
            hint=f"Check that {store.path} is readable and not open in another program, "
            "then reload the page.",
        )
        return
    if raw.empty:
        _empty_page()
        return

    frame = history_frame(raw, _custom_presets(store))
    filtered = _filters(frame)
    _latest_hidden_note(frame, filtered)
    if filtered.empty:
        with components.empty_card():
            components.empty_state(
                ":material/filter_alt_off:",
                "No evaluations match these filters",
                "Choose another dataset, task or gold column.",
            )
            with st.container(horizontal_alignment="center"):
                st.button("Clear filters", icon=":material/close:", on_click=_clear_filters)
        return

    _comparability_note(filtered)
    board = _leaderboard_section(filtered)
    # Colours follow every model in history, so filters never repaint one.
    colors = charts.model_colors(sorted(frame["model"].unique(), key=str.casefold), _dark())
    _charts_section(filtered, colors, list(board["model"]))
    _history_section(filtered, len(frame))


# ==========================================================================
# Data preparation (pure)
# ==========================================================================


def history_frame(raw: pd.DataFrame, custom: list[presets.Preset] | None = None) -> pd.DataFrame:
    """``Store.list_evaluations()`` plus display/filter columns:

    ``when`` (datetime), ``model`` (display name, one per model: names
    shared by different models get the end of their path), ``model_key``
    (model id or label - what "best per model" groups by), ``task`` (preset
    title, with the variable when there is one), ``dataset_key`` / ``gold_key``
    (:data:`UNKNOWN` when not recorded), ``source_word`` ("App", "CLI",
    "v1"), and the metric columns as floats. Order is kept (newest first).
    """
    frame = raw.copy().reset_index(drop=True)
    for column in store_mod.EVALUATION_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    frame["when"] = pd.to_datetime(frame["created_at"], errors="coerce")
    for column in ("accuracy", "macro_f1", "kappa", "rows_per_s"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(float)
    for column in ("n_rows", "n_scored", "n_correct", "n_invalid", "n_excluded"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Float64")

    labels = [_text(v) for v in frame["model_label"]]
    ids = [_text(v) for v in frame["model_id"]]
    frame["model_key"] = [mid or label or "?" for label, mid in zip(labels, ids, strict=True)]
    # One name per model: same-named local folders (checkpoint-500, final...)
    # would otherwise share a row label and a chart line.
    frame["model"] = fmt.distinct_model_names(
        [_model_name(label, mid) for label, mid in zip(labels, ids, strict=True)],
        list(frame["model_key"]),
    )
    frame["task_key"] = [_text(v) or UNKNOWN for v in frame["preset"]]
    frame["task_name"] = [task_title(key, "", custom) for key in frame["preset"]]
    frame["task"] = [
        task_title(key, _text(var), custom)
        for key, var in zip(frame["preset"], frame["var"], strict=True)
    ]
    frame["dataset_key"] = [_text(v) or UNKNOWN for v in frame["dataset"]]
    frame["gold_key"] = [_text(v) or UNKNOWN for v in frame["gold_column"]]
    frame["fingerprint"] = [_text(v) for v in frame["dataset_fingerprint"]]
    frame["source_word"] = [fmt.source_word(v) for v in frame["source"]]
    frame["precision_word"] = [fmt.precision(_text(v) or None) for v in frame["precision"]]
    frame["prompt_word"] = [
        prompt_word(edited, run_id)
        for edited, run_id in zip(frame["template_edited"], frame["run_id"], strict=True)
    ]
    return frame


def prompt_word(edited: Any, run_id: Any) -> str:
    """How a history line's prompt reads: "Preset", "Edited · 3f2a91" (the
    start of the run id, which names the exact prompt + settings) or "–"
    when it wasn't recorded (older lines)."""
    if edited is None or _text(edited) == "":
        return fmt.DASH
    try:
        is_edited = bool(int(float(edited)))
    except (TypeError, ValueError):
        is_edited = str(edited).strip().lower() in {"true", "yes"}
    if not is_edited:
        return "Preset"
    code = _text(run_id)[:6]
    return f"Edited · {code}" if code else "Edited"


def task_title(key: Any, var: str = "", custom: list[presets.Preset] | None = None) -> str:
    """Preset title for a history ``preset`` key ("Sentiment"), with the
    variable appended when the task used one ("Custom labels · refund, other")."""
    key_text = _text(key)
    if not key_text:
        return "Unknown task"
    preset = presets.get_preset(key_text, custom)
    title = preset.title if preset is not None else key_text
    return f"{title} · {fmt.truncate(var, 32)}" if var else title


def apply_filters(
    frame: pd.DataFrame,
    dataset: str = ALL,
    task: str = ALL,
    gold: str = ALL,
    selection: str = ALL,
) -> pd.DataFrame:
    """Rows of a :func:`history_frame` matching every filter that isn't
    :data:`ALL` (``selection`` is a dataset fingerprint)."""
    mask = pd.Series(True, index=frame.index)
    for column, wanted in (
        ("dataset_key", dataset),
        ("task_key", task),
        ("gold_key", gold),
        ("fingerprint", selection),
    ):
        if wanted and wanted != ALL:
            mask &= frame[column] == wanted
    return frame.loc[mask]


def leaderboard(frame: pd.DataFrame, best_per_model: bool = True) -> pd.DataFrame:
    """History rows ranked by macro-F1, then accuracy, then recency (rows
    without a macro-F1, e.g. version-1 lines, rank after those with one).
    With ``best_per_model`` only each model's top line is kept - one per
    prompt when a model was run with more than one (so prompt variants can
    be compared) - and a ``runs`` column counts the lines each one stands for."""
    if frame.empty:
        return frame.assign(runs=pd.Series(dtype="int64"))
    ranked = frame.sort_values(
        ["macro_f1", "accuracy", "when"], ascending=False, na_position="last"
    )
    prompts = (
        frame["prompt_word"]
        if "prompt_word" in frame.columns
        else pd.Series(fmt.DASH, index=frame.index)
    )
    variants = prompts.groupby(frame["model_key"]).nunique()
    split = set(variants[variants > 1].index)
    group = [
        f"{key}\x1f{prompt}" if key in split else key
        for key, prompt in zip(frame["model_key"], prompts, strict=True)
    ]
    groups = pd.Series(group, index=frame.index)
    ranked["_group"] = groups.loc[ranked.index]
    ranked["runs"] = ranked["_group"].map(groups.value_counts()).fillna(0).astype("int64")
    if best_per_model:
        ranked = ranked.drop_duplicates(subset="_group", keep="first")
    return ranked.drop(columns="_group").reset_index(drop=True)


def selection_options(frame: pd.DataFrame) -> dict[str, str]:
    """Distinct dataset fingerprints in ``frame`` -> a human label ("60 rows",
    "First 20 of 60 rows" when the selection was recorded). Where two labels
    would read the same (say the file was edited between runs), each gets the
    date it was last scored: "62 rows · last scored 13 Oct". Newest first."""
    base: dict[str, str] = {}
    last: dict[str, Any] = {}
    has_selection = "selection" in frame.columns
    for _, row in frame.iterrows():  # newest first: the first line is the latest
        fp = row["fingerprint"]
        if not fp or fp in base:
            continue
        label = _text(row["selection"]) if has_selection else ""
        if _ALL_ROWS.fullmatch(label):  # "All 62 rows" is "62 rows", as older lines say
            label = ""
        if not label:
            n_rows = row["n_rows"]
            label = fmt.plural(int(n_rows), "row") if pd.notna(n_rows) else "Unknown rows"
        base[fp] = label
        last[fp] = row["when"]

    out = dict(base)
    for label in {v for v in base.values() if list(base.values()).count(v) > 1}:
        clashing = [fp for fp, v in base.items() if v == label]
        for spell in (_day, _day_time):
            named = {fp: f"{label} · last scored {_when(last[fp], spell)}" for fp in clashing}
            if len(set(named.values())) == len(named):
                break
        else:  # same minute: number them, newest first
            named = {fp: f"{named[fp]} ({n})" for n, fp in enumerate(clashing, start=1)}
        out.update(named)
    return out


_ALL_ROWS = re.compile(r"all\s+[\d,]+\s+rows?", re.IGNORECASE)


def _day(value: datetime) -> str:
    return f"{value.day} {value:%b}"


def _day_time(value: datetime) -> str:
    return f"{value.day} {value:%b %H:%M}"


def _when(value: Any, spell: Callable[[datetime], str]) -> str:
    if value is None or pd.isna(value):
        return "on an unknown date"
    return spell(pd.Timestamp(value).to_pydatetime())


# ==========================================================================
# Version-1 import
# ==========================================================================


def _legacy_import(store: store_mod.Store) -> None:
    """The "found version-1 files" card, or the report of an import just done."""
    report = st.session_state.get(_K_IMPORT_REPORT)
    if report is not None:
        _import_report(report)
        return
    error = st.session_state.get(_K_IMPORT_ERROR)
    if error:
        components.error_with_details(
            "Couldn't import the files from the previous version.",
            details=error,
            hint="The files themselves are unchanged. Check they aren't open in another "
            "program, then try again.",
        )
    if st.session_state.get(_K_LEGACY_HIDDEN):
        return
    settings = resources.get_settings()
    try:
        if store.get_meta("legacy_imported_at"):
            return
        scan = store_mod.scan_legacy(settings.results_dir)
    except Exception:  # a status card must never break the page
        return
    if not scan.found:
        return

    with st.container(
        border=True,
        key="history_legacy_card",
        horizontal=True,
        vertical_alignment="center",
        gap="medium",
    ):
        components.decorative_icon(":material/move_to_inbox:", width="content")
        with st.container(gap="xsmall", width="stretch"):
            st.markdown(f"**{_legacy_headline(scan)}**")
            st.caption(
                "Import them to see past scores here and reuse saved outputs instead of "
                "generating them again. The original files stay as they are."
            )
        st.button(
            "Not now",
            key="history.legacy_hide",
            type="tertiary",
            on_click=_hide_legacy,
        )
        st.button(
            "Import",
            key="history.legacy_import",
            type="primary",
            icon=":material/download:",
            on_click=_run_import,
            args=(str(scan.results_dir),),
        )


def _legacy_headline(scan: store_mod.LegacyScan) -> str:
    runs = len(scan.manifests)
    if runs and scan.evals_csv is not None:
        found = f"{fmt.plural(runs, 'saved run')} and an evaluation history"
    elif runs:
        found = fmt.plural(runs, "saved run")
    else:
        found = "an evaluation history"
    return f"Found {found} from the previous version"


def _hide_legacy() -> None:
    st.session_state[_K_LEGACY_HIDDEN] = True


def _run_import(results_dir: str) -> None:
    st.session_state.pop(_K_IMPORT_ERROR, None)
    try:
        report = store_mod.import_legacy(resources.get_store(), results_dir)
    except Exception as exc:
        st.session_state[_K_IMPORT_ERROR] = f"{type(exc).__name__}: {exc}"
        return
    st.session_state[_K_IMPORT_REPORT] = report
    message, icon = _import_summary(report)
    components.queue_toast(message, icon)


def _import_summary(report: store_mod.LegacyImportReport) -> tuple[str, str]:
    parts = []
    if report.runs_imported:
        parts.append(
            f"{fmt.plural(report.runs_imported, 'saved run')} "
            f"({fmt.plural(report.outputs_imported, 'output')})"
        )
    if report.evaluations_imported:
        parts.append(fmt.plural(report.evaluations_imported, "evaluation"))
    if parts:
        return ("Imported " + " and ".join(parts), ":material/check_circle:")
    return ("Nothing new to import", ":material/info:")


def import_notes(report: store_mod.LegacyImportReport) -> tuple[list[str], list[str]]:
    """(skipped, merged): what couldn't be imported, apart from version 1's
    repeated evaluation log lines - those were merged into one line each,
    nothing was lost - which the store reports among the skipped items."""
    merged = [str(n) for n in getattr(report, "notes", None) or []]
    skipped: list[str] = []
    for reason in report.skipped:
        (merged if _MERGED in str(reason) else skipped).append(str(reason))
    return skipped, merged


_MERGED = "kept once"
"""How the store words a merge of repeated version-1 log lines."""
_REPEATS = re.compile(r"(\d[\d,]*) rows? repeats?")


def merged_note(merged: list[str]) -> str | None:
    """ "3 repeated log lines from version 1 were merged (it logged an
    evaluation again on every rerun)." or None."""
    if not merged:
        return None
    counts = [_REPEATS.search(m) for m in merged]
    n = sum(int(c.group(1).replace(",", "")) for c in counts if c)
    what = fmt.plural(n, "repeated log line") if n else "Repeated log lines"
    verb = "was" if n == 1 else "were"
    return (
        f"{what} from version 1 {verb} merged into the evaluation they repeat (version 1 "
        "logged an evaluation again on every rerun)."
    )


def _import_report(report: store_mod.LegacyImportReport) -> None:
    """What the import just did, in the card that offered it."""
    message, _ = _import_summary(report)
    skipped, merged = import_notes(report)
    imported = bool(report.runs_imported or report.evaluations_imported)
    details = []
    if report.evaluations_imported:
        details.append(
            "Version 1 didn't record which file was scored, so its evaluations are listed "
            "under Dataset: Not recorded."
        )
    note = merged_note(merged)
    if note:
        details.append(note)
    if skipped:
        details.append(f"{fmt.plural(len(skipped), 'item')} couldn't be imported (see below).")
    if not imported and not skipped:
        details.append("Everything in the results folder had been imported already.")

    with st.container(border=True, key="history_import_report"):
        with st.container(horizontal=True, vertical_alignment="center", gap="medium"):
            components.decorative_icon(
                ":material/check_circle:" if imported else ":material/info:", width="content"
            )
            with st.container(gap="xsmall", width="stretch"):
                st.markdown(f"**{message} from the previous version**")
                if details:
                    st.caption(" ".join(details))
            if report.evaluations_imported:
                st.button(
                    "Show imported scores",
                    key="history.import_show",
                    icon=":material/filter_alt:",
                    on_click=_show_legacy,
                )
            st.button(
                "Dismiss",
                key="history.import_dismiss",
                type="tertiary",
                on_click=lambda: st.session_state.pop(_K_IMPORT_REPORT, None),
            )
        if skipped:
            with st.expander(f"Not imported · {len(skipped)}", icon=":material/info:"):
                st.markdown("\n".join(f"- {fmt.md(skip_reason(reason))}" for reason in skipped))


def skip_reason(reason: str) -> str:
    """An import report's skip reason in the app's words ("saved outputs",
    never "checkpoint")."""
    return reason.replace("unreadable checkpoint", "unreadable saved-outputs file")


def _show_legacy() -> None:
    """Point the filters at the imported version-1 evaluations. Task and
    gold column are left to default to the newest version-1 line, so
    different tasks aren't ranked on one leaderboard."""
    _chose_filters()
    st.session_state[K_DATASET] = UNKNOWN
    for key in (K_TASK, K_GOLD, K_SELECTION):
        st.session_state[key] = None  # not an option: resolved to the default


# ==========================================================================
# Filters
# ==========================================================================


def _filters(frame: pd.DataFrame) -> pd.DataFrame:
    """The filter row; returns the matching rows.

    The filters cascade (each one offers only values present under the ones
    before it). Until the user picks one they follow the newest evaluation -
    its dataset, task, gold column and rows - so the page opens on a
    like-for-like comparison that includes the run just scored. A link opens
    the view it names: the URL always spells out every filter."""
    newest_id = int(frame["id"].iloc[0])
    st.session_state[_K_NEWEST] = newest_id
    from_url: dict[str, str] = {}
    if _K_STARTED not in st.session_state:  # first visit in this browser session
        st.session_state[_K_STARTED] = True
        st.session_state[_K_SEEN] = newest_id
        from_url = _url_filters()
        st.session_state[_K_AUTO] = not from_url
    # Following filters move to a newly logged evaluation - but not when one
    # is deleted, and not while rows are ticked in the table: the view never
    # changes under a selection. Then the filters stay put and a note offers
    # the new evaluation instead.
    auto = False
    if st.session_state.get(_K_AUTO, True):
        followed = st.session_state.get(_K_FOLLOWED)
        if followed is not None and newest_id > followed and _rows_ticked():
            st.session_state[_K_AUTO] = False
            st.session_state[_K_SEEN] = followed
        else:
            auto = followed is None or newest_id > followed
            st.session_state[_K_FOLLOWED] = newest_id

    cols = st.columns(4, gap="medium")
    datasets = [ALL, *_options(frame["dataset_key"])]
    _resolve(K_DATASET, datasets, default_filter(frame, "dataset_key"), auto, from_url)
    with cols[0]:
        dataset = st.selectbox(
            "Dataset",
            datasets,
            key=K_DATASET,
            persist_state="session",
            on_change=_chose_filters,
            format_func=lambda v: _filter_label(v, "All datasets"),
        )
    in_dataset = apply_filters(frame, dataset=dataset)

    tasks = [ALL, *_options(in_dataset["task_key"])]
    task_names = dict(zip(in_dataset["task_key"], in_dataset["task_name"], strict=True))
    _resolve(K_TASK, tasks, default_filter(in_dataset, "task_key"), auto, from_url)
    with cols[1]:
        task = st.selectbox(
            "Task",
            tasks,
            key=K_TASK,
            persist_state="session",
            on_change=_chose_filters,
            format_func=lambda v: _filter_label(v, "All tasks", task_names),
        )
    in_task = apply_filters(in_dataset, task=task)

    golds = [ALL, *_options(in_task["gold_key"])]
    _resolve(K_GOLD, golds, default_filter(in_task, "gold_key"), auto, from_url)
    with cols[2]:
        gold = st.selectbox(
            "Gold column",
            golds,
            key=K_GOLD,
            persist_state="session",
            on_change=_chose_filters,
            format_func=lambda v: _filter_label(v, "All gold columns"),
        )
    narrowed = apply_filters(in_task, gold=gold)

    selection = ALL
    rows_param: str | None = None
    choices = selection_options(narrowed) if dataset != ALL else {}
    if len(choices) > 1:
        options = [ALL, *choices]
        _resolve(
            K_SELECTION,
            options,
            default_selection(narrowed),
            auto,
            from_url,
            match=_match_fingerprint,
        )
        with cols[3]:
            selection = st.selectbox(
                "Rows",
                options,
                key=K_SELECTION,
                persist_state="session",
                on_change=_chose_filters,
                format_func=lambda v: "All row selections" if v == ALL else choices.get(v, v),
                help="Scores on different rows of the same file (say the first 20 rows and "
                "all of them) aren't directly comparable. Opens on the rows of the newest "
                "evaluation.",
            )
        rows_param = _url_value(selection, selection[:_FP_CHARS])
    elif len(choices) == 1:  # pin it, so a link keeps meaning these rows
        rows_param = next(iter(choices))[:_FP_CHARS]

    _sync_url(
        {
            "dataset": _url_value(dataset),
            "task": _url_value(task),
            "gold": _url_value(gold),
            "rows": rows_param,
        }
    )
    return apply_filters(narrowed, selection=selection)


def _rows_ticked() -> bool:
    """Whether rows of the full-history table are ticked (as of this rerun)."""
    for key in list(st.session_state.keys()):
        if str(key).startswith(K_TABLE + "::"):
            return bool(_selected_rows(str(key), 1_000_000))
    return bool(st.session_state.get(_K_SELECTED))


def _resolve(
    key: str,
    options: list[str],
    default: str,
    auto: bool,
    from_url: dict[str, str],
    match: Callable[[str, list[str]], str | None] | None = None,
) -> None:
    """Give the filter ``key`` its value for this run, before its widget is
    drawn: ``default`` while the filters follow the newest evaluation, the
    URL's value on a link's first load (``default`` if the link names
    something that no longer exists), else the user's choice - or
    ``default`` once that choice has vanished from ``options`` because an
    earlier filter changed."""
    if auto:
        st.session_state[key] = default
        return
    raw = from_url.get(URL_PARAMS[key])
    if raw is not None:
        value = _from_url(raw, options, match)
        st.session_state[key] = default if value is None else value
        return
    if st.session_state.get(key) not in options:
        st.session_state[key] = default


def _from_url(raw: str, options: list[str], match: Callable | None) -> str | None:
    """The option a URL value names, :data:`ALL` for "all", or None."""
    if raw in options:
        return raw
    if match is not None:
        found = match(raw, options)
        if found is not None:
            return found
    spelled = raw.strip().casefold()
    if spelled == URL_ALL:
        return ALL
    if spelled == URL_UNKNOWN and UNKNOWN in options:
        return UNKNOWN
    return None


def _match_fingerprint(raw: str, options: list[str]) -> str | None:
    """The one fingerprint option that starts with ``raw`` (URLs carry a prefix)."""
    token = raw.strip()
    if len(token) < 6:
        return None
    found = [o for o in options if o != ALL and o.startswith(token)]
    return found[0] if len(found) == 1 else None


def _url_value(value: str, spelled: str | None = None) -> str:
    if value == ALL:
        return URL_ALL
    if value == UNKNOWN:
        return URL_UNKNOWN
    return spelled or value


def _url_filters() -> dict[str, str]:
    """The filters a link names (query parameter -> raw value)."""
    out: dict[str, str] = {}
    for name in URL_PARAMS.values():
        try:
            value = st.query_params.get(name)
        except Exception:
            value = None
        if value:
            out[name] = str(value)
    return out


def _sync_url(values: dict[str, str | None]) -> None:
    """Mirror the view in the URL (only the parameters that changed)."""
    for name, value in values.items():
        try:
            if value is None:
                if name in st.query_params:
                    del st.query_params[name]
            elif st.query_params.get(name) != value:
                st.query_params[name] = value
        except Exception:  # the URL is a convenience; never break the page
            pass


def _chose_filters() -> None:
    """The user picked a filter: stop following the newest evaluation."""
    st.session_state[_K_STARTED] = True
    st.session_state[_K_AUTO] = False
    st.session_state[_K_SEEN] = st.session_state.get(_K_NEWEST)


def default_filter(frame: pd.DataFrame, column: str) -> str:
    """Default for a filter over ``column``: the newest evaluation's value
    (``frame`` is newest first), so the filter names what is shown; :data:`ALL`
    when there is nothing to choose from."""
    values = {v for v in frame[column] if isinstance(v, str) and v}
    if not values:
        return ALL
    newest = frame[column].iloc[0]
    return newest if newest in values else ALL


def default_selection(frame: pd.DataFrame) -> str:
    """The newest evaluation's dataset fingerprint (the same rule as the
    other filters, so the run just scored is always in the default view), or
    :data:`ALL` when it has none."""
    if frame.empty:
        return ALL
    newest = frame["fingerprint"].iloc[0]
    return newest if isinstance(newest, str) and newest else ALL


def _options(values: pd.Series) -> list[str]:
    """Distinct filter values, alphabetical, with :data:`UNKNOWN` last."""
    distinct = {str(v) for v in values if isinstance(v, str) and v}
    known = sorted(distinct - {UNKNOWN}, key=str.casefold)
    return known + ([UNKNOWN] if UNKNOWN in distinct else [])


def _filter_label(value: str, all_label: str, names: dict[str, str] | None = None) -> str:
    if value == ALL:
        return all_label
    if value == UNKNOWN:
        return "Not recorded (version 1)"
    return (names or {}).get(value, value)


def _clear_filters() -> None:
    _chose_filters()
    for key in (K_DATASET, K_TASK, K_GOLD, K_SELECTION):
        st.session_state[key] = ALL


def _latest_hidden_note(frame: pd.DataFrame, filtered: pd.DataFrame) -> None:
    """A one-click way back when an evaluation logged since the user chose
    these filters (say the run just scored) falls outside them."""
    newest = frame.iloc[0]
    newest_id = int(newest["id"])
    if st.session_state.get(_K_AUTO, True) or st.session_state.get(_K_SEEN) == newest_id:
        return
    if newest_id in set(filtered["id"].astype(int)):
        return
    what = f"{newest['model']} · {newest['task']}"
    if newest["dataset_key"] != UNKNOWN:
        what += f" on {newest['dataset_key']}"
    when = fmt.timestamp(newest["when"].to_pydatetime()) if pd.notna(newest["when"]) else ""
    with st.container(
        horizontal=True, vertical_alignment="center", gap="small", key="history_latest_note"
    ):
        st.caption(
            f":material/new_releases: The latest evaluation, {fmt.md(what)}"
            + (f" ({when})" if when else "")
            + ", isn't in this view.",
            width="content",
        )
        st.button(
            "Show it",
            key="history.show_latest",
            type="tertiary",
            icon=":material/arrow_forward:",
            icon_position="right",
            on_click=_follow_newest,
        )


def _follow_newest() -> None:
    """Back to the default view: the filters follow the newest evaluation."""
    st.session_state[_K_AUTO] = True
    st.session_state.pop(_K_FOLLOWED, None)
    st.session_state[_K_SEEN] = st.session_state.get(_K_NEWEST)


def _comparability_note(frame: pd.DataFrame) -> None:
    """A muted warning when the view mixes scores that measure different things."""
    datasets = frame["dataset_key"].nunique()
    golds = frame["gold_key"].nunique()
    tasks = frame["task_key"].nunique()
    fingerprints = frame.loc[frame["fingerprint"] != "", "fingerprint"].nunique()
    if datasets > 1:
        note = (
            f"This view mixes {datasets} datasets. Scores from different data aren't "
            "directly comparable: pick a dataset to compare like with like."
        )
    elif golds > 1:
        note = (
            f"This view mixes {golds} gold columns. Pick one to compare models on the same labels."
        )
    elif tasks > 1 and (frame["gold_key"] == UNKNOWN).any():
        note = (
            f"This view mixes {tasks} tasks, and version 1 didn't record which gold column "
            "each was scored against, so these scores may measure different things. Pick a "
            "task to compare like with like."
        )
    elif fingerprints > 1:
        note = (
            f"These evaluations cover {fingerprints} different row selections of the data "
            "(for example the first 20 rows and all rows). Pick one under Rows to compare "
            "like with like."
        )
    else:
        return
    st.caption(f":material/info: {note}")


# ==========================================================================
# Leaderboard + charts
# ==========================================================================


def _leaderboard_section(frame: pd.DataFrame) -> pd.DataFrame:
    """The leaderboard; returns it (its order ranks the models for the
    charts' colour key)."""
    st.space("small")
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.subheader("Leaderboard", anchor=False, width="content")
        st.space("stretch")
        view = st.segmented_control(
            "View",
            [VIEW_BEST, VIEW_ALL],
            default=VIEW_BEST,
            required=True,
            key=K_VIEW,
            bind="query-params",
            label_visibility="collapsed",
        )
    board = leaderboard(frame, best_per_model=view != VIEW_ALL)
    varies = _varying_columns(frame)
    folded, same = constant_columns(board)

    table = pd.DataFrame(
        {
            "Model": board["model"],
            "Task": board["task"],
            "Dataset": board["dataset_key"].map(_dataset_word),
            "Prompt": board["prompt_word"],
            "Rows scored": board["n_scored"],
            "Accuracy": board["accuracy"] * 100,
            "Macro-F1": board["macro_f1"] * 100,
            "Cohen's κ": board["kappa"],
            "Invalid": board["n_invalid"],
            "Weights": board["precision_word"],
            "Runs": board["runs"],
            "Date": board["when"],
            "Source": board["source_word"],
        }
    )
    # Sort order is the rank: no rank column.
    order = [
        "Model",
        *[c for c in ("Task", "Dataset") if c in varies],
        "Prompt",
        "Rows scored",
        "Accuracy",
        "Macro-F1",
        "Cohen's κ",
        "Invalid",
        "Weights",
        *(["Runs"] if view != VIEW_ALL else []),
        "Date",
        *(["Source"] if "Source" in varies else []),
    ]
    st.dataframe(
        table,
        hide_index=True,
        placeholder=fmt.DASH,
        height="content" if len(table) <= 15 else 560,
        # Columns with one value for every run are said once in the caption,
        # which keeps the table narrow enough for a laptop screen.
        column_order=[c for c in order if c not in folded],
        column_config=_metric_columns(compact=True)
        | {
            # Medium, not large: at 1024 px every column still fits, and the
            # grid stretches it on wider screens.
            "Model": st.column_config.TextColumn(width="medium"),
            "Runs": st.column_config.NumberColumn(
                format="%d",
                width=56,
                help="Scored runs this line stands for (its best is shown)",
            ),
        },
        key=f"history.leaderboard::{view}",
        alt="Leaderboard of evaluations ranked by macro-F1, then accuracy",
    )
    note = (
        "Each model's best run in this view, ranked by macro-F1 then accuracy."
        if view != VIEW_ALL
        else "Every run in this view, ranked by macro-F1 then accuracy."
    )
    if view != VIEW_ALL and "Prompt" not in folded:
        note = note.replace("Each model's best run", "Each model's best run per prompt")
    if same:
        note += f" {'All runs here' if len(board) > 1 else 'This run'}: {' · '.join(same)}."
    if (frame["source_word"] == "v1").any():
        note += " Version-1 runs recorded accuracy only, so they rank after the others."
    st.caption(note)
    return board


def constant_columns(board: pd.DataFrame, *, excluded: bool = False) -> tuple[set[str], list[str]]:
    """Table columns with the same value on every line, and how to say each
    once instead: ({"Weights", ...}, ["4-bit weights", "61 rows scored", ...]).
    ``excluded`` also folds the history table's Excluded column."""
    folded: set[str] = set()
    words: list[str] = []
    if board.empty:
        return folded, words
    precisions = set(board["precision_word"])
    if len(precisions) == 1:
        folded.add("Weights")
        (only,) = precisions
        if only == fmt.DASH:
            words.append("weights not recorded")
        elif only != fmt.PRECISION_NAMES["none"]:  # the demo model has no weights
            words.append(f"{only} weights")
    if "prompt_word" in board.columns:
        prompts = set(board["prompt_word"])
        if len(prompts) == 1:
            folded.add("Prompt")
            (only,) = prompts
            if only == "Preset":
                words.append("preset prompt")
            elif only != fmt.DASH:
                words.append(f"{only.lower()} prompt".replace("edited · ", "edited "))
    counts: list[tuple[str, str]] = [("n_scored", "Rows scored"), ("n_invalid", "Invalid")]
    if excluded:
        counts.append(("n_excluded", "Excluded"))
    each = " each" if len(board) > 1 else ""
    for column, name in counts:
        values = board[column]
        if values.isna().any() or values.nunique() != 1:
            continue
        number = int(values.iloc[0])
        folded.add(name)
        if column == "n_invalid":
            words.append(
                "no invalid answers" if number == 0 else fmt.plural(number, "invalid answer") + each
            )
        elif column == "n_excluded":
            words.append(
                "no rows left out"
                if number == 0
                else fmt.plural(number, "row") + " left out of scoring" + each
            )
        else:
            words.append(f"{fmt.plural(number, 'row')} scored")
    # Version-1 lines recorded accuracy only; a column with nothing in it goes.
    for column, name in (("macro_f1", "Macro-F1"), ("kappa", "Cohen's κ")):
        if board[column].isna().all():
            folded.add(name)
    return folded, words


def _metric_columns(compact: bool = False) -> dict[str, Any]:
    """Column config shared by the leaderboard and the full history table."""
    bar_width = 132 if compact else None
    return {
        "Accuracy": _PERCENT(
            format="%.1f%%",
            min_value=0.0,
            max_value=100.0,
            help=_HELP_ACCURACY,
            width=bar_width,
        ),
        "Macro-F1": _PERCENT(
            format="%.1f%%", min_value=0.0, max_value=100.0, help=_HELP_F1, width=bar_width
        ),
        "Cohen's κ": st.column_config.NumberColumn(format="%.2f", help=components.HELP_KAPPA),
        "Rows scored": st.column_config.NumberColumn(format="localized"),
        "Invalid": st.column_config.NumberColumn(
            format="localized",
            help="Answers outside the label set. They count as wrong.",
        ),
        "Date": st.column_config.DatetimeColumn(format=_DATE_FORMAT),
        "Source": st.column_config.TextColumn(
            help="App, command line (CLI), or imported from version 1 (v1)."
        ),
        "Weights": st.column_config.TextColumn(
            help="Numeric precision the model weights were loaded in."
        ),
        "Prompt": st.column_config.TextColumn(
            help="Edited means the prompt was changed from the preset; the code identifies "
            "the exact prompt + settings."
        ),
    }


def _charts_section(frame: pd.DataFrame, colors: dict[str, str], ranked: list[str]) -> None:
    """Model quality beside score over time, under one colour key (models in
    leaderboard order; colours fixed per model across filters)."""
    plot = frame.assign(model_label=frame["model"], preset=frame["task"])
    plot["dataset"] = plot["dataset_key"].map(_dataset_word)
    for column in ("n_rows", "n_scored"):
        plot[column] = plot[column].astype(float)  # plain floats serialise cleanly
    both = plot.dropna(subset=["accuracy", "macro_f1"])
    metric = "macro_f1" if plot["macro_f1"].notna().any() else "accuracy"
    timed = plot.dropna(subset=[metric, "when"])
    if len(both) < 2 and len(timed) < 2:
        return  # one point is no comparison; the table says it all

    st.space("small")
    in_view = [m for m in dict.fromkeys(ranked) if m in set(plot["model"])]
    components.color_key({m: colors[m] for m in in_view if m in colors}, label="Models")
    # Version-1 lines have accuracy only: then the timeline gets the full width.
    if both.empty:
        _timeline_chart(plot, timed, metric, colors)
        return
    if timed.empty:
        _quality_chart(plot, colors)
        return
    left, right = st.columns(2, gap="large")
    with left:
        _quality_chart(plot, colors)
    with right:
        _timeline_chart(plot, timed, metric, colors)


def _dark() -> bool:
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return False


def _quality_chart(plot: pd.DataFrame, colors: dict[str, str]) -> None:
    has_speed = "rows_per_s" in plot.columns and plot["rows_per_s"].notna().any()
    # Fixed height (no border), like the timeline's heading: the two charts
    # start level whether or not the speed toggle is shown.
    with st.container(
        horizontal=True, vertical_alignment="center", gap="small", height=40, border=False
    ):
        st.subheader("Model quality", anchor=False, width="content")
        st.space("stretch")
        mode = _SCATTER_QUALITY
        if has_speed:
            mode = (
                st.segmented_control(
                    "Compare",
                    [_SCATTER_QUALITY, _SCATTER_SPEED],
                    default=_SCATTER_QUALITY,
                    required=True,
                    key=K_SCATTER,
                    label_visibility="collapsed",
                )
                or _SCATTER_QUALITY
            )
    if mode == _SCATTER_SPEED:
        y = "macro_f1" if plot["macro_f1"].notna().any() else "accuracy"
        x = "rows_per_s"
        caption = "Higher is better, further right is faster. One point per run."
    else:
        x, y = "accuracy", "macro_f1"
        caption = (
            "One point per run. Points below the dashed line do better on accuracy than on "
            "macro-F1: they get the common labels right and miss rare ones."
        )
    points = plot.dropna(subset=[x, y])
    if points.empty:
        st.caption("No run in this view has both scores recorded.")
        return
    # Name the points when each model has one and there are few of them.
    labelled = len(points) <= 8 and points["model"].is_unique
    st.altair_chart(
        charts.history_scatter(plot, x, y, dark=_dark(), colors=colors, point_labels=labelled),
        width="stretch",
        alt=f"Scatter plot of {charts.METRIC_TITLES.get(y, y)} against "
        f"{charts.METRIC_TITLES.get(x, x)}, one point per run, coloured by model",
    )
    st.caption(caption)


def _timeline_chart(
    plot: pd.DataFrame, timed: pd.DataFrame, metric: str, colors: dict[str, str]
) -> None:
    title = charts.METRIC_TITLES.get(metric, metric)
    # Fixed height (no border): level with the quality chart's heading row.
    with st.container(
        horizontal=True, vertical_alignment="center", gap="small", height=40, border=False
    ):
        st.subheader(f"{title} over time", anchor=False, width="content")
    if timed.empty:
        st.caption("No dated runs in this view.")
        return
    st.altair_chart(
        charts.history_timeline(plot, metric, dark=_dark(), colors=colors),
        width="stretch",
        alt=f"Line chart of {title.lower()} over time, one line per model",
    )
    st.caption(
        "One line per model, the same score the leaderboard ranks by. Hover a point for the run."
    )


_OPTIONAL = ("Task", "Dataset", "Gold column", "Source")
"""Table columns shown only when they differ between rows in view."""


def _varying_columns(frame: pd.DataFrame) -> set[str]:
    """Which of :data:`_OPTIONAL` are worth a column for these rows: those
    with more than one value, and Source when anything isn't from the app."""
    out = {
        name
        for name, column in (
            ("Task", "task"),
            ("Dataset", "dataset_key"),
            ("Gold column", "gold_key"),
        )
        if frame[column].nunique() > 1
    }
    if (frame["source_word"] != "App").any():
        out.add("Source")
    return out


def _dataset_word(value: str) -> str:
    return "Not recorded" if value == UNKNOWN else value


# ==========================================================================
# Full history + deletion
# ==========================================================================


def history_table(frame: pd.DataFrame) -> pd.DataFrame:
    """The full-history table for a :func:`history_frame` (any filter
    applied), newest first, scores as fractions (0-1)."""
    frame = frame.reset_index(drop=True)
    return pd.DataFrame(
        {
            "Date": frame["when"],
            "Model": frame["model"],
            "Task": frame["task"],
            "Dataset": frame["dataset_key"].map(_dataset_word),
            "Gold column": frame["gold_key"].map(_dataset_word),
            "Rows scored": frame["n_scored"],
            "Accuracy": frame["accuracy"],
            "Macro-F1": frame["macro_f1"],
            "Cohen's κ": frame["kappa"],
            "Invalid": frame["n_invalid"],
            "Weights": frame["precision_word"],
            "Prompt": frame["prompt_word"],
            "Source": frame["source_word"],
            "Rows": frame["n_rows"],
            "Excluded": frame["n_excluded"],
            "Note": [_text(v) for v in frame["note"]],
            "Run id": [_text(v) for v in frame["run_id"]],
        }
    )


def _history_section(frame: pd.DataFrame, total: int) -> None:
    frame = frame.reset_index(drop=True)
    table = history_table(frame)
    ids = [int(v) for v in frame["id"]]
    varies = _varying_columns(frame)
    folded, same = constant_columns(frame, excluded=True)
    if not any(table["Note"]):
        folded.add("Note")
    # The CSV keeps everything; on screen "Rows" repeats Rows scored + Excluded.
    folded.add("Rows")

    # The selection is remembered by evaluation id, not by row position: the
    # table's key changes whenever the rows in view change (a filter, a
    # deletion, a run logged from another tab), and the new table starts
    # with the same evaluations ticked - never with whatever now sits at the
    # old positions.
    key = table_key(ids)
    if key in st.session_state:  # same rows as last run: the table's own state is current
        chosen = [ids[i] for i in _selected_rows(key, len(ids))]
    else:
        wanted = set(st.session_state.get(_K_SELECTED) or [])
        chosen = [i for i in ids if i in wanted]
    st.session_state[_K_SELECTED] = chosen
    busy = resources.get_job_manager().active() is not None

    st.space("small")
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.subheader("Evaluations", anchor=False, width="content")
        shown = fmt.plural(len(table), "evaluation")
        st.caption(shown if len(table) == total else f"{shown} of {total:,}", width="content")
        if len(table) < total:
            st.button(
                f"Show all {total:,}",
                key="history.show_all",
                type="tertiary",
                icon=":material/filter_alt_off:",
                on_click=_clear_filters,
            )
        st.space("stretch")
        st.download_button(
            "Download CSV",
            data=results.to_csv_bytes(_export_frame(table)),
            file_name=f"evaluation-history_{datetime.now():%Y%m%d-%H%M}.csv",
            mime="text/csv",
            icon=":material/download:",
            on_click="ignore",
            key="history.download",
        )
        n_selected = len(chosen)
        # Keyed by what it deletes, so a click made on a stale page can't
        # land on a different set of evaluations.
        clicked = st.button(
            f"Delete {fmt.plural(n_selected, 'evaluation')}" if n_selected else "Delete selected",
            key=delete_key(chosen),
            icon=":material/delete:",
            disabled=busy or not n_selected,
            help="Wait for the current run to finish."
            if busy
            else (None if n_selected else "Select rows in the table first."),
        )

    shown = table.assign(Accuracy=table["Accuracy"] * 100)
    shown["Macro-F1"] = table["Macro-F1"] * 100
    st.dataframe(
        shown,
        hide_index=True,
        placeholder=fmt.DASH,
        key=key,
        on_select="rerun",
        selection_mode="multi-row",
        selection_default={"selection": {"rows": [ids.index(i) for i in chosen]}}
        if chosen
        else None,
        # Columns the filters pin to one value, or that hold one value for
        # every line, are left out (they're in the CSV, and said below).
        column_order=[
            c
            for c in table.columns
            if (c not in _OPTIONAL or c in varies) and c not in folded and c != "Run id"
        ],
        column_config=_metric_columns()
        | {
            "Accuracy": st.column_config.NumberColumn(format="%.1f%%", help=_HELP_ACCURACY),
            "Macro-F1": st.column_config.NumberColumn(format="%.1f%%", help=_HELP_F1),
            "Rows": st.column_config.NumberColumn(format="localized"),
            "Excluded": st.column_config.NumberColumn(
                format="localized",
                help="Rows left out of scoring: blank text, failed generation, or a gold "
                "label outside the label set.",
            ),
            "Model": st.column_config.TextColumn(width="large"),
            "Note": st.column_config.TextColumn(width="medium"),
        },
        alt="Every evaluation in the current view, newest first",
    )
    note = "Select rows to delete them. Deleting history lines keeps saved outputs."
    if same:
        note = f"{'All' if len(table) > 1 else 'This one'}: {' · '.join(same)}. " + note
    st.caption(note)

    if clicked and chosen and not busy:
        position = {eval_id: n for n, eval_id in enumerate(ids)}
        summary = [_row_summary(table.iloc[position[i]]) for i in chosen]
        _confirm_delete(chosen, summary)


def table_key(ids: list[int]) -> str:
    """Key of the full-history table showing evaluations ``ids`` (in order)."""
    return f"{K_TABLE}::{_signature(ids)}"


def delete_key(ids: list[int]) -> str:
    """Key of the delete button while evaluations ``ids`` are selected."""
    return f"{K_DELETE}::{_signature(ids) if ids else 'none'}"


def _signature(values: Iterable[Any]) -> str:
    return f"{zlib.crc32(','.join(map(str, values)).encode()):08x}"


def _selected_rows(key: str, n_rows: int) -> list[int]:
    """Valid selected row positions of the history table (from widget state)."""
    event: Any = st.session_state.get(key)
    try:
        rows = event["selection"]["rows"]
    except (KeyError, TypeError):
        rows = getattr(getattr(event, "selection", None), "rows", None) or []
    return sorted({int(r) for r in rows if 0 <= int(r) < n_rows})


def _row_summary(row: pd.Series) -> str:
    when = fmt.timestamp(row["Date"].to_pydatetime()) if pd.notna(row["Date"]) else ""
    score = fmt.percent(row["Accuracy"])
    parts = [str(row["Model"]), str(row["Task"]), f"accuracy {score}"]
    if when:
        parts.append(when)
    return " · ".join(parts)


def _export_frame(table: pd.DataFrame) -> pd.DataFrame:
    """The table as plain values (dates as ISO text, scores as fractions)."""
    out = table.copy()
    out["Date"] = [d.isoformat(sep=" ") if pd.notna(d) else "" for d in out["Date"]]
    return out


@st.dialog("Delete evaluations?")
def _confirm_delete(ids: list[int], summary: list[str]) -> None:
    if st.session_state.pop(_K_DELETED, None):  # done by the callback below
        st.rerun()
    n = len(ids)
    st.write(
        f"Remove {fmt.plural(n, 'evaluation')} from history? Saved outputs are kept. "
        "This can't be undone."
    )
    lines = [f"- {fmt.md(line)}" for line in summary[:6]]
    if n > 6:
        lines.append(f"- … and {n - 6:,} more")
    st.markdown("\n".join(lines))
    with st.container(horizontal=True, horizontal_alignment="right", gap="small"):
        if st.button("Cancel", key="history.delete_cancel"):
            st.rerun()
        st.button(
            f"Delete {fmt.plural(n, 'evaluation')}",
            key="history.delete_confirm",
            type="primary",
            icon=":material/delete:",
            on_click=_delete_evaluations,
            args=(ids,),
        )
    message = st.session_state.pop(_K_DELETE_MSG, None)
    if message:
        st.warning(message, icon=":material/warning:")


def _delete_evaluations(ids: Iterable[int]) -> None:
    if resources.get_job_manager().active() is not None:
        st.session_state[_K_DELETE_MSG] = "A run is in progress. Try again when it has finished."
        return
    store = resources.get_store()
    deleted = 0
    try:
        for eval_id in ids:
            deleted += bool(store.delete_evaluation(int(eval_id)))
    except Exception as exc:
        st.session_state[_K_DELETE_MSG] = (
            f"Deleted {fmt.plural(deleted, 'evaluation')}, then failed: {exc}"
        )
        return
    st.session_state[_K_DELETED] = True
    st.session_state[_K_SELECTED] = []
    components.queue_toast(
        f"Deleted {fmt.plural(deleted, 'evaluation')} from history", ":material/delete:"
    )


# ==========================================================================
# Empty page and small helpers
# ==========================================================================


def _empty_page() -> None:
    with components.empty_card():
        components.empty_state(
            ":material/leaderboard:",
            "No evaluations yet",
            "Pick a gold-label column on the Run page to score your next run. "
            "Each scored run is saved here so you can compare models and prompts.",
        )
        with st.container(horizontal_alignment="center"):
            components.page_button("run", "Go to the Run page", ":material/play_circle:")


def _custom_presets(store: store_mod.Store) -> list[presets.Preset]:
    try:
        return store.list_presets()
    except Exception:
        return []


def _model_name(label: str, model_id: str) -> str:
    """The catalogue's current name for a known model (so version-1 lines
    read "Qwen2.5-3B-Instruct", not "Qwen2.5-3B-Instruct (faster)"), else
    the recorded label, else the last part of the id."""
    spec = models.find_model(model_id) if model_id else None
    if spec is not None:
        return spec.label
    if label:
        return label
    if model_id:
        return fmt.model_name(model_id)
    return "Unknown model"


def _text(value: Any) -> str:
    """A cell as stripped text; "" for None / NaN / NA."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()
