"""Altair charts for the Results and History pages.

Pure functions: each takes a DataFrame (or a list) and returns a finished
Altair chart; nothing here imports Streamlit. Render with
``st.altair_chart(chart, width="stretch")`` and the default
``theme="streamlit"`` (fonts and sizing follow the app theme).

Every chart goes through :func:`finish`, so axes, legends and gridlines look
the same everywhere and axis text keeps AA contrast in both themes (the
Streamlit theme's grey is only 3.7:1 on white). Colours that carry meaning are
set explicitly rather than left to the theme, and text marks take a ``dark``
flag - pass ``st.context.theme.type == "dark"``: Vega draws them black unless
told otherwise.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

import altair as alt
import pandas as pd

from batchtextreview.labels import INVALID
from batchtextreview.ui.analysis import display_label

# ---- palette ----------------------------------------------------------------

ACCENT = "#4F46E5"
"""Indigo (the light theme's primary colour) for bars: white text on it passes AA."""
ACCENT_DARK = "#818CF8"
"""The dark theme's indigo (its primary colour): solid bars on near-black."""
ON_ACCENT = "#FFFFFF"
"""Text on :data:`ACCENT` (6.3:1)."""
ON_ACCENT_DARK = "#111827"
"""Text on :data:`ACCENT_DARK` (about 6:1; white would be 3:1)."""
TEXT_LIGHT = "#1F2937"
TEXT_DARK = "#E6EDF3"
"""Text colours for marks on the page background (light / dark theme)."""
MUTED_LIGHT = "#4B5563"
MUTED_DARK = "#9CA3AF"
"""Axis labels and secondary marks (7.6:1 / 7.5:1 on the page background)."""
SURFACE_LIGHT, SURFACE_DARK = "#F6F7F9", "#161B22"
BORDER_LIGHT, BORDER_DARK = "#E5E7EB", "#30363D"
"""The theme's secondary background and border colours (empty heatmap cells)."""

CATEGORICAL_LIGHT: tuple[str, ...] = (
    "#4F46E5",
    "#E69F00",
    "#009E73",
    "#56B4E9",
    "#D55E00",
    "#CC79A7",
    "#0072B2",
    "#B59B00",
    "#6B7280",
    "#8B5CF6",
)
CATEGORICAL_DARK: tuple[str, ...] = (
    "#818CF8",
    "#F5B342",
    "#34C79A",
    "#7CC6F0",
    "#F07A3A",
    "#E39AC4",
    "#4A9EDB",
    "#E0CB4A",
    "#9CA3AF",
    "#B39DFA",
)
"""The theme's categorical colours (``chartCategoricalColors`` in
``.streamlit/config.toml``), for charts whose colours must match a key drawn
outside the chart (:func:`model_colors`)."""

# Confusion-matrix ramps, light to strong, eight steps each. Correct answers
# (the diagonal) are indigo and errors orange, the app's blue / orange pairing
# for right / wrong. Each step's text colour passes AA (checked in the tests).
CORRECT_RAMP_LIGHT = (
    "#EEF2FF",
    "#E0E7FF",
    "#C7D2FE",
    "#A5B4FC",
    "#818CF8",
    "#4F46E5",
    "#4338CA",
    "#3730A3",
)
ERROR_RAMP_LIGHT = (
    "#FFF7ED",
    "#FFEDD5",
    "#FED7AA",
    "#FDBA74",
    "#FB923C",
    "#C2410C",
    "#9A3412",
    "#7C2D12",
)
CORRECT_RAMP_DARK = (
    "#1E1B4B",
    "#272463",
    "#312E81",
    "#3730A3",
    "#4338CA",
    "#4F46E5",
    "#818CF8",
    "#A5B4FC",
)
ERROR_RAMP_DARK = (
    "#2A1708",
    "#431D07",
    "#5C2A0C",
    "#7C2D12",
    "#9A3412",
    "#EA580C",
    "#FB923C",
    "#FDBA74",
)
_RAMP_STEPS = 8
# First ramp step that takes the "inverse" text colour (white in light mode,
# near-black in dark mode).
_INVERSE_FROM = {
    (False, True): 5,  # light, correct
    (False, False): 5,  # light, errors
    (True, True): 6,  # dark, correct
    (True, False): 5,  # dark, errors
}

CONFUSION_SELECTION = "cell"
"""Name of the point selection in :func:`confusion_heatmap`. With
``st.altair_chart(..., on_select="rerun")`` the clicked cell arrives as
``event.selection[CONFUSION_SELECTION] == [{"gold": ..., "predicted": ...}]``."""

COMPACT_LABELS = 8
"""Matrices with more labels than this get smaller, angled axis labels and
cell values without the % sign, so nothing overlaps or is dropped."""

_BAR_PX = 28
"""Row pitch of horizontal bar charts. Heights are given as ``alt.Step``:
the Streamlit theme autosizes with "fit", so a fixed height would include
the axes and squash the bars."""
_CELL_PX = 40
_CELL_PX_COMPACT = 32
"""Row pitch of the confusion heatmap (normal / many labels)."""
HISTORY_HEIGHT = 300
"""Height of both History charts: neither has a legend inside the chart (the
page draws one key for both), so their plot areas match."""

METRIC_TITLES: dict[str, str] = {
    "accuracy": "Accuracy",
    "macro_f1": "Macro-F1",
    "weighted_f1": "Weighted F1",
    "kappa": "Cohen's κ",
    "rows_per_s": "Rows per second",
    "n_scored": "Rows scored",
    "n_rows": "Rows",
    "created_at": "Date",
}
"""Axis titles for history columns."""

_PERCENT_METRICS = frozenset({"accuracy", "macro_f1", "weighted_f1"})


# --------------------------------------------------------------------------
# Shared styling
# --------------------------------------------------------------------------


def muted(dark: bool) -> str:
    return MUTED_DARK if dark else MUTED_LIGHT


def ink(dark: bool) -> str:
    return TEXT_DARK if dark else TEXT_LIGHT


def finish(chart: Any, dark: bool = False) -> Any:
    """One quiet style for every chart: 12 px axis text in an AA grey, light
    gridlines, no axis lines or ticks, no frame. Chart-level configuration
    overrides the Streamlit theme's."""
    return (
        chart.configure_axis(
            labelFontSize=12,
            titleFontSize=12,
            titleFontWeight=500,
            labelColor=muted(dark),
            titleColor="#D1D5DB" if dark else "#374151",
            gridColor="#21262D" if dark else "#EEF0F3",
            domain=False,
            ticks=False,
        )
        .configure_legend(
            labelFontSize=12,
            titleFontSize=12,
            labelColor=muted(dark),
            titleColor=muted(dark),
        )
        .configure_view(stroke=None)
    )


# --------------------------------------------------------------------------
# Results page
# --------------------------------------------------------------------------


def _ramp_step(share: float) -> int:
    return max(0, min(_RAMP_STEPS - 1, int(share * _RAMP_STEPS)))


def confusion_frame(
    confusion: pd.DataFrame, normalize: bool = False, *, dark: bool = False
) -> pd.DataFrame:
    """Long form of a confusion matrix (gold labels as index, predicted labels
    as columns): one row per cell with ``gold``, ``predicted``, ``count``,
    ``share`` (of the gold row), ``value`` (share when ``normalize`` else
    count), ``text`` (what the cell prints), ``correct``,
    ``predicted_label`` (the predicted label as shown: INVALID reads
    "Invalid"), ``step`` (0-7, the colour step for ``share``), ``fill``
    (None for empty cells), ``ink`` (text colour on that fill), ``strong``
    (filled dark enough to take the inverse text colour) and ``desc`` (what
    a screen reader announces).

    Colour follows the share of the gold row in both modes - so a small
    class that is mostly wrong stands out as much as a large one - and the
    Counts / % choice only changes the printed number."""
    columns = [
        "gold",
        "predicted",
        "count",
        "share",
        "value",
        "text",
        "correct",
        "predicted_label",
        "step",
        "fill",
        "ink",
        "strong",
        "desc",
    ]
    matrix = confusion.copy()
    matrix.index = [str(v) for v in matrix.index]
    matrix.columns = [str(v) for v in matrix.columns]
    matrix = matrix.fillna(0).astype(int)
    long = matrix.rename_axis(index="gold", columns="predicted").stack().rename("count")
    long = long.reset_index()
    if long.empty:
        return pd.DataFrame(columns=columns)
    long["gold"] = long["gold"].astype(str)
    long["predicted"] = long["predicted"].astype(str)
    long["count"] = long["count"].astype(int)
    totals = long.groupby("gold")["count"].transform("sum")
    long["share"] = (long["count"] / totals.where(totals > 0)).fillna(0.0).astype(float)
    long["value"] = long["share"] if normalize else long["count"].astype(float)
    compact = len(matrix.columns) > COMPACT_LABELS
    if normalize:
        # Many columns: the unit goes in the caption, so "47" fits the cell.
        spell = "{:.0f}" if compact else "{:.0f}%"
        long["text"] = [spell.format(s * 100) for s in long["share"]]
    else:
        long["text"] = [f"{c:,}" for c in long["count"]]
    long["correct"] = long["gold"] == long["predicted"]
    names = list(matrix.columns)
    long["predicted_label"] = [display_label(p, names) for p in long["predicted"]]

    steps: list[int] = []
    fills: list[str | None] = []
    inks: list[str] = []
    strong: list[bool] = []
    for count, share, correct in zip(long["count"], long["share"], long["correct"], strict=True):
        step = _ramp_step(float(share))
        inverse = step >= _INVERSE_FROM[(dark, bool(correct))]
        if count:
            ramp = (
                (CORRECT_RAMP_DARK if dark else CORRECT_RAMP_LIGHT)
                if correct
                else (ERROR_RAMP_DARK if dark else ERROR_RAMP_LIGHT)
            )
            fills.append(ramp[step])
            if inverse:
                inks.append(ON_ACCENT_DARK if dark else ON_ACCENT)
            else:
                inks.append(ink(dark))
        else:
            fills.append(None)
            inks.append(muted(dark))
        steps.append(step)
        strong.append(bool(count) and inverse)
    long["step"] = steps
    long["fill"] = pd.Series(fills, index=long.index, dtype=object)
    long["ink"] = inks
    long["strong"] = strong
    long["desc"] = [
        f"Gold {g} predicted {p}: {c:,} {'row' if c == 1 else 'rows'}, {s:.0%} of the gold row"
        for g, p, c, s in zip(
            long["gold"], long["predicted_label"], long["count"], long["share"], strict=True
        )
    ]
    return long[columns]


def confusion_heatmap(
    confusion: pd.DataFrame,
    normalize: bool = False,
    *,
    dark: bool = False,
    selected: tuple[str, str] | None = None,
) -> alt.LayerChart:
    """Heatmap of a confusion matrix (``EvalReport.confusion()``).

    Gold labels down the side, predictions across the top, both in the
    matrix's own order (the task's label set, INVALID last, shown as
    "Invalid"). Every cell sits on a light grid tile; filled cells are indigo
    on the diagonal (correct) and orange off it (errors), stronger for a
    larger share of the gold row. Every cell prints its count (or its share
    when ``normalize``). Clicking a filled cell selects it
    (:data:`CONFUSION_SELECTION`, fields ``gold`` and ``predicted`` - the raw
    labels, INVALID included). ``selected`` outlines a (gold, predicted)
    cell chosen earlier, e.g. one remembered while the chart wasn't on
    screen. ``dark`` picks the dark theme's ramps and text colours.

    More than :data:`COMPACT_LABELS` labels: angled, smaller axis labels
    that are never dropped, and smaller cell text."""
    long = confusion_frame(confusion, normalize, dark=dark)
    if selected is not None:
        long["picked"] = (long["gold"] == str(selected[0])) & (
            long["predicted"] == str(selected[1])
        )
    else:
        long["picked"] = False
    gold_order = [str(v) for v in confusion.index]
    names = [str(v) for v in confusion.columns]
    pred_order = [display_label(v, names) for v in names]
    n = len(pred_order)
    compact = n > COMPACT_LABELS
    longest = max((len(p) for p in pred_order), default=0)
    if longest <= 9 and n <= 7:
        angle = 0
    elif n > 10:
        angle = -90
    else:
        angle = -45
    font = 11 if compact else 12

    selection = alt.selection_point(
        name=CONFUSION_SELECTION, fields=["gold", "predicted"], on="click", clear="dblclick"
    )
    x = alt.X(
        "predicted_label:N",
        sort=pred_order,
        title="Predicted",
        scale=alt.Scale(paddingInner=0.06),
        axis=alt.Axis(
            orient="top",
            labelAngle=angle,
            labelAlign="left" if angle else "center",
            labelOverlap=False,
            labelLimit=110 if angle else 140,
            labelFontSize=font,
            ticks=False,
            domain=False,
        ),
    )
    y = alt.Y(
        "gold:N",
        sort=gold_order,
        title="Gold label",
        scale=alt.Scale(paddingInner=0.06),
        axis=alt.Axis(
            labelLimit=160, labelOverlap=False, labelFontSize=font, ticks=False, domain=False
        ),
    )
    tooltip = [
        alt.Tooltip("gold:N", title="Gold"),
        alt.Tooltip("predicted_label:N", title="Predicted"),
        alt.Tooltip("count:Q", title="Rows", format=","),
        alt.Tooltip("share:Q", title="Share of gold row", format=".0%"),
    ]
    page_ink = ink(dark)
    # The grid: a tile under every cell, so empty cells (and the matrix's
    # shape) stay visible in both themes. The tiles carry each cell's
    # description, so a screen reader announces every cell once.
    tiles = (
        alt.Chart()
        .mark_rect(
            cornerRadius=3,
            fill=SURFACE_DARK if dark else SURFACE_LIGHT,
            stroke=BORDER_DARK if dark else BORDER_LIGHT,
            strokeWidth=1,
        )
        .encode(x=x, y=y, tooltip=tooltip, description=alt.Description("desc:N"))
    )
    cells = (
        alt.Chart()
        .mark_rect(cornerRadius=3)
        .encode(
            x=x,
            y=y,
            color=alt.Color("fill:N", scale=None, legend=None),
            stroke=alt.value(page_ink),
            strokeWidth=alt.condition(selection, alt.value(3), alt.value(0), empty=False),
            tooltip=tooltip,
        )
        .transform_filter(alt.datum.count > 0)
        .add_params(selection)
    )
    size = 11 if compact else 13
    filled_text = (
        alt.Chart()
        .mark_text(fontSize=size, fontWeight=600)
        .encode(
            x=x,
            y=y,
            text="text:N",
            color=alt.Color("ink:N", scale=None, legend=None),
            tooltip=tooltip,
        )
        .transform_filter(alt.datum.count > 0)
    )
    empty_text = (
        alt.Chart()
        .mark_text(fontSize=size, opacity=0.6 if dark else 0.65)
        .encode(x=x, y=y, text="text:N", color=alt.value(muted(dark)), tooltip=tooltip)
        .transform_filter(alt.datum.count == 0)
    )
    # A remembered cell gets the same outline as a clicked one.
    picked = (
        alt.Chart()
        .mark_rect(cornerRadius=3, filled=False, stroke=page_ink, strokeWidth=3)
        .encode(x=x, y=y)
        .transform_filter(alt.datum.picked)
    )
    chart = alt.layer(tiles, cells, filled_text, empty_text, picked, data=long).properties(
        height=alt.Step(_CELL_PX_COMPACT if compact else _CELL_PX)
    )
    return finish(chart, dark)


def label_distribution(
    counts: pd.DataFrame, *, dark: bool = False, gold: dict[str, int] | None = None
) -> alt.LayerChart:
    """Horizontal bars of predicted-label counts (``JobAnalysis.label_counts()``:
    columns ``label`` and ``count``), in the frame's order, each bar annotated
    with its count and share; no value axis (the labels carry the numbers).
    INVALID is drawn faded and named "Invalid" (column ``name`` holds the
    names as shown).

    ``gold`` (label -> rows with that gold label) adds a dashed outline at
    each label's gold count, so over- and under-prediction show at a glance."""
    data = counts[["label", "count"]].copy()
    data["label"] = data["label"].astype(str)
    data["count"] = pd.to_numeric(data["count"], errors="coerce").fillna(0).astype(int)
    total = int(data["count"].sum())
    data["share"] = data["count"] / total if total else 0.0
    data["text"] = [f"{c:,} · {s:.0%}" for c, s in zip(data["count"], data["share"], strict=True)]
    data["invalid"] = data["label"] == INVALID
    names = data["label"].tolist()
    data["name"] = [display_label(v, names) for v in names]
    has_gold = bool(gold)
    data["gold"] = [float(gold.get(v, 0)) if gold else math.nan for v in names]
    data["end"] = data[["count", "gold"]].max(axis=1, skipna=True)
    data["desc"] = [
        f"{n}: {c:,} predicted ({s:.0%})" + (f", {int(g):,} in gold" if has_gold else "")
        for n, c, s, g in zip(data["name"], data["count"], data["share"], data["gold"], strict=True)
    ]
    order = data["name"].tolist()
    # Headroom on the right so the end-of-bar labels never leave the plot.
    top = max(1.0, float(data["end"].max()) if len(data) else 1.0)

    tooltip = [
        alt.Tooltip("name:N", title="Label"),
        alt.Tooltip("count:Q", title="Predicted", format=","),
        alt.Tooltip("share:Q", title="Share", format=".0%"),
    ]
    if has_gold:
        tooltip.append(alt.Tooltip("gold:Q", title="In gold", format=","))
    base = alt.Chart(data).encode(
        y=alt.Y(
            "name:N",
            sort=order,
            title=None,
            axis=alt.Axis(labelLimit=140, ticks=False, domain=False, labelPadding=6),
        ),
        tooltip=tooltip,
    )
    scale = alt.Scale(domain=[0, top * 1.9], nice=False)
    bars = base.mark_bar(cornerRadiusEnd=3, color=ACCENT_DARK if dark else ACCENT).encode(
        x=alt.X("count:Q", axis=None, scale=scale),
        opacity=alt.condition(alt.datum.invalid, alt.value(0.4), alt.value(1.0)),
        description=alt.Description("desc:N"),
    )
    layers: list[Any] = [bars]
    if has_gold:
        layers.append(
            base.mark_bar(
                filled=False,
                stroke=muted(dark),
                strokeDash=[3, 2],
                strokeWidth=1.25,
                cornerRadiusEnd=3,
            )
            .encode(x=alt.X("gold:Q", axis=None, scale=scale))
            .transform_filter(alt.datum.gold > 0)
        )
    layers.append(
        base.mark_text(align="left", dx=5, fontSize=12).encode(
            x=alt.X("end:Q", axis=None, scale=scale), text="text:N", color=alt.value(ink(dark))
        )
    )
    return finish(alt.layer(*layers).properties(height=alt.Step(_BAR_PX)), dark)


def length_histogram(lengths: Sequence[int] | pd.Series, *, dark: bool = False) -> alt.Chart:
    """Histogram of output lengths in characters (at most about 24 bins).

    Binned here rather than by Vega, so the x axis is an ordinary number axis
    with a handful of ticks (a Vega-binned axis labels every bin edge)."""
    values = pd.to_numeric(pd.Series(list(lengths), dtype=object), errors="coerce").dropna()
    values = values.astype(int)
    if values.empty:
        data = pd.DataFrame({"start": [], "end": [], "rows": []})
    else:
        low, high = int(values.min()), int(values.max())
        step = max(1, math.ceil((high - low + 1) / 24))
        starts = (values - low) // step * step + low
        counts = starts.value_counts().sort_index()
        data = pd.DataFrame(
            {
                "start": counts.index.astype(int),
                "end": counts.index.astype(int) + step,
                "rows": counts.to_numpy(dtype=int),
            }
        )
    data["range"] = [
        f"{a:,}" if b - a == 1 else f"{a:,}-{b - 1:,}"
        for a, b in zip(data["start"], data["end"], strict=True)
    ]
    chart = (
        alt.Chart(data)
        .mark_bar(
            cornerRadiusTopLeft=2,
            cornerRadiusTopRight=2,
            color=ACCENT_DARK if dark else ACCENT,
            binSpacing=1,
        )
        .encode(
            x=alt.X(
                "start:Q",
                title="Output length (characters)",
                scale=alt.Scale(zero=False, nice=False),
                axis=alt.Axis(format="d", tickCount=8, labelOverlap=True, tickMinStep=1),
            ),
            x2="end:Q",
            y=alt.Y("rows:Q", title="Rows", axis=alt.Axis(tickMinStep=1, tickCount=3, format="d")),
            tooltip=[
                alt.Tooltip("range:N", title="Characters"),
                alt.Tooltip("rows:Q", title="Rows", format=","),
            ],
        )
        .properties(height=180)
    )
    return finish(chart, dark)


# --------------------------------------------------------------------------
# History page
# --------------------------------------------------------------------------


def model_colors(models: Iterable[str], dark: bool = False) -> dict[str, str]:
    """A fixed colour per model, in the order given (the theme's categorical
    colours, repeating after ten). Pass every model in history, not just the
    ones in view, so a model keeps its colour when the filters change; draw
    the key from the same mapping."""
    palette = CATEGORICAL_DARK if dark else CATEGORICAL_LIGHT
    out: dict[str, str] = {}
    for name in models:
        if name not in out:
            out[name] = palette[len(out) % len(palette)]
    return out


def _metric_axis(column: str) -> alt.Axis:
    title = METRIC_TITLES.get(column, column)
    if column in _PERCENT_METRICS:
        return alt.Axis(title=title, format="%", tickCount=5)
    return alt.Axis(title=title, tickCount=5)


def _history_tooltip(df: pd.DataFrame, extra: Sequence[str]) -> list[alt.Tooltip]:
    tips: list[alt.Tooltip] = []
    for column, title in (
        ("model_label", "Model"),
        ("preset", "Task"),
        ("dataset", "Dataset"),
        ("gold_column", "Gold column"),
    ):
        if column in df.columns:
            tips.append(alt.Tooltip(f"{column}:N", title=title))
    for column in extra:
        if column == "created_at":
            continue
        fmt = ".1%" if column in _PERCENT_METRICS else (",.2f" if column != "n_scored" else ",")
        tips.append(alt.Tooltip(f"{column}:Q", title=METRIC_TITLES.get(column, column), format=fmt))
    if "n_scored" in df.columns and "n_scored" not in extra:
        tips.append(alt.Tooltip("n_scored:Q", title="Rows scored", format=","))
    if "created_at" in df.columns:
        tips.append(alt.Tooltip("created_at:T", title="Date", format="%-d %b %Y, %H:%M"))
    return tips


def _history_data(df: pd.DataFrame, numeric: Sequence[str]) -> pd.DataFrame:
    data = df.copy()
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="coerce").astype(float)
    data = data.dropna(subset=list(numeric))
    if "created_at" in data.columns:
        data["created_at"] = pd.to_datetime(data["created_at"], errors="coerce")
    if "model_label" in data.columns:
        data["model_label"] = data["model_label"].fillna("(unknown)").astype(str)
    # Altair serialises the frame; keep only plain columns (labels may be lists).
    keep = [
        c
        for c in data.columns
        if c in METRIC_TITLES
        or c in {"model_label", "preset", "dataset", "gold_column", "source", *numeric}
    ]
    data = data[keep].reset_index(drop=True)
    data["desc"] = [_point_description(row, numeric) for _, row in data.iterrows()]
    return data


def _point_description(row: pd.Series, numeric: Sequence[str]) -> str:
    parts = [str(row.get("model_label", "Run"))]
    when = row.get("created_at")
    if when is not None and not pd.isna(when):
        # Not "%-d": that strftime flag is glibc-only and raises on Windows.
        stamp = pd.Timestamp(when)
        parts[0] += f", {stamp.day} {stamp:%b}"
    values = []
    for column in numeric:
        value = row.get(column)
        if value is None or pd.isna(value):
            continue
        title = METRIC_TITLES.get(column, column)
        spelled = f"{value:.1%}" if column in _PERCENT_METRICS else f"{value:,.2f}"
        values.append(f"{title.lower() if column != 'kappa' else title} {spelled}")
    return parts[0] + (": " + ", ".join(values) if values else "")


def _percent_domain(values: pd.Series, pad: float = 0.05) -> tuple[float, float]:
    """A 0-1 axis range around ``values``: padded, rounded out to 5-point
    steps and clipped to [0, 1]; the full range when there are no values."""
    numbers = pd.to_numeric(values, errors="coerce").dropna()
    if numbers.empty:
        return 0.0, 1.0
    # round() first: 0.7 - 0.05 is 0.6499999... in floating point.
    low = max(0.0, math.floor(round((float(numbers.min()) - pad) * 20, 6)) / 20)
    high = min(1.0, math.ceil(round((float(numbers.max()) + pad) * 20, 6)) / 20)
    return low, high


def _time_axis(dates: pd.Series) -> alt.Axis:
    """A date axis that never repeats a label: times of day when every run
    is from the same day, else one label per day (or per few days)."""
    when = pd.to_datetime(dates, errors="coerce").dropna()
    if not when.empty and when.dt.normalize().nunique() == 1:
        return alt.Axis(
            title=f"Time on {when.min():%d %b %Y}".replace(" 0", " ", 1),
            format="%H:%M",
            labelAngle=0,
            tickCount=6,
        )
    days = 1 if when.empty else (when.max().normalize() - when.min().normalize()).days + 1
    return alt.Axis(
        title="Date",
        format="%-d %b",
        labelAngle=0,
        labelOverlap=True,
        tickCount=alt.TimeIntervalStep(interval="day", step=max(1, math.ceil(days / 8))),
    )


def _model_color(data: pd.DataFrame, colors: dict[str, str] | None, dark: bool) -> Any:
    if "model_label" not in data.columns:
        return alt.value(ACCENT_DARK if dark else ACCENT)
    mapping = dict(colors or model_colors(sorted(data["model_label"].unique()), dark))
    # Models missing from the mapping still get a colour (appended).
    for name in data["model_label"].unique():
        if name not in mapping:
            mapping = model_colors([*mapping, name], dark) | mapping
    return alt.Color(
        "model_label:N",
        title="Model",
        scale=alt.Scale(domain=list(mapping), range=list(mapping.values())),
        legend=None,
    )


def history_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    dark: bool = False,
    colors: dict[str, str] | None = None,
    point_labels: bool = False,
) -> alt.LayerChart:
    """One point per history row (``Store.list_evaluations()`` frame, any
    filter applied): ``x`` and ``y`` are numeric columns such as
    ``"accuracy"``, ``"macro_f1"``, ``"kappa"`` or ``"rows_per_s"``. Points are
    coloured by ``model_label`` with ``colors`` (:func:`model_colors`; no
    legend in the chart - the page draws the key) and carry model / task /
    dataset / date tooltips. Rows missing ``x`` or ``y`` are dropped.

    Two percentage axes share one range (zoomed to the scores in view, in
    5-point steps within 0-100 %) with a dashed x = y reference line, so the
    distance from it - the gap between the two scores - reads true.
    ``point_labels`` names each point (use it for a handful of points)."""
    data = _history_data(df, [x, y])
    both_percent = x in _PERCENT_METRICS and y in _PERCENT_METRICS
    if both_percent:
        lo, hi = _percent_domain(pd.concat([data[x], data[y]]))
        x_scale = y_scale = alt.Scale(domain=[lo, hi])
    else:

        def _scale(column: str) -> alt.Scale:
            if column in _PERCENT_METRICS:
                return alt.Scale(domain=list(_percent_domain(data[column])))
            return alt.Scale(zero=False, nice=True)

        x_scale, y_scale = _scale(x), _scale(y)

    enc_x = alt.X(f"{x}:Q", axis=_metric_axis(x), scale=x_scale)
    enc_y = alt.Y(f"{y}:Q", axis=_metric_axis(y), scale=y_scale)
    layers: list[Any] = []
    if both_percent:
        line = pd.DataFrame({x: [lo, hi], y: [lo, hi]})
        layers.append(
            alt.Chart(line)
            .mark_line(strokeDash=[4, 4], strokeWidth=1, color=muted(dark))
            .encode(x=enc_x, y=enc_y)
        )
        # In the top-left corner, which stays empty (a run rarely scores
        # higher on macro-F1 than on accuracy), clear of the line itself.
        note = f"Dashed line: {METRIC_TITLES.get(x, x)} = {METRIC_TITLES.get(y, y)}"
        layers.append(
            alt.Chart(pd.DataFrame({x: [lo], y: [hi], "note": [note]}))
            .mark_text(align="left", baseline="top", dx=6, dy=6, fontSize=11, color=muted(dark))
            .encode(x=enc_x, y=enc_y, text="note:N")
        )
    points = alt.Chart(data).encode(
        x=enc_x,
        y=enc_y,
        color=_model_color(data, colors, dark),
        tooltip=_history_tooltip(data, [x, y]),
        description=alt.Description("desc:N"),
    )
    layers.append(points.mark_circle(size=90, opacity=0.9))
    if point_labels and "model_label" in data.columns:
        layers.append(
            points.mark_text(align="left", dx=8, fontSize=11, color=muted(dark)).encode(
                text="model_label:N", color=alt.value(muted(dark))
            )
        )
    return finish(alt.layer(*layers).properties(height=HISTORY_HEIGHT), dark)


def history_timeline(
    df: pd.DataFrame,
    y: str = "accuracy",
    *,
    dark: bool = False,
    colors: dict[str, str] | None = None,
) -> alt.LayerChart:
    """``y`` (default accuracy) over ``created_at``, one line with points per
    ``model_label`` (coloured like :func:`history_scatter`, no legend). Rows
    missing a date or ``y`` are dropped. A percentage axis is zoomed like
    :func:`history_scatter`'s; the time axis is inset so the first and last
    points aren't cut by the plot's edge."""
    data = _history_data(df, [y])
    if "created_at" in data.columns:
        data = data.dropna(subset=["created_at"]).sort_values("created_at")
    else:
        data["created_at"] = pd.NaT
    scale = (
        alt.Scale(domain=list(_percent_domain(data[y])))
        if y in _PERCENT_METRICS
        else alt.Scale(zero=False)
    )
    base = alt.Chart(data).encode(
        x=alt.X("created_at:T", axis=_time_axis(data["created_at"]), scale=alt.Scale(padding=16)),
        y=alt.Y(f"{y}:Q", axis=_metric_axis(y), scale=scale),
        color=_model_color(data, colors, dark),
        tooltip=_history_tooltip(data, [y]),
    )
    chart = base.mark_line(strokeWidth=2, opacity=0.7) + base.mark_point(
        filled=True, size=60, opacity=0.95
    ).encode(description=alt.Description("desc:N"))
    return finish(chart.properties(height=HISTORY_HEIGHT), dark)
