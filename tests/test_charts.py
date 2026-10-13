"""Tests for batchtextreview.ui.charts (pure Altair builders)."""

from __future__ import annotations

import altair as alt
import pandas as pd
import pytest

from batchtextreview.labels import INVALID
from batchtextreview.ui import charts


@pytest.fixture
def confusion() -> pd.DataFrame:
    labels = ["positive", "negative", "mixed", "neutral"]
    frame = pd.DataFrame(
        [[5, 1, 0, 0], [2, 7, 0, 1], [0, 0, 0, 0], [1, 0, 0, 3]],
        index=pd.Index(labels, name="gold"),
        columns=pd.Index(labels, name="predicted"),
    )
    frame[INVALID] = [0, 1, 0, 0]
    return frame


def _contrast(a: str, b: str) -> float:
    def lum(colour: str) -> float:
        rgb = [int(colour.lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4)]
        lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
        return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]

    high, low = sorted([lum(a), lum(b)], reverse=True)
    return (high + 0.05) / (low + 0.05)


def _layer_data(chart: alt.TopLevelMixin) -> pd.DataFrame:
    """The chart's data frame (Altair hoists data shared by every layer to
    the top-level chart)."""
    data = chart.data
    if not isinstance(data, pd.DataFrame) and getattr(chart, "layer", None):
        data = chart.layer[0].data
    assert isinstance(data, pd.DataFrame)
    return data


def _rows(spec: dict, layer: dict) -> list[dict]:
    """A layer's data rows in a to_dict() spec (its own data, else the chart's)."""
    return spec["datasets"][(layer.get("data") or spec["data"])["name"]]


# ---- confusion matrix --------------------------------------------------------


def test_confusion_frame_counts(confusion):
    long = charts.confusion_frame(confusion)
    assert len(long) == 4 * 5
    assert list(long.columns) == [
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
    cell = long[(long.gold == "negative") & (long.predicted == "negative")].iloc[0]
    assert cell["count"] == 7
    assert cell["text"] == "7"
    assert bool(cell["correct"])
    # Correct answers are indigo, errors orange; empty cells have no fill.
    assert cell["fill"] in charts.CORRECT_RAMP_LIGHT
    error = long[(long.gold == "negative") & (long.predicted == "positive")].iloc[0]
    assert error["fill"] in charts.ERROR_RAMP_LIGHT
    empty = long[(long.gold == "positive") & (long.predicted == "mixed")].iloc[0]
    assert empty["fill"] is None and empty["ink"] == charts.MUTED_LIGHT
    # An all-zero gold row has share 0, not NaN.
    assert long.loc[long.gold == "mixed", "share"].eq(0).all()
    # INVALID keeps its value (for the click selection) but reads "Invalid".
    invalid = long[long.predicted == INVALID]
    assert set(invalid["predicted_label"]) == {"Invalid"}
    assert cell["desc"] == "Gold negative predicted negative: 7 rows, 64% of the gold row"


def test_confusion_colour_follows_share_not_count():
    """A small class that is mostly wrong is as visible as a large one."""
    matrix = pd.DataFrame([[90, 10], [1, 3]], index=["big", "small"], columns=["big", "small"])
    long = charts.confusion_frame(matrix).set_index(["gold", "predicted"])
    assert long.loc[("small", "small"), "step"] == long.loc[("big", "big"), "step"] - 1
    assert long.loc[("small", "big"), "step"] > long.loc[("big", "small"), "step"]
    # The Counts / % choice changes the printed number only.
    shares = charts.confusion_frame(matrix, normalize=True)
    assert list(shares["fill"]) == list(charts.confusion_frame(matrix)["fill"])


@pytest.mark.parametrize("dark", [False, True])
def test_every_heatmap_step_has_readable_text(dark):
    matrix = pd.DataFrame(
        [[i, 8 - i] for i in range(9)],
        index=[f"g{i}" for i in range(9)],
        columns=["g0", "other"],
    )
    long = charts.confusion_frame(matrix, dark=dark)
    for fill, ink in zip(long["fill"], long["ink"], strict=True):
        if fill is not None:
            assert _contrast(fill, ink) >= 4.5, (fill, ink)
    page = "#0D1117" if dark else "#FFFFFF"
    assert _contrast(charts.muted(dark), page) >= 4.5


def test_confusion_frame_normalised(confusion):
    long = charts.confusion_frame(confusion, normalize=True)
    pos = long[long.gold == "positive"].set_index("predicted")
    assert pos.loc["positive", "share"] == pytest.approx(5 / 6)
    assert pos.loc["positive", "text"] == "83%"
    assert pos["value"].equals(pos["share"])


def test_confusion_heatmap_structure(confusion):
    chart = charts.confusion_heatmap(confusion)
    assert isinstance(chart, alt.LayerChart)
    data = _layer_data(chart)
    assert len(data) == 20
    spec = chart.to_dict()
    names = [p["name"] for p in spec.get("params", [])]
    assert charts.CONFUSION_SELECTION in names
    select = next(p for p in spec["params"] if p["name"] == charts.CONFUSION_SELECTION)
    assert select["select"]["fields"] == ["gold", "predicted"]
    # A grid tile under every cell, so empty cells stay visible.
    tiles = spec["layer"][0]
    assert tiles["mark"]["fill"] == charts.SURFACE_LIGHT and "transform" not in tiles
    # Axes follow the matrix order, INVALID last and shown as "Invalid".
    x = spec["layer"][0]["encoding"]["x"]
    assert x["field"] == "predicted_label"
    assert x["sort"] == ["positive", "negative", "mixed", "neutral", "Invalid"]
    assert x["axis"]["labelOverlap"] is False  # never drop a label
    assert spec["layer"][0]["encoding"]["y"]["sort"] == list(confusion.index)
    # Nothing remembered: the outline layer has no cell to draw.
    assert not _layer_data(chart)["picked"].any()
    # One chart style everywhere: AA axis text, no frame.
    assert spec["config"]["axis"]["labelColor"] == charts.MUTED_LIGHT
    assert spec["config"]["view"]["stroke"] is None


def test_confusion_heatmap_many_labels_stay_readable():
    labels = [f"emotion {i}" for i in range(12)]
    matrix = pd.DataFrame(1, index=labels, columns=labels)
    spec = charts.confusion_heatmap(matrix, normalize=True).to_dict()
    axis = spec["layer"][0]["encoding"]["x"]["axis"]
    assert axis["labelOverlap"] is False
    assert axis["labelAngle"] == -90
    assert axis["labelFontSize"] == 11
    texts = set(charts.confusion_frame(matrix, normalize=True)["text"])
    assert texts == {"8"}  # the % sign moves to the caption


def test_confusion_heatmap_remembered_cell(confusion):
    chart = charts.confusion_heatmap(confusion, selected=("negative", INVALID))
    data = _layer_data(chart)
    picked = data[data["picked"]]
    assert list(zip(picked["gold"], picked["predicted"], strict=True)) == [("negative", INVALID)]
    outline = chart.to_dict()["layer"][-1]
    assert outline["mark"]["filled"] is False
    assert outline["mark"]["stroke"] == charts.TEXT_LIGHT  # not orange: that means "error"
    assert outline["transform"] == [{"filter": "datum.picked"}]


def test_invalid_name_avoids_a_real_invalid_label():
    matrix = pd.DataFrame(
        [[1, 0, 1], [0, 2, 0]], index=["valid", "invalid"], columns=["valid", "invalid", INVALID]
    )
    long = charts.confusion_frame(matrix)
    assert set(long.loc[long.predicted == INVALID, "predicted_label"]) == {"Not in label set"}
    assert set(long.loc[long.predicted == "invalid", "predicted_label"]) == {"invalid"}


def test_dark_theme_colours(confusion):
    spec = charts.confusion_heatmap(confusion, dark=True).to_dict()
    assert spec["layer"][0]["mark"]["fill"] == charts.SURFACE_DARK
    assert spec["config"]["axis"]["labelColor"] == charts.MUTED_DARK
    long = charts.confusion_frame(confusion, dark=True)
    filled = long.dropna(subset=["fill"])
    assert set(filled.loc[filled["correct"], "fill"]) <= set(charts.CORRECT_RAMP_DARK)
    assert set(filled.loc[~filled["correct"], "fill"]) <= set(charts.ERROR_RAMP_DARK)


def test_confusion_heatmap_empty():
    empty = pd.DataFrame()
    chart = charts.confusion_heatmap(empty)
    chart.to_dict()


# ---- label distribution / histogram ------------------------------------------


def test_label_distribution():
    counts = pd.DataFrame({"label": ["positive", "negative", INVALID], "count": [6, 3, 1]})
    chart = charts.label_distribution(counts)
    data = _layer_data(chart)
    assert list(data["name"]) == ["positive", "negative", "Invalid"]
    assert data["share"].sum() == pytest.approx(1.0)
    assert list(data["text"]) == ["6 · 60%", "3 · 30%", "1 · 10%"]
    assert len(chart.to_dict()["layer"]) == 2  # bars + labels


def test_label_distribution_outlines_gold_counts():
    counts = pd.DataFrame({"label": ["positive", "negative"], "count": [8, 2]})
    chart = charts.label_distribution(counts, gold={"positive": 5, "negative": 5})
    layers = chart.to_dict()["layer"]
    assert len(layers) == 3
    outline = layers[1]
    assert outline["mark"]["filled"] is False and outline["mark"]["strokeDash"] == [3, 2]
    data = _layer_data(chart)
    assert list(data["end"]) == [8, 5]  # labels sit past the longer of bar and outline
    text = layers[2]
    assert text["encoding"]["color"]["value"] == charts.TEXT_LIGHT
    dark = charts.label_distribution(counts, dark=True).to_dict()["layer"]
    assert dark[-1]["encoding"]["color"]["value"] == charts.TEXT_DARK


def test_label_distribution_all_zero():
    chart = charts.label_distribution(pd.DataFrame({"label": ["a", "b"], "count": [0, 0]}))
    chart.to_dict()


def test_length_histogram():
    chart = charts.length_histogram([1, 5, 10, 20, 200, None])
    data = chart.data
    assert int(data["rows"].sum()) == 5
    assert (data["end"] - data["start"]).nunique() == 1  # equal-width bins
    spec = chart.to_dict()
    assert spec["height"] == 180
    # A plain number axis with a few ticks, not one label per bin.
    assert spec["encoding"]["x"]["axis"]["tickCount"] == 8
    assert "bin" not in spec["encoding"]["x"]
    narrow = charts.length_histogram(list(range(65, 85))).data
    assert len(narrow) == 20 and set(narrow["end"] - narrow["start"]) == {1}
    assert charts.length_histogram([]).data.empty


# ---- history -----------------------------------------------------------------


@pytest.fixture
def history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [3, 2, 1],
            "model_label": ["Qwen", "Granite", None],
            "preset": ["Sentiment"] * 3,
            "dataset": ["feedback.csv"] * 3,
            "accuracy": [0.8, 0.7, None],
            "macro_f1": [0.75, None, 0.5],
            "kappa": [0.6, 0.4, 0.2],
            "n_scored": [60, 60, 20],
            "labels": [["positive", "negative"], None, ["a"]],
            "created_at": ["2026-10-03T12:00:00", "2026-10-02T11:00:00", "2026-10-01T10:00:00"],
        }
    )


def _points(spec: dict) -> dict:
    """The point layer of a history chart (the one with a colour field)."""
    return next(layer for layer in spec["layer"] if "field" in layer["encoding"].get("color", {}))


def test_model_colors_are_stable():
    colors = charts.model_colors(["b", "a", "b", "c"])
    assert list(colors) == ["b", "a", "c"]
    assert colors["b"] == charts.CATEGORICAL_LIGHT[0]
    assert charts.model_colors(["b"], dark=True)["b"] == charts.CATEGORICAL_DARK[0]
    many = charts.model_colors([str(i) for i in range(12)])
    assert many["10"] == many["0"]  # the palette repeats after ten


def test_history_scatter_drops_missing(history):
    chart = charts.history_scatter(history, "accuracy", "macro_f1")
    assert isinstance(chart, alt.LayerChart)
    spec = chart.to_dict()
    points = _points(spec)
    data = _rows(spec, points)
    assert len(data) == 1
    assert "labels" not in data[0]  # list values are not serialised
    assert points["encoding"]["x"]["axis"]["format"] == "%"
    assert points["encoding"]["color"]["field"] == "model_label"
    assert points["encoding"]["color"]["legend"] is None  # the page draws one key


def test_history_scatter_shares_percent_range_and_draws_reference(history):
    full = pd.DataFrame({"model_label": ["a", "b"], "accuracy": [0.6, 0.8], "macro_f1": [0.4, 0.7]})
    spec = charts.history_scatter(full, "accuracy", "macro_f1").to_dict()
    points = _points(spec)
    assert points["encoding"]["x"]["scale"] == points["encoding"]["y"]["scale"]
    assert points["encoding"]["x"]["scale"]["domain"] == [0.35, 0.85]
    assert points["encoding"]["x"]["axis"]["tickCount"] == 5
    line = spec["layer"][0]
    assert line["mark"]["type"] == "line" and line["mark"]["strokeDash"] == [4, 4]
    labelled = charts.history_scatter(full, "accuracy", "macro_f1", point_labels=True)
    assert labelled.to_dict()["layer"][-1]["mark"]["type"] == "text"


def test_history_scatter_uses_given_colours(history):
    colors = {"Granite": "#111111", "Qwen": "#222222", "(unknown)": "#333333"}
    spec = charts.history_scatter(history, "kappa", "accuracy", colors=colors).to_dict()
    scale = _points(spec)["encoding"]["color"]["scale"]
    assert scale == {"domain": list(colors), "range": list(colors.values())}


def test_history_scatter_non_percent_axis(history):
    spec = charts.history_scatter(history, "kappa", "accuracy").to_dict()
    points = _points(spec)
    assert len(_rows(spec, points)) == 2
    assert "format" not in points["encoding"]["x"]["axis"]
    assert len(spec["layer"]) == 1  # no x = y line between different measures


def test_history_scatter_zooms_percent_axes(history):
    spec = _points(charts.history_scatter(history, "accuracy", "kappa").to_dict())
    # Accuracy 0.7-0.8, padded by 5 points and rounded out to 5-point steps.
    assert spec["encoding"]["x"]["scale"]["domain"] == [0.65, 0.85]
    full = pd.DataFrame({"model_label": ["a", "b"], "accuracy": [0.02, 0.99], "kappa": [0, 1]})
    spec = _points(charts.history_scatter(full, "accuracy", "kappa").to_dict())
    assert spec["encoding"]["x"]["scale"]["domain"] == [0.0, 1.0]


def test_history_timeline(history):
    chart = charts.history_timeline(history)
    assert isinstance(chart, alt.LayerChart)
    data = _layer_data(chart)
    assert len(data) == 2
    assert data["created_at"].is_monotonic_increasing
    f1 = charts.history_timeline(history, y="macro_f1")
    assert len(_layer_data(f1)) == 2
    assert set(_layer_data(f1)["model_label"]) == {"Qwen", "(unknown)"}
    spec = chart.to_dict()
    x = spec["layer"][0]["encoding"]["x"]
    assert x["scale"] == {"padding": 16}  # first and last points aren't cut
    assert spec["layer"][0]["encoding"]["color"]["legend"] is None
    assert spec["height"] == charts.HISTORY_HEIGHT


def test_history_timeline_axis_never_repeats_a_day(history):
    spec = charts.history_timeline(history).to_dict()
    axis = spec["layer"][0]["encoding"]["x"]["axis"]
    assert axis["format"] == "%-d %b"
    assert axis["tickCount"] == {"interval": "day", "step": 1}  # 3 days: one label each
    same_day = history.assign(
        created_at=["2026-10-03T12:00:00", "2026-10-03T11:00:00", "2026-10-03T09:30:00"]
    )
    axis = charts.history_timeline(same_day).to_dict()["layer"][0]["encoding"]["x"]["axis"]
    assert axis["format"] == "%H:%M"
    assert axis["title"] == "Time on 3 Oct 2026"
    month = history.assign(
        accuracy=[0.8, 0.7, 0.6], created_at=["2026-10-31", "2026-10-15", "2026-10-01"]
    )
    axis = charts.history_timeline(month).to_dict()["layer"][0]["encoding"]["x"]["axis"]
    assert axis["tickCount"] == {"interval": "day", "step": 4}


def test_history_charts_empty():
    empty = pd.DataFrame(columns=["model_label", "accuracy", "macro_f1", "created_at"])
    charts.history_scatter(empty, "accuracy", "macro_f1").to_dict()
    charts.history_timeline(empty).to_dict()
