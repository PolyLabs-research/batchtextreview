"""Pure formatting helpers for UI copy (no Streamlit import)."""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Sequence
from datetime import datetime
from typing import Any

DASH = "–"
"""Shown for missing / undefined numbers."""

_DATA_EXTENSIONS = (".csv", ".tsv", ".txt", ".xlsx", ".xlsm", ".xls", ".parquet", ".jsonl", ".json")

PRECISION_NAMES = {
    "auto": "Auto",
    "4bit": "4-bit",
    "fp16": "fp16",
    "bf16": "bf16",
    "fp32": "fp32",
    "none": "n/a",
}
"""Short precision names for chips, tables and controls ("none" is the demo
model, which has no precision)."""

SOURCE_WORDS = {"app": "App", "cli": "CLI", "legacy": "v1"}
"""Where a run or evaluation came from (``source`` in the results database)."""

_MARKDOWN_SPECIAL = "\\`*_{}[]()#+-.!|<>~:$"


def _finite(value: Any) -> float | None:
    """``value`` as a finite float, or None for None / NaN / NA / inf / junk."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def plural(n: int, noun: str, plural_form: str | None = None) -> str:
    """``plural(1, "row")`` -> "1 row"; ``plural(2000, "row")`` -> "2,000 rows"."""
    word = noun if n == 1 else (plural_form or f"{noun}s")
    return f"{n:,} {word}"


def duration(seconds: float | None) -> str:
    """Compact human duration: "under 1 s", "42 s", "4 min 12 s", "1 h 05 min".

    Whole seconds below a minute: a progress estimate or a load time reads
    better as "2 s" than "2.1 s", and tenths would change on every refresh."""
    value = _finite(seconds)
    if value is None:
        return DASH
    value = max(0.0, value)
    if value == 0:
        return "0 s"
    if value < 1:
        return "under 1 s"
    # Decide the unit on the rounded value, so 59.6 s is "1 min 00 s" (not "60 s").
    whole = round(value)
    if whole < 60:
        return f"{whole} s"
    minutes, secs = divmod(whole, 60)
    if minutes < 60:
        return f"{minutes} min {secs:02d} s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes:02d} min"


def estimate(seconds: float | None) -> str:
    """A time estimate: "about 2 min 50 s", or "under 1 s" (never "about
    under 1 s")."""
    value = _finite(seconds)
    if value is None:
        return DASH
    if value < 1:
        return "under 1 s"
    return f"about {duration(value)}"


def percent(value: float | None, digits: int = 1) -> str:
    number = _finite(value)
    if number is None:
        return DASH
    return f"{number * 100:.{digits}f}%"


def number(value: float | None, digits: int = 2) -> str:
    num = _finite(value)
    if num is None:
        return DASH
    return f"{num:.{digits}f}"


def rate(rows_per_s: float | None) -> str:
    value = _finite(rows_per_s)
    if not value or value < 0:
        return DASH
    if value >= 1:
        return f"{value:.1f} rows/s"
    return f"{60 * value:.1f} rows/min"


def timestamp(value: datetime | str | None) -> str:
    """ "12 Oct 2026, 14:03" for datetimes / ISO strings; "" for None / NaT.

    The same spelling as the tables' date columns ("D MMM YYYY, HH:mm"), so a
    run on the Results page can be matched to its History line at a glance."""
    if value is None:
        return ""
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return value
        value = parsed
    try:
        return f"{value.day} {value:%b %Y, %H:%M}"
    except (AttributeError, TypeError, ValueError):  # pd.NaT and other non-dates
        return ""


def slug(value: str, max_len: int = 40) -> str:
    """Filesystem-safe slug: "Qwen2.5-3B · Sentiment" -> "qwen2.5-3b-sentiment".

    Accented letters are transliterated ("Über Café" -> "uber-cafe")."""
    ascii_text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^A-Za-z0-9.]+", "-", ascii_text).strip("-.").lower()
    return s[:max_len].strip("-.") or "results"


def export_filename(
    dataset: str, task: str, model: str, ext: str, when: datetime | None = None
) -> str:
    """ "{dataset}_{task}_{model}_{YYYYMMDD-HHMM}.{ext}" with each part slugged.

    Only a known data-file extension is dropped from ``dataset``, so a name
    like "Qwen2.5 run" keeps its dot."""
    when = when or datetime.now()
    stem = dataset
    for known in _DATA_EXTENSIONS:
        if stem.lower().endswith(known):
            stem = stem[: -len(known)]
            break
    parts = [slug(stem, 30), slug(task, 24), slug(model, 30), f"{when:%Y%m%d-%H%M}"]
    return "_".join(parts) + f".{ext.lstrip('.')}"


def truncate(text: str, limit: int = 80) -> str:
    """Collapse whitespace and cut to at most ``limit`` characters (with "…")."""
    text = " ".join(str(text).split())
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def file_size(n_bytes: float | None) -> str:
    """ "0 B", "900 B", "12.3 KB", "4.5 MB", "1.2 GB"."""
    size = _finite(n_bytes)
    if size is None:
        return DASH
    size = max(0.0, size)
    for unit in ("B", "KB", "MB"):
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def precision(value: Any) -> str:
    """ "4bit" -> "4-bit", "none" (demo model) -> "n/a"; missing or unknown
    values -> "–"."""
    if value is None:
        return DASH
    text = str(value).strip()
    return PRECISION_NAMES.get(text, DASH)


def source_word(value: Any) -> str:
    """ "app" -> "App", "cli" -> "CLI", "legacy" -> "v1"; missing -> "App"."""
    text = "" if value is None or _is_na(value) else str(value).strip()
    if not text:
        return SOURCE_WORDS["app"]
    return SOURCE_WORDS.get(text, text.capitalize())


def model_name(model_id: str) -> str:
    """A model id as a short name: the last part of a repo id or folder path
    ("Qwen/Qwen2.5-3B-Instruct" -> "Qwen2.5-3B-Instruct"), and a cloud model
    without its provider ("anthropic:claude-opus-4-8" -> "claude-opus-4-8")."""
    text = str(model_id)
    prefix, sep, rest = text.partition(":")
    if sep and prefix in ("anthropic", "openai") and rest:
        text = rest
    text = text.replace("\\", "/").rstrip("/")
    return text.rsplit("/", 1)[-1] or str(model_id)


def price(per_million: tuple[float, float] | None) -> str:
    """List price per million input / output tokens:
    "$0.10 / $0.50 per M tokens"; "–" when unknown."""
    if not per_million:
        return DASH
    sent, generated = per_million
    return f"{_dollars(sent)} / {_dollars(generated)} per M tokens"


def money(value: float | None) -> str:
    """US dollars: "$0.04", "$12.30", "$1,204"; "< $0.01" for a positive
    amount below a cent; "–" when unknown."""
    amount = _finite(value)
    if amount is None:
        return DASH
    if 0 < amount < 0.01:
        return "< $0.01"
    return _dollars(amount)


def _dollars(amount: float) -> str:
    if amount >= 1000:
        return f"${amount:,.0f}"
    return f"${amount:,.2f}"


def no_math(text: str) -> str:
    """``text`` with its dollar signs escaped, for Markdown: Streamlit reads
    "$0.10 / $0.50" as a LaTeX formula."""
    return str(text).replace("$", "\\$")


def token_count(n: float | None) -> str:
    """Tokens, rounded the way people say them: "940", "4.2k", "310k", "1.3M"."""
    value = _finite(n)
    if value is None:
        return DASH
    value = max(0.0, value)
    if value < 1000:
        return f"{value:,.0f}"
    if value < 10_000:
        return f"{value / 1000:.1f}k"
    if value < 1_000_000:
        return f"{value / 1000:,.0f}k"
    return f"{value / 1_000_000:,.1f}M"


def distinct_model_names(names: Sequence[str], keys: Sequence[str]) -> list[str]:
    """``names`` with every name that stands for more than one model (key =
    model id or folder path) told apart by the end of its path: three local
    folders all called "tinymodel" read "a/tinymodel", "other/tinymodel" and
    "third/tinymodel". Names used by one model only are left alone."""
    groups: dict[str, set[str]] = {}
    for name, key in zip(names, keys, strict=True):
        groups.setdefault(name, set()).add(str(key))
    renamed: dict[tuple[str, str], str] = {}
    for name, group in groups.items():
        if len(group) < 2:
            continue
        parts = {k: [p for p in k.replace("\\", "/").rstrip("/").split("/") if p] for k in group}
        longest = max((len(p) for p in parts.values()), default=1)
        labels = {k: k for k in group}
        for depth in range(2, longest + 1):
            tried = {k: "/".join(p[-depth:]) or k for k, p in parts.items()}
            if len(set(tried.values())) == len(tried):
                labels = tried
                break
        for key in group:
            label = labels[key]
            same_end = label.rsplit("/", 1)[-1].casefold() == name.casefold()
            renamed[(name, key)] = label if same_end or label == name else f"{name} ({label})"
    return [renamed.get((n, str(k)), n) for n, k in zip(names, keys, strict=True)]


def md(text: object) -> str:
    """User data made safe for inline Markdown: special characters escaped
    and whitespace (newlines included) collapsed to single spaces."""
    out = str(text)
    for ch in _MARKDOWN_SPECIAL:
        out = out.replace(ch, "\\" + ch)
    return " ".join(out.split())


def _is_na(value: Any) -> bool:
    """True for NaN / NaT (not equal to themselves) and pd.NA (whose
    comparisons have no truth value)."""
    try:
        return bool(value != value)
    except (TypeError, ValueError):
        return True
