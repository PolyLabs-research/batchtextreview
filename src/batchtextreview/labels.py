"""Cleaning model outputs and resolving them to a closed label set."""

from __future__ import annotations

import functools
import json
import numbers
import re
import unicodedata
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

INVALID = "[invalid]"
"""Shown in the LABEL column / confusion matrix when an output doesn't
resolve to exactly one allowed label."""

# --------------------------------------------------------------------------
# Patterns
# --------------------------------------------------------------------------

_THINK_BLOCK = re.compile(r"<\s*think\s*>.*?<\s*/\s*think\s*>", re.IGNORECASE | re.DOTALL)
_THINK_CLOSE = re.compile(r"<\s*/\s*think\s*>", re.IGNORECASE)
_THINK_OPEN = re.compile(r"<\s*think\s*>", re.IGNORECASE)

# Invisible characters that NFKC leaves alone but that break matching.
_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff"))

# "_" and "-" (plus the Unicode hyphens) joining two word characters.
_JOINER = re.compile(r"(?<=\w)[_\-\u2010\u2011](?=\w)")
_SPACES = re.compile(r"\s+")

# Stripped from both ends of a key: whitespace, sentence punctuation,
# quotes, brackets and markdown emphasis. Deliberately *not* + - / # % $ & @
# so labels such as "c++", "-1", "n/a" and "c#" survive.
_WRAP = (
    " \t\n\r\f\v"  # whitespace
    + ".,;:!?¡¿…"  # sentence punctuation
    + "\"'`´‘’‚‛“”„‟«»‹›"  # quotes
    + "()[]{}<>"  # brackets
    + "*_~"  # markdown emphasis
    + "。、「」『』【】"  # CJK punctuation (NFKC already maps the fullwidth forms)
)
_BULLET = re.compile(r"^(?:[-•·‣▪]\s+)+")
_INTEGRAL_DECIMAL = re.compile(r"([+-]?\d+)\.0+")

_NUMBER = r"[+-]?\d+(?:\.\d+)?"
# "3/5", "3 / 5", "3 out of 5", "**3**/5", "3 (out of 5)" - the whole
# candidate (label_key has already stripped the outer emphasis / bracket).
_OUT_OF = r"[*_`]*\s*\(?\s*(?:/|out of)\s*[*_`]*\d+(?:\.\d+)?"
_SCORE_FULL = re.compile(rf"({_NUMBER}){_OUT_OF}")
# The same form anywhere in a sentence, collapsed to "<n>" before containment
# so the denominator isn't mistaken for a second label.
_SCORE_INLINE = re.compile(rf"(?<![^\W_])({_NUMBER}){_OUT_OF}(?![^\W_])")

# "Answer:", "Sentiment:", "Score (1-5):" ... must start with a letter so
# "10:30" isn't read as a prefix.
_PREFIX = re.compile(
    r"([^\W\d_][\w '’\-]{0,48}?(?:\s*\([\w '’\-/]{0,24}\))?)\s*:\s*(.+)", re.DOTALL
)

# "non" because label_key turns "non-urgent" into "non urgent"; a label that
# is itself "non-urgent" still wins, its longer hit swallowing "urgent".
_NEGATOR = (
    r"(?:\b(?:not|non|never|no|neither|nor|without|cannot|hardly|barely|scarcely|far from"
    r"|anything but)"
    r"|\b(?:is|are|was|were|do|does|did|ca|wo|could|would|should|has|have|had|ai|must|need)n['’]?t)"
)
# Allows "not a complaint", "isn't very positive", "not at all positive"; the
# label must otherwise follow the negator directly (quotes / emphasis markers
# are fine).
_FILLER = (
    r"(?:a|an|very|really|entirely|completely|particularly|overly|too|that|so|quite|strictly"
    r"|exactly|necessarily|truly|at all|even|remotely)"
)
_NEGATED_TAIL = re.compile(_NEGATOR + rf"(?:\s+{_FILLER}){{0,2}}[\s\"'`*_“”‘’]+$")
# How far back from a hit to look for a negation: the longest negator plus two
# fillers is under 40 characters, which leaves room for quotes / emphasis.
_NEGATION_WINDOW = 64

# What may follow a leading answer ("Yes, ...", "Yes. ...", "Yes - ...",
# "Yes: ...", "Yes\n..."): closing emphasis / quotes, then a clause break.
# Punctuation needs a space after it so "2.5" and "3,5" aren't "2" and "3";
# "?" is left out on purpose ("Positive? Negative? Hard to say.").
_CLAUSE_BREAK = r"""[*_`"'”’»]*(?:[.,;:!]+\s|\s*[–—]+\s*|\s+-+\s)"""
# A line break is a clause break too; label_key would fold it into a space.
_LINE_BREAK = re.compile(r"[ \t]*(?:\r\n?|\n)\s*")
# Words that may join labels in a list of options or a hedge ("positive or
# negative", "3, maybe 4"): a leading label followed only by these and other
# labels isn't a clear answer.
_LIST_GLUE = frozenset(
    {"and", "or", "nor", "both", "either", "neither", "maybe", "perhaps", "possibly", "probably"}
)
_WORD = re.compile(r"[^\W_]+")

_FENCE = re.compile(r"```[ \t]*[\w+-]*[ \t]*\r?\n?(.*?)```", re.DOTALL)
_OPEN_FENCE = re.compile(r"^\s*```[ \t]*[\w+-]*[ \t]*\r?\n?")
_JSON_OPENER = re.compile(r"[{\[]")
_JSON_SCAN_BUDGET = 2_000_000
# What json.loads stops at when a cut-off document ends inside a literal or a
# number ("tr", "nul", "-", "1." leaves ".", "1e+" leaves "e+"), or nothing.
_PARTIAL_TOKEN = re.compile(r"t(?:ru?)?|f(?:a(?:ls?)?)?|n(?:ul?)?|-|\.|[eE][+-]?|")


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def is_missing(value: Any) -> bool:
    """None, NaN, pd.NA / NaT, or a whitespace-only string."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if not pd.api.types.is_scalar(value):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def clean_output(raw: Any) -> str:
    """Tidy a raw generation for storage and display.

    * ``None`` -> "".
    * Remove complete ``<think>...</think>`` blocks (any case, multiline),
      and anything up to and including a dangling ``</think>`` at the start
      (a template may have opened the block). An unterminated ``<think>``
      with nothing after it -> "".
    * Strip surrounding whitespace.
    Nothing else - this is what the user sees in OUTPUT.
    """
    if is_missing(raw):
        return ""
    text = raw if isinstance(raw, str) else str(raw)
    if "think" not in text.lower():
        return text.strip()
    text = _THINK_BLOCK.sub("", text)
    # Any close tag left has no opener: the chat template opened the block,
    # so everything before it is reasoning.
    closes = list(_THINK_CLOSE.finditer(text))
    if closes:
        text = text[closes[-1].end() :]
    # An opener with no close tag: generation stopped mid-thought.
    opened = _THINK_OPEN.search(text)
    if opened:
        text = text[: opened.start()]
    return text.strip()


def label_key(value: Any) -> str | None:
    """Canonical comparison form of a label or a raw value.

    * Missing (see :func:`is_missing`) -> None.
    * Integral floats / numpy numbers become ints first (3.0 -> "3"), so a
      numeric gold column read as float still matches "3".
    * Lowercase, Unicode NFKC, collapse internal whitespace, treat ``_`` and
      ``-`` between word characters as a space ("not_mentioned" ->
      "not mentioned"), and strip wrapping punctuation, quotes, markdown
      emphasis (``*``, ``_``, backticks) and whitespace from both ends.
    """
    if is_missing(value):
        return None
    text = _number_text(value)
    # NFKC on both sides of lower() so the key is a fixed point of label_key.
    text = unicodedata.normalize("NFKC", text).translate(_ZERO_WIDTH).lower()
    text = unicodedata.normalize("NFKC", text)
    text = _JOINER.sub(" ", text)
    text = _SPACES.sub(" ", text).strip()
    stripped, previous = text, None
    while stripped != previous:  # "**- positive**" needs two rounds
        previous = stripped
        stripped = _BULLET.sub("", stripped.strip(_WRAP))
    # A value made only of punctuation ("?", "...") keeps its spelling rather
    # than collapsing to nothing.
    text = stripped or text
    # Same idea as for floats, for numbers that arrived as strings ("3.0").
    integral = _INTEGRAL_DECIMAL.fullmatch(text)
    if integral:
        text = integral.group(1)
    return text or None


def normalise_label(raw: Any, allowed: Sequence[str] | None = None) -> str | None:
    """Resolve a model output (or a gold value) for comparison.

    Without ``allowed``: return :func:`label_key` (None if missing) - used for
    exact-match scoring of open-ended tasks.

    With ``allowed``: return the *canonical* label (``label_key`` of the
    matching allowed entry) or None. Never guesses away ambiguity:

    1. Missing -> None.
    2. Run :func:`clean_output` first, then compare via ``label_key``.
    3. Exact match with an allowed label -> that label.
    4. Leading prefix like "Answer:", "Label:", "Sentiment:" or any
       "<word(s)>:" before the answer is dropped and steps 3 is retried.
    5. Score form "<n>/<m>" or "<n> out of <m>" where n is allowed -> n.
    6. Leading answer: the text (or what follows a step-4 prefix) starts
       with an allowed label (longest wins) and then a clause break - ". ",
       ", ", "; ", ": ", "! ", a dash or a line break - and the rest isn't
       just more labels ("Yes, the customer had no reply." -> yes, but
       "Yes, no" and "Positive. Negative." go on to step 7).
    7. Whole-word containment: find every allowed label occurring in the
       cleaned text bounded by non-word characters (use lookarounds
       ``(?<!\\w)`` / ``(?!\\w)``, not ``\\b``, so labels with punctuation
       like "c++" work). Discard a hit that lies entirely inside a longer
       hit's span ("very positive" beats "positive"), and discard a hit
       directly negated ("not positive", "isn't positive", "never
       positive", "non-urgent"). Exactly one distinct label left -> that
       label; zero or several -> None.
    """
    if is_missing(raw):
        return None
    if allowed is None:
        # Strings are cleaned first; numbers go straight to label_key so 3.0
        # keeps its float -> int treatment.
        return label_key(clean_output(raw) if isinstance(raw, str) else raw)
    return _resolve_value(raw, _label_set(allowed))


def canonical_labels(allowed: Sequence[str] | None) -> list[str]:
    """``label_key`` of each allowed label, blanks dropped, duplicates
    removed (first occurrence wins), order kept."""
    if allowed is None:
        return []
    return list(_label_set(allowed).labels)


def resolve_labels(outputs: Sequence[Any], allowed: Sequence[str]) -> list[str]:
    """``normalise_label`` for each output, with :data:`INVALID` for None.

    Blank outputs (rows that were blank or errored) map to "" rather than
    INVALID, so the LABEL column doesn't blame the model for them.
    """
    labels = _label_set(allowed)
    resolved: list[str] = []
    for output in outputs:
        if is_missing(output):
            resolved.append("")
            continue
        label = _resolve_value(output, labels)
        resolved.append(INVALID if label is None else label)
    return resolved


def parse_json_output(raw: Any) -> dict | list | None:
    """Best-effort JSON from a model output.

    Strips ```json fences, then tries ``json.loads`` on the whole string,
    then on the balanced ``{...}`` / ``[...]`` spans (respecting
    strings/escapes): the first one that is a JSON object, else the first
    JSON array. Only top-level spans count - a malformed or cut-off object
    never yields one of its own values - and an output that ends inside a
    cut-off JSON document gives None. Returns None if nothing parses.
    """
    text = clean_output(raw)
    if not text:
        return None
    candidates: list[str] = []
    fenced = _FENCE.search(text)
    if fenced:
        candidates.append(fenced.group(1))
    elif text.lstrip().startswith("```"):
        # Opening fence only: the output was cut off before the closing one.
        candidates.append(_OPEN_FENCE.sub("", text))
    candidates.append(text)

    for candidate in candidates:
        value = _loads_container(candidate.strip())
        if value is not None:
            return value
    for candidate in candidates:
        value = _first_balanced_json(candidate)
        if value is not None:
            return value
    return None


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _number_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, numbers.Integral):
        return str(int(value))
    if isinstance(value, numbers.Real):
        as_float = float(value)
        if as_float.is_integer():
            return str(int(as_float))
    return str(value)


@dataclass(frozen=True)
class _LabelSet:
    labels: tuple[str, ...]
    members: frozenset[str]
    patterns: tuple[tuple[str, re.Pattern[str]], ...]
    leading: re.Pattern[str] | None
    """A label at the very start, followed by a clause break."""
    any_label: re.Pattern[str] | None
    """Any label as a whole word, longest first."""


def _label_set(allowed: Sequence[Any]) -> _LabelSet:
    entries = tuple(allowed)
    if all(isinstance(entry, str) for entry in entries):
        return _cached_label_set(entries)
    # Off-contract entries skip the cache: equal values of different types
    # (True == 1) would share a cache slot yet have different keys.
    return _build_label_set(entries)


@functools.lru_cache(maxsize=64)
def _cached_label_set(allowed: tuple[Any, ...]) -> _LabelSet:
    return _build_label_set(allowed)


def _build_label_set(allowed: tuple[Any, ...]) -> _LabelSet:
    seen: set[str] = set()
    labels: list[str] = []
    for label in allowed:
        key = label_key(label)
        if key is not None and key not in seen:
            seen.add(key)
            labels.append(key)
    # A word boundary that also treats "_" as a boundary: inside a key "_"
    # only survives as markdown emphasis ("_positive_").
    patterns = tuple(
        (label, re.compile(rf"(?<![^\W_]){re.escape(label)}(?![^\W_])")) for label in labels
    )
    leading = any_label = None
    if labels:
        alternation = "|".join(re.escape(label) for label in sorted(labels, key=len, reverse=True))
        leading = re.compile(rf"(?:{alternation})(?={_CLAUSE_BREAK})")
        any_label = re.compile(rf"(?<![^\W_])(?:{alternation})(?![^\W_])")
    return _LabelSet(tuple(labels), frozenset(labels), patterns, leading, any_label)


def _resolve_value(value: Any, labels: _LabelSet) -> str | None:
    """Steps 2-7 of :func:`normalise_label` for a value that isn't missing."""
    text = clean_output(value) if isinstance(value, str) else value
    key = label_key(text)
    if key is None:
        return None
    lead_key = key
    if isinstance(text, str) and "\n" in text:
        lead_key = label_key(_LINE_BREAK.sub(". ", text)) or key
    return _resolve_key(key, labels, lead_key)


def _resolve_key(key: str, labels: _LabelSet, lead_key: str | None = None) -> str | None:
    """Steps 3-7 of :func:`normalise_label` for an already-keyed output.

    ``lead_key`` is the key with line breaks kept as clause breaks, for
    step 6; it defaults to ``key``.
    """
    if not labels.labels:
        return None
    # Exact / score form on the key, then again with each "<words>:" prefix
    # peeled off.
    for candidate in _peeled(key):
        if candidate in labels.members:
            return candidate
        score = _SCORE_FULL.fullmatch(candidate)
        if score:
            n = label_key(score.group(1))
            if n in labels.members:
                return n
    for candidate in _peeled(key if lead_key is None else lead_key):
        label = _leading_label(candidate, labels)
        if label is not None:
            return label
    return _contained_label(key, labels)


def _peeled(key: str) -> Iterator[str]:
    """``key``, then up to three more with a "<words>:" prefix peeled off."""
    candidate: str | None = key
    for _ in range(4):
        if candidate is None:
            return
        yield candidate
        prefix = _PREFIX.fullmatch(candidate)
        if not prefix:
            return
        candidate = label_key(prefix.group(2))


def _leading_label(text: str, labels: _LabelSet) -> str | None:
    """The label ``text`` opens with, when a clause break follows it and the
    rest isn't merely more labels ("Yes, no", "positive, negative")."""
    if labels.leading is None or labels.any_label is None:
        return None
    lead = labels.leading.match(text)
    if lead is None:
        return None
    rest = labels.any_label.sub(" ", text[lead.end() :])
    if all(word in _LIST_GLUE for word in _WORD.findall(rest)):
        return None
    return lead.group()


def _contained_label(text: str, labels: _LabelSet) -> str | None:
    text = _SCORE_INLINE.sub(r"\1", text)
    # By start, longest first: a hit then lies inside a longer one exactly
    # when an earlier hit reaches at least as far (two distinct labels never
    # share a span). Linear, so a degenerate "4 4 4 ..." output stays cheap.
    hits = sorted(
        (
            (m.start(), m.end(), label)
            for label, pattern in labels.patterns
            for m in pattern.finditer(text)
        ),
        key=lambda hit: (hit[0], -hit[1]),
    )
    found: set[str] = set()
    reach = -1
    for start, end, label in hits:
        swallowed = end <= reach
        reach = max(reach, end)
        if swallowed or _NEGATED_TAIL.search(text, max(0, start - _NEGATION_WINDOW), start):
            continue
        found.add(label)
    return found.pop() if len(found) == 1 else None


def _loads_container(text: str) -> dict | list | None:
    if not text:
        return None
    try:
        value = json.loads(text)
    except (ValueError, RecursionError):
        return None
    return value if isinstance(value, (dict, list)) else None


def _first_balanced_json(text: str) -> dict | list | None:
    """The first top-level span that is a JSON object, else the first one
    that is a JSON array (the JSON presets ask for objects, and a preamble
    may echo a list: 'Aspects: ["food", "service"]').

    A bracket inside a span that closed - or stopped at a mismatched closer -
    belongs to that span and is never tried on its own, so a malformed
    object ("{'a': [1]}", a trailing comma) doesn't yield one of its values.
    A span that runs to the end of the text is a cut-off document when it
    holds valid JSON so far: the answer was cut off, so None. Any other
    unclosed span (a stray "{" in prose) is passed over bracket by bracket.

    Restarting after every opener is quadratic on pathological input
    ("{{{{..."), so the total scanning work is capped.
    """
    budget = _JSON_SCAN_BUDGET
    found_list: list | None = None
    pos = 0
    while budget > 0:
        opener = _JSON_OPENER.search(text, pos)
        if opener is None:
            break
        start = opener.start()
        stop, closed = _balanced_span(text, start, limit=budget)
        # Charge what was actually scanned: a stray "[ ...}" stops early and
        # mustn't use up the budget meant for a real object further on.
        budget -= stop - start
        if stop < len(text) or closed:
            if closed and (found_list is None or text[start] == "{"):
                value = _loads_container(text[start:stop])
                if isinstance(value, dict):
                    return value
                if value is not None:
                    found_list = value
            pos = stop
        elif _cut_off_json(text[start:]):
            return None
        else:
            pos = start + 1
    return found_list


def _cut_off_json(fragment: str) -> bool:
    """Whether ``fragment``, a span that never closed, is JSON cut off at
    the end (rather than prose with a stray bracket)."""
    try:
        # strict=False: a string cut off after a raw newline should read as
        # unterminated, not as an invalid control character.
        json.loads(fragment, strict=False)
    except json.JSONDecodeError as exc:
        if exc.msg.startswith("Unterminated string"):
            return True
        # Cut off in a literal or a number: '{"a": tr', '{"a": 1.'.
        return _PARTIAL_TOKEN.fullmatch(fragment[exc.pos :].rstrip()) is not None
    except RecursionError:
        # Nested deeper than the parser goes, and never closed.
        return True
    except ValueError:
        return False
    return False


def _balanced_span(text: str, start: int, limit: int) -> tuple[int, bool]:
    """Scan at most ``limit`` characters from the bracket at ``text[start]``.

    Returns ``(stop, closed)``: ``stop`` is the index just past the last
    character scanned, ``closed`` whether that character balanced
    ``text[start]`` (False on a mismatched closer or running out).
    """
    closer = {"{": "}", "[": "]"}
    stack: list[str] = []
    in_string = False
    escaped = False
    end = min(len(text), start + limit)
    for i in range(start, end):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in closer:
            stack.append(closer[ch])
        elif ch in "}]":
            if not stack or stack.pop() != ch:
                return i + 1, False
            if not stack:
                return i + 1, True
    return end, False
