"""Task presets: prompt templates plus the metadata the app needs around them.

Placeholders in a template:

``{text}``  replaced with each row's text (required).
``{var}``   replaced once with the value typed into the task's variable box
            (only parameterised presets use it).

``Preset.key`` is the stable identifier stored in evaluation history. Built-in
keys are the version-1 preset names, so old history lines up; ``title`` is
what the UI shows.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

OutputKind = Literal["label", "json", "text"]

CATEGORIES: tuple[str, ...] = ("Classify", "Extract", "Generate", "Custom")


def parse_label_list(value: str | None) -> list[str]:
    """Split a comma / semicolon / newline separated list of labels.

    Whitespace is trimmed, blanks dropped, and duplicates removed
    case-insensitively (first spelling wins); order is preserved.
    """
    if not value:
        return []
    return _clean_labels(re.split(r"[,;\n]", value))


def _clean_labels(labels: list[str]) -> list[str]:
    """Trim, drop blanks, de-duplicate case-insensitively (first spelling wins)."""
    seen: set[str] = set()
    out: list[str] = []
    for part in labels:
        label = part.strip()
        if label and label.lower() not in seen:
            seen.add(label.lower())
            out.append(label)
    return out


@dataclass(frozen=True)
class Preset:
    key: str
    title: str
    category: str
    template: str
    max_new_tokens: int
    description: str = ""
    labels: tuple[str, ...] | None = None
    """Closed label set - enables the LABEL column and evaluation."""
    labels_from_var: bool = False
    """Derive the label set from the comma-separated variable value."""
    var_label: str | None = None
    """Caption for the variable input, e.g. "Categories"."""
    var_hint: str | None = None
    """Placeholder example for the variable input."""
    output_kind: OutputKind = "text"
    builtin: bool = True
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def needs_var(self) -> bool:
        return "{var}" in self.template

    def resolve_labels(self, var: str | None = None) -> list[str] | None:
        """The closed label set for this preset, or ``None`` for open output."""
        if self.labels_from_var:
            return parse_label_list(var) or None
        return list(self.labels) if self.labels else None


BUILTIN_PRESETS: tuple[Preset, ...] = (
    Preset(
        key="Sentiment",
        title="Sentiment",
        category="Classify",
        description="Positive, negative, mixed or neutral.",
        template=(
            "Classify the sentiment of the text below as positive, negative, mixed or "
            "neutral. Reply with exactly one word.\n\nTEXT:\n{text}"
        ),
        max_new_tokens=8,
        labels=("positive", "negative", "mixed", "neutral"),
        output_kind="label",
    ),
    Preset(
        key="Emotion",
        title="Emotion",
        category="Classify",
        description="Dominant emotion from a fixed set of seven.",
        template=(
            "Identify the dominant emotion in the text below from: joy, anger, "
            "sadness, fear, surprise, disgust, neutral. Reply with one word only."
            "\n\nTEXT:\n{text}"
        ),
        max_new_tokens=8,
        labels=("joy", "anger", "sadness", "fear", "surprise", "disgust", "neutral"),
        output_kind="label",
    ),
    Preset(
        key="Intent",
        title="Intent",
        category="Classify",
        description="Complaint, query, request, praise or other.",
        template=(
            "Classify the primary intent of the text below as one of: complaint, "
            "query, request, praise, other. Reply with one word only."
            "\n\nTEXT:\n{text}"
        ),
        max_new_tokens=8,
        labels=("complaint", "query", "request", "praise", "other"),
        output_kind="label",
    ),
    Preset(
        key="Urgency (1-5)",
        title="Urgency (1-5)",
        category="Classify",
        description="Urgency score from 1 (not urgent) to 5 (extremely urgent).",
        template=(
            "Rate how urgent the text below is from 1 (not urgent) to 5 "
            "(extremely urgent). Reply with only the number.\n\nTEXT:\n{text}"
        ),
        max_new_tokens=8,
        labels=("1", "2", "3", "4", "5"),
        output_kind="label",
    ),
    Preset(
        key="Yes/No flag (set {var})",
        title="Yes/No question",
        category="Classify",
        description="Answer your own yes/no question about each text.",
        template=(
            "Answer the question about the text below with only Yes or No.\n\n"
            "QUESTION: {var}\n\nTEXT:\n{text}"
        ),
        max_new_tokens=8,
        labels=("yes", "no"),
        var_label="Question",
        var_hint="e.g. Does the text describe a service failure?",
        output_kind="label",
    ),
    Preset(
        key="Categorise (set {var})",
        title="Categorise",
        category="Classify",
        description="Assign each text to one of your own categories.",
        template=(
            "Classify the text below into exactly one of these categories: {var}. "
            "Reply with only the category name.\n\nTEXT:\n{text}"
        ),
        max_new_tokens=16,
        labels_from_var=True,
        var_label="Categories (comma-separated)",
        var_hint="e.g. invoice, contract, email, report, other",
        output_kind="label",
    ),
    Preset(
        key="Language",
        title="Language",
        category="Classify",
        description="The language the text is written in.",
        template=(
            "Identify the language of the text below. Reply with only the English "
            "name of the language.\n\nTEXT:\n{text}"
        ),
        max_new_tokens=8,
        output_kind="text",
    ),
    Preset(
        key="Entity extraction",
        title="Entity extraction",
        category="Extract",
        description="People, organisations, locations and dates as JSON.",
        template=(
            "Extract the named entities from the text below. Return ONLY a JSON "
            'object with keys "people", "organisations", "locations", '
            '"dates", each a list of strings (empty list if none).\n\n'
            "TEXT:\n{text}"
        ),
        max_new_tokens=256,
        output_kind="json",
    ),
    Preset(
        key="Keyword extraction",
        title="Keywords",
        category="Extract",
        description="The 3-5 most important keywords or phrases.",
        template=(
            "Extract the 3-5 most important keywords or key phrases from the text "
            "below. Return a comma-separated list and nothing else.\n\nTEXT:\n{text}"
        ),
        max_new_tokens=32,
        output_kind="text",
    ),
    Preset(
        key="Extract field (set {var})",
        title="Extract a field",
        category="Extract",
        description="Pull out one value you describe, or NONE.",
        template=(
            "Extract the following from the text below: {var}. If it is not "
            "present, reply NONE. Return only the extracted value.\n\nTEXT:\n{text}"
        ),
        max_new_tokens=32,
        var_label="Field to extract",
        var_hint="e.g. the monetary amount",
        output_kind="text",
    ),
    Preset(
        key="Aspect sentiment (set {var})",
        title="Aspect sentiment",
        category="Extract",
        description="Sentiment towards each aspect you list, as JSON.",
        template=(
            "For each aspect in this list - {var} - give the sentiment toward it "
            "in the text below as positive, negative, neutral, or not_mentioned. "
            "Return ONLY a JSON object mapping each aspect to its sentiment."
            "\n\nTEXT:\n{text}"
        ),
        max_new_tokens=96,
        var_label="Aspects (comma-separated)",
        var_hint="e.g. food, service, price",
        output_kind="json",
    ),
    Preset(
        key="PII detection",
        title="PII detection",
        category="Extract",
        description="Personal or sensitive details found, as JSON.",
        template=(
            "List any personal or sensitive details in the text below (names, "
            "addresses, phone numbers, emails, account numbers, dates of birth). "
            'Return ONLY a JSON object with key "pii" as a list of strings '
            "(empty list if none).\n\nTEXT:\n{text}"
        ),
        max_new_tokens=256,
        output_kind="json",
    ),
    Preset(
        key="Topic analysis",
        title="Topics",
        category="Generate",
        description="1-3 short topic labels.",
        template=(
            "Give 1-3 short topic labels that best describe the text below, "
            "comma-separated, with no other words.\n\nTEXT:\n{text}"
        ),
        max_new_tokens=24,
        output_kind="text",
    ),
    Preset(
        key="One-line summary",
        title="One-line summary",
        category="Generate",
        description="A single sentence of 20 words or fewer.",
        template=(
            "Summarise the text below in a single sentence of 20 words or fewer. "
            "Return only the summary.\n\nTEXT:\n{text}"
        ),
        max_new_tokens=48,
        output_kind="text",
    ),
    Preset(
        key="PII redaction",
        title="PII redaction",
        category="Generate",
        description="The text rewritten with personal data replaced by [REDACTED].",
        template=(
            "Rewrite the text below with every piece of personal data (names, "
            "addresses, phone numbers, emails, account numbers) replaced by "
            "[REDACTED]. Return only the rewritten text.\n\nTEXT:\n{text}"
        ),
        max_new_tokens=1024,
        output_kind="text",
    ),
    Preset(
        key="Custom",
        title="Custom prompt",
        category="Custom",
        description="Write your own prompt. Use {text} where each row goes.",
        template="{text}",
        max_new_tokens=256,
        output_kind="text",
    ),
)

_BY_KEY = {p.key: p for p in BUILTIN_PRESETS}


def builtin_presets() -> list[Preset]:
    return list(BUILTIN_PRESETS)


def get_preset(key: str, extra: list[Preset] | None = None) -> Preset | None:
    """Find a preset by key among the built-ins and any ``extra`` (custom) ones."""
    if key in _BY_KEY:
        return _BY_KEY[key]
    for p in extra or ():
        if p.key == key:
            return p
    return None


def make_custom_preset(
    title: str,
    template: str,
    max_new_tokens: int,
    labels: list[str] | None = None,
    var_hint: str | None = None,
) -> Preset:
    """Build a user-saved preset. Its key is the title (unique among customs)."""
    title = title.strip()
    if not title:
        raise ValueError("Give the preset a name.")
    if title in _BY_KEY:
        raise ValueError(f'"{title}" is a built-in preset name - choose another.')
    if "{text}" not in template:
        raise ValueError("The template must contain the {text} placeholder.")
    tokens = int(max_new_tokens)
    if tokens < 1:
        raise ValueError("Max output tokens must be at least 1.")
    cleaned = _clean_labels(labels or [])
    has_labels = bool(cleaned)
    return Preset(
        key=title,
        title=title,
        category="Custom",
        template=template,
        max_new_tokens=tokens,
        description="Saved preset",
        labels=tuple(cleaned) if has_labels else None,
        var_label="Variable" if "{var}" in template else None,
        var_hint=var_hint,
        output_kind="label" if has_labels else "text",
        builtin=False,
    )
