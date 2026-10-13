"""Tests for batchtextreview.presets."""

from __future__ import annotations

import dataclasses

import pytest

from batchtextreview.presets import (
    BUILTIN_PRESETS,
    CATEGORIES,
    Preset,
    builtin_presets,
    get_preset,
    make_custom_preset,
    parse_label_list,
)
from batchtextreview.prompting import apply_variable, template_problems

# --------------------------------------------------------------------------
# Version 1's app.py tables, verbatim. Built-in keys are stored in evaluation
# history, and templates / token budgets feed run ids, so the built-ins must
# reproduce these exactly.
# --------------------------------------------------------------------------

# fmt: off
V1_PRESETS = {
    "Sentiment": (
        "Classify the sentiment of the text below as positive, negative, mixed or "
        "neutral. Reply with exactly one word.\n\nTEXT:\n{text}"
    ),
    "Entity extraction": (
        "Extract the named entities from the text below. Return ONLY a JSON "
        "object with keys \"people\", \"organisations\", \"locations\", "
        "\"dates\", each a list of strings (empty list if none).\n\n"
        "TEXT:\n{text}"
    ),
    "Topic analysis": (
        "Give 1-3 short topic labels that best describe the text below, "
        "comma-separated, with no other words.\n\nTEXT:\n{text}"
    ),
    "One-line summary": (
        "Summarise the text below in a single sentence of 20 words or fewer. "
        "Return only the summary.\n\nTEXT:\n{text}"
    ),
    "Keyword extraction": (
        "Extract the 3-5 most important keywords or key phrases from the text "
        "below. Return a comma-separated list and nothing else.\n\nTEXT:\n{text}"
    ),
    "Emotion": (
        "Identify the dominant emotion in the text below from: joy, anger, "
        "sadness, fear, surprise, disgust, neutral. Reply with one word only."
        "\n\nTEXT:\n{text}"
    ),
    "Intent": (
        "Classify the primary intent of the text below as one of: complaint, "
        "query, request, praise, other. Reply with one word only."
        "\n\nTEXT:\n{text}"
    ),
    "Urgency (1-5)": (
        "Rate how urgent the text below is from 1 (not urgent) to 5 "
        "(extremely urgent). Reply with only the number.\n\nTEXT:\n{text}"
    ),
    "Language": (
        "Identify the language of the text below. Reply with only the English "
        "name of the language.\n\nTEXT:\n{text}"
    ),
    "PII detection": (
        "List any personal or sensitive details in the text below (names, "
        "addresses, phone numbers, emails, account numbers, dates of birth). "
        "Return ONLY a JSON object with key \"pii\" as a list of strings "
        "(empty list if none).\n\nTEXT:\n{text}"
    ),
    "PII redaction": (
        "Rewrite the text below with every piece of personal data (names, "
        "addresses, phone numbers, emails, account numbers) replaced by "
        "[REDACTED]. Return only the rewritten text.\n\nTEXT:\n{text}"
    ),
    # --- parameterised: fill the Variable box with what goes in {var} ---
    "Yes/No flag (set {var})": (
        "Answer the question about the text below with only Yes or No.\n\n"
        "QUESTION: {var}\n\nTEXT:\n{text}"
    ),
    "Categorise (set {var})": (
        "Classify the text below into exactly one of these categories: {var}. "
        "Reply with only the category name.\n\nTEXT:\n{text}"
    ),
    "Extract field (set {var})": (
        "Extract the following from the text below: {var}. If it is not "
        "present, reply NONE. Return only the extracted value.\n\nTEXT:\n{text}"
    ),
    "Aspect sentiment (set {var})": (
        "For each aspect in this list - {var} - give the sentiment toward it "
        "in the text below as positive, negative, neutral, or not_mentioned. "
        "Return ONLY a JSON object mapping each aspect to its sentiment."
        "\n\nTEXT:\n{text}"
    ),
    "Custom": "{text}",
}

# Sensible default output length per task - short tasks generate far faster.
V1_SUGGESTED_TOKENS = {
    "Sentiment": 8, "Emotion": 8, "Intent": 8, "Urgency (1-5)": 8,
    "Language": 8, "Yes/No flag (set {var})": 8,
    "Categorise (set {var})": 16, "Topic analysis": 24,
    "Keyword extraction": 32, "Extract field (set {var})": 32,
    "One-line summary": 48, "Aspect sentiment (set {var})": 96,
    "Entity extraction": 256, "PII detection": 256, "Custom": 256,
    "PII redaction": 1024,
}

# Closed label sets per task: enables the LABEL column and evaluation.
# "FROM_VAR" derives the set from the comma-separated Variable box.
V1_LABEL_SETS = {
    "Sentiment": ["positive", "negative", "mixed", "neutral"],
    "Emotion": ["joy", "anger", "sadness", "fear", "surprise", "disgust",
                "neutral"],
    "Intent": ["complaint", "query", "request", "praise", "other"],
    "Urgency (1-5)": ["1", "2", "3", "4", "5"],
    "Yes/No flag (set {var})": ["yes", "no"],
    "Categorise (set {var})": "FROM_VAR",
}

# Example hint shown in the Variable box for parameterised tasks.
V1_VAR_HINT = {
    "Yes/No flag (set {var})": "e.g. Does the text describe a service failure?",
    "Categorise (set {var})": "e.g. invoice, contract, email, report, other",
    "Extract field (set {var})": "e.g. the monetary amount",
    "Aspect sentiment (set {var})": "e.g. food, service, price",
}
# fmt: on

PRESETS = {p.key: p for p in BUILTIN_PRESETS}
VAR_KEYS = [k for k in V1_PRESETS if "{var}" in V1_PRESETS[k]]


# --------------------------------------------------------------------------
# Built-ins vs version 1
# --------------------------------------------------------------------------


def test_sixteen_builtins_with_unique_keys_and_titles() -> None:
    assert len(BUILTIN_PRESETS) == 16
    assert len(PRESETS) == 16
    assert len({p.title for p in BUILTIN_PRESETS}) == 16


def test_builtin_keys_are_version_1_names() -> None:
    assert set(PRESETS) == set(V1_PRESETS)
    assert set(V1_SUGGESTED_TOKENS) == set(V1_PRESETS)  # sanity check on the copy


@pytest.mark.parametrize("key", list(V1_PRESETS))
def test_template_matches_version_1(key: str) -> None:
    assert PRESETS[key].template == V1_PRESETS[key]


@pytest.mark.parametrize("key", list(V1_PRESETS))
def test_max_new_tokens_matches_version_1(key: str) -> None:
    assert PRESETS[key].max_new_tokens == V1_SUGGESTED_TOKENS[key]


@pytest.mark.parametrize("key", list(V1_PRESETS))
def test_label_set_matches_version_1(key: str) -> None:
    preset = PRESETS[key]
    v1 = V1_LABEL_SETS.get(key)
    if v1 == "FROM_VAR":
        assert preset.labels_from_var
        assert preset.labels is None
    else:
        assert not preset.labels_from_var
        assert preset.resolve_labels() == v1
        assert preset.labels == (tuple(v1) if v1 else None)


@pytest.mark.parametrize("key", list(V1_PRESETS))
def test_var_hint_matches_version_1(key: str) -> None:
    assert PRESETS[key].var_hint == V1_VAR_HINT.get(key)


@pytest.mark.parametrize("preset", BUILTIN_PRESETS, ids=lambda p: p.key)
def test_builtin_is_well_formed(preset: Preset) -> None:
    assert "{text}" in preset.template
    assert preset.template.count("{text}") == 1
    assert preset.builtin is True
    assert preset.category in CATEGORIES
    assert preset.title and preset.description
    assert preset.max_new_tokens >= 1
    assert preset.tags == ()
    # A variable caption exactly when the template takes a variable.
    assert (preset.var_label is not None) == preset.needs_var
    assert (preset.var_hint is not None) == preset.needs_var
    # Closed label sets are what make a preset a "label" task.
    has_label_set = bool(preset.labels) or preset.labels_from_var
    assert (preset.output_kind == "label") == has_label_set
    if preset.labels:
        assert all(label == label.strip().lower() for label in preset.labels)
        assert len(set(preset.labels)) == len(preset.labels)


def test_needs_var_matches_version_1_parameterised_names() -> None:
    for key, preset in PRESETS.items():
        assert preset.needs_var == ("(set {var})" in key), key
    assert sorted(VAR_KEYS) == sorted(k for k, p in PRESETS.items() if p.needs_var)


def test_json_presets() -> None:
    json_keys = {k for k, p in PRESETS.items() if p.output_kind == "json"}
    assert json_keys == {"Entity extraction", "Aspect sentiment (set {var})", "PII detection"}
    for key in json_keys:
        assert "JSON" in PRESETS[key].template


def test_only_categorise_derives_labels_from_var() -> None:
    assert [k for k, p in PRESETS.items() if p.labels_from_var] == ["Categorise (set {var})"]


def test_every_category_has_builtins() -> None:
    used = {p.category for p in BUILTIN_PRESETS}
    assert used == set(CATEGORIES)
    assert [p.key for p in BUILTIN_PRESETS if p.category == "Custom"] == ["Custom"]
    assert PRESETS["Custom"].template == "{text}"


def test_builtin_presets_returns_a_fresh_list() -> None:
    first = builtin_presets()
    assert first == list(BUILTIN_PRESETS)
    first.clear()
    assert len(builtin_presets()) == 16


def test_preset_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        PRESETS["Sentiment"].template = "{text}"  # type: ignore[misc]


# --------------------------------------------------------------------------
# resolve_labels
# --------------------------------------------------------------------------


def test_resolve_labels_returns_a_copy() -> None:
    preset = PRESETS["Sentiment"]
    labels = preset.resolve_labels()
    assert labels == ["positive", "negative", "mixed", "neutral"]
    assert labels is not None
    labels.append("sarcastic")
    assert preset.resolve_labels() == ["positive", "negative", "mixed", "neutral"]


def test_resolve_labels_ignores_var_for_fixed_sets() -> None:
    assert PRESETS["Yes/No flag (set {var})"].resolve_labels("Is it late?") == ["yes", "no"]


@pytest.mark.parametrize("key", ["Language", "Entity extraction", "Custom", "One-line summary"])
def test_resolve_labels_open_output(key: str) -> None:
    assert PRESETS[key].resolve_labels() is None
    assert PRESETS[key].resolve_labels("a, b") is None


def test_resolve_labels_from_var_dedupes_and_trims() -> None:
    preset = PRESETS["Categorise (set {var})"]
    assert preset.resolve_labels(" invoice, Contract ,email,, contract; INVOICE\nreport ") == [
        "invoice",
        "Contract",
        "email",
        "report",
    ]


@pytest.mark.parametrize("var", [None, "", "   ", " , ; \n"])
def test_resolve_labels_from_blank_var_is_open(var: str | None) -> None:
    assert PRESETS["Categorise (set {var})"].resolve_labels(var) is None


def test_resolve_labels_empty_tuple_is_open() -> None:
    preset = dataclasses.replace(PRESETS["Sentiment"], labels=())
    assert preset.resolve_labels() is None


# --------------------------------------------------------------------------
# parse_label_list
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, []),
        ("", []),
        ("   ", []),
        (",,;\n", []),
        ("a", ["a"]),
        ("a,b,c", ["a", "b", "c"]),
        ("a; b;c", ["a", "b", "c"]),
        ("a\nb\r\nc", ["a", "b", "c"]),
        (" a , b ;\n c ", ["a", "b", "c"]),
        ("Yes, yes, YES, no", ["Yes", "no"]),
        ("b, a, B, c, A", ["b", "a", "c"]),
        ("not mentioned, n/a", ["not mentioned", "n/a"]),
        ("1, 2, 3, 2", ["1", "2", "3"]),
    ],
)
def test_parse_label_list(value: str | None, expected: list[str]) -> None:
    assert parse_label_list(value) == expected


# --------------------------------------------------------------------------
# get_preset
# --------------------------------------------------------------------------


def test_get_preset_builtin() -> None:
    assert get_preset("Sentiment") is PRESETS["Sentiment"]
    assert get_preset("Yes/No flag (set {var})") is PRESETS["Yes/No flag (set {var})"]


def test_get_preset_by_title_is_not_a_key_lookup() -> None:
    # Keys, not display titles, are the stable identifiers.
    assert get_preset("Keywords") is None
    assert get_preset("Keyword extraction") is PRESETS["Keyword extraction"]


def test_get_preset_from_extra() -> None:
    mine = make_custom_preset("Mine", "Do it: {text}", 16)
    assert get_preset("Mine", [mine]) is mine
    assert get_preset("Mine") is None
    assert get_preset("Mine", []) is None
    assert get_preset("Nope", [mine]) is None


def test_get_preset_builtin_wins_over_extra() -> None:
    impostor = dataclasses.replace(PRESETS["Sentiment"], template="x {text}", builtin=False)
    assert get_preset("Sentiment", [impostor]) is PRESETS["Sentiment"]


# --------------------------------------------------------------------------
# make_custom_preset
# --------------------------------------------------------------------------


def test_make_custom_preset_plain() -> None:
    preset = make_custom_preset("  Bullet summary ", "Summarise as bullets:\n{text}", 128)
    assert preset.key == preset.title == "Bullet summary"
    assert preset.category == "Custom"
    assert preset.template == "Summarise as bullets:\n{text}"
    assert preset.max_new_tokens == 128
    assert preset.labels is None
    assert preset.resolve_labels() is None
    assert preset.output_kind == "text"
    assert preset.builtin is False
    assert preset.var_label is None
    assert not preset.needs_var
    assert not preset.labels_from_var


def test_make_custom_preset_with_labels_and_var() -> None:
    preset = make_custom_preset(
        "Ticket triage",
        "Route this ticket to {var}.\n\n{text}",
        16,
        labels=["billing", "tech", "other"],
        var_hint="e.g. a team",
    )
    assert preset.labels == ("billing", "tech", "other")
    assert preset.resolve_labels() == ["billing", "tech", "other"]
    assert preset.output_kind == "label"
    assert preset.needs_var
    assert preset.var_label == "Variable"
    assert preset.var_hint == "e.g. a team"


def test_make_custom_preset_empty_labels_is_open() -> None:
    preset = make_custom_preset("Open", "{text}", 8, labels=[])
    assert preset.labels is None
    assert preset.output_kind == "text"


def test_make_custom_preset_cleans_labels() -> None:
    preset = make_custom_preset("Clean", "{text}", 8, labels=[" a ", "", "B", "A", "  "])
    assert preset.labels == ("a", "B")
    blank = make_custom_preset("Blank", "{text}", 8, labels=["", "   "])
    assert blank.labels is None
    assert blank.output_kind == "text"


def test_make_custom_preset_coerces_tokens() -> None:
    assert make_custom_preset("T", "{text}", 32.0).max_new_tokens == 32  # type: ignore[arg-type]
    assert make_custom_preset("T", "{text}", "64").max_new_tokens == 64  # type: ignore[arg-type]


@pytest.mark.parametrize("tokens", [0, -1])
def test_make_custom_preset_rejects_non_positive_tokens(tokens: int) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        make_custom_preset("T", "{text}", tokens)


@pytest.mark.parametrize("title", ["", "   ", "\n"])
def test_make_custom_preset_requires_a_name(title: str) -> None:
    with pytest.raises(ValueError, match="name"):
        make_custom_preset(title, "{text}", 8)


@pytest.mark.parametrize("title", list(V1_PRESETS))
def test_make_custom_preset_rejects_builtin_keys(title: str) -> None:
    with pytest.raises(ValueError, match="built-in"):
        make_custom_preset(title, "{text}", 8)
    with pytest.raises(ValueError, match="built-in"):
        make_custom_preset(f"  {title} ", "{text}", 8)


@pytest.mark.parametrize("template", ["", "Summarise this.", "{TEXT}", "{ text }", "{var}"])
def test_make_custom_preset_requires_text_placeholder(template: str) -> None:
    with pytest.raises(ValueError, match=r"\{text\}"):
        make_custom_preset("Mine", template, 8)


def test_make_custom_preset_allows_literal_json_braces() -> None:
    template = 'Return {"label": "x"} for:\n{text}'
    assert make_custom_preset("Json", template, 64).template == template


@pytest.mark.parametrize(
    ("key", "var", "labels"),
    [
        ("Categorise (set {var})", "invoice, contract, email", ["invoice", "contract", "email"]),
        ("Yes/No flag (set {var})", "Does it mention a refund?", ["Yes", "No"]),
    ],
)
def test_saving_a_var_preset_from_its_filled_template_is_self_consistent(
    key: str, var: str, labels: list[str]
) -> None:
    # A Preset can't store a variable value (and make_custom_preset can't set
    # labels_from_var), so a {var} preset must be saved from its *filled*
    # template: then the prompt and the fixed label set always agree.
    source = get_preset(key)
    assert source is not None and source.needs_var
    saved = make_custom_preset(
        "Mine", apply_variable(source.template, var), 16, source.resolve_labels(var)
    )
    assert not saved.needs_var
    assert saved.var_label is None
    assert template_problems(saved.template) == []
    assert var in saved.template
    assert [x.lower() for x in saved.resolve_labels() or []] == [x.lower() for x in labels]
