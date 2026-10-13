"""Tests for batchtextreview.prompting."""

from __future__ import annotations

import pytest

from batchtextreview.presets import BUILTIN_PRESETS
from batchtextreview.prompting import (
    TEXT,
    VAR,
    TemplateError,
    apply_variable,
    render_prompt,
    template_problems,
)

JSON_TEMPLATE = 'Reply as JSON like {"label": "x", "score": 1} or {}.\n\nTEXT:\n{text}'

# --------------------------------------------------------------------------
# template_problems
# --------------------------------------------------------------------------


def test_placeholder_tokens() -> None:
    assert TEXT == "{text}"
    assert VAR == "{var}"


@pytest.mark.parametrize(
    "template",
    ["{text}", "Classify:\n{text}", JSON_TEMPLATE, "{text} and again {text}", "{{text}}"],
)
def test_valid_templates_have_no_problems(template: str) -> None:
    assert template_problems(template) == []
    assert template_problems(template, "unused variable") == []


@pytest.mark.parametrize("template", ["", " ", "\n\t "])
def test_empty_template_is_a_single_problem(template: str) -> None:
    assert template_problems(template) == ["The prompt template is empty."]
    assert template_problems(template, "x") == ["The prompt template is empty."]


@pytest.mark.parametrize("template", ["Summarise this.", "{TEXT}", "{ text }", "{text", "text}"])
def test_missing_text_placeholder(template: str) -> None:
    problems = template_problems(template)
    assert len(problems) == 1
    assert "{text}" in problems[0]


@pytest.mark.parametrize("var", [None, "", "   ", "\n"])
def test_var_template_needs_a_value(var: str | None) -> None:
    problems = template_problems("Q: {var}\n{text}", var)
    assert len(problems) == 1
    assert "{var}" in problems[0] and "variable" in problems[0]


@pytest.mark.parametrize("var", ["{text}", "about {text} here", " {text} "])
def test_var_containing_text_placeholder_is_rejected(var: str) -> None:
    problems = template_problems("Q: {var}\n{text}", var)
    assert problems == ["The variable can't contain {text}."]


def test_var_containing_text_is_fine_when_template_has_no_var() -> None:
    # The variable is ignored entirely, so its content can't matter.
    assert template_problems("{text}", "{text}") == []


def test_all_problems_are_reported_together() -> None:
    problems = template_problems("Question: {var}", None)
    assert len(problems) == 2
    assert any("{text}" in p for p in problems)
    assert any("variable" in p for p in problems)


@pytest.mark.parametrize("preset", BUILTIN_PRESETS, ids=lambda p: p.key)
def test_every_builtin_template_is_valid_with_a_variable(preset) -> None:
    var = "food, service" if preset.needs_var else None
    assert template_problems(preset.template, var) == []
    if preset.needs_var:
        assert template_problems(preset.template, None) != []


# --------------------------------------------------------------------------
# apply_variable
# --------------------------------------------------------------------------


def test_apply_variable_without_var_returns_template_unchanged() -> None:
    assert apply_variable(JSON_TEMPLATE) == JSON_TEMPLATE
    assert apply_variable(JSON_TEMPLATE, "ignored") == JSON_TEMPLATE


def test_apply_variable_substitutes_trimmed_value() -> None:
    final = apply_variable("Categories: {var}.\n\n{text}", "  a, b, c \n")
    assert final == "Categories: a, b, c.\n\n{text}"


def test_apply_variable_substitutes_every_occurrence() -> None:
    final = apply_variable("{var}? Answer about {var}.\n{text}", "refunds")
    assert final == "refunds? Answer about refunds.\n{text}"


def test_apply_variable_keeps_json_braces_literal() -> None:
    template = 'Map each of {var} to a sentiment, e.g. {"food": "positive"}.\n{text}'
    final = apply_variable(template, "food, price")
    assert final == 'Map each of food, price to a sentiment, e.g. {"food": "positive"}.\n{text}'


@pytest.mark.parametrize("var", ["{0}", "{}", "{name}", "{{x}}", "%s", r"\1", "$&"])
def test_apply_variable_inserts_special_characters_verbatim(var: str) -> None:
    assert apply_variable("Q: {var}\n{text}", var) == f"Q: {var}\n{{text}}"


def test_apply_variable_does_not_rescan_the_value() -> None:
    # A variable mentioning {var} is inserted literally, not expanded again.
    assert apply_variable("Q: {var}\n{text}", "what is {var}?") == "Q: what is {var}?\n{text}"


def test_apply_variable_rejects_var_containing_text() -> None:
    with pytest.raises(TemplateError, match=r"can't contain \{text\}"):
        apply_variable("Q: {var}\n{text}", "summarise {text}")


def test_apply_variable_raises_every_problem() -> None:
    with pytest.raises(TemplateError) as info:
        apply_variable("Question: {var}", "")
    message = str(info.value)
    assert "{text}" in message
    assert "variable" in message


def test_apply_variable_rejects_empty_template() -> None:
    with pytest.raises(TemplateError, match="empty"):
        apply_variable("   ")


def test_template_error_is_a_value_error() -> None:
    assert issubclass(TemplateError, ValueError)
    with pytest.raises(ValueError):
        apply_variable("no placeholder")


# --------------------------------------------------------------------------
# render_prompt
# --------------------------------------------------------------------------


def test_render_prompt_inserts_text() -> None:
    assert render_prompt("Classify:\n{text}", "Great service") == "Classify:\nGreat service"


def test_render_prompt_keeps_json_braces_literal() -> None:
    prompt = render_prompt(JSON_TEMPLATE, "hello")
    assert prompt == 'Reply as JSON like {"label": "x", "score": 1} or {}.\n\nTEXT:\nhello'


@pytest.mark.parametrize(
    "text",
    [
        "{text}",
        "{var}",
        "literal {var} and {text} inside",
        '{"already": "json"}',
        "{0} {} {{}} %s %(name)s",
        r"backslashes \1 \g<0> \n",
        "$& $1",
        "",
        "   ",
        "émoji ✓ — ünïcode\r\nwith CRLF",
    ],
)
def test_render_prompt_inserts_text_verbatim(text: str) -> None:
    assert render_prompt("A:{text}:B", text) == f"A:{text}:B"


def test_render_prompt_fills_every_occurrence() -> None:
    assert render_prompt("{text} / {text}", "x") == "x / x"


def test_render_prompt_does_not_touch_var() -> None:
    # render_prompt expects a final template; a stray {var} stays literal.
    assert render_prompt("{var}: {text}", "x") == "{var}: x"


def test_two_step_render_never_expands_row_text() -> None:
    final = apply_variable("Question: {var}\n\nTEXT:\n{text}", "Is it {late}?")
    prompts = [render_prompt(final, row) for row in ("{var}", "{text}", "plain")]
    assert prompts == [
        "Question: Is it {late}?\n\nTEXT:\n{var}",
        "Question: Is it {late}?\n\nTEXT:\n{text}",
        "Question: Is it {late}?\n\nTEXT:\nplain",
    ]


def test_render_builtin_parameterised_preset() -> None:
    preset = next(p for p in BUILTIN_PRESETS if p.key == "Categorise (set {var})")
    final = apply_variable(preset.template, "invoice, contract, other")
    prompt = render_prompt(final, "Please find attached invoice #42.")
    assert prompt == (
        "Classify the text below into exactly one of these categories: "
        "invoice, contract, other. Reply with only the category name.\n\n"
        "TEXT:\nPlease find attached invoice #42."
    )
