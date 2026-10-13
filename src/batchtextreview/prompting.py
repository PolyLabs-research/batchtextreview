"""Prompt-template validation and rendering.

Only the exact tokens ``{text}`` and ``{var}`` are special. Every other brace
is literal, so templates can contain JSON examples such as ``{"a": 1}``.

Rendering is two-step: :func:`apply_variable` substitutes ``{var}`` once to
give the *final template* (which is what the run id hashes), then
:func:`render_prompt` substitutes ``{text}`` per row. Row text is inserted
verbatim and never re-scanned, so text containing ``{var}`` or ``{text}`` is
harmless.
"""

from __future__ import annotations

TEXT = "{text}"
VAR = "{var}"


class TemplateError(ValueError):
    """The template / variable combination can't be run."""


def template_problems(template: str, var: str | None = None) -> list[str]:
    """Human-readable reasons the template can't be run (empty list = OK)."""
    problems: list[str] = []
    if not template or not template.strip():
        return ["The prompt template is empty."]
    if TEXT not in template:
        problems.append("The prompt template must include the {text} placeholder.")
    if VAR in template:
        value = (var or "").strip()
        if not value:
            problems.append("This task uses {var} - fill in the variable first.")
        elif TEXT in value:
            problems.append("The variable can't contain {text}.")
    return problems


def apply_variable(template: str, var: str | None = None) -> str:
    """Return the final template with ``{var}`` substituted.

    Raises :class:`TemplateError` with every problem found.
    """
    problems = template_problems(template, var)
    if problems:
        raise TemplateError(" ".join(problems))
    if VAR in template:
        return template.replace(VAR, (var or "").strip())
    return template


def render_prompt(final_template: str, text: str) -> str:
    """Insert one row's text into a final template."""
    return final_template.replace(TEXT, text)
