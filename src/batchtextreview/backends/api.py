"""Cloud API backends: Anthropic (Claude) and OpenAI (GPT).

The texts leave this computer: each prompt is sent to the provider as one
request. The prompts of a batch are sent in parallel - the Run page's batch
size is the number of requests in flight - and the provider's SDK retries
each request on rate limits, overload, timeouts and network errors, with a
backoff that honours the provider's retry-after.

A prompt that fails on its own (declined, too long, still rate-limited after
the retries) is reported in ``last_row_errors``, and the rest of its batch is
kept. ``generate()`` raises only for what no prompt can get past - a rejected
key, a model the key can't use - as a fatal :class:`GenerationError`, which
stops the run.

Request settings favour short, cheap answers: the lowest reasoning effort
each model accepts (a reasoning model's thinking counts against the output
limit, so it gets :data:`REASONING_HEADROOM` extra tokens) and, where the
model allows it, temperature 0.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from batchtextreview import api_keys
from batchtextreview.backends.base import GenerationError, ModelLoadError
from batchtextreview.models import API_PROVIDERS, api_provider

log = logging.getLogger(__name__)

REASONING_HEADROOM = 2048
"""Output tokens allowed on top of the answer's ``max_new_tokens`` for a
model that reasons (thinks) before it answers: the reasoning counts against
the same limit."""
MAX_PARALLEL = 64
"""Most requests in flight at once (the Run page's largest batch size)."""
MAX_RETRIES = 4
"""SDK retries per request, with exponential backoff."""
REQUEST_TIMEOUT_S = 600.0
"""Seconds before one request is abandoned (then retried)."""

_OPENAI_EFFORTS = ("none", "minimal", "low")
"""reasoning_effort values tried in turn, cheapest first: not every reasoning
model accepts every value, and a model that doesn't reason accepts none."""


@dataclass
class _Answer:
    text: str = ""
    hit_limit: bool = False
    error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0


class _Fatal(Exception):
    """A failure no request can get past (bad key, model not available)."""

    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.hint = hint


def make_api_backend(model_id: str) -> ApiBackend:
    """The backend for a ``provider:model`` id. Raises ModelLoadError when
    the SDK isn't installed, no key is set, or the provider rejects the key
    or doesn't offer the model."""
    provider = api_provider(model_id)
    if provider == "anthropic":
        return AnthropicBackend(model_id)
    if provider == "openai":
        return OpenAIBackend(model_id)
    raise ModelLoadError(f"{model_id!r} isn't a cloud API model (anthropic:… or openai:…).")


class ApiBackend:
    """Shared machinery: parallel requests, per-row errors, usage totals."""

    provider = ""
    precision = "none"

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.model = model_id.split(":", 1)[1]
        self.last_hit_limit: list[bool] = []
        self.last_row_errors: list[str | None] = []
        self.last_usage: tuple[int, int] = (0, 0)
        self.closed = False

    @property
    def provider_label(self) -> str:
        return API_PROVIDERS.get(self.provider, self.provider)

    def generate(self, prompts: Sequence[str], max_new_tokens: int) -> list[str]:
        self.last_hit_limit, self.last_row_errors, self.last_usage = [], [], (0, 0)
        if self.closed:
            raise GenerationError(f"The {self.provider_label} client has been closed.")
        batch = list(prompts)
        if not batch:
            return []
        limit = max(1, int(max_new_tokens))

        def ask(prompt: str) -> _Answer:
            return self._safe_ask(prompt, limit)

        try:
            if len(batch) == 1:
                answers = [ask(batch[0])]
            else:
                workers = min(len(batch), MAX_PARALLEL)
                with ThreadPoolExecutor(workers, thread_name_prefix="btr-api") as pool:
                    answers = list(pool.map(ask, batch))
        except _Fatal as exc:
            raise GenerationError(str(exc), exc.hint, fatal=True) from None
        self.last_usage = (
            sum(a.input_tokens for a in answers),
            sum(a.output_tokens for a in answers),
        )
        self.last_hit_limit = [a.hit_limit for a in answers]
        self.last_row_errors = [a.error for a in answers]
        return [a.text for a in answers]

    def close(self) -> None:
        self.closed = True
        client = getattr(self, "_client", None)
        close = getattr(client, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                log.debug("Error closing the %s client", self.provider_label, exc_info=True)

    # ---- per request -----------------------------------------------------
    def _safe_ask(self, prompt: str, max_new_tokens: int) -> _Answer:
        try:
            return self._ask(prompt, max_new_tokens)
        except _Fatal:
            raise
        except Exception as exc:  # a bug or an unexpected SDK error: this row only
            log.warning("%s request failed unexpectedly", self.provider_label, exc_info=True)
            return _Answer(error=f"{type(exc).__name__}: {exc}")

    def _ask(self, prompt: str, max_new_tokens: int) -> _Answer:
        raise NotImplementedError

    def _failed(self, exc: BaseException) -> _Answer:
        """A request that failed after the SDK's retries: raises _Fatal for
        what every request would hit, else this row's error."""
        status = _status(exc)
        detail = _api_message(exc)
        who = self.provider_label
        if status == 401:
            raise _Fatal(f"{who} rejected the API key ({detail}).", _key_hint(self.provider))
        if status == 403:
            raise _Fatal(
                f"{who} refused the request: this key can't use {self.model} ({detail}).",
                "Check the key's permissions and the account's model access.",
            )
        if status == 404:
            raise _Fatal(
                f"{who} has no model “{self.model}” that this key can use ({detail}).",
                "Check the model id on the provider's model list.",
            )
        if status == 429:
            return _Answer(
                error=f"{who} rate limit still exceeded after {MAX_RETRIES} retries "
                f"({detail}). Lower the batch size (requests sent at once), then run "
                "again to retry the failed rows."
            )
        if status is not None and status >= 500:
            return _Answer(error=f"{who} server error {status} ({detail}). Run again to retry.")
        if status is not None:
            return _Answer(error=f"{who} rejected the request: {detail}")
        name = type(exc).__name__
        if "Timeout" in name:
            return _Answer(error=f"No answer from {who} in time. Run again to retry.")
        if "Connection" in name:
            return _Answer(
                error=f"Couldn't reach {who} ({detail}). Check the network connection or "
                "proxy (see Corporate networks in the README)."
            )
        return _Answer(error=f"{name}: {detail}")


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


class OpenAIBackend(ApiBackend):
    """Chat Completions, which OpenAI and OpenAI-compatible servers
    (``OPENAI_BASE_URL``) both offer."""

    provider = "openai"

    def __init__(self, model_id: str, client: Any = None) -> None:
        super().__init__(model_id)
        self._client = client if client is not None else _openai_client()
        self._lock = threading.Lock()
        self._effort: str | None = _OPENAI_EFFORTS[0]
        """reasoning_effort to send (None: don't send it)."""
        self._temperature = True
        """Send temperature 0 (reasoning models reject it)."""
        self._token_param = "max_completion_tokens"
        self._headroom = False
        """Reasoning observed although the effort sent should have none."""
        self._check_model()

    def _check_model(self) -> None:
        try:
            self._client.models.retrieve(self.model)
        except Exception as exc:
            if _status(exc) == 404 and _openai_base_url():
                return  # a compatible server need not list its models
            raise _load_error(self, exc) from None

    def _settings(self) -> tuple[str | None, bool, str, bool]:
        with self._lock:
            return self._effort, self._temperature, self._token_param, self._headroom

    def _ask(self, prompt: str, max_new_tokens: int) -> _Answer:
        for _ in range(8):
            effort, temperature, token_param, headroom = self._settings()
            reasoning = headroom or effort not in (None, "none")
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                token_param: max_new_tokens + (REASONING_HEADROOM if reasoning else 0),
            }
            if effort is not None:
                kwargs["reasoning_effort"] = effort
            if temperature:
                kwargs["temperature"] = 0
            try:
                response = self._client.chat.completions.create(**kwargs)
            except Exception as exc:
                if _status(exc) == 400 and self._adapt(exc, (effort, temperature, token_param)):
                    continue
                return self._failed(exc)
            answer, reasoned = _openai_answer(response)
            if reasoned and answer.hit_limit and not answer.text.strip() and not reasoning:
                with self._lock:
                    self._headroom = True
                log.info(
                    "%s reasons even at reasoning_effort=%s: allowing %s more output tokens",
                    self.model,
                    effort,
                    f"{REASONING_HEADROOM:,}",
                )
                continue
            return answer
        return _Answer(error=f"Couldn't find request settings that {self.model} accepts.")

    def _adapt(self, exc: BaseException, sent: tuple[str | None, bool, str]) -> bool:
        """Change the request settings after a 400 that names one of them.
        True when the request should be sent again."""
        param = str(getattr(exc, "param", None) or "").lower()
        code = str(getattr(exc, "code", None) or "").lower()
        text = _api_message(exc).lower()

        def names(field: str) -> bool:
            return param == field if param else field in text

        with self._lock:
            if (self._effort, self._temperature, self._token_param) != sent:
                return True  # another request has already adapted the settings
            effort, temperature, token_param = sent
            if effort is not None and names("reasoning_effort"):
                unsupported = code == "unsupported_parameter" or "unsupported parameter" in text
                following = _OPENAI_EFFORTS.index(effort) + 1 if effort in _OPENAI_EFFORTS else 99
                self._effort = (
                    None
                    if unsupported or following >= len(_OPENAI_EFFORTS)
                    else _OPENAI_EFFORTS[following]
                )
                log.info(
                    "%s rejected reasoning_effort=%s; trying %s", self.model, effort, self._effort
                )
                return True
            if temperature and names("temperature"):
                self._temperature = False
                log.info("%s rejected temperature=0; sending none", self.model)
                return True
            if token_param == "max_completion_tokens" and names("max_completion_tokens"):
                self._token_param = "max_tokens"
                log.info("%s rejected max_completion_tokens; using max_tokens", self.model)
                return True
        return False


def _openai_answer(response: Any) -> tuple[_Answer, bool]:
    """(the answer, whether the model reported reasoning tokens)."""
    usage = getattr(response, "usage", None)
    sent = int(getattr(usage, "prompt_tokens", 0) or 0)
    generated = int(getattr(usage, "completion_tokens", 0) or 0)
    details = getattr(usage, "completion_tokens_details", None)
    reasoned = bool(getattr(details, "reasoning_tokens", 0) or 0)
    choices = getattr(response, "choices", None) or []
    if not choices:
        return _Answer(error="OpenAI returned no answer.", input_tokens=sent), reasoned
    choice = choices[0]
    message = getattr(choice, "message", None)
    text = str(getattr(message, "content", None) or "")
    refusal = getattr(message, "refusal", None)
    finish = getattr(choice, "finish_reason", None)
    answer = _Answer(text=text, input_tokens=sent, output_tokens=generated)
    if refusal and not text.strip():
        answer.error = f"The model declined to answer this text: {refusal}"
    elif finish == "content_filter":
        answer.error = "OpenAI's content filter withheld the answer to this text."
    else:
        answer.hit_limit = finish == "length"
    return answer, reasoned


def _openai_base_url() -> str:
    return api_keys._clean(os.environ.get("OPENAI_BASE_URL"))


def _openai_client() -> Any:
    try:
        import openai
    except ImportError:
        raise ModelLoadError(
            "The openai package isn't installed.",
            'Install the cloud API extra: pip install -e ".[api]", then restart the app.',
        ) from None
    key = api_keys.get_key("openai")
    base_url = _openai_base_url()
    if not key and not base_url:
        raise ModelLoadError("No OpenAI API key is set.", _key_hint("openai"))
    # A local OpenAI-compatible server often checks no key at all.
    return openai.OpenAI(
        api_key=key or "not-needed",
        base_url=base_url or None,
        max_retries=MAX_RETRIES,
        timeout=REQUEST_TIMEOUT_S,
    )


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


class AnthropicBackend(ApiBackend):
    """The Messages API, with the request settings chosen from the model's
    capabilities (Models API): the lowest effort it supports, and room for
    thinking where it thinks. Sampling parameters aren't sent - current
    models reject them."""

    provider = "anthropic"

    def __init__(self, model_id: str, client: Any = None) -> None:
        super().__init__(model_id)
        self._client = client if client is not None else _anthropic_client()
        self._lock = threading.Lock()
        self._effort: str | None = "low"
        self._thinks = True
        self.display_name = self.model
        self._read_capabilities()

    def _read_capabilities(self) -> None:
        try:
            info = self._client.models.retrieve(self.model)
        except Exception as exc:
            raise _load_error(self, exc) from None
        self.display_name = str(getattr(info, "display_name", None) or self.model)
        caps = getattr(info, "capabilities", None)
        if caps is None:
            return  # unknown: try effort "low" and drop it if refused
        with contextlib.suppress(Exception):  # a capability tree of another shape
            effort = caps.effort
            self._effort = "low" if effort.supported and effort.low.supported else None
        with contextlib.suppress(Exception):
            self._thinks = bool(caps.thinking.supported)

    def _ask(self, prompt: str, max_new_tokens: int) -> _Answer:
        for _ in range(2):
            with self._lock:
                effort = self._effort
            kwargs: dict[str, Any] = {
                "model": self.model,
                "max_tokens": max_new_tokens + (REASONING_HEADROOM if self._thinks else 0),
                "messages": [{"role": "user", "content": prompt}],
            }
            if effort is not None:
                kwargs["output_config"] = {"effort": effort}
            try:
                message = self._client.messages.create(**kwargs)
            except Exception as exc:
                if (
                    effort is not None
                    and _status(exc) == 400
                    and "effort" in _api_message(exc).lower()
                ):
                    with self._lock:
                        self._effort = None
                    log.info("%s rejected effort=%s; sending none", self.model, effort)
                    continue
                return self._failed(exc)
            return _anthropic_answer(message)
        return _Answer(error=f"Couldn't find request settings that {self.model} accepts.")


def _anthropic_answer(message: Any) -> _Answer:
    usage = getattr(message, "usage", None)
    sent = sum(
        int(getattr(usage, field, 0) or 0)
        for field in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
    )
    generated = int(getattr(usage, "output_tokens", 0) or 0)
    text = "".join(
        str(getattr(block, "text", "") or "")
        for block in getattr(message, "content", None) or []
        if getattr(block, "type", None) == "text"
    )
    stop = getattr(message, "stop_reason", None)
    answer = _Answer(text=text, input_tokens=sent, output_tokens=generated)
    if stop == "refusal":
        details = getattr(message, "stop_details", None)
        category = getattr(details, "category", None)
        answer.text = ""
        answer.error = "Claude declined to answer this text" + (
            f" (safety category: {category})." if category else "."
        )
    elif stop == "model_context_window_exceeded":
        answer.error = "This text is too long for the model. Lower Max input characters."
    else:
        answer.hit_limit = stop == "max_tokens"
    return answer


def _anthropic_client() -> Any:
    try:
        import anthropic
    except ImportError:
        raise ModelLoadError(
            "The anthropic package isn't installed.",
            'Install the cloud API extra: pip install -e ".[api]", then restart the app.',
        ) from None
    key = api_keys.get_key("anthropic")
    if not key and not any(
        os.environ.get(name) for name in ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE")
    ):
        raise ModelLoadError("No Anthropic API key is set.", _key_hint("anthropic"))
    try:
        return anthropic.Anthropic(api_key=key, max_retries=MAX_RETRIES, timeout=REQUEST_TIMEOUT_S)
    except Exception as exc:
        raise ModelLoadError(
            f"Couldn't set up the Anthropic client ({exc}).", _key_hint("anthropic")
        ) from None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _key_hint(provider: str) -> str:
    return (
        f"Add a key on the Models & settings page, or set {api_keys.ENV_VARS[provider]} "
        "and restart the app."
    )


def _status(exc: BaseException) -> int | None:
    status = getattr(exc, "status_code", None)
    return status if isinstance(status, int) else None


def _api_message(exc: BaseException) -> str:
    """The provider's own error message, without the SDK's "Error code: 400 -
    {...}" wrapping."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        nested = error.get("message") if isinstance(error, dict) else None
        message = nested or body.get("message")
        if isinstance(message, str) and message.strip():
            return " ".join(message.split())
    text = str(getattr(exc, "message", None) or exc or type(exc).__name__)
    return " ".join(text.split())


def _load_error(backend: ApiBackend, exc: BaseException) -> ModelLoadError:
    """The error to show when checking the model at load time failed."""
    try:
        answer = backend._failed(exc)
    except _Fatal as fatal:
        return ModelLoadError(str(fatal), fatal.hint)
    message = answer.error or str(exc)
    return ModelLoadError(
        f"Couldn't check {backend.model} with {backend.provider_label}: {message}"
    )


__all__ = [
    "MAX_PARALLEL",
    "REASONING_HEADROOM",
    "AnthropicBackend",
    "ApiBackend",
    "OpenAIBackend",
    "make_api_backend",
]
