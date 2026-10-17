"""Tests for the cloud API backends (OpenAI, Anthropic), API keys and the
API part of the model catalogue.

Requests never leave the process: each SDK client gets an ``httpx2``
transport that answers like the provider would.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from typing import Any

import pytest

from batchtextreview import api_keys, models
from batchtextreview.backends import manager as manager_module
from batchtextreview.backends.base import GenerationError, ModelLoadError

openai = pytest.importorskip("openai")
anthropic = pytest.importorskip("anthropic")
httpx2 = pytest.importorskip("httpx2")

from batchtextreview.backends import api  # noqa: E402  (needs the SDKs)

Handler = Callable[[Any], Any]


@pytest.fixture(autouse=True)
def _no_keys(monkeypatch: pytest.MonkeyPatch):
    for name in (*api_keys.ENV_VARS.values(), "OPENAI_BASE_URL", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("ANTHROPIC_PROFILE", raising=False)
    api_keys._reset_for_tests()
    yield
    api_keys._reset_for_tests()


def _body(request: Any) -> dict[str, Any]:
    return json.loads(request.content or b"{}")


def _error(status: int, message: str, **fields: Any) -> Any:
    return httpx2.Response(status, json={"error": {"message": message, "type": "x", **fields}})


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


def _completion(
    text: str | None = "positive",
    finish: str = "stop",
    refusal: str | None = None,
    reasoning: int = 0,
    prompt_tokens: int = 10,
    completion_tokens: int = 2,
) -> Any:
    return httpx2.Response(
        200,
        json={
            "id": "c1",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish,
                    "message": {"role": "assistant", "content": text, "refusal": refusal},
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "completion_tokens_details": {"reasoning_tokens": reasoning},
            },
        },
    )


class OpenAIServer:
    """A scripted OpenAI: ``chat`` answers each completion request."""

    def __init__(self, chat: Handler, model_status: int = 200) -> None:
        self.chat = chat
        self.model_status = model_status
        self.requests: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def __call__(self, request: Any) -> Any:
        if request.method == "GET":
            if self.model_status != 200:
                return _error(self.model_status, "The model `x` does not exist.")
            return httpx2.Response(
                200, json={"id": "m", "object": "model", "created": 0, "owned_by": "openai"}
            )
        body = _body(request)
        with self._lock:
            self.requests.append(body)
        return self.chat(body)

    @property
    def completions(self) -> list[dict[str, Any]]:
        return list(self.requests)


def openai_backend(server: OpenAIServer, model: str = "gpt-6-luna") -> api.OpenAIBackend:
    client = openai.OpenAI(
        api_key="sk-test",
        http_client=httpx2.Client(transport=httpx2.MockTransport(server)),
        max_retries=0,
    )
    return api.OpenAIBackend(f"openai:{model}", client=client)


def test_openai_answers_in_order_with_usage_and_cheapest_settings():
    def chat(body):
        return _completion(text=body["messages"][0]["content"].upper())

    server = OpenAIServer(chat)
    backend = openai_backend(server)
    prompts = [f"text {i}" for i in range(20)]
    assert backend.generate(prompts, 8) == [p.upper() for p in prompts]
    assert backend.last_row_errors == [None] * 20
    assert backend.last_hit_limit == [False] * 20
    assert backend.last_usage == (200, 40)
    first = server.completions[0]
    assert first["reasoning_effort"] == "none" and first["temperature"] == 0
    assert first["max_completion_tokens"] == 8  # no reasoning: no headroom
    assert first["messages"] == [{"role": "user", "content": first["messages"][0]["content"]}]


def test_openai_learns_which_settings_the_model_accepts():
    def chat(body):
        if "temperature" in body:
            return _error(
                400, "Unsupported parameter: 'temperature'.", param="temperature",
                code="unsupported_parameter",
            )  # fmt: skip
        if body.get("reasoning_effort") in ("none", "minimal"):
            return _error(
                400, "Unsupported value: 'reasoning_effort'.", param="reasoning_effort",
                code="unsupported_value",
            )  # fmt: skip
        return _completion()

    server = OpenAIServer(chat)
    backend = openai_backend(server)
    assert backend.generate(["a", "b", "c"], 8) == ["positive"] * 3
    assert backend.last_row_errors == [None] * 3
    accepted = [r for r in server.completions if "temperature" not in r][-1]
    assert accepted["reasoning_effort"] == "low"
    # A reasoning effort above "none" leaves room for the reasoning.
    assert accepted["max_completion_tokens"] == 8 + api.REASONING_HEADROOM
    sent_before = len(server.completions)
    backend.generate(["d"], 8)  # the settings are remembered
    assert len(server.completions) == sent_before + 1


def test_openai_model_without_reasoning_gets_no_effort_and_no_headroom():
    def chat(body):
        if "reasoning_effort" in body:
            return _error(
                400, "Unsupported parameter: 'reasoning_effort' is not supported with this model.",
                param="reasoning_effort", code="unsupported_parameter",
            )  # fmt: skip
        return _completion()

    server = OpenAIServer(chat)
    backend = openai_backend(server, "gpt-4.1")
    backend.generate(["a"], 16)
    last = server.completions[-1]
    assert "reasoning_effort" not in last
    assert last["temperature"] == 0 and last["max_completion_tokens"] == 16


def test_openai_falls_back_to_max_tokens_for_older_servers():
    def chat(body):
        if "max_completion_tokens" in body:
            return _error(400, "Unrecognized request argument supplied: max_completion_tokens")
        if "reasoning_effort" in body:
            return _error(400, "Unrecognized request argument supplied: reasoning_effort")
        return _completion()

    server = OpenAIServer(chat)
    backend = openai_backend(server)
    assert backend.generate(["a"], 8) == ["positive"]
    last = server.completions[-1]
    assert last["max_tokens"] == 8 and "max_completion_tokens" not in last


def test_openai_adds_headroom_when_the_model_reasons_anyway():
    def chat(body):
        if body["max_completion_tokens"] <= 8:
            return _completion(text="", finish="length", reasoning=8, completion_tokens=8)
        return _completion(reasoning=40, completion_tokens=42)

    server = OpenAIServer(chat)
    backend = openai_backend(server)
    assert backend.generate(["a"], 8) == ["positive"]
    assert backend.last_hit_limit == [False]
    assert server.completions[-1]["max_completion_tokens"] == 8 + api.REASONING_HEADROOM


def test_openai_row_errors_keep_the_rest_of_the_batch():
    def chat(body):
        text = body["messages"][0]["content"]
        if text == "bad":
            return _error(400, "Invalid prompt: your prompt was flagged.", code="invalid_prompt")
        if text == "refused":
            return _completion(text=None, refusal="I can't help with that.")
        if text == "filtered":
            return _completion(text="", finish="content_filter")
        if text == "long":
            return _completion(text="positive and", finish="length")
        if text == "busy":
            return _error(429, "Rate limit reached for requests.")
        return _completion()

    backend = openai_backend(OpenAIServer(chat))
    outputs = backend.generate(["ok", "bad", "refused", "filtered", "long", "busy"], 8)
    assert outputs[0] == "positive" and outputs[4] == "positive and"
    errors = backend.last_row_errors
    assert errors[0] is None and errors[4] is None
    assert "Invalid prompt" in str(errors[1])
    assert "declined" in str(errors[2]) and "can't help" in str(errors[2])
    assert "content filter" in str(errors[3])
    assert "rate limit" in str(errors[5]).lower() and "batch size" in str(errors[5])
    assert backend.last_hit_limit == [False, False, False, False, True, False]


def test_openai_rejected_key_mid_run_is_fatal():
    def chat(body):
        return _error(401, "Incorrect API key provided.")

    backend = openai_backend(OpenAIServer(chat))
    with pytest.raises(GenerationError) as caught:
        backend.generate(["a", "b"], 8)
    assert caught.value.fatal is True
    assert "rejected the API key" in str(caught.value)
    assert "OPENAI_API_KEY" in str(caught.value)


def test_openai_unknown_model_fails_at_load():
    with pytest.raises(ModelLoadError, match="no model “gpt-nope”"):
        openai_backend(OpenAIServer(lambda body: _completion(), model_status=404), "gpt-nope")


def test_openai_compatible_server_need_not_list_its_models(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    backend = openai_backend(OpenAIServer(lambda body: _completion(), model_status=404), "llama")
    assert backend.generate(["a"], 4) == ["positive"]


def test_openai_client_needs_a_key_unless_a_server_is_set(monkeypatch):
    with pytest.raises(ModelLoadError, match="No OpenAI API key"):
        api._openai_client()
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
    client = api._openai_client()
    assert str(client.base_url).startswith("http://localhost:8000/v1")
    assert client.max_retries == api.MAX_RETRIES


def test_closed_backend_refuses_to_generate():
    backend = openai_backend(OpenAIServer(lambda body: _completion()))
    backend.close()
    with pytest.raises(GenerationError):
        backend.generate(["a"], 4)


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


def _capabilities(effort: bool = True, thinking: bool = True) -> dict[str, Any]:
    level = {"supported": effort}
    return {
        "batch": {"supported": True},
        "citations": {"supported": True},
        "code_execution": {"supported": False},
        "context_management": {"supported": False},
        "effort": {
            "supported": effort,
            "low": level,
            "medium": level,
            "high": level,
            "max": level,
        },
        "image_input": {"supported": True},
        "pdf_input": {"supported": True},
        "server_tools": {"supported": False},
        "structured_outputs": {"supported": True},
        "thinking": {
            "supported": thinking,
            "types": {"adaptive": {"supported": thinking}, "enabled": {"supported": False}},
        },
    }


def _message(
    text: str | None = "negative",
    stop: str = "end_turn",
    category: str | None = None,
    usage: dict[str, int] | None = None,
) -> Any:
    content: list[dict[str, Any]] = [{"type": "thinking", "thinking": "", "signature": "s"}]
    if text is not None:
        content.append({"type": "text", "text": text})
    return httpx2.Response(
        200,
        json={
            "id": "msg",
            "type": "message",
            "role": "assistant",
            "model": "claude",
            "content": content,
            "stop_reason": stop,
            "stop_sequence": None,
            "stop_details": (
                {"type": "refusal", "category": category, "explanation": None}
                if stop == "refusal"
                else None
            ),
            "usage": usage or {"input_tokens": 20, "output_tokens": 3},
        },
    )


class AnthropicServer:
    def __init__(
        self, messages: Handler, capabilities: dict[str, Any] | None = None, status: int = 200
    ) -> None:
        self.messages = messages
        self.capabilities = capabilities
        self.status = status
        self.requests: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def __call__(self, request: Any) -> Any:
        if request.method == "GET":
            if self.status != 200:
                return httpx2.Response(
                    self.status,
                    json={"type": "error", "error": {"type": "x", "message": "nope"}},
                )
            info: dict[str, Any] = {
                "id": "claude-haiku-5-5",
                "type": "model",
                "display_name": "Claude Haiku 5.5",
                "created_at": "2026-09-01T00:00:00Z",
                "lifecycle": "active",
                "max_input_tokens": 1_000_000,
                "max_tokens": 128_000,
            }
            if self.capabilities is not None:
                info["capabilities"] = self.capabilities
            return httpx2.Response(200, json=info)
        body = _body(request)
        with self._lock:
            self.requests.append(body)
        return self.messages(body)


def anthropic_backend(server: AnthropicServer) -> api.AnthropicBackend:
    client = anthropic.Anthropic(
        api_key="sk-ant-test",
        http_client=httpx2.Client(transport=httpx2.MockTransport(server)),
        max_retries=0,
    )
    return api.AnthropicBackend("anthropic:claude-haiku-5-5", client=client)


def test_anthropic_settings_follow_the_model_capabilities():
    server = AnthropicServer(lambda body: _message(), _capabilities())
    backend = anthropic_backend(server)
    assert backend.display_name == "Claude Haiku 5.5"
    assert backend.generate(["one", "two"], 8) == ["negative", "negative"]
    request = server.requests[0]
    assert request["output_config"] == {"effort": "low"}
    assert request["max_tokens"] == 8 + api.REASONING_HEADROOM
    assert "temperature" not in request and "thinking" not in request
    assert backend.last_usage == (40, 6)


def test_anthropic_model_without_effort_or_thinking():
    server = AnthropicServer(lambda body: _message(), _capabilities(effort=False, thinking=False))
    anthropic_backend(server).generate(["one"], 8)
    request = server.requests[0]
    assert "output_config" not in request and request["max_tokens"] == 8


def test_anthropic_drops_effort_when_refused():
    def messages(body):
        if "output_config" in body:
            return httpx2.Response(
                400,
                json={
                    "type": "error",
                    "error": {"type": "invalid_request_error", "message": "effort: not supported"},
                },
            )
        return _message()

    server = AnthropicServer(messages)  # no capabilities: tries "low" first
    backend = anthropic_backend(server)
    assert backend.generate(["one"], 8) == ["negative"]
    assert backend.last_row_errors == [None]
    assert "output_config" not in server.requests[-1]


def test_anthropic_refusals_long_texts_and_cut_answers():
    def messages(body):
        text = body["messages"][0]["content"]
        if text == "refuse":
            return _message(text=None, stop="refusal", category="cyber")
        if text == "huge":
            return _message(text=None, stop="model_context_window_exceeded")
        if text == "cut":
            return _message(text="neg", stop="max_tokens")
        return _message(usage={"input_tokens": 5, "cache_read_input_tokens": 7, "output_tokens": 1})

    backend = anthropic_backend(AnthropicServer(messages, _capabilities()))
    outputs = backend.generate(["fine", "refuse", "huge", "cut"], 8)
    assert outputs == ["negative", "", "", "neg"]
    errors = backend.last_row_errors
    assert errors[0] is None and errors[3] is None
    assert errors[1] == "Claude declined to answer this text (safety category: cyber)."
    assert "too long" in str(errors[2])
    assert backend.last_hit_limit == [False, False, False, True]
    assert backend.last_usage[0] == 12 + 20 * 3


@pytest.mark.parametrize(
    ("status", "words"),
    [(401, "rejected the API key"), (403, "can't use"), (404, "has no model")],
)
def test_anthropic_load_errors(status, words):
    with pytest.raises(ModelLoadError, match=words):
        anthropic_backend(AnthropicServer(lambda body: _message(), status=status))


def test_anthropic_client_needs_a_key():
    with pytest.raises(ModelLoadError, match="No Anthropic API key"):
        api._anthropic_client()
    api_keys.set_session_key("anthropic", "sk-ant-session")
    assert api._anthropic_client().api_key == "sk-ant-session"


def test_error_message_is_the_providers_own_text():
    def messages(body):
        return httpx2.Response(
            400,
            json={
                "type": "error",
                "error": {"type": "invalid_request_error", "message": "prompt is too long"},
            },
        )

    backend = anthropic_backend(AnthropicServer(messages, _capabilities()))
    backend.generate(["x"], 8)
    assert backend.last_row_errors == ["Anthropic rejected the request: prompt is too long"]


def test_factory_dispatches_api_models(monkeypatch):
    made: list[str] = []

    def fake(model_id: str) -> Any:
        made.append(model_id)
        return object()

    monkeypatch.setattr(api, "make_api_backend", fake)
    manager_module.default_factory("anthropic:claude-haiku-5-5", "none")
    manager_module.default_factory("openai:gpt-6-luna", "none")
    assert made == ["anthropic:claude-haiku-5-5", "openai:gpt-6-luna"]


def test_make_api_backend_rejects_other_ids():
    with pytest.raises(ModelLoadError, match="isn't a cloud API model"):
        api.make_api_backend("Qwen/Qwen2.5-1.5B-Instruct")


# ---------------------------------------------------------------------------
# API keys
# ---------------------------------------------------------------------------


class MemoryKeyring:
    """A keyring backend that keeps passwords in a dict."""

    def __init__(self) -> None:
        import keyring.backend

        class Backend(keyring.backend.KeyringBackend):
            priority = 1  # type: ignore[assignment]

            def __init__(self) -> None:
                super().__init__()
                self.store: dict[tuple[str, str], str] = {}

            def get_password(self, service: str, username: str) -> str | None:
                return self.store.get((service, username))

            def set_password(self, service: str, username: str, password: str) -> None:
                self.store[(service, username)] = password

            def delete_password(self, service: str, username: str) -> None:
                import keyring.errors

                if (service, username) not in self.store:
                    raise keyring.errors.PasswordDeleteError("missing")
                del self.store[(service, username)]

        self.backend = Backend()


@pytest.fixture
def memory_keyring(monkeypatch, real_keyring_lookup):
    keyring = pytest.importorskip("keyring")
    previous = keyring.get_keyring()
    memory = MemoryKeyring()
    keyring.set_keyring(memory.backend)
    yield memory.backend
    keyring.set_keyring(previous)


def test_key_precedence_session_then_env_then_keyring(monkeypatch, memory_keyring):
    assert api_keys.lookup("openai") is None and api_keys.key_source("openai") is None
    assert api_keys.keyring_available()
    api_keys.save_key("openai", "  sk-saved  ")
    assert memory_keyring.store[(api_keys.KEYRING_SERVICE, "openai")] == "sk-saved"
    assert api_keys.lookup("openai") == ("sk-saved", "keyring")
    monkeypatch.setenv("OPENAI_API_KEY", '"sk-env"')  # cmd keeps the quotes
    assert api_keys.lookup("openai") == ("sk-env", "env")
    api_keys.set_session_key("openai", "sk-session")
    assert api_keys.lookup("openai") == ("sk-session", "session")
    api_keys.set_session_key("openai", "")
    assert api_keys.get_key("openai") == "sk-env"
    assert api_keys.delete_saved_key("openai") is True
    assert api_keys.delete_saved_key("openai") is False
    monkeypatch.delenv("OPENAI_API_KEY")
    assert api_keys.get_key("openai") is None


def test_saving_without_a_keyring_explains_the_alternative(monkeypatch):
    monkeypatch.setattr(api_keys, "_keyring_backend", lambda: None)
    assert not api_keys.keyring_available()
    with pytest.raises(RuntimeError, match="environment variable"):
        api_keys.save_key("anthropic", "sk-ant-x")
    assert api_keys.saved_key("anthropic") is None


def test_keys_are_shown_masked():
    assert api_keys.masked("sk-ant-api03-abcdefghijklmnop1234") == "sk-ant-…1234"
    assert api_keys.masked("sk-proj-abcdefghijklmnopqrstu9876") == "sk-…9876"
    assert api_keys.masked("short") == "…rt"
    assert api_keys.masked("") == ""


def test_unknown_provider_is_rejected():
    with pytest.raises(ValueError):
        api_keys.get_key("mistral")


# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------


def test_builtin_api_models():
    specs = models.api_models()
    assert {s.provider for s in specs} == {"anthropic", "openai"}
    assert all(s.is_api and not s.is_local and s.hf_url is None for s in specs)
    assert [s.repo for s in models.api_models("anthropic")] == [
        "anthropic:claude-haiku-5-5",
        "anthropic:claude-sonnet-5-5",
        "anthropic:claude-opus-5-5",
    ]
    assert models.find_model("Claude Opus 5.5") is models.find_model("anthropic:claude-opus-5-5")
    assert all(s.price for s in models.api_models("anthropic"))
    # API models never mix into the local list.
    assert not any(s.is_api for s in models.available_models(include_demo=True))


@pytest.mark.parametrize(
    ("value", "repo", "label"),
    [
        ("anthropic:claude-opus-4-8", "anthropic:claude-opus-4-8", "claude-opus-4-8"),
        ("Anthropic: claude-haiku-5-5", "anthropic:claude-haiku-5-5", "Claude Haiku 5.5"),
        (
            "openai:ft:gpt-6-luna:acme::a1",
            "openai:ft:gpt-6-luna:acme::a1",
            "ft:gpt-6-luna:acme::a1",
        ),
        (
            "openai:meta-llama/Llama-3.1-8B",
            "openai:meta-llama/Llama-3.1-8B",
            "meta-llama/Llama-3.1-8B",
        ),
    ],
)
def test_custom_api_model_ids(value, repo, label):
    spec = models.custom_model(value)
    assert (spec.repo, spec.label) == (repo, label)
    assert spec.is_api and spec.api_model == repo.split(":", 1)[1]
    assert models.model_revision(spec.repo) == ""


@pytest.mark.parametrize("value", ["openai:", "anthropic: two words", "openai:-dash"])
def test_bad_api_model_ids(value):
    with pytest.raises(ValueError, match="model id after"):
        models.custom_model(value)


def test_run_config_of_an_api_model_has_no_precision():
    from batchtextreview.engine import RunConfig

    spec = models.find_model("anthropic:claude-haiku-5-5")
    assert spec is not None
    config = RunConfig.build(spec, "Q: {text}", 8, precision="4bit")
    assert config.precision == "none" and config.model_revision == ""


# ---------------------------------------------------------------------------
# Review fixes
# ---------------------------------------------------------------------------


class _Fake:
    """A cloud backend stand-in that remembers the key it was made with."""

    def __init__(self, model_id: str, key: str | None) -> None:
        self.model_id, self.precision, self.key, self.closed = model_id, "none", key, False

    def generate(self, prompts, max_new_tokens):
        return ["x"] * len(prompts)

    def close(self) -> None:
        self.closed = True


def test_manager_reconnects_when_the_key_changes_and_keeps_the_local_model():
    made: list[_Fake] = []

    def factory(model_id: str, precision: str) -> Any:
        backend = _Fake(model_id, api_keys.get_key("openai"))
        made.append(backend)
        return backend

    manager = manager_module.ModelManager(factory)
    with manager.acquire("Qwen/Qwen2.5-1.5B-Instruct", "4bit") as local:
        pass
    api_keys.set_session_key("openai", "sk-personal")
    with manager.acquire("openai:gpt-6-luna", "none") as first:
        assert first.key == "sk-personal"
    with manager.acquire("openai:gpt-6-luna", "none") as again:
        assert again is first  # same key: reused
    api_keys.set_session_key("openai", "sk-company")
    assert not manager.is_loaded("openai:gpt-6-luna", "none")
    with manager.acquire("openai:gpt-6-luna", "none") as second:
        assert second is not first and second.key == "sk-company" and first.closed
    # The local model was never evicted, and "loaded" only names it.
    assert manager.loaded == ("Qwen/Qwen2.5-1.5B-Instruct", "4bit") and not local.closed
    second.closed = True  # e.g. Ctrl+C closed it: a new client next time
    with manager.acquire("openai:gpt-6-luna", "none") as third:
        assert third is not second
    manager.unload()
    assert third.closed and local.closed and manager.loaded is None


@pytest.mark.parametrize(
    "error",
    [
        {"message": "max_completion_tokens is too large: 34816.", "param": "max_completion_tokens",
         "code": "invalid_value"},
        {"message": "'max_tokens' or 'max_completion_tokens' is too large: the context is 4096."},
        {"message": "temperature must be at most 2.", "param": "temperature", "code": "invalid_value"},
    ],
)  # fmt: skip
def test_openai_keeps_its_settings_after_errors_that_are_not_unsupported(error):
    def chat(body):
        if body["messages"][0]["content"] == "bad":
            return httpx2.Response(400, json={"error": {"type": "invalid_request_error", **error}})
        return _completion()

    server = OpenAIServer(chat)
    backend = openai_backend(server)
    backend.generate(["bad"], 8)
    assert backend.last_row_errors[0] and "rejected the request" in backend.last_row_errors[0]
    assert backend._settings() == ("none", True, "max_completion_tokens", False)
    assert backend.generate(["fine"], 8) == ["positive"]


def test_openai_sends_no_reasoning_effort_to_another_server(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
    server = OpenAIServer(lambda body: _completion())
    backend = openai_backend(server, "my-local-model")
    backend.generate(["a"], 8)
    request = server.completions[-1]
    assert "reasoning_effort" not in request and request["max_completion_tokens"] == 8


def test_out_of_credit_is_fatal():
    def chat(body):
        return _error(429, "You exceeded your current quota.", code="insufficient_quota")

    with pytest.raises(GenerationError, match="out of credit") as caught:
        openai_backend(OpenAIServer(chat)).generate(["a", "b"], 8)
    assert caught.value.fatal

    def messages(body):
        return httpx2.Response(
            400,
            json={
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": "Your credit balance is too low to access the Anthropic API.",
                },
            },
        )

    with pytest.raises(GenerationError, match="out of credit"):
        anthropic_backend(AnthropicServer(messages, _capabilities())).generate(["a"], 8)


def anthropic_model(server: AnthropicServer, model: str) -> api.AnthropicBackend:
    client = anthropic.Anthropic(
        api_key="sk-ant-test",
        http_client=httpx2.Client(transport=httpx2.MockTransport(server)),
        max_retries=0,
    )
    return api.AnthropicBackend(f"anthropic:{model}", client=client)


def test_anthropic_headroom_only_for_models_that_think():
    server = AnthropicServer(lambda body: _message(), _capabilities())
    anthropic_model(server, "claude-opus-4-8").generate(["a"], 8)
    assert server.requests[-1]["max_tokens"] == 8  # 4.x models don't think unless asked
    anthropic_model(server, "claude-opus-5-5").generate(["a"], 8)
    assert server.requests[-1]["max_tokens"] == 8 + api.REASONING_HEADROOM


def test_anthropic_adds_headroom_once_thinking_used_up_the_limit():
    def messages(body):
        if body["max_tokens"] <= 8:
            return _message(text=None, stop="max_tokens")  # a thinking block, no text
        return _message()

    server = AnthropicServer(messages, _capabilities())
    backend = anthropic_model(server, "claude-sonnet-4-6")
    assert backend.generate(["a"], 8) == ["negative"]
    assert [r["max_tokens"] for r in server.requests] == [8, 8 + api.REASONING_HEADROOM]


def test_anthropic_budget_never_exceeds_the_models_output_limit():
    class Small(AnthropicServer):
        def __call__(self, request):
            response = super().__call__(request)
            if request.method == "GET":
                info = json.loads(response.content)
                info["max_tokens"] = 1000
                return httpx2.Response(200, json=info)
            return response

    server = Small(lambda body: _message(), _capabilities())
    anthropic_model(server, "claude-haiku-5-5").generate(["a"], 8)
    assert server.requests[-1]["max_tokens"] == 1000


def test_ctrl_c_closes_the_client_instead_of_waiting():
    backend = openai_backend(OpenAIServer(lambda body: _completion()))

    def interrupted(prompt, max_new_tokens):
        raise KeyboardInterrupt

    backend._ask = interrupted  # type: ignore[method-assign]
    with pytest.raises(KeyboardInterrupt):
        backend.generate(["a", "b", "c"], 8)
    assert backend.closed


def test_recipient_and_fingerprint(monkeypatch):
    assert api_keys.recipient("openai") == "OpenAI"
    before = api_keys.fingerprint("openai")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example.com/v1")
    assert api_keys.recipient("openai") == "gateway.example.com"
    assert api_keys.fingerprint("openai") != before
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "llm-proxy.internal:8443")
    assert api_keys.recipient("anthropic") == "llm-proxy.internal:8443"
    key_less = api_keys.fingerprint("anthropic")
    api_keys.set_session_key("anthropic", "sk-ant-x")
    assert api_keys.fingerprint("anthropic") != key_less
    assert "sk-ant-x" not in api_keys.fingerprint("anthropic")


def test_a_failed_keyring_read_is_tried_again(monkeypatch, memory_keyring):
    memory_keyring.store[(api_keys.KEYRING_SERVICE, "openai")] = "sk-saved"
    calls = {"n": 0}
    real_get = memory_keyring.get_password

    def locked_once(service, name):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("keyring is locked")
        return real_get(service, name)

    monkeypatch.setattr(memory_keyring, "get_password", locked_once)
    assert api_keys.saved_key("openai") is None  # locked: not remembered
    assert api_keys.saved_key("openai") == "sk-saved"


def test_a_refused_removal_is_reported(monkeypatch, memory_keyring):
    api_keys.save_key("anthropic", "sk-ant-saved")

    def refuse(service, name):
        raise RuntimeError("access denied")

    monkeypatch.setattr(memory_keyring, "delete_password", refuse)
    with pytest.raises(RuntimeError, match="Couldn't remove the saved key"):
        api_keys.delete_saved_key("anthropic")


def test_plaintext_keyrings_are_not_offered(real_keyring_lookup, monkeypatch):
    keyring = pytest.importorskip("keyring")

    class Plaintext:  # what keyrings.alt's file backend looks like
        pass

    Plaintext.__module__ = "keyrings.alt.file"
    monkeypatch.setattr(keyring, "get_keyring", lambda: Plaintext())
    assert not api_keys.keyring_available()


def test_masked_short_keys_show_little():
    assert api_keys.masked("abcdefghijklm") == "…jklm"
    assert api_keys.masked("sk-ant-api03-abcdefghijklmnop1234") == "sk-ant-…1234"


def test_cloud_batches_ignore_the_character_budget():
    from batchtextreview import engine

    calls: list[int] = []

    class Counting(_Fake):
        def generate(self, prompts, max_new_tokens):
            calls.append(len(prompts))
            return ["x"] * len(prompts)

    class Manager:
        import contextlib

        @contextlib.contextmanager
        def acquire(self, model_id, precision):
            yield Counting(model_id, None)

    spec = models.find_model("openai:gpt-6-luna")
    assert spec is not None
    config = engine.RunConfig.build(spec, "Q: {text}", 4)
    texts = [f"{i} " + "y" * 3000 for i in range(10)]
    engine.execute_run(config, texts, store=None, manager=Manager(), batch_size=8)
    assert calls == [8, 2]  # 24,000 characters in a batch: fine for an API
