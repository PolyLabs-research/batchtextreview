"""FakeBackend, demo_responder and ModelManager (no torch needed)."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from batchtextreview.backends import Backend, GenerationError, ModelLoadError
from batchtextreview.backends import manager as manager_module
from batchtextreview.backends import transformers_backend as tb
from batchtextreview.backends.fake import FakeBackend, demo_responder
from batchtextreview.backends.manager import ModelManager, default_factory, default_manager
from batchtextreview.models import DEMO_REPO
from batchtextreview.presets import BUILTIN_PRESETS, get_preset
from batchtextreview.prompting import apply_variable, render_prompt

# --------------------------------------------------------------------------
# FakeBackend
# --------------------------------------------------------------------------


def test_fake_backend_defaults_and_protocol():
    backend = FakeBackend()
    assert isinstance(backend, Backend)
    assert backend.model_id == DEMO_REPO
    assert backend.precision == "none"
    assert backend.calls == []
    assert backend.closed is False


def test_fake_backend_responder_and_calls():
    backend = FakeBackend(responder=str.upper)
    assert backend.generate(["ab", "cd"], 8) == ["AB", "CD"]
    assert backend.generate(("ef",), 8) == ["EF"]
    assert backend.calls == [["ab", "cd"], ["ef"]]
    assert all(type(batch) is list for batch in backend.calls)


def test_fake_backend_empty_batch():
    backend = FakeBackend()
    assert backend.generate([], 8) == []
    assert backend.calls == [[]]


def test_fake_backend_default_responder_is_demo_responder():
    prompt = render_prompt(get_preset("Sentiment").template, "Excellent service, thank you!")
    assert FakeBackend().generate([prompt], 8) == [demo_responder(prompt)]


def test_fake_backend_delay():
    backend = FakeBackend(responder=str.lower, delay_s=0.05)
    start = time.perf_counter()
    backend.generate(["A"], 4)
    assert time.perf_counter() - start >= 0.045


def test_fake_backend_fail_when_records_then_raises():
    backend = FakeBackend(responder=str.lower, fail_when=lambda batch: "boom" in batch)
    assert backend.generate(["ok"], 4) == ["ok"]
    with pytest.raises(RuntimeError, match="fake failure"):
        backend.generate(["fine", "boom"], 4)
    assert backend.calls == [["ok"], ["fine", "boom"]]
    # A failure doesn't poison the backend.
    assert backend.generate(["again"], 4) == ["again"]


def test_fake_backend_reports_outputs_that_reach_the_limit():
    backend = FakeBackend(responder=lambda prompt: prompt)
    assert backend.last_hit_limit == []
    outputs = backend.generate(["one two three", "one two", "", "a b c d"], 3)
    assert outputs == ["one two three", "one two", "", "a b c d"]
    assert backend.last_hit_limit == [True, False, False, True]
    backend.generate(["one two three"], 8)
    assert backend.last_hit_limit == [False]
    backend.generate([], 3)
    assert backend.last_hit_limit == []


def test_fake_backend_hit_limit_is_reset_by_a_failed_call():
    backend = FakeBackend(responder=str.lower, fail_when=lambda batch: "boom" in batch)
    backend.generate(["a b"], 1)
    assert backend.last_hit_limit == [True]
    with pytest.raises(RuntimeError, match="fake failure"):
        backend.generate(["boom"], 1)
    assert backend.last_hit_limit == []
    backend.generate(["a b"], 1)
    backend.close()
    with pytest.raises(GenerationError):
        backend.generate(["a b"], 1)
    assert backend.last_hit_limit == []


def test_fake_backend_close_makes_it_unusable():
    backend = FakeBackend()
    backend.close()
    assert backend.closed is True
    with pytest.raises(GenerationError):
        backend.generate(["x"], 4)
    backend.close()  # idempotent


# --------------------------------------------------------------------------
# demo_responder
# --------------------------------------------------------------------------

SAMPLE = (
    "This is unacceptable. My claim LIB-40517 for a damaged book has gone missing twice and "
    "nobody answers my calls. Contact Mrs Jane Smith on 07700 900123 or jane.smith@example.com "
    "before 3 March 2025."
)

VARS = {
    "Yes/No flag (set {var})": "Does the text describe a service failure?",
    "Categorise (set {var})": "payment, claim, address change, other",
    "Extract field (set {var})": "the claim number",
    "Aspect sentiment (set {var})": "claim, calls, price",
}


def _ask(key: str, text: str, var: str | None = None) -> str:
    preset = get_preset(key)
    assert preset is not None
    final = apply_variable(preset.template, var if var is not None else VARS.get(key))
    return demo_responder(render_prompt(final, text))


@pytest.mark.parametrize("preset", BUILTIN_PRESETS, ids=lambda p: p.key)
def test_demo_responder_answers_every_builtin_preset(preset):
    var = VARS.get(preset.key)
    if preset.needs_var:
        assert var, f"add a sample variable for {preset.key!r}"
    prompt = render_prompt(apply_variable(preset.template, var), SAMPLE)

    answer = demo_responder(prompt)
    assert isinstance(answer, str)
    assert answer.strip()
    assert answer == demo_responder(prompt)  # deterministic

    allowed = preset.resolve_labels(var)
    if preset.output_kind == "label":
        assert allowed
        assert answer.lower() in {label.lower() for label in allowed}
    elif preset.output_kind == "json":
        assert isinstance(json.loads(answer), dict)


def test_demo_responder_label_presets_on_sample():
    assert _ask("Sentiment", SAMPLE) == "negative"
    assert _ask("Emotion", SAMPLE) == "anger"
    assert _ask("Intent", SAMPLE) == "complaint"
    assert _ask("Yes/No flag (set {var})", SAMPLE) == "Yes"
    assert _ask("Categorise (set {var})", SAMPLE) == "claim"
    assert int(_ask("Urgency (1-5)", SAMPLE)) >= 3


@pytest.mark.parametrize(
    ("text", "sentiment", "intent"),
    [
        ("Thank you so much, the staff were excellent and very helpful.", "positive", "praise"),
        ("Could you tell me when the summer reading challenge closes?", "neutral", "query"),
        ("Please update the email address on my library account.", "neutral", "request"),
        ("Attached is the signed room booking form.", "neutral", "other"),
        ("Thanks for the quick reply, but my fine is still wrong.", "mixed", "complaint"),
        ("I am not happy with this.", "negative", "complaint"),
    ],
)
def test_demo_responder_sentiment_and_intent(text, sentiment, intent):
    assert _ask("Sentiment", text) == sentiment
    assert _ask("Intent", text) == intent


def test_demo_responder_language():
    assert _ask("Language", "The weather is nice and I would like to visit.") == "English"
    assert (
        _ask("Language", "Bonjour, je vous remercie pour votre aide avec le dossier.") == "French"
    )
    assert _ask("Language", "Привет, как дела?") == "Russian"
    assert _ask("Language", "Diolch yn fawr am eich help gyda'r cais.") == "Welsh"


def test_demo_responder_categorise_uses_the_variable():
    var = "invoice, contract, other"
    assert _ask("Categorise (set {var})", "Invoice 443 is overdue.", var) == "invoice"
    assert _ask("Categorise (set {var})", "Hello there.", var) == "other"
    # No "other" bucket: still one of the categories, picked stably.
    no_other = "red, green, blue"
    answer = _ask("Categorise (set {var})", "Hello there.", no_other)
    assert answer in {"red", "green", "blue"}
    assert answer == _ask("Categorise (set {var})", "Hello there.", no_other)
    # Multi-word categories are matched on their words.
    multi = "payment, address change, other"
    assert _ask("Categorise (set {var})", "Please change my address.", multi) == "address change"


def test_demo_responder_json_tasks():
    entities = json.loads(_ask("Entity extraction", SAMPLE))
    assert set(entities) == {"people", "organisations", "locations", "dates"}
    assert "Mrs Jane Smith" in entities["people"]
    assert "3 March 2025" in entities["dates"]

    pii = json.loads(_ask("PII detection", SAMPLE))["pii"]
    assert "jane.smith@example.com" in pii
    assert "07700 900123" in pii
    assert "Mrs Jane Smith" in pii
    assert json.loads(_ask("PII detection", "Nothing personal here.")) == {"pii": []}

    aspects = json.loads(_ask("Aspect sentiment (set {var})", SAMPLE))
    assert aspects == {"claim": "negative", "calls": "negative", "price": "not_mentioned"}


def test_demo_responder_text_tasks():
    redacted = _ask("PII redaction", SAMPLE)
    assert "[REDACTED]" in redacted
    assert "jane.smith@example.com" not in redacted
    assert "07700 900123" not in redacted
    assert redacted.startswith("This is unacceptable.")

    assert _ask("Extract field (set {var})", SAMPLE) == "LIB-40517"
    assert _ask("Extract field (set {var})", "Nothing to see.") == "NONE"
    assert _ask("Extract field (set {var})", "We paid £1,250.00 last week.", "the amount") == (
        "£1,250.00"
    )

    keywords = _ask("Keyword extraction", SAMPLE).split(", ")
    assert 3 <= len(keywords) <= 5
    assert "claim" in keywords

    topics = _ask("Topic analysis", SAMPLE).split(", ")
    assert 1 <= len(topics) <= 3
    assert "claims" in topics

    summary = _ask("One-line summary", SAMPLE)
    assert summary == "This is unacceptable."
    long_text = " ".join(f"word{i}" for i in range(40)) + "."
    assert len(_ask("One-line summary", long_text).split()) == 20


def test_demo_responder_unknown_prompt_gets_first_12_words():
    text = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen"
    assert demo_responder(text) == "one two three four five six seven eight nine ten eleven twelve"
    # The bare {text} template: row text that mentions a task word is not a task.
    assert demo_responder("I love the new language course") == "I love the new language course"
    assert demo_responder("") == ""


def test_demo_responder_custom_template_loose_cues():
    template = "What is the sentiment of this review? Answer in one word.\n\nTEXT:\n{text}"
    assert demo_responder(render_prompt(template, "Terrible, slow and rude.")) == "negative"
    # Text after the *last* marker is the text, even if the row contains one.
    tricky = render_prompt(get_preset("Sentiment").template, "TEXT:\nGreat stuff")
    assert demo_responder(tricky) == "positive"


def test_demo_responder_ignores_task_words_inside_the_text():
    prompt = render_prompt(get_preset("Sentiment").template, "Identify the language: wonderful!")
    assert demo_responder(prompt) == "positive"


# --------------------------------------------------------------------------
# ModelManager
# --------------------------------------------------------------------------


class RecordingBackend(FakeBackend):
    """A FakeBackend that writes load/close events into a shared log."""

    def __init__(self, model_id: str, precision: str, log: list[str]) -> None:
        super().__init__(model_id=model_id, precision=precision, responder=str.upper)
        self.log = log
        log.append(f"load {model_id}")

    def close(self) -> None:
        self.log.append(f"close {self.model_id}")
        super().close()


class Factory:
    """Counts loads; ``fail`` holds model ids whose load raises."""

    def __init__(self, delay_s: float = 0.0) -> None:
        self.log: list[str] = []
        self.fail: set[str] = set()
        self.loads = 0
        self.delay_s = delay_s
        self.built: list[RecordingBackend] = []

    def __call__(self, model_id: str, precision: str) -> Backend:
        self.loads += 1
        if self.delay_s:
            time.sleep(self.delay_s)
        if model_id in self.fail:
            self.log.append(f"fail {model_id}")
            raise ModelLoadError(f"cannot load {model_id}", hint="try another")
        backend = RecordingBackend(model_id, precision, self.log)
        self.built.append(backend)
        return backend


def test_manager_loads_lazily_and_reuses():
    factory = Factory()
    manager = ModelManager(factory)
    assert manager.loaded is None
    assert factory.loads == 0
    assert manager.busy is False

    with manager.acquire("a", "fp32") as backend:
        assert backend.generate(["x"], 4) == ["X"]
    with manager.acquire("a", "fp32") as again:
        assert again is backend
    assert factory.loads == 1
    assert manager.loaded == ("a", "fp32")


def test_manager_reloads_a_local_folder_whose_files_changed(tmp_path):
    """Regression: a long-lived server kept generating with the old weights
    of a fine-tune saved again into the same folder."""
    import os

    folder = tmp_path / "my-classifier"
    folder.mkdir()
    weights = folder / "model.safetensors"
    weights.write_text("v1", encoding="utf-8")
    os.utime(weights, ns=(1_700_000_000_000_000_000,) * 2)
    model_id = str(folder)

    factory = Factory()
    manager = ModelManager(factory)
    with manager.acquire(model_id, "fp32") as first:
        pass
    assert manager.is_loaded(model_id, "fp32")
    with manager.acquire(model_id, "fp32") as again:
        assert again is first
    assert factory.loads == 1

    weights.write_text("v2", encoding="utf-8")
    os.utime(weights, ns=(1_800_000_000_000_000_000,) * 2)
    assert manager.loaded == (model_id, "fp32")
    assert not manager.is_loaded(model_id, "fp32")
    with manager.acquire(model_id, "fp32") as reloaded:
        assert reloaded is not first
    assert factory.loads == 2
    assert factory.log == [f"load {model_id}", f"close {model_id}", f"load {model_id}"]
    assert manager.is_loaded(model_id, "fp32")


def test_manager_is_loaded():
    manager = ModelManager(Factory())
    assert not manager.is_loaded("a", "fp32")
    with manager.acquire("a", "fp32"):
        pass
    assert manager.is_loaded("a", "fp32")
    assert not manager.is_loaded("a", "fp16")
    assert not manager.is_loaded("b", "fp32")
    manager.unload()
    assert not manager.is_loaded("a", "fp32")


def test_manager_precision_is_part_of_the_key():
    factory = Factory()
    manager = ModelManager(factory)
    with manager.acquire("a", "fp32"):
        pass
    with manager.acquire("a", "fp16") as backend:
        assert backend.precision == "fp16"
    assert factory.log == ["load a", "close a", "load a"]
    assert manager.loaded == ("a", "fp16")


def test_manager_swap_closes_old_backend_before_loading_new(monkeypatch):
    factory = Factory()
    manager = ModelManager(factory)
    freed: list[str] = []
    monkeypatch.setattr(
        manager_module,
        "_free_accelerator_memory",
        lambda: freed.append(factory.log[-1]),
    )
    with manager.acquire("a", "fp32") as first:
        pass
    with manager.acquire("b", "fp32") as second:
        assert second is not first
    assert factory.log == ["load a", "close a", "load b"]
    assert first.closed and not second.closed
    # Memory is released after closing the old model, before the new load.
    assert freed == ["close a"]
    assert manager.loaded == ("b", "fp32")


def test_manager_failed_load_leaves_nothing_loaded_and_retries():
    """Regression: v1 kept the old label while its weights were already freed."""
    factory = Factory()
    manager = ModelManager(factory)
    with manager.acquire("good", "fp32") as good:
        pass

    factory.fail.add("bad")
    with pytest.raises(ModelLoadError, match="cannot load bad"), manager.acquire("bad", "fp32"):
        pytest.fail("the body must not run when loading fails")  # pragma: no cover
    assert manager.loaded is None
    assert good.closed
    assert manager.busy is False

    # The next acquire of the same model tries the load again.
    with pytest.raises(ModelLoadError), manager.acquire("bad", "fp32"):
        pass  # pragma: no cover
    assert factory.log.count("fail bad") == 2

    # ...and succeeds once the cause is fixed.
    factory.fail.clear()
    with manager.acquire("bad", "fp32") as fixed:
        assert fixed.generate(["ok"], 4) == ["OK"]
    assert manager.loaded == ("bad", "fp32")

    # Going back to the first model loads a fresh backend, never the closed one.
    with manager.acquire("good", "fp32") as reloaded:
        assert reloaded is not good
        assert not reloaded.closed
        assert reloaded.generate(["x"], 4) == ["X"]


def test_manager_failed_load_frees_memory(monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(manager_module, "_free_accelerator_memory", lambda: calls.append(1))
    factory = Factory()
    factory.fail.add("bad")
    manager = ModelManager(factory)
    with pytest.raises(ModelLoadError), manager.acquire("bad", "fp32"):
        pass  # pragma: no cover
    assert calls == [1]


def test_manager_close_error_does_not_block_the_swap():
    class BadClose(FakeBackend):
        def close(self) -> None:
            raise RuntimeError("close failed")

    built: list[str] = []

    def factory(model_id: str, precision: str) -> Backend:
        built.append(model_id)
        return BadClose(model_id=model_id) if model_id == "a" else FakeBackend(model_id=model_id)

    manager = ModelManager(factory)
    with manager.acquire("a", "fp32"):
        pass
    with manager.acquire("b", "fp32") as backend:
        assert backend.model_id == "b"
    assert built == ["a", "b"]
    assert manager.loaded == ("b", "fp32")


def test_manager_exception_in_body_keeps_model_and_releases_lock():
    factory = Factory()
    manager = ModelManager(factory)
    with pytest.raises(ValueError), manager.acquire("a", "fp32"):
        raise ValueError("generation blew up")
    assert manager.busy is False
    assert manager.loaded == ("a", "fp32")
    with manager.acquire("a", "fp32"):
        pass
    assert factory.loads == 1


def test_manager_busy_flag():
    manager = ModelManager(Factory())
    assert manager.busy is False
    with manager.acquire("a", "fp32"):
        assert manager.busy is True
    assert manager.busy is False


def test_manager_lock_serialises_threads():
    factory = Factory()
    manager = ModelManager(factory)
    entered = threading.Event()
    release = threading.Event()
    order: list[str] = []

    def first() -> None:
        with manager.acquire("a", "fp32"):
            order.append("first in")
            entered.set()
            release.wait(5)
            order.append("first out")

    def second() -> None:
        with manager.acquire("b", "fp32") as backend:
            order.append("second in")
            backend.generate(["x"], 4)

    t1 = threading.Thread(target=first)
    t1.start()
    assert entered.wait(5)
    t2 = threading.Thread(target=second)
    t2.start()
    time.sleep(0.1)
    # The second thread is blocked: it neither ran nor swapped the model.
    assert order == ["first in"]
    assert manager.busy is True
    assert manager.loaded == ("a", "fp32")
    assert factory.log == ["load a"]

    release.set()
    t1.join(5)
    t2.join(5)
    assert order == ["first in", "first out", "second in"]
    assert factory.log == ["load a", "close a", "load b"]
    assert manager.busy is False


def test_manager_concurrent_requests_for_one_model_load_it_once():
    factory = Factory(delay_s=0.05)
    manager = ModelManager(factory)
    results: list[Backend] = []

    def worker() -> None:
        with manager.acquire("a", "fp32") as backend:
            results.append(backend)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
    assert factory.loads == 1
    assert len(results) == 4
    assert all(r is results[0] for r in results)


def test_manager_unload():
    factory = Factory()
    manager = ModelManager(factory)
    assert manager.unload() is False
    with manager.acquire("a", "fp32") as backend:
        pass
    assert manager.unload() is True
    assert backend.closed
    assert manager.loaded is None
    assert manager.unload() is False
    with manager.acquire("a", "fp32") as fresh:
        assert fresh is not backend
    assert factory.loads == 2


def test_manager_unload_waits_for_the_lock():
    manager = ModelManager(Factory())
    entered = threading.Event()
    release = threading.Event()
    done: list[bool] = []

    def holder() -> None:
        with manager.acquire("a", "fp32"):
            entered.set()
            release.wait(5)

    t1 = threading.Thread(target=holder)
    t1.start()
    assert entered.wait(5)
    t2 = threading.Thread(target=lambda: done.append(manager.unload()))
    t2.start()
    time.sleep(0.05)
    assert done == []
    assert manager.loaded == ("a", "fp32")
    release.set()
    t1.join(5)
    t2.join(5)
    assert done == [True]
    assert manager.loaded is None


def test_default_factory_demo_model():
    backend = default_factory(DEMO_REPO, "none")
    assert isinstance(backend, FakeBackend)
    assert backend.model_id == DEMO_REPO
    assert backend.precision == "none"


def test_default_factory_demo_model_does_not_keep_prompts():
    # The manager holds the demo backend for the life of the server; recording
    # every prompt of every run would grow without bound.
    backend = default_factory(DEMO_REPO, "none")
    assert backend.generate(["Classify the sentiment.\nTEXT:\ngreat"], 4) == ["positive"]
    assert backend.calls == []
    recording = FakeBackend()
    recording.generate(["x"], 4)
    assert recording.calls == [["x"]]


def test_default_factory_real_model_uses_transformers_backend(monkeypatch):
    from batchtextreview.backends import transformers_backend

    seen: list[tuple[str, str]] = []

    class Stub:
        def __init__(self, model_id: str, precision: str) -> None:
            seen.append((model_id, precision))
            self.model_id, self.precision = model_id, precision

    monkeypatch.setattr(transformers_backend, "TransformersBackend", Stub)
    backend = default_factory("org/model", "fp16")
    assert isinstance(backend, Stub)
    assert seen == [("org/model", "fp16")]


def test_default_manager_is_a_singleton():
    assert default_manager() is default_manager()
    assert isinstance(default_manager(), ModelManager)


def test_importing_backends_does_not_import_torch():
    code = (
        "import sys\n"
        "import batchtextreview.backends.manager as m\n"
        "import batchtextreview.backends.fake\n"
        "import batchtextreview.backends.transformers_backend\n"
        "with m.ModelManager().acquire('demo://fake', 'none') as b:\n"
        "    b.generate(['hello'], 4)\n"
        "m._free_accelerator_memory()\n"
        "assert 'torch' not in sys.modules, 'torch was imported'\n"
        "assert 'transformers' not in sys.modules, 'transformers was imported'\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr


# --------------------------------------------------------------------------
# transformers_backend helpers that need no torch
# --------------------------------------------------------------------------


def _named(name: str, base: type[BaseException] = Exception) -> type[BaseException]:
    """An exception class with a given name, like huggingface_hub's."""
    return type(name, (base,), {})


def _caused(outer: BaseException, inner: BaseException) -> BaseException:
    outer.__cause__ = inner
    return outer


@pytest.mark.parametrize(
    ("exc", "message", "hint"),
    [
        (
            _caused(
                OSError("You are trying to access a gated repo."), _named("GatedRepoError")("401")
            ),
            "gated",
            "HF_TOKEN",
        ),
        (
            OSError("401 Client Error: Unauthorized for url"),
            "refused",
            "HF_TOKEN",
        ),
        (
            # A typo in the repo id: huggingface_hub's text mentions gated
            # repos and the HTTP layer says 401 Unauthorized - still "not found".
            _caused(
                OSError("org/x is not a local folder and is not a valid model identifier"),
                _caused(
                    _named("RepositoryNotFoundError", OSError)(
                        "401 Client Error. Repository Not Found for url: https://huggingface.co/"
                        "org/x/resolve/main/config.json. If you are trying to access a private "
                        "or gated repo, make sure you are authenticated."
                    ),
                    _named("HTTPStatusError")("Client error '401 Unauthorized' for url"),
                ),
            ),
            "not found",
            "repo id",
        ),
        (
            # A proxy / firewall block page: no gated-repo or not-found code.
            _caused(
                _named("HfHubHTTPError", OSError)("403 Forbidden: None. Cannot access content"),
                _named("HTTPStatusError")("Client error '403 Forbidden' for url"),
            ),
            "refused",
            "proxy",
        ),
        (
            ValueError(
                "Unrecognized configuration class <class 'transformers.models.t5."
                "configuration_t5.T5Config'> for this kind of AutoModel: AutoModelForCausalLM."
            ),
            "T5Config) isn't a text-generation",
            "instruct",
        ),
        (
            OSError("org/x is not a local folder and is not a valid model identifier listed on"),
            "not found",
            "repo id",
        ),
        (_named("RepositoryNotFoundError", OSError)("404"), "not found", "repo id"),
        (
            _caused(OSError("There was a specific connection error"), _named("SSLError")("x")),
            "Couldn't download",
            "proxy",
        ),
        (
            OSError("We couldn't connect to 'https://huggingface.co' to load the files"),
            "Couldn't download",
            "truststore",
        ),
        (
            _named("OutOfMemoryError", RuntimeError)("CUDA out of memory"),
            "GPU memory",
            "smaller model",
        ),
        (
            ValueError("Some modules are dispatched on the CPU or the disk. Make sure you have"),
            "GPU memory",
            "smaller model",
        ),
        (
            ModuleNotFoundError("No module named 'accelerate'", name="accelerate"),
            "missing",
            "pip install -U accelerate",
        ),
        (
            ImportError("LlamaTokenizer requires the SentencePiece library but it was not found"),
            "missing",
            "pip install -U sentencepiece",
        ),
        (
            ValueError(
                "The checkpoint has model type `qwen9` but Transformers does not recognize this architecture."
            ),
            "can't load",
            "pip install -U transformers",
        ),
        (
            ValueError(
                "Loading this model requires you to execute custom code; set trust_remote_code=True"
            ),
            "custom code",
            "transformers",
        ),
    ],
)
def test_load_errors_are_mapped_to_hints(exc, message, hint):
    err = tb._load_error(exc, "org/x", "4bit")
    assert isinstance(err, ModelLoadError)
    assert message.lower() in str(err).lower()
    assert hint.lower() in (err.hint or "").lower()


def test_load_error_architecture_upgrade_hint_is_kept_for_unknown_types():
    exc = ValueError("Transformers does not recognize this architecture.")
    err = tb._load_error(exc, "org/x", "fp16")
    assert "pip install -U transformers" in (err.hint or "")
    t5 = ValueError("Unrecognized configuration class T5Config for this kind of AutoModel: X")
    assert "pip install" not in (tb._load_error(t5, "org/x", "fp16").hint or "")


@pytest.mark.parametrize(
    ("precision", "suggests_4bit", "suggests_fp16"),
    [("fp16", True, False), ("4bit", False, False), ("fp32", True, True)],
)
def test_load_error_oom_hint_depends_on_precision(precision, suggests_4bit, suggests_fp16):
    exc = _named("OutOfMemoryError", RuntimeError)("CUDA out of memory")
    err = tb._load_error(exc, "org/x", precision)
    assert ("4-bit" in (err.hint or "")) is suggests_4bit
    assert ("fp16" in (err.hint or "")) is suggests_fp16
    assert "smaller model" in (err.hint or "")


def test_offloaded_modules_from_the_device_map():
    from types import SimpleNamespace

    def model(device_map):
        return SimpleNamespace(hf_device_map=device_map)

    assert tb._offloaded(model({"": 0})) == []
    assert tb._offloaded(model({"a": 0, "b": "cuda:0", "c": SimpleNamespace(type="cuda")})) == []
    assert tb._offloaded(
        model({"a": 0, "b": "cpu", "c": "disk", "d": SimpleNamespace(type="cpu")})
    ) == [("b", "cpu"), ("c", "disk"), ("d", "cpu")]
    assert tb._offloaded(SimpleNamespace()) == []  # loaded without a device map
    assert tb._offloaded(model(None)) == []


@pytest.mark.parametrize(
    ("architectures", "encoder_decoder", "refused"),
    [
        (["LlamaForCausalLM"], False, None),
        (["GPT2LMHeadModel"], False, None),
        (["ReformerModelWithLMHead"], False, None),
        (["Gemma3ForConditionalGeneration"], False, None),
        ([], False, None),
        (["XLMRobertaForSequenceClassification"], False, "XLMRobertaForSequenceClassification"),
        (["ElectraForPreTraining"], False, "ElectraForPreTraining"),
        (["BartForConditionalGeneration"], True, "BartForConditionalGeneration"),
    ],
)
def test_not_text_generation_without_torch(architectures, encoder_decoder, refused):
    from types import SimpleNamespace

    config = SimpleNamespace(architectures=architectures, is_encoder_decoder=encoder_decoder)
    assert tb._not_text_generation(config) == refused


def test_chat_templates_get_one_fixed_day():
    assert tb._fixed_strftime("%d %b %Y") == tb._CHAT_DATE
    assert tb._fixed_strftime("%B %d, %Y") == "July 26, 2024"  # Granite 3.x's format


def _hub_error(status: int, error_code: str | None) -> BaseException:
    """What transformers raises for a Hub reply, built by huggingface_hub itself."""
    hub_utils = pytest.importorskip("huggingface_hub.utils")
    http = None
    for name in ("httpx2", "httpx"):
        try:
            http = __import__(name)
            break
        except ImportError:
            continue
    if http is None:
        pytest.skip("no httpx-style client for huggingface_hub")
    url = "https://huggingface.co/org/x/resolve/main/config.json"
    headers = {"X-Error-Code": error_code} if error_code else {}
    response = http.Response(status, headers=headers, request=http.Request("GET", url))
    try:
        try:
            hub_utils.hf_raise_for_status(response)
        except Exception as hub_exc:
            raise OSError("org/x: transformers could not fetch config.json") from hub_exc
    except OSError as exc:
        return exc
    pytest.fail(f"hf_raise_for_status accepted HTTP {status}")


@pytest.mark.parametrize(
    ("status", "error_code", "message", "hint"),
    [
        (404, "RepoNotFound", "not found", "repo id"),
        (401, "RepoNotFound", "not found", "repo id"),  # what the Hub sends anonymously
        (401, "GatedRepo", "gated", "accept the licence"),
        (403, "GatedRepo", "gated", "accept the licence"),
        (403, None, "refused", "proxy"),
    ],
)
def test_load_error_maps_real_hub_errors(status, error_code, message, hint):
    err = tb._load_error(_hub_error(status, error_code), "org/x", "4bit")
    assert message in str(err).lower()
    assert hint.lower() in (err.hint or "").lower()
    if status == 403 and error_code is None:
        assert "(HTTP 403)" in str(err)


def test_load_error_offline_mode(monkeypatch):
    monkeypatch.setattr(tb, "_offline_mode", lambda: True)
    exc = OSError("We couldn't connect to 'https://huggingface.co' to load the files")
    err = tb._load_error(exc, "org/x", "fp32")
    assert "offline" in str(err)
    assert "HF_HUB_OFFLINE" in (err.hint or "")


def test_load_error_unknown_keeps_the_detail(monkeypatch):
    monkeypatch.setattr(tb, "current_log_file", lambda: None)
    err = tb._load_error(KeyError("weird"), "org/x", "fp32")
    assert "KeyError" in str(err)
    # Regression: an unrecognised error came with no next step at all.
    assert err.hint == "Check the model name and precision, or choose another model."
    long = tb._load_error(RuntimeError("x" * 5000), "org/x", "fp32")
    assert len(str(long)) < 400


def test_load_error_unknown_points_at_the_log_file(monkeypatch, tmp_path):
    log_file = tmp_path / "logs" / "batchtextreview.log"
    monkeypatch.setattr(tb, "current_log_file", lambda: log_file)
    err = tb._load_error(KeyError("weird"), "org/x", "fp32")
    assert f"The full message is in {log_file}." in (err.hint or "")


def _cpu_allocator_error() -> RuntimeError:
    """The RuntimeError torch's CPU allocator really raises (Linux wording)."""
    torch = pytest.importorskip("torch")
    try:
        torch.empty((10**13,))  # 40 TB
    except RuntimeError as exc:
        return exc
    pytest.skip("this machine allocated 40 TB")


@pytest.mark.parametrize(
    "make_exc",
    [
        _cpu_allocator_error,
        MemoryError,
        lambda: RuntimeError(  # torch on Windows
            "[enforce fail at alloc_cpu.cpp:114] data. DefaultCPUAllocator: not enough memory: "
            "you tried to allocate 30000000000 bytes."
        ),
        lambda: OSError(12, "Cannot allocate memory"),  # mapping the weight files
        lambda: _caused(OSError("loading failed"), MemoryError()),
    ],
    ids=["torch-linux", "MemoryError", "torch-windows", "ENOMEM", "chained"],
)
def test_load_error_running_out_of_ram_on_the_cpu(make_exc):
    """Regression: these fell through to "Could not load X: RuntimeError:
    [enforce fail at alloc_cpu.cpp ...]" with no hint."""
    err = tb._load_error(make_exc(), "Qwen/Qwen2.5-7B-Instruct", "fp32", "cpu")
    text = str(err)
    assert text.startswith(
        "Not enough memory (RAM) to load Qwen/Qwen2.5-7B-Instruct at fp32 on the CPU: its 7.6B "
        "parameters need about 30 GB at 4 bytes each."
    )
    assert "GPU" not in text and "enforce fail" not in text
    assert (
        err.hint
        == "Choose a smaller model (a 1.5B one needs about 6 GB at fp32), and close other programs."
    )


def test_load_error_ram_message_without_a_known_size():
    err = tb._load_error(MemoryError(), "/models/mine", "fp32", "cpu")
    assert str(err).startswith(
        "Not enough memory (RAM) to load /models/mine at fp32 on the CPU: the weights take "
        "about 4 bytes per parameter."
    )
    assert "about 6 GB at fp32" in (err.hint or "")


def test_load_error_ram_on_a_gpu_machine_is_not_a_gpu_error():
    """With CUDA the weights still pass through RAM; the allocator error
    then means RAM, not GPU memory, ran out."""
    exc = RuntimeError("DefaultCPUAllocator: can't allocate memory: you tried to allocate 1 bytes")
    err = tb._load_error(exc, "org/x", "4bit", "cuda")
    assert str(err).startswith("Not enough memory (RAM) to load org/x at 4bit:")
    assert "on the CPU" not in str(err)
    assert err.hint == "Choose a smaller model, and close other programs."


@pytest.mark.parametrize("device", ["cuda", "mps", None])
def test_load_error_gpu_out_of_memory_stays_a_gpu_error(device):
    err = tb._load_error(RuntimeError("CUDA out of memory."), "org/x", "fp16", device)
    assert str(err).startswith("Not enough GPU memory")


def test_load_error_out_of_memory_on_the_cpu_is_ram():
    err = tb._load_error(RuntimeError("out of memory"), "org/x", "fp32", "cpu")
    assert str(err).startswith("Not enough memory (RAM)")


def test_drop_tracebacks_clears_the_whole_chain():
    try:
        try:
            raise ValueError("inner")
        except ValueError as inner:
            raise RuntimeError("outer") from inner
    except RuntimeError as outer:
        caught = outer
    assert caught.__traceback__ is not None
    tb._drop_tracebacks(caught)
    assert caught.__traceback__ is None
    assert caught.__cause__.__traceback__ is None


@pytest.mark.parametrize(
    ("precision", "device", "expected"),
    [
        ("fp32", "cpu", "fp32"),
        ("fp32", "cuda", "fp32"),
        ("fp16", "cuda", "fp16"),
        ("bf16", "mps", "bf16"),
        ("fp16", "mps", "fp16"),
        ("auto", "cpu", "fp32"),
        ("auto", "mps", "fp16"),
        ("FP32", "cpu", "fp32"),
    ],
)
def test_check_precision_accepts(precision, device, expected):
    assert tb._check_precision(precision, device) == expected


@pytest.mark.parametrize(
    ("precision", "device", "hint"),
    [
        ("4bit", "cpu", "fp32"),
        ("4bit", "mps", "fp16"),
        ("fp16", "cpu", "fp32"),
        ("bf16", "cpu", "fp32"),
        ("none", "cpu", "fp32"),
        ("int8", "cuda", "4bit"),
    ],
)
def test_check_precision_refuses_with_hint(precision, device, hint):
    with pytest.raises(ModelLoadError) as info:
        tb._check_precision(precision, device)
    assert hint in (info.value.hint or "")


def test_check_precision_4bit_needs_bitsandbytes(monkeypatch):
    monkeypatch.setitem(sys.modules, "bitsandbytes", None)
    with pytest.raises(ModelLoadError) as info:
        tb._check_precision("4bit", "cuda")
    assert "pip install -U bitsandbytes" in (info.value.hint or "")
    assert tb._check_precision("auto", "cuda") == "fp16"


@pytest.mark.parametrize(
    ("value", "is_path"),
    [
        ("Qwen/Qwen2.5-1.5B-Instruct", False),
        ("gpt2", False),
        ("/opt/models/qwen", True),
        ("~/models/qwen", True),
        ("./qwen", True),
        ("../qwen", True),
        ("C:\\models\\qwen", True),
        ("D:/models/qwen", True),
        ("models\\qwen", True),
    ],
)
def test_looks_like_path(value, is_path):
    assert tb._looks_like_path(value) is is_path


def test_model_source(tmp_path):
    assert tb._model_source("Qwen/Qwen2.5-1.5B-Instruct") == "Qwen/Qwen2.5-1.5B-Instruct"
    with pytest.raises(ModelLoadError, match="not found"):
        tb._model_source(str(tmp_path / "missing"))
    (tmp_path / "empty").mkdir()
    with pytest.raises(ModelLoadError, match=r"config\.json"):
        tb._model_source(str(tmp_path / "empty"))
    (tmp_path / "ok").mkdir()
    (tmp_path / "ok" / "config.json").write_text("{}")
    # An incomplete copy (weights but no tokenizer) fails fast, rather than
    # transformers blaming a missing sentencepiece package.
    with pytest.raises(ModelLoadError, match="no tokenizer files") as info:
        tb._model_source(str(tmp_path / "ok"))
    assert "tokenizer.json" in (info.value.hint or "")
    (tmp_path / "ok" / "tokenizer.json").write_text("{}")
    assert tb._model_source(f"  {tmp_path / 'ok'}  ") == str(tmp_path / "ok")


@pytest.mark.parametrize(
    "files",
    [
        ["tokenizer_config.json"],
        ["tokenizer.model"],
        ["vocab.json", "merges.txt"],
        ["vocab.txt"],
        ["spiece.model"],
        ["tekken.json"],
        ["qwen.tiktoken"],
    ],
)
def test_model_source_accepts_any_tokenizer_format(tmp_path, files):
    (tmp_path / "config.json").write_text("{}")
    for name in files:
        (tmp_path / name).write_text("")
    assert tb._model_source(str(tmp_path)) == str(tmp_path)


class _FakeAuto:
    """Stands in for an AutoModel class; records kwargs, raises on demand."""

    def __init__(self, behaviour):
        self.behaviour = behaviour
        self.calls: list[dict] = []

    def from_pretrained(self, source, **kwargs):
        self.calls.append(kwargs)
        return self.behaviour(kwargs)


def test_from_pretrained_falls_back_to_torch_dtype():
    def behaviour(kwargs):
        if "dtype" in kwargs:
            raise TypeError("__init__() got an unexpected keyword argument 'dtype'")
        return "model"

    auto = _FakeAuto(behaviour)
    assert tb._from_pretrained_dtype(auto, "x", {"dtype": "f16", "device_map": "auto"}) == "model"
    assert auto.calls == [
        {"dtype": "f16", "device_map": "auto"},
        {"torch_dtype": "f16", "device_map": "auto"},
    ]


def test_from_pretrained_retries_image_text_class_on_value_error():
    def unrecognised(kwargs):
        raise ValueError("Unrecognized configuration class Qwen3_5Config for AutoModelForCausalLM")

    class Transformers:
        AutoModelForCausalLM = _FakeAuto(unrecognised)
        AutoModelForImageTextToText = _FakeAuto(lambda kwargs: "multimodal")

    assert tb._from_pretrained(Transformers, "x", {"dtype": 1}) == "multimodal"


def test_from_pretrained_reports_the_first_error_when_both_fail():
    def first(kwargs):
        raise ValueError("first problem")

    def second(kwargs):
        raise ValueError("second problem")

    class Transformers:
        AutoModelForCausalLM = _FakeAuto(first)
        AutoModelForImageTextToText = _FakeAuto(second)

    with pytest.raises(ValueError, match="first problem"):
        tb._from_pretrained(Transformers, "x", {})


def test_from_pretrained_skips_pointless_retry():
    def unknown(kwargs):
        raise ValueError("model type `x` but Transformers does not recognize this architecture")

    class Transformers:
        AutoModelForCausalLM = _FakeAuto(unknown)
        AutoModelForImageTextToText = _FakeAuto(lambda kwargs: pytest.fail("retried"))

    with pytest.raises(ValueError, match="does not recognize"):
        tb._from_pretrained(Transformers, "x", {})


def test_accepts_dtype_kwarg():
    assert tb._accepts_dtype_kwarg(type("T", (), {"__version__": "4.56.0"}))
    assert tb._accepts_dtype_kwarg(type("T", (), {"__version__": "5.19.0"}))
    assert not tb._accepts_dtype_kwarg(type("T", (), {"__version__": "4.55.2"}))
    assert tb._accepts_dtype_kwarg(type("T", (), {"__version__": "dev"}))


def test_is_oom():
    class FakeTorch:
        class cuda:
            OutOfMemoryError = _named("OutOfMemoryError", RuntimeError)

    assert tb._is_oom(FakeTorch.cuda.OutOfMemoryError("x"), FakeTorch)
    assert tb._is_oom(RuntimeError("CUDA error: out of memory"), FakeTorch)
    assert not tb._is_oom(RuntimeError("something else"), FakeTorch)
    assert not tb._is_oom(ValueError("out of memory"), FakeTorch)
    assert tb._is_oom(RuntimeError("out of memory"), None)


def test_stop_ids_cover_every_end_token():
    tokenizer = SimpleNamespace(eos_token_id=2, pad_token_id=0)
    llama3 = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[128001, 128009]))
    assert tb._stop_ids(tokenizer, llama3) == {0, 2, 128001, 128009}
    single = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=2))
    blank = SimpleNamespace(eos_token_id=None, pad_token_id=None)
    assert tb._stop_ids(blank, single) == {2}
    assert tb._stop_ids(tokenizer, SimpleNamespace()) == {0, 2}
    assert tb._stop_ids(blank, SimpleNamespace(generation_config=None)) == set()
