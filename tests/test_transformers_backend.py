"""TransformersBackend against tiny random Llama models built on the fly.

No download ever happens: :mod:`tiny_model` writes the checkpoints into a
temporary folder. Every model is loaded on the CPU in fp32, even on a
machine with a GPU, so results are deterministic.
"""

from __future__ import annotations

import contextlib
import gc
import logging
import re
import shutil
import sys
import time
import weakref
from collections.abc import Iterator
from pathlib import Path

import pytest

from batchtextreview.backends import GenerationError, ModelLoadError
from batchtextreview.backends import transformers_backend as tb
from batchtextreview.backends.transformers_backend import TransformersBackend
from tiny_model import CHAT_TEMPLATE, build_tiny_model, build_tokenizer

pytestmark = pytest.mark.llm

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

PROMPTS = [
    "hello",
    "classify the sentiment of this text",
    "please update my address with the new one today thank you very much",
    "is the service good or bad ?",
    "a short test",
]


@contextlib.contextmanager
def cpu_only() -> Iterator[None]:
    """Make the backend pick the CPU even where CUDA / MPS exist."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(torch.cuda, "is_available", lambda: False)
        mps = getattr(torch.backends, "mps", None)
        if mps is not None:
            mp.setattr(mps, "is_available", lambda: False)
        yield


def load(path: Path | str, precision: str = "fp32") -> TransformersBackend:
    with cpu_only():
        return TransformersBackend(str(path), precision)


@pytest.fixture(scope="session")
def model_dirs(tmp_path_factory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("tiny-models")
    return {
        "chat": build_tiny_model(root / "chat", seed=0),
        "plain": build_tiny_model(root / "plain", chat_template=False, seed=1),
        "nopad": build_tiny_model(root / "nopad", pad_token=False, seed=2),
    }


@pytest.fixture(scope="session")
def backends(model_dirs) -> Iterator[dict[str, TransformersBackend]]:
    loaded = {name: load(path) for name, path in model_dirs.items()}
    yield loaded
    for backend in loaded.values():
        backend.close()


@pytest.fixture
def chat(backends) -> TransformersBackend:
    return backends["chat"]


@contextlib.contextmanager
def capture_generate(backend: TransformersBackend) -> Iterator[list[dict]]:
    """Record the kwargs and output of every ``model.generate`` call."""
    calls: list[dict] = []
    model = backend.model
    original = model.generate

    def spy(**kwargs):
        output = original(**kwargs)
        calls.append({**kwargs, "_output": output})
        return output

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(model, "generate", spy)
        yield calls


def one_at_a_time(backend: TransformersBackend, prompts: list[str], n: int) -> list[str]:
    return [backend.generate([p], n)[0] for p in prompts]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def test_fp32_load_on_cpu(chat):
    assert chat.device == "cpu"
    assert chat.precision == "fp32"
    assert next(chat.model.parameters()).dtype == torch.float32
    assert next(chat.model.parameters()).device.type == "cpu"
    assert chat.model.training is False
    assert chat.tokenizer.padding_side == "left"
    assert chat.uses_chat_template is True


def test_auto_precision_resolves_to_fp32_on_cpu(model_dirs):
    backend = load(model_dirs["chat"], "auto")
    try:
        assert backend.precision == "fp32"
    finally:
        backend.close()


@pytest.mark.parametrize("precision", ["fp16", "bf16"])
def test_half_precision_on_cpu_is_refused_with_hint(model_dirs, precision):
    with pytest.raises(ModelLoadError) as info:
        load(model_dirs["chat"], precision)
    assert "fp32" in (info.value.hint or "")


def test_4bit_on_cpu_is_refused_with_hint(model_dirs):
    with pytest.raises(ModelLoadError) as info:
        load(model_dirs["chat"], "4bit")
    assert "CUDA" in str(info.value)
    assert "fp32" in (info.value.hint or "")


def test_4bit_without_bitsandbytes_is_refused_with_hint(model_dirs, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setitem(sys.modules, "bitsandbytes", None)  # import now fails
    with pytest.raises(ModelLoadError) as info:
        TransformersBackend(str(model_dirs["chat"]), "4bit")
    assert "pip install" in (info.value.hint or "")
    assert "bitsandbytes" in (info.value.hint or "")


@pytest.fixture
def offline(monkeypatch):
    """Belt and braces: the Hub must never be contacted from tests."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)


def test_unknown_repo_raises_model_load_error(offline):
    start = time.perf_counter()
    with pytest.raises(ModelLoadError) as info:
        load("btr-test-nonexistent-org/no-such-model-0b1c")
    assert time.perf_counter() - start < 15
    assert "btr-test-nonexistent-org/no-such-model-0b1c" in str(info.value)
    assert info.value.hint


def test_missing_folder_raises_model_load_error(offline, tmp_path):
    with pytest.raises(ModelLoadError, match="not found"):
        load(tmp_path / "missing-model-folder")


def test_folder_without_config_raises_model_load_error(offline, tmp_path):
    (tmp_path / "not-a-model").mkdir()
    with pytest.raises(ModelLoadError, match=r"config\.json"):
        load(tmp_path / "not-a-model")


def test_folder_without_tokenizer_files_is_refused_before_loading(offline, model_dirs, tmp_path):
    copy = tmp_path / "incomplete"
    shutil.copytree(model_dirs["chat"], copy)
    for item in copy.iterdir():
        if item.name.startswith("tokenizer"):
            item.unlink()
    with pytest.raises(ModelLoadError, match="no tokenizer files") as info:
        load(copy)
    assert "sentencepiece" not in (info.value.hint or "")


def test_seq2seq_model_is_refused_as_not_text_generation(offline, tmp_path):
    from transformers import T5Config, T5ForConditionalGeneration

    tokenizer = build_tokenizer()
    config = T5Config(
        vocab_size=len(tokenizer), d_model=16, d_ff=32, num_layers=1, num_heads=2, d_kv=8
    )
    T5ForConditionalGeneration(config).save_pretrained(tmp_path / "t5")
    tokenizer.save_pretrained(tmp_path / "t5")
    with pytest.raises(ModelLoadError, match="isn't a text-generation") as info:
        load(tmp_path / "t5")
    assert "pip install" not in (info.value.hint or "")


def _bert_classifier(vocab_size: int):
    from transformers import BertConfig, BertForSequenceClassification

    config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        num_labels=3,
    )
    return BertForSequenceClassification(config)


def _roberta_masked_lm(vocab_size: int):
    from transformers import RobertaConfig, RobertaForMaskedLM

    config = RobertaConfig(
        vocab_size=vocab_size,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
    )
    return RobertaForMaskedLM(config)


def _bart(vocab_size: int):
    from transformers import BartConfig, BartForConditionalGeneration

    config = BartConfig(
        vocab_size=vocab_size,
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
    )
    return BartForConditionalGeneration(config)


def _gpt_neox(vocab_size: int):
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    config = GPTNeoXConfig(
        vocab_size=vocab_size,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=128,
    )
    return GPTNeoXForCausalLM(config)


def _save_checkpoint(dest: Path, build) -> Path:
    tokenizer = build_tokenizer()
    torch.manual_seed(0)
    build(len(tokenizer)).save_pretrained(dest)
    tokenizer.save_pretrained(dest)
    return dest


@pytest.mark.parametrize(
    ("build", "kind"),
    [
        (_bert_classifier, "BertForSequenceClassification"),
        (_roberta_masked_lm, "RobertaForMaskedLM"),
        (_bart, "BartForConditionalGeneration"),
    ],
)
def test_encoder_checkpoints_are_refused_before_loading(
    offline, tmp_path, monkeypatch, build, kind
):
    """Regression: AutoModelForCausalLM loads these as BertLMHeadModel /
    RobertaForCausalLM / BartForCausalLM with a random head, so every row
    came back "generated" as junk instead of the error below."""

    def no_weights(*args, **kwargs):
        raise AssertionError("weights loaded for a checkpoint that can't generate text")

    monkeypatch.setattr(tb, "_from_pretrained", no_weights)
    with pytest.raises(ModelLoadError, match="isn't a text-generation") as info:
        load(_save_checkpoint(tmp_path / kind, build))
    assert f"({kind})" in str(info.value)
    assert "BERT-style" in (info.value.hint or "")


def test_running_out_of_ram_while_loading_on_the_cpu(model_dirs, monkeypatch):
    """Regression: torch's raw allocator error reached the user with no
    advice; the backend now says RAM ran out on the CPU."""

    def out_of_ram(*args, **kwargs):
        torch.empty((10**13,))  # 40 TB: the real DefaultCPUAllocator error
        raise AssertionError("allocated 40 TB")

    monkeypatch.setattr(tb, "_from_pretrained", out_of_ram)
    with pytest.raises(ModelLoadError) as info:
        load(model_dirs["chat"])
    message = str(info.value)
    assert message.startswith("Not enough memory (RAM) to load ")
    assert " at fp32 on the CPU: the weights take about 4 bytes per parameter." in message
    assert "smaller model" in (info.value.hint or "")
    assert isinstance(info.value.__cause__, RuntimeError)


def test_decoder_whose_config_says_is_decoder_false_still_loads(offline, tmp_path):
    """GPT-NeoX configs carry is_decoder=False (an encoder-decoder setting),
    so that flag mustn't be what refuses BERT."""
    backend = load(_save_checkpoint(tmp_path / "neox", _gpt_neox))
    try:
        assert backend.model.config.is_decoder is False
        assert len(backend.generate(["hello"], 2)) == 1
    finally:
        backend.close()


@pytest.mark.parametrize(
    ("architectures", "encoder_decoder", "refused"),
    [
        (["LlamaForCausalLM"], False, None),
        (["GPT2LMHeadModel"], False, None),
        (["Qwen3_5ForConditionalGeneration"], False, None),  # multimodal, decoder-only
        (["IdeficsForVisionText2Text"], False, None),  # known to the auto classes
        (None, False, None),  # nothing to go by: let transformers decide
        (["BertForSequenceClassification"], False, "BertForSequenceClassification"),
        (["LlamaForSequenceClassification"], False, "LlamaForSequenceClassification"),
        (["T5ForConditionalGeneration"], True, "T5ForConditionalGeneration"),
        (None, True, "SimpleNamespace"),
    ],
)
def test_not_text_generation(architectures, encoder_decoder, refused):
    from types import SimpleNamespace

    config = SimpleNamespace(architectures=architectures, is_encoder_decoder=encoder_decoder)
    assert tb._not_text_generation(config) == refused


# --------------------------------------------------------------------------
# A model too big for the GPU
# --------------------------------------------------------------------------


def _load_on_pretend_gpu(monkeypatch, path: Path, precision: str, device_map: dict):
    """Load ``path`` "on CUDA": the weights really go to the CPU, then the
    model carries ``device_map`` as if ``device_map="auto"`` had placed it.

    Returns the kwargs from_pretrained was given and a weak reference to the
    model, plus the backend (or the error) from loading.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    original = tb._from_pretrained
    seen: dict = {}
    models: list[weakref.ref] = []

    def from_pretrained(transformers, source, kwargs):
        seen.update(kwargs)
        model = original(transformers, source, {})
        model.hf_device_map = dict(device_map)
        models.append(weakref.ref(model))
        return model

    monkeypatch.setattr(tb, "_from_pretrained", from_pretrained)
    try:
        outcome = TransformersBackend(str(path), precision)
    except ModelLoadError as exc:
        outcome = exc
    return outcome, seen, models[0]


@pytest.mark.parametrize(
    ("precision", "where", "suggests"),
    [("fp16", "cpu", "4-bit"), ("bf16", "disk", "4-bit"), ("fp32", "cpu", "fp16")],
)
def test_model_spilling_off_the_gpu_is_refused_as_too_big(
    model_dirs, monkeypatch, logs, precision, where, suggests
):
    """Regression: a 7B model at fp16 on an 8 GB card loaded "fine" with most
    layers on the CPU, and the run went ahead tens of times slower."""
    device_map = {
        "model.embed_tokens": 0,
        "model.layers.0": 0,
        "model.layers.1": where,
        "model.norm": where,
        "lm_head": where,
    }
    with logs.at_level(logging.WARNING, logger=tb.__name__):
        error, seen, model = _load_on_pretend_gpu(
            monkeypatch, model_dirs["chat"], precision, device_map
        )
    assert seen["device_map"] == "auto"
    assert isinstance(error, ModelLoadError)
    assert f"Not enough GPU memory to load {model_dirs['chat']} at {precision}" in str(error)
    assert "CPU" in str(error)
    assert suggests in (error.hint or "")
    # Freed although the error (and its traceback) is still around.
    gc.collect()
    assert model() is None
    (warning,) = _messages(logs, logging.WARNING)
    assert f"put 3 of its 5 modules on the {where}" in warning


def test_model_that_fits_on_the_gpu_loads(model_dirs, monkeypatch):
    backend, seen, _ = _load_on_pretend_gpu(monkeypatch, model_dirs["chat"], "fp16", {"": 0})
    assert isinstance(backend, TransformersBackend)
    try:
        assert seen["device_map"] == "auto"
        assert backend.device == "cuda"
        assert backend.model is not None
    finally:
        backend.close()


def test_missing_pad_token_falls_back_to_eos(backends):
    tokenizer = backends["nopad"].tokenizer
    assert tokenizer.pad_token == tokenizer.eos_token
    assert tokenizer.pad_token_id == tokenizer.eos_token_id


@pytest.fixture
def logs(caplog, monkeypatch):
    """caplog, also when an earlier test configured logging with the package
    logger not propagating to the root (where caplog listens)."""
    monkeypatch.setattr(logging.getLogger("batchtextreview"), "propagate", True)
    return caplog


def _messages(caplog, level: int) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == tb.__name__ and r.levelno == level]


def test_load_is_logged_with_device_precision_and_time(model_dirs, logs):
    with logs.at_level(logging.INFO, logger=tb.__name__):
        load(model_dirs["plain"]).close()
    (loaded,) = [m for m in _messages(logs, logging.INFO) if m.startswith("Loaded")]
    assert str(model_dirs["plain"]) in loaded
    assert re.search(r" on cpu \(fp32\) in \d+\.\d s\.$", loaded)
    # No chat template: prompts go in raw, which deserves a warning.
    (warning,) = _messages(logs, logging.WARNING)
    assert "no chat template" in warning
    assert str(model_dirs["plain"]) in warning


def test_chat_model_load_logs_no_warning(model_dirs, logs):
    with logs.at_level(logging.INFO, logger=tb.__name__):
        load(model_dirs["chat"]).close()
    assert _messages(logs, logging.INFO)
    assert _messages(logs, logging.WARNING) == []


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------


def test_generate_returns_one_stripped_string_per_prompt(chat):
    outputs = chat.generate(PROMPTS, 8)
    assert len(outputs) == len(PROMPTS)
    assert all(isinstance(o, str) and o == o.strip() for o in outputs)
    assert any(outputs)
    assert chat.generate([], 8) == []


@pytest.mark.parametrize("name", ["chat", "plain", "nopad"])
def test_batched_generation_matches_one_at_a_time(backends, name):
    """Prompts of different lengths are left-padded; the slice must line up."""
    backend = backends[name]
    batched = backend.generate(PROMPTS, 10)
    assert batched == one_at_a_time(backend, PROMPTS, 10)
    assert len(set(batched)) > 1  # not one output copied to every row
    # Order follows the input order.
    assert backend.generate(PROMPTS[::-1], 10) == batched[::-1]


def test_max_new_tokens_is_respected(chat):
    tokenizer = chat.tokenizer
    with capture_generate(chat) as calls:
        for limit in (1, 3):
            for output in chat.generate(PROMPTS, limit):
                assert len(tokenizer(output, add_special_tokens=False)["input_ids"]) <= limit
            generated = calls[-1]["_output"].shape[1] - calls[-1]["input_ids"].shape[1]
            assert generated <= limit
    longer = chat.generate(PROMPTS, 12)
    assert max(len(tokenizer(o, add_special_tokens=False)["input_ids"]) for o in longer) > 3


def test_generate_call_arguments(chat):
    """Greedy, the tokenizer's own pad id (not eos), only ids + mask passed."""
    tokenizer = chat.tokenizer
    assert tokenizer.pad_token_id != tokenizer.eos_token_id
    with capture_generate(chat) as calls:
        chat.generate(PROMPTS[:2], 4)
    (call,) = calls
    assert call["pad_token_id"] == tokenizer.pad_token_id
    assert call["do_sample"] is False
    assert call["max_new_tokens"] == 4
    assert {k for k in call if not k.startswith("_")} >= {"input_ids", "attention_mask"}
    assert "token_type_ids" not in call
    # Left padding: the shorter prompt's padding is at the start.
    ids, mask = call["input_ids"], call["attention_mask"]
    assert mask[:, -1].all()
    assert (ids[mask == 0] == tokenizer.pad_token_id).all()


def test_chat_template_is_applied_once(chat):
    assert chat._format(["hello"]) == ["<s>user: hello</s><s>assistant:"]
    tokenizer = chat.tokenizer
    with capture_generate(chat) as calls:
        chat.generate(["hello"], 2)
    ids = calls[0]["input_ids"][0].tolist()
    # The template brings its own BOS; the tokenizer must not add another.
    assert ids[0] == tokenizer.bos_token_id
    assert ids[1] != tokenizer.bos_token_id


# Like Llama 3.2's template: today's date unless the caller passes one.
DATED_TEMPLATE = (
    "{% if date_string is not defined %}"
    "{% set date_string = strftime_now('%d %b %Y') %}"
    "{% endif %}"
    "<s>system: today is {{ date_string }} thinking {{ enable_thinking }}</s>" + CHAT_TEMPLATE
)


def test_chat_template_gets_a_fixed_date(tmp_path):
    """Saved outputs mustn't depend on the day the prompt was formatted."""
    backend = load(build_tiny_model(tmp_path / "dated", chat_template=DATED_TEMPLATE, seed=3))
    try:
        messages = [{"role": "user", "content": "hello"}]
        undated = backend.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        assert "today is " in undated
        assert "26 Jul 2024" not in undated  # left alone, it would use today's date
        assert backend._format(["hello"]) == [
            "<s>system: today is 26 Jul 2024 thinking False</s><s>user: hello</s><s>assistant:"
        ]
        assert len(backend.generate(["hello"], 3)) == 1
    finally:
        backend.close()


# Like IBM Granite 3.x: calls strftime_now() itself, never reads date_string.
STRFTIME_TEMPLATE = "<s>system: Today's Date: {{ strftime_now('%B %d, %Y') }}.</s>" + CHAT_TEMPLATE


def test_chat_template_calling_strftime_now_gets_the_fixed_date(tmp_path):
    """Regression: such templates put today's date in every prompt, so a
    re-run on another day mixed cached outputs made with different prompts."""
    backend = load(build_tiny_model(tmp_path / "granite3", chat_template=STRFTIME_TEMPLATE, seed=4))
    try:
        messages = [{"role": "user", "content": "hello"}]
        undated = backend.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        assert "Today's Date: " in undated
        assert "July 26, 2024" not in undated  # left alone, it would use today's date
        assert backend._format(["hello"]) == [
            "<s>system: Today's Date: July 26, 2024.</s><s>user: hello</s><s>assistant:"
        ]
        assert len(backend.generate(["hello"], 3)) == 1
    finally:
        backend.close()


def test_without_chat_template_the_raw_prompt_is_used(backends):
    plain = backends["plain"]
    assert plain.uses_chat_template is False
    assert plain._format(["hello world"]) == ["hello world"]
    tokenizer = plain.tokenizer
    with capture_generate(plain) as calls:
        outputs = plain.generate(["hello world"], 4)
    assert len(outputs) == 1
    ids = calls[0]["input_ids"][0].tolist()
    # Special tokens are added when no template did it (BOS here).
    assert ids == [
        tokenizer.bos_token_id,
        *tokenizer("hello world", add_special_tokens=False)["input_ids"],
    ]


# --------------------------------------------------------------------------
# Output-limit reporting
# --------------------------------------------------------------------------


def generated_ids(backend: TransformersBackend, prompts: list[str], limit: int) -> list[list[int]]:
    """The new token ids of each row of one batch."""
    with capture_generate(backend) as calls:
        backend.generate(prompts, limit)
    (call,) = calls
    return call["_output"][:, call["input_ids"].shape[1] :].tolist()


@pytest.fixture
def stop_on(chat, monkeypatch):
    """Make generation also stop on the given ids, the way Llama 3's
    generation config lists several end-of-turn tokens."""
    config = chat.model.generation_config

    def stop(*ids: int) -> None:
        monkeypatch.setattr(config, "eos_token_id", [chat.tokenizer.eos_token_id, *ids])

    return stop


def test_hit_limit_from_generated_ids():
    eos, pad = 1, 3
    new_tokens = torch.tensor(
        [
            [5, 6, 7],  # still going at the limit: cut off
            [5, 6, eos],  # finished on the very last step
            [5, eos, pad],  # finished early, then padded
            [5, 6, 128009],  # a second end-of-turn id (Llama 3)
        ]
    )
    stops = {eos, pad, 128009}
    assert tb._hit_limit(new_tokens, 3, stops) == [True, False, False, False]
    # The batch stopped before the limit: every row finished.
    assert tb._hit_limit(new_tokens[:, :2], 3, stops) == [False] * 4
    assert tb._hit_limit(new_tokens[:0], 3, stops) == []


def test_every_row_hits_the_limit_when_none_finishes(chat):
    rows = generated_ids(chat, PROMPTS, 5)
    # The random tiny model never produces EOS on its own.
    assert all(chat.tokenizer.eos_token_id not in row for row in rows)
    assert chat.last_hit_limit == [True] * len(PROMPTS)


@pytest.mark.parametrize("limit", [3, 6])
def test_rows_that_stop_on_an_end_token_are_not_flagged(chat, stop_on, limit):
    """limit=3: row 0's stop token is its last step; limit=6: it stops early."""
    rows = generated_ids(chat, PROMPTS, limit)
    stop = rows[0][2]
    stop_on(stop)
    outputs = chat.generate(PROMPTS, limit)
    expected = [stop not in row for row in rows]
    assert expected[0] is False
    assert True in expected
    assert chat.last_hit_limit == expected
    tokenizer = chat.tokenizer
    stop_word = tokenizer.convert_ids_to_tokens(stop)  # an ordinary word here, so decoded
    for output, cut in zip(outputs, expected, strict=True):
        if cut:
            assert len(tokenizer(output, add_special_tokens=False)["input_ids"]) == limit
        else:
            assert output.split()[-1] == stop_word


def test_no_row_is_flagged_when_all_stop_early(chat, stop_on):
    rows = generated_ids(chat, PROMPTS, 8)
    stop_on(*{row[0] for row in rows})
    outputs = chat.generate(PROMPTS, 8)
    assert [len(output.split()) for output in outputs] == [1] * len(PROMPTS)
    assert chat.last_hit_limit == [False] * len(PROMPTS)


def test_hit_limit_is_reset_by_an_empty_or_failed_call(chat, monkeypatch):
    chat.generate(PROMPTS[:2], 3)
    assert chat.last_hit_limit == [True, True]
    assert chat.generate([], 3) == []
    assert chat.last_hit_limit == []
    chat.generate(PROMPTS[:2], 3)
    _flaky_chunk(chat, monkeypatch, lambda batch: True, ValueError("bad input"))
    with pytest.raises(ValueError, match="bad input"):
        chat.generate(PROMPTS[:2], 3)
    assert chat.last_hit_limit == []


# --------------------------------------------------------------------------
# Out-of-memory halving
# --------------------------------------------------------------------------

# Tokens per PROMPTS entry once the tiny chat template is applied; the
# chunk sizes asserted below follow from these.
PROMPT_TOKENS = [8, 13, 20, 14, 10]


def _flaky_chunk(backend, monkeypatch, fails, error=None):
    """Make ``_generate_chunk`` raise OOM when ``fails(batch)``; return the sizes seen.

    Also resets the token budget for the test (restored afterwards), since
    the session's backends are shared.
    """
    original = backend._generate_chunk
    sizes: list[int] = []
    monkeypatch.setattr(backend, "_token_budget", None)

    def chunk(prompts, max_new_tokens):
        sizes.append(len(prompts))
        if fails(prompts):
            raise error or torch.cuda.OutOfMemoryError(
                "CUDA out of memory. Tried to allocate 2 GiB"
            )
        return original(prompts, max_new_tokens)

    monkeypatch.setattr(backend, "_generate_chunk", chunk)
    return sizes


def test_prompt_token_counts(chat):
    assert chat._token_counts(PROMPTS) == PROMPT_TOKENS
    with capture_generate(chat) as calls:
        chat.generate(PROMPTS, 2)
    assert calls[0]["input_ids"].shape[1] == max(PROMPT_TOKENS)  # the padded width


def test_estimate_and_chunk_size():
    assert tb._estimate([8, 13], 6) == 2 * (13 + 6)
    assert tb._chunk_size(PROMPT_TOKENS, 6, 40) == 2  # 2 x 19 fits, 3 x 26 doesn't
    assert tb._chunk_size(PROMPT_TOKENS, 6, 10_000) == 5
    assert tb._chunk_size(PROMPT_TOKENS, 6, 1) == 1  # always at least one prompt
    assert tb._chunk_size([20, 8, 8, 8], 6, 78) == 3  # the longest prompt sets the width


def test_oom_halves_the_batch_and_keeps_order(chat, monkeypatch, logs):
    expected = one_at_a_time(chat, PROMPTS, 6)
    sizes = _flaky_chunk(chat, monkeypatch, lambda batch: len(batch) > 1)
    with logs.at_level(logging.WARNING, logger=tb.__name__):
        assert chat.generate(PROMPTS, 6) == expected
    assert sizes == [5, 2, 1, 1, 3, 1, 2, 1, 1]
    assert chat.last_hit_limit == [True] * len(PROMPTS)
    # Every prompt fitted alone and no pair did: the longest single prompt.
    assert chat.token_budget == max(PROMPT_TOKENS) + 6
    warnings = _messages(logs, logging.WARNING)
    assert "Out of GPU memory generating 5 prompts" in warnings[0]
    assert sum("retrying in halves" in m for m in warnings) == 4
    assert "split to stay within 26 tokens" in warnings[-1]


def test_runtime_error_mentioning_oom_is_also_halved(chat, monkeypatch):
    expected = chat.generate(PROMPTS, 6)
    sizes = _flaky_chunk(
        chat,
        monkeypatch,
        lambda batch: len(batch) > 2,
        RuntimeError("CUDA error: out of memory"),
    )
    assert chat.generate(PROMPTS, 6) == expected
    assert sizes == [5, 2, 3, 1, 2]


def test_single_prompt_oom_reraises(chat, monkeypatch):
    sizes = _flaky_chunk(chat, monkeypatch, lambda batch: True)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        chat.generate(PROMPTS[:2], 4)
    assert sizes == [2, 1]
    # Nothing fitted, so nothing was learned: the GPU may just have been busy.
    assert chat.token_budget is None
    assert chat.last_hit_limit == []


def test_other_errors_are_not_retried(chat, monkeypatch):
    sizes = _flaky_chunk(chat, monkeypatch, lambda batch: True, ValueError("bad input"))
    with pytest.raises(ValueError, match="bad input"):
        chat.generate(PROMPTS, 4)
    assert sizes == [5]
    assert chat.token_budget is None


def test_oom_retry_runs_after_the_failed_attempt_is_released(chat, monkeypatch):
    """The halves are generated outside the except block: by then the failed
    attempt's traceback, and the tensors its frames held, must be gone."""
    expected = one_at_a_time(chat, PROMPTS, 4)
    original = chat._generate_chunk
    monkeypatch.setattr(chat, "_token_budget", None)
    failed: list[weakref.ref] = []
    seen: list[tuple[BaseException | None, bool]] = []

    def chunk(prompts, max_new_tokens):
        seen.append((sys.exc_info()[1], all(ref() is None for ref in failed)))
        if len(prompts) > 2:
            activations = torch.ones(8)  # held by this frame, so by the traceback
            failed.append(weakref.ref(activations))
            raise torch.cuda.OutOfMemoryError("CUDA out of memory.")
        return original(prompts, max_new_tokens)

    emptied: list[BaseException | None] = []
    monkeypatch.setattr(tb, "_empty_cache", lambda torch: emptied.append(sys.exc_info()[1]))
    monkeypatch.setattr(chat, "_generate_chunk", chunk)
    assert chat.generate(PROMPTS, 4) == expected
    assert len(seen) == 5  # [5, 2, 3, 1, 2]
    assert seen[1:] == [(None, True)] * 4
    assert emptied and emptied == [None] * len(emptied)


def test_oom_budget_sticks_for_later_calls(chat, monkeypatch, logs):
    expected = chat.generate(PROMPTS, 6)
    failing = True
    sizes = _flaky_chunk(chat, monkeypatch, lambda batch: failing and len(batch) > 2)
    assert chat.token_budget is None
    with logs.at_level(logging.WARNING, logger=tb.__name__):
        assert chat.generate(PROMPTS, 6) == expected
    assert sizes == [5, 2, 3, 1, 2]
    # Fitted: 2 x (13+6), 1 x (20+6), 2 x (14+6); failed: 5 x 26, 3 x 26.
    assert chat.token_budget == 40
    assert "split to stay within 40 tokens" in _messages(logs, logging.WARNING)[-1]

    # Later calls are pre-split, even once a bigger batch would fit again.
    failing = False
    sizes.clear()
    logs.clear()
    with logs.at_level(logging.WARNING, logger=tb.__name__):
        assert chat.generate(PROMPTS, 6) == expected
        assert chat.generate(PROMPTS[::-1], 6) == expected[::-1]
        assert chat.generate(PROMPTS[1:4], 6) == expected[1:4]
    assert sizes == [2, 1, 2, 2, 1, 2, 1, 1, 1]
    assert chat.last_hit_limit == [True] * 3
    assert chat.token_budget == 40
    assert _messages(logs, logging.WARNING) == []


def test_budget_learned_on_long_prompts_leaves_short_ones_in_full_batches(chat, monkeypatch):
    """The point of counting tokens rather than prompts: running out of
    memory on long texts mustn't slow every later batch of short ones."""
    capacity = 150  # tokens the pretend GPU holds

    def too_big(batch):
        return tb._estimate(chat._token_counts(batch), 2) > capacity

    long_prompts = [" ".join(["the"] * 60)] * 4
    short_prompts = ["hello"] * 8
    expected_long = chat.generate(long_prompts, 2)
    expected_short = chat.generate(short_prompts, 2)
    sizes = _flaky_chunk(chat, monkeypatch, too_big)
    assert chat.generate(long_prompts, 2) == expected_long
    assert sizes == [4, 2, 2]
    assert chat.token_budget is not None and chat.token_budget <= capacity

    sizes.clear()
    assert chat.generate(short_prompts, 2) == expected_short
    assert sizes == [8]  # a prompt-count limit would have made this 2, 2, 2, 2
    sizes.clear()
    assert chat.generate(long_prompts, 2) == expected_long
    assert sizes == [2, 2]  # no second out-of-memory error


def test_oom_budget_only_goes_down_and_applies_mid_call(chat, monkeypatch):
    expected = chat.generate(PROMPTS, 6)
    fits = 2
    sizes = _flaky_chunk(chat, monkeypatch, lambda batch: len(batch) > fits)
    chat.generate(PROMPTS, 6)
    assert chat.token_budget == 40
    fits = 1  # memory got tighter
    sizes.clear()
    assert chat.generate(PROMPTS, 6) == expected
    # The first pre-split chunk fails; the budget it teaches covers the rest.
    assert sizes == [2, 1, 1, 1, 1, 1]
    assert chat.token_budget == 13 + 6
    fits = 5  # room again: the budget stays where it is
    sizes.clear()
    assert chat.generate(PROMPTS, 6) == expected
    assert sizes == [1] * 5
    assert chat.token_budget == 19


def test_oom_budget_stays_below_every_size_that_failed(chat, monkeypatch):
    """Two prompts with the long one failed (2 x 24 tokens) and three short
    ones fitted (3 x 18): 54 is no safe budget."""
    prompts = [*PROMPTS, "hello world"]
    long_prompt = PROMPTS[2]
    expected = chat.generate(prompts, 4)
    sizes = _flaky_chunk(chat, monkeypatch, lambda b: long_prompt in b and len(b) > 1)
    assert chat.generate(prompts, 4) == expected
    assert sizes == [6, 3, 1, 2, 1, 1, 3]
    assert chat.token_budget == 20 + 4  # the long prompt alone


def test_a_call_that_raises_still_lowers_the_budget(chat, monkeypatch):
    """A prompt too long to fit even alone fails the call, but what fitted
    on the way there still counts."""
    long_prompt = PROMPTS[2]
    sizes = _flaky_chunk(chat, monkeypatch, lambda b: len(b) > 1 or long_prompt in b)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        chat.generate(PROMPTS, 6)
    assert sizes == [5, 2, 1, 1, 3, 1]
    assert chat.token_budget == 13 + 6  # the largest that fitted, below 20 + 6 alone


# --------------------------------------------------------------------------
# Closing
# --------------------------------------------------------------------------


def test_close_releases_the_model(model_dirs):
    backend = load(model_dirs["chat"])
    assert backend.generate(["hello"], 2)
    model_ref = weakref.ref(backend.model)
    backend.close()
    gc.collect()
    assert model_ref() is None
    assert backend.model is None
    assert backend.tokenizer is None
    with pytest.raises(GenerationError):
        backend.generate(["hello"], 2)
    backend.close()  # idempotent
