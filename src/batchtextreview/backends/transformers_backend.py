"""Hugging Face transformers backend (local inference, GPU or CPU).

Only this module (and code it calls) imports torch / transformers, and only
when a real model is constructed.
"""

from __future__ import annotations

import gc
import logging
import os
import re
import time
from collections.abc import Collection, Sequence
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from batchtextreview.backends.base import (
    AUTO,
    DeviceInfo,
    GenerationError,
    ModelLoadError,
    resolve_precision,
)
from batchtextreview.logs import current_log_file
from batchtextreview.models import find_model

_log = logging.getLogger(__name__)

_LOADABLE = ("4bit", "fp16", "bf16", "fp32")
# ValueErrors where retrying with the image-text class would only fail again
# (slowly, after reloading every weight).
_NO_RETRY = ("does not recognize this architecture", "dispatched on the cpu or the disk")
_MAX_DETAIL = 300
# Passed to every chat template. Llama 3.x templates otherwise put today's
# date in the system prompt, so the same prompt would give different outputs
# on different days. This is the date Llama 3.1's own template defaults to.
# Templates that call strftime_now() themselves (Granite 3.x: "Today's Date:
# ...") get the same day from _fixed_strftime.
_CHAT_DAY = datetime(2024, 7, 26)
_CHAT_DATE = "26 Jul 2024"
# Class-name endings of checkpoints that generate text: decoder-only LMs, and
# multimodal models whose text side is one.
_GENERATIVE_SUFFIXES = ("ForCausalLM", "LMHeadModel", "WithLMHead", "ForConditionalGeneration")


class TransformersBackend:
    """Greedy, batched, chat-templated generation with a local HF model.

    Loading (in ``__init__``):

    * Device: CUDA if available, else Apple MPS, else CPU.
    * Precision (already resolved, never "auto"):
        - "4bit": bitsandbytes NF4 + double quant, fp16 compute,
          ``device_map="auto"``. Requires CUDA and bitsandbytes, otherwise
          raise ModelLoadError with a hint ("choose fp16/fp32" or
          "pip install bitsandbytes").
        - "fp16" / "bf16": that dtype on CUDA/MPS (``device_map="auto"`` on
          CUDA, ``.to("mps")`` on MPS). On CPU raise ModelLoadError (half
          precision on CPU is unusably slow) with hint "use fp32".
        - "fp32": float32 on whatever device.
    * Pass the dtype as ``dtype=`` (transformers >= 4.56); fall back to
      ``torch_dtype=`` if the installed transformers rejects ``dtype``.
    * ``AutoModelForCausalLM`` first; on ``ValueError`` (early-fusion
      multimodal configs such as Qwen3.5) retry with
      ``AutoModelForImageTextToText`` - text-only generation with input_ids
      works the same way.
    * Tokenizer: ``padding_side="left"``; if it has no pad token, use eos.
    * Refused before any weights load: checkpoints that can't generate text
      this way - encoder-decoders (T5, BART...) and encoder-only or
      task-head ones (BERT / RoBERTa classifiers...), which the causal-LM
      class would otherwise load with a random head and run, producing junk.
    * Refused after loading on CUDA: a model ``device_map="auto"`` couldn't
      fit on the GPU and spread onto the CPU or disk (tens of times slower);
      that is reported as not enough GPU memory.
    * ``model.eval()``.
    * Map failures to ModelLoadError with actionable hints:
        gated/401/403 ("accept the licence on huggingface.co and set HF_TOKEN"),
        repo not found / no local folder, network/SSL ("check proxy; truststore
        is enabled"), CUDA OOM ("choose a smaller model or 4-bit"), missing
        packages, unsupported architecture ("pip install -U transformers").
      Free any partially loaded model before raising.
    * Logs the model id, device, precision and load time (INFO), and a
      WARNING when the tokenizer has no chat template.

    Generation (``generate``):

    * Each prompt is wrapped with ``tokenizer.apply_chat_template(
      [{"role": "user", "content": p}], tokenize=False,
      add_generation_prompt=True, enable_thinking=False,
      date_string="26 Jul 2024", strftime_now=<that day>)`` when the
      tokenizer has a chat template (``enable_thinking=False`` keeps Qwen3
      from thinking; the fixed ``date_string`` stops Llama 3.x templates, and
      the fixed ``strftime_now`` those that call it directly such as Granite
      3.x, from putting today's date in the prompt; templates ignore
      variables they don't use). Without a chat
      template the raw prompt is used. Tokenise with ``padding=True``,
      ``add_special_tokens=False`` when a chat template was applied (the
      template already adds BOS), True otherwise.
    * ``model.generate(**inputs, max_new_tokens=..., do_sample=False,
      temperature=None, top_p=None, top_k=None,
      pad_token_id=tokenizer.pad_token_id)`` under ``torch.inference_mode()``.
    * Slice off the (uniform, left-padded) prompt length, decode with
      ``skip_special_tokens=True``, strip.
    * ``last_hit_limit`` is set from the generated ids, one flag per prompt.
    * CUDA OOM (``torch.cuda.OutOfMemoryError`` or a RuntimeError whose
      message contains "out of memory"): leave the ``except`` block (so the
      failed attempt's traceback, and the activations its frames hold, are
      released), empty the cache, split the batch in half and retry each
      half recursively; a single prompt that still OOMs re-raises. What
      fitted and what didn't sets ``token_budget``, measured in tokens as
      prompts x (longest prompt + ``max_new_tokens``) since that is what
      memory grows with: for the life of the backend, later calls are
      pre-split into chunks within it, so one OOM isn't paid for again on
      every call - while short prompts still go in full batches after long
      ones ran out of memory.
    """

    model_id: str
    precision: str
    device: str
    last_hit_limit: list[bool]
    """One flag per prompt of the latest ``generate()`` call, in order: True
    where the completion was cut off at ``max_new_tokens`` rather than
    finished. Empty before the first call and after a failed one."""

    def __init__(self, model_id: str, precision: str) -> None:
        self.model_id = model_id
        self.precision = precision
        self.device = "cpu"
        self.last_hit_limit = []
        self._torch: Any = None
        self._model: Any = None
        self._tokenizer: Any = None
        self._chat = False
        self._token_budget: int | None = None

        failure: ModelLoadError | None = None
        started = time.perf_counter()
        try:
            self._load()
        except ModelLoadError:
            self.close()
            raise
        except Exception as exc:
            _log.warning(
                "Loading %s (%s) failed: %s: %s", model_id, precision, type(exc).__name__, exc
            )
            _log.debug("Load failure traceback", exc_info=True)
            failure = _load_error(exc, model_id, self.precision, self.device)
            # The traceback pins the frames of from_pretrained, and with them
            # any half-loaded weights; drop it so close() can actually free
            # the memory. The message survives as __cause__.
            _drop_tracebacks(exc)
            failure.__cause__ = exc
        except BaseException:  # KeyboardInterrupt and friends
            self.close()
            raise
        if failure is not None:
            self.close()
            raise failure
        _log.info(
            "Loaded %s on %s (%s) in %.1f s.",
            model_id,
            self.device,
            self.precision,
            time.perf_counter() - started,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def tokenizer(self) -> Any:
        """The loaded tokenizer (``None`` once closed)."""
        return self._tokenizer

    @property
    def model(self) -> Any:
        """The loaded model (``None`` once closed)."""
        return self._model

    @property
    def uses_chat_template(self) -> bool:
        return self._chat

    @property
    def token_budget(self) -> int | None:
        """How big one ``model.generate`` call may be - prompts x (longest
        prompt + ``max_new_tokens``), in tokens - learned from running out
        of GPU memory: the largest call that fitted below the smallest that
        didn't. None until that happens; it only ever goes down (loading the
        model again starts afresh)."""
        return self._token_budget

    def generate(self, prompts: Sequence[str], max_new_tokens: int) -> list[str]:
        self.last_hit_limit = []
        if self._model is None or self._tokenizer is None:
            raise GenerationError(f"The model {self.model_id} has been unloaded.")
        batch = [str(p) for p in prompts]
        if not batch:
            return []
        limit = max(1, int(max_new_tokens))
        budget = self._token_budget
        lengths: list[int] | None = None  # tokens per prompt, once a budget applies
        texts: list[str] = []
        hit_limit: list[bool] = []
        try:
            start = 0
            while start < len(batch):
                # Planned every time round: a budget learned on this call's
                # first chunk already applies to the rest of it.
                end = len(batch)
                if self._token_budget is not None:
                    if lengths is None:
                        lengths = self._token_counts(batch)
                    end = start + _chunk_size(lengths[start:], limit, self._token_budget)
                chunk_texts, chunk_hits = self._generate_and_learn(
                    batch[start:end], limit, None if lengths is None else lengths[start:end]
                )
                texts += chunk_texts
                hit_limit += chunk_hits
                start = end
        finally:
            if self._token_budget != budget:
                _log.warning(
                    "%s ran out of GPU memory; from now on its batches are split to stay within "
                    "%s tokens (prompts x (longest prompt + max new tokens)).",
                    self.model_id,
                    f"{self._token_budget:,}",
                )
        self.last_hit_limit = hit_limit
        return texts

    def close(self) -> None:
        """Drop model/tokenizer references, gc, empty CUDA cache."""
        self._model = None
        self._tokenizer = None
        _release_memory(self._torch)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        torch, transformers = _import_stack()
        self._torch = torch
        self.device = _pick_device(torch)
        self.precision = precision = _check_precision(self.precision, self.device)
        source = _model_source(self.model_id)

        # The tokenizer first: it is small, so a wrong name, a gated repo or a
        # network problem is reported before gigabytes of weights download.
        tokenizer = transformers.AutoTokenizer.from_pretrained(source)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            fallback = tokenizer.eos_token or tokenizer.unk_token
            if fallback is None:
                raise ModelLoadError(
                    f"The tokenizer of {self.model_id} has no pad, end-of-sequence or "
                    "unknown token, so prompts can't be batched."
                )
            tokenizer.pad_token = fallback
        self._tokenizer = tokenizer
        self._chat = bool(getattr(tokenizer, "chat_template", None))

        # config.json is tiny; reading it here means a BERT classifier or a
        # T5 is refused before its weights are loaded.
        kind = _not_text_generation(transformers.AutoConfig.from_pretrained(source))
        if kind is not None:
            raise _not_text_generation_error(self.model_id, kind)

        kwargs = _model_kwargs(torch, transformers, precision, self.device)
        model = _from_pretrained(transformers, source, kwargs)
        offloaded = _offloaded(model) if self.device == "cuda" else []
        if offloaded:
            total = len(model.hf_device_map)
            # Drop the only reference before raising: the error's traceback
            # keeps this frame alive, and with it the weights on the GPU.
            del model
            _log.warning(
                "%s at %s doesn't fit on the GPU: device_map='auto' put %d of its %d modules "
                "on the %s, which would run tens of times slower; refusing to load it.",
                self.model_id,
                precision,
                len(offloaded),
                total,
                " / ".join(sorted({where for _, where in offloaded})),
            )
            raise _gpu_memory_error(
                self.model_id,
                precision,
                "part of it would have to run from the CPU, many times slower",
            )
        if self.device == "mps":
            model = model.to("mps")
        model.eval()
        self._model = model
        if not self._chat:
            _log.warning(
                "%s has no chat template, so prompts are sent to it as raw text.", self.model_id
            )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _generate_and_learn(
        self, prompts: list[str], max_new_tokens: int, lengths: list[int] | None
    ) -> tuple[list[str], list[bool]]:
        """``_generate_safe``, then lower ``token_budget`` to the largest
        call that fitted below the smallest one that ran out of memory - also
        when a prompt that doesn't fit even alone makes this raise. A call
        where nothing fitted teaches nothing: the GPU may just have been busy,
        and a budget of next to nothing would last as long as the model."""
        fitted: list[int] = []
        failed: list[int] = []
        try:
            return self._generate_safe(prompts, max_new_tokens, lengths, fitted, failed)
        finally:
            if failed and fitted:
                smallest_failure = min(failed)
                below = [size for size in fitted if size < smallest_failure]
                budget = max(below) if below else smallest_failure - 1
                if self._token_budget is None or budget < self._token_budget:
                    self._token_budget = budget

    def _generate_safe(
        self,
        prompts: list[str],
        max_new_tokens: int,
        lengths: list[int] | None,
        fitted: list[int],
        failed: list[int],
    ) -> tuple[list[str], list[bool]]:
        """Texts and hit-limit flags, halving the batch on CUDA OOM.

        ``lengths`` are the prompts' token counts (None: not counted yet).
        The size (see :func:`_estimate`) of every ``model.generate`` call
        that fitted / ran out of memory is appended to ``fitted`` /
        ``failed`` - except a call whose prompts were never counted (one
        that fitted at once, with no budget yet), which teaches nothing.
        """
        result: tuple[list[str], list[bool]] | None = None
        try:
            result = self._generate_chunk(prompts, max_new_tokens)
        except Exception as exc:
            if not _is_oom(exc, self._torch):
                raise
            if len(prompts) == 1:
                counted = lengths if lengths is not None else self._token_counts(prompts)
                failed.append(_estimate(counted, max_new_tokens))
                raise
        if result is not None:
            if lengths is not None:
                fitted.append(_estimate(lengths, max_new_tokens))
            return result
        # Out of memory. Retrying only now, outside the except block, matters:
        # until the handler exits, the exception's traceback keeps the failed
        # attempt's frames - and the activations they hold - alive, so
        # emptying the cache in there gives nothing back.
        _release_memory(self._torch)
        _log.warning(
            "Out of GPU memory generating %d prompts with %s; retrying in halves.",
            len(prompts),
            self.model_id,
        )
        if lengths is None:
            lengths = self._token_counts(prompts)
        failed.append(_estimate(lengths, max_new_tokens))
        mid = len(prompts) // 2
        first = self._generate_safe(prompts[:mid], max_new_tokens, lengths[:mid], fitted, failed)
        second = self._generate_safe(prompts[mid:], max_new_tokens, lengths[mid:], fitted, failed)
        return first[0] + second[0], first[1] + second[1]

    def _token_counts(self, prompts: Sequence[str]) -> list[int]:
        """Tokens per prompt as ``_generate_chunk`` sends it (before padding)."""
        encoded = self._tokenizer(self._format(prompts), add_special_tokens=not self._chat)
        return [len(ids) for ids in encoded["input_ids"]]

    def _format(self, prompts: Sequence[str]) -> list[str]:
        """The exact strings that get tokenised."""
        if not self._chat:
            return list(prompts)
        return [
            self._tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
                date_string=_CHAT_DATE,
                # Overrides the template global of the same name.
                strftime_now=_fixed_strftime,
            )
            for p in prompts
        ]

    def _generate_chunk(
        self, prompts: Sequence[str], max_new_tokens: int
    ) -> tuple[list[str], list[bool]]:
        """One batched ``model.generate`` call, no OOM handling: the texts and
        whether each one hit ``max_new_tokens``."""
        torch = self._torch
        tokenizer, model = self._tokenizer, self._model
        encoded = tokenizer(
            self._format(prompts),
            return_tensors="pt",
            padding=True,
            add_special_tokens=not self._chat,
        )
        # Only what a decoder needs: some tokenizers also emit token_type_ids,
        # which most causal LMs' generate() rejects.
        device = model.device
        inputs = {
            "input_ids": encoded["input_ids"].to(device),
            "attention_mask": encoded["attention_mask"].to(device),
        }
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.pad_token_id,
            )
        sequences = getattr(output, "sequences", output)
        # Left padding makes every prompt end at the same column.
        new_tokens = sequences[:, inputs["input_ids"].shape[1] :]
        hit_limit = _hit_limit(new_tokens, max_new_tokens, _stop_ids(tokenizer, model))
        texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        return [text.strip() for text in texts], hit_limit


def _estimate(lengths: Sequence[int], max_new_tokens: int) -> int:
    """The size of one ``model.generate`` call as ``token_budget`` counts it:
    prompts x (longest prompt + ``max_new_tokens``), what the left-padded
    batch and its cache grow to."""
    return len(lengths) * (max(lengths) + max_new_tokens)


def _chunk_size(lengths: Sequence[int], max_new_tokens: int, budget: int) -> int:
    """How many prompts from the front of ``lengths`` (token counts) the next
    call takes: as many as stay within ``budget``, and always at least one."""
    longest = 0
    for count, length in enumerate(lengths, start=1):
        longest = max(longest, length)
        if count > 1 and count * (longest + max_new_tokens) > budget:
            return count - 1
    return len(lengths)


def _fixed_strftime(format: str) -> str:
    """``strftime_now`` for chat templates: always :data:`_CHAT_DAY`."""
    return _CHAT_DAY.strftime(format)


def _stop_ids(tokenizer: Any, model: Any) -> set[int]:
    """Ids a finished row can end with: every end-of-sequence id the
    generation config lists (Llama 3 has several), the tokenizer's own, and
    the pad id that fills a row after it finishes."""
    config = getattr(model, "generation_config", None)
    ids: set[int] = set()
    for value in (
        getattr(config, "eos_token_id", None),
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
    ):
        for item in value if isinstance(value, (list, tuple, set)) else [value]:
            if isinstance(item, int):
                ids.add(item)
    return ids


def _hit_limit(new_tokens: Any, max_new_tokens: int, stop_ids: Collection[int]) -> list[bool]:
    """Per row of the generated ids: True when generation was cut off.

    Rows that finish early are padded to the longest one, so a row was cut
    exactly when the batch ran all ``max_new_tokens`` steps and the row's
    last token is neither an end-of-sequence nor a pad token.
    """
    rows, steps = int(new_tokens.shape[0]), int(new_tokens.shape[1])
    if steps < max_new_tokens:
        return [False] * rows
    return [token not in stop_ids for token in new_tokens[:, -1].tolist()]


# ----------------------------------------------------------------------
# Loading helpers
# ----------------------------------------------------------------------


def _import_stack() -> tuple[Any, Any]:
    try:
        import torch
    except Exception as exc:
        raise ModelLoadError(
            f"PyTorch could not be imported ({type(exc).__name__}: {exc}).",
            hint='Install PyTorch and the "llm" extra as README.md describes for your machine.',
        ) from exc
    try:
        import transformers
    except Exception as exc:
        raise ModelLoadError(
            f"The transformers package could not be imported ({type(exc).__name__}: {exc}).",
            hint="pip install -U transformers accelerate",
        ) from exc
    return torch, transformers


def _pick_device(torch: Any) -> str:
    try:
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _check_precision(precision: str, device: str) -> str:
    """Validate ``precision`` for ``device``; raises ModelLoadError."""
    value = (precision or AUTO).lower()
    if value == AUTO:
        # Only probe bitsandbytes where it matters: importing it pulls in torch.
        bnb = device == "cuda" and _has_bitsandbytes()
        value = resolve_precision(AUTO, DeviceInfo(kind=device, name=device, bitsandbytes=bnb))
    if value not in _LOADABLE:
        raise ModelLoadError(
            f"Unknown precision {precision!r} for a local model.",
            hint=f"Choose one of {', '.join(_LOADABLE)}.",
        )
    if value == "4bit":
        if device != "cuda":
            where = "an Apple GPU" if device == "mps" else "the CPU"
            better = "fp16" if device == "mps" else "fp32"
            raise ModelLoadError(
                f"4-bit quantisation needs an NVIDIA GPU (CUDA), but this machine is using {where}.",
                hint=f"Choose {better} precision instead.",
            )
        if not _has_bitsandbytes():
            raise ModelLoadError(
                "4-bit quantisation needs the bitsandbytes package, which isn't installed "
                "(or failed to import).",
                hint="pip install -U bitsandbytes, or choose fp16 precision.",
            )
    elif value in ("fp16", "bf16") and device == "cpu":
        raise ModelLoadError(
            f"{value} on the CPU is unusably slow (no GPU was detected).",
            hint="Use fp32 precision on the CPU.",
        )
    return value


def _has_bitsandbytes() -> bool:
    try:
        import bitsandbytes  # noqa: F401
    except Exception:
        return False
    return True


def _model_source(model_id: str) -> str:
    """Expand a local folder path; leave repo ids alone.

    A missing folder, or one without ``config.json`` or any tokenizer file,
    fails fast with a clear message instead of a confusing tokenizer error
    from transformers (which blames a missing sentencepiece package).
    """
    value = model_id.strip()
    path = Path(value).expanduser()
    try:
        is_dir = path.is_dir()
    except OSError:
        is_dir = False
    if not is_dir:
        if _looks_like_path(value):
            raise ModelLoadError(
                f"Model folder not found: {value}",
                hint="Check the path, or enter a Hugging Face repo id such as "
                "Qwen/Qwen2.5-1.5B-Instruct.",
            )
        return value
    if not (path / "config.json").is_file():
        raise ModelLoadError(
            f"{value} is not a model folder (it has no config.json).",
            hint="Choose the folder that contains config.json, the tokenizer files and the "
            "*.safetensors weights.",
        )
    if not _has_tokenizer_files(path):
        raise ModelLoadError(
            f"{value} has no tokenizer files, so it looks like an incomplete copy of the model.",
            hint="Copy the whole model folder, including tokenizer.json / tokenizer_config.json "
            "(or tokenizer.model), next to config.json.",
        )
    return str(path)


def _has_tokenizer_files(path: Path) -> bool:
    """Any file a transformers tokenizer could be built from (deliberately loose)."""
    try:
        names = [item.name.lower() for item in path.iterdir() if item.is_file()]
    except OSError:
        return True  # let transformers report whatever is wrong
    known = {"vocab.json", "vocab.txt", "merges.txt", "spiece.model", "tekken.json"}
    return any(
        name.startswith("tokenizer") or name in known or name.endswith((".model", ".tiktoken"))
        for name in names
    )


def _looks_like_path(value: str) -> bool:
    """True for things that can only be a filesystem path, never a repo id."""
    if value.startswith(("/", "~", "./", "../", ".\\", "..\\", "\\")) or "\\" in value:
        return True
    return bool(re.match(r"^[A-Za-z]:[\\/]", value))  # Windows drive letter


def _model_kwargs(torch: Any, transformers: Any, precision: str, device: str) -> dict[str, Any]:
    dtypes = {
        "4bit": torch.float16,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    kwargs: dict[str, Any] = {"dtype": dtypes[precision]}
    if precision == "4bit":
        kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    if device == "cuda":
        kwargs["device_map"] = "auto"
    if not _accepts_dtype_kwarg(transformers):
        kwargs["torch_dtype"] = kwargs.pop("dtype")
    return kwargs


def _accepts_dtype_kwarg(transformers: Any) -> bool:
    """``dtype=`` replaced ``torch_dtype=`` in transformers 4.56."""
    match = re.match(r"(\d+)\.(\d+)", getattr(transformers, "__version__", ""))
    if not match:
        return True
    return (int(match.group(1)), int(match.group(2))) >= (4, 56)


def _from_pretrained(transformers: Any, source: str, kwargs: dict[str, Any]) -> Any:
    try:
        return _from_pretrained_dtype(transformers.AutoModelForCausalLM, source, kwargs)
    except ValueError as first:
        fallback = getattr(transformers, "AutoModelForImageTextToText", None)
        if fallback is None or any(p in str(first).lower() for p in _NO_RETRY):
            raise
        _log.info("%s is not a causal LM config (%s); trying the image-text class.", source, first)
        try:
            return _from_pretrained_dtype(fallback, source, kwargs)
        except ValueError:
            # Not a multimodal model either: the first error says why.
            raise first from None


def _from_pretrained_dtype(auto_class: Any, source: str, kwargs: dict[str, Any]) -> Any:
    try:
        return auto_class.from_pretrained(source, **kwargs)
    except TypeError as exc:
        if "dtype" not in kwargs or "dtype" not in str(exc):
            raise
    legacy = dict(kwargs)
    legacy["torch_dtype"] = legacy.pop("dtype")
    return auto_class.from_pretrained(source, **legacy)


def _not_text_generation(config: Any) -> str | None:
    """The class to name when ``config`` is a model this backend can't
    generate with; None when it looks like a decoder (or can't be told).

    ``AutoModelForCausalLM`` happily loads a BERT / RoBERTa / XLM-R /
    ELECTRA checkpoint as a ``*LMHeadModel`` (bidirectional, with a random
    head) and a BART-style one as its decoder alone: the load "works" and
    every row comes back as junk. Encoder-decoders are refused outright
    (their outputs don't start with the prompt, which generation slices
    off); otherwise a checkpoint saved as anything but a text-generation
    class (``BertForSequenceClassification``, ``RobertaForMaskedLM``...).
    """
    architectures = [str(name) for name in getattr(config, "architectures", None) or []]
    kind = architectures[0] if architectures else type(config).__name__
    if getattr(config, "is_encoder_decoder", False) is True:
        return kind
    if architectures and not any(_is_generation_class(name) for name in architectures):
        return kind
    return None


def _is_generation_class(name: str) -> bool:
    if name.endswith(_GENERATIVE_SUFFIXES):
        return True
    try:
        return name in _generation_classes()
    except Exception:  # transformers moved its mappings: the suffixes will do
        return False


@lru_cache(maxsize=1)
def _generation_classes() -> frozenset[str]:
    """Every class the two auto classes used here can load, for the few
    whose names don't end in one of :data:`_GENERATIVE_SUFFIXES`
    (``IdeficsForVisionText2Text``...)."""
    from transformers.models.auto import modeling_auto

    names: set[str] = set()
    for mapping_name in (
        "MODEL_FOR_CAUSAL_LM_MAPPING_NAMES",
        "MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES",
    ):
        for value in getattr(modeling_auto, mapping_name, {}).values():
            names.update(value if isinstance(value, (list, tuple)) else [value])
    return frozenset(str(name) for name in names)


def _offloaded(model: Any) -> list[tuple[str, str]]:
    """(module, "cpu" / "disk") for every module ``device_map="auto"``
    placed off the GPU because it didn't fit; empty when all of the model
    is on the GPU (or no device map was used)."""
    device_map = getattr(model, "hf_device_map", None)
    if not isinstance(device_map, dict):
        return []
    out: list[tuple[str, str]] = []
    for name, where in device_map.items():
        # An int (GPU index), "cuda:0", "cpu", "disk", or a torch.device.
        kind = str(getattr(where, "type", where)).split(":")[0].strip().lower()
        if kind in ("cpu", "disk", "meta"):
            out.append((str(name), kind))
    return out


# ----------------------------------------------------------------------
# Error mapping
# ----------------------------------------------------------------------


def _chain(exc: BaseException) -> list[BaseException]:
    out: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and all(current is not seen for seen in out):
        out.append(current)
        current = current.__cause__ or current.__context__
    return out


def _drop_tracebacks(exc: BaseException) -> None:
    for item in _chain(exc):
        item.__traceback__ = None


def _offline_mode() -> bool:
    try:
        from huggingface_hub import constants

        return bool(constants.HF_HUB_OFFLINE)
    except Exception:
        flag = os.environ.get("HF_HUB_OFFLINE") or os.environ.get("TRANSFORMERS_OFFLINE") or ""
        return flag.strip().lower() in {"1", "true", "yes", "on"}


def _load_error(
    exc: BaseException, model_id: str, precision: str, device: str | None = None
) -> ModelLoadError:
    """Translate a loading exception into a ModelLoadError with a hint.

    Matches on exception class *names* along the cause/context chain (so
    huggingface_hub, requests and httpx needn't be imported) and on message
    text. Order matters: the most specific diagnosis wins. ``device`` is
    where the weights were going ("cuda", "mps", "cpu"; None if unknown).
    """
    chain = _chain(exc)
    names = {type(e).__name__ for e in chain}
    low = " ".join(str(e) for e in chain).lower()
    detail = f"{type(exc).__name__}: {exc}".strip().replace("\n", " ")
    if len(detail) > _MAX_DETAIL:
        detail = detail[: _MAX_DETAIL - 1] + "…"

    def has(*phrases: str) -> bool:
        return any(phrase in low for phrase in phrases)

    # torch's CPU allocator: "DefaultCPUAllocator: can't allocate memory"
    # (Linux, macOS) or "...: not enough memory" (Windows); Python's own
    # MemoryError; ENOMEM from mapping the weight files.
    if (
        "MemoryError" in names
        or has("defaultcpuallocator", "can't allocate memory", "cannot allocate memory")
        or has("not enough memory")
        or (device == "cpu" and has("out of memory"))
    ):
        return _ram_error(model_id, precision, device)
    if "OutOfMemoryError" in names or has("out of memory", "dispatched on the cpu or the disk"):
        return _gpu_memory_error(model_id, precision)
    # Only a real gated-repo response counts: huggingface_hub's "repository
    # not found" text also mentions gated repos, and a typo in the repo id
    # comes back as 401/404 RepositoryNotFoundError.
    if "GatedRepoError" in names or has(
        "cannot access gated repo", "trying to access a gated repo"
    ):
        return ModelLoadError(
            f"{model_id} is a gated model and this machine doesn't have access to it.",
            hint=f"Accept the licence on https://huggingface.co/{model_id} and set HF_TOKEN "
            "(or run `hf auth login`).",
        )
    if names & {"RepositoryNotFoundError", "HFValidationError", "RevisionNotFoundError"} or has(
        "is not a local folder",
        "not a valid model identifier",
        "repo id must",
        "repository not found",
        "revision not found",
        "does not appear to have",
    ):
        return ModelLoadError(
            f"Model not found: {model_id}",
            hint="Check the Hugging Face repo id (e.g. Qwen/Qwen2.5-1.5B-Instruct), or that the "
            "local folder contains config.json, the tokenizer files and the weights. A private "
            "repo also needs HF_TOKEN.",
        )
    refused = re.search(r"\b(401|403)\b\W*(client error|unauthorized|forbidden)", low)
    if "ProxyError" not in names and (refused or has("unauthorized", "forbidden")):
        status = f" (HTTP {refused.group(1)})" if refused else ""
        return ModelLoadError(
            f"huggingface.co refused to send {model_id}{status}.",
            hint="If HF_TOKEN is set, check that it is valid (or run `hf auth login`); otherwise "
            "a corporate proxy or firewall may be blocking huggingface.co.",
        )
    if has("trust_remote_code"):
        return ModelLoadError(
            f"{model_id} needs custom code from its repository, which this app doesn't run.",
            hint="Choose a model that transformers supports natively, or pip install -U transformers.",
        )
    if has("for this kind of automodel"):
        # Mostly caught by _not_text_generation before loading; this is for
        # configs it lets through (no architectures listed).
        config = re.search(r"configuration class <class '(?:\w+\.)*(\w+)'>", str(exc))
        return _not_text_generation_error(model_id, config.group(1) if config else None)
    if has(
        "does not recognize this architecture",
        "unrecognized configuration class",
        "unrecognized model",
    ):
        return ModelLoadError(
            f"This version of transformers can't load {model_id} ({detail}).",
            hint="Update it: pip install -U transformers",
        )
    if names & {"ModuleNotFoundError", "ImportError"} or has(
        "pip install", "requires the", "you need to have"
    ):
        missing = next((e.name for e in chain if isinstance(e, ImportError) and e.name), None)
        package = missing.split(".")[0] if missing else None
        if package is None:
            known = ("bitsandbytes", "accelerate", "sentencepiece", "protobuf", "tiktoken")
            package = next((name for name in known if name in low), None)
        return ModelLoadError(
            f"A package needed to load {model_id} is missing ({detail}).",
            hint=f"pip install -U {package}"
            if package
            else "pip install -U transformers accelerate",
        )
    if _offline_mode() and (
        names & {"LocalEntryNotFoundError", "OfflineModeIsEnabled"}
        or has("couldn't connect", "offline")
    ):
        return ModelLoadError(
            f"{model_id} isn't in the local Hugging Face cache, and offline mode is on.",
            hint="Unset HF_HUB_OFFLINE to allow the download, or check the model name.",
        )
    network_types = {
        "SSLError",
        "SSLCertVerificationError",
        "ConnectionError",
        "ConnectError",
        "ProxyError",
        "Timeout",
        "ReadTimeout",
        "ConnectTimeout",
        "TimeoutException",
        "LocalEntryNotFoundError",
    }
    if (
        names & network_types
        or has(
            "couldn't connect",
            "certificate verify failed",
            "max retries exceeded",
            "connection error",
            "timed out",
            "proxy",
        )
        or re.search(r"\bssl", low)
    ):
        return ModelLoadError(
            f"Couldn't download {model_id} from huggingface.co ({detail}).",
            hint="Check your internet connection and proxy settings (HTTPS_PROXY). Corporate TLS "
            "inspection is trusted through the operating-system certificate store (truststore) "
            "unless BTR_NO_TRUSTSTORE is set.",
        )
    hint = "Check the model name and precision, or choose another model."
    log_file = current_log_file()
    if log_file is not None:
        hint += f" The full message is in {log_file}."
    return ModelLoadError(f"Could not load {model_id}: {detail}", hint=hint)


# Weight bytes per parameter at each precision (4-bit NF4 plus its scales
# and the layers it leaves at 16 bits).
_BYTES_PER_PARAM = {"fp32": 4.0, "fp16": 2.0, "bf16": 2.0, "4bit": 0.7}


def _ram_error(model_id: str, precision: str, device: str | None) -> ModelLoadError:
    """Main memory (not GPU memory) ran out while loading."""
    per_param = _BYTES_PER_PARAM.get(precision)
    where = " on the CPU" if device == "cpu" else ""
    message = f"Not enough memory (RAM) to load {model_id} at {precision}{where}"
    spec = find_model(model_id)
    if per_param is not None and spec is not None and spec.params_b:
        needed = max(1, round(spec.params_b * per_param))
        message += (
            f": its {spec.size_label} parameters need about {needed} GB at "
            f"{per_param:g} bytes each."
        )
    elif per_param is not None:
        message += f": the weights take about {per_param:g} bytes per parameter."
    else:
        message += "."
    if device == "cpu" and per_param is not None:
        hint = (
            f"Choose a smaller model (a 1.5B one needs about {round(1.5 * per_param)} GB at "
            f"{precision}), and close other programs."
        )
    else:
        hint = "Choose a smaller model, and close other programs."
    return ModelLoadError(message, hint=hint)


def _gpu_memory_error(model_id: str, precision: str, why: str = "") -> ModelLoadError:
    if precision == "4bit":
        smaller = "a smaller model"
    elif precision == "fp32":
        smaller = "a smaller model, or fp16 or 4-bit precision"
    else:
        smaller = "a smaller model or 4-bit precision"
    return ModelLoadError(
        f"Not enough GPU memory to load {model_id} at {precision}" + (f": {why}." if why else "."),
        hint=f"Choose {smaller}, and close other programs using the GPU.",
    )


def _not_text_generation_error(model_id: str, kind: str | None) -> ModelLoadError:
    """A known architecture that can't generate text: BERT-style encoders,
    T5-style encoder-decoders. Upgrading transformers won't change that."""
    named = f" ({kind})" if kind else ""
    return ModelLoadError(
        f"{model_id}{named} isn't a text-generation (decoder-only) model, so it can't answer "
        "prompts here.",
        hint="Choose an instruct or chat model such as Qwen/Qwen2.5-1.5B-Instruct; "
        "BERT-style and T5-style models aren't supported.",
    )


# ----------------------------------------------------------------------
# Memory helpers
# ----------------------------------------------------------------------


def _is_oom(exc: BaseException, torch: Any) -> bool:
    oom_type = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)
    if isinstance(oom_type, type) and isinstance(exc, oom_type):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _release_memory(torch: Any) -> None:
    """Collect garbage (tensors in reference cycles), then empty the cache."""
    gc.collect()
    _empty_cache(torch)


def _empty_cache(torch: Any) -> None:
    if torch is None:
        return
    try:
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        mps = getattr(torch, "mps", None)
        backend = getattr(torch.backends, "mps", None)
        if mps is not None and backend is not None and backend.is_available():
            mps.empty_cache()
    except Exception:  # best effort only
        pass
