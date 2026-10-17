"""Model catalogue.

Each local entry is a Hugging Face repo id; every built-in local model fits
an 8 GB GPU at 4-bit. Cloud API models are written ``provider:model``
(``anthropic:claude-haiku-5-5``, ``openai:gpt-6-luna``) - a colon can't
appear in a repo id, so the two never clash. Cached outputs and evaluation
history are keyed by ``repo`` (never by the display label), so labels can
be edited freely.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

DEMO_REPO = "demo://fake"

API_PROVIDERS: dict[str, str] = {"anthropic": "Anthropic", "openai": "OpenAI"}
"""Cloud API providers: the ``provider:`` prefix of a model id -> display name."""
PRICES_AS_OF = "October 2026"
"""When :attr:`ModelSpec.price` values were last checked against the
providers' list prices."""

_MODEL_FILE_SUFFIXES = frozenset(
    {
        ".json",  # config, generation config, tokenizer, safetensors index
        ".safetensors",
        ".bin",
        ".pt",
        ".pth",
        ".gguf",
        ".msgpack",
        ".h5",
        ".model",  # sentencepiece
        ".tiktoken",
        ".txt",  # merges.txt, vocab.txt
        ".jinja",  # chat template
        ".py",  # remote code
    }
)
"""Top-level files of a local model folder that :func:`model_revision`
fingerprints (README, images and hidden files are left out)."""


@dataclass(frozen=True)
class ModelSpec:
    label: str
    """Display name."""
    repo: str
    """Hugging Face repo id, a local model directory, or ``demo://fake``."""
    params_b: float | None = None
    """Parameter count in billions (for display and rough speed hints)."""
    tier: str = ""
    """Short speed/quality hint shown as a badge, e.g. "fastest"."""
    gated: bool = False
    """Needs the licence accepted on huggingface.co and ``HF_TOKEN`` set."""
    experimental: bool = False
    """New architecture; needs an up-to-date ``transformers``."""
    builtin: bool = True
    price: tuple[float, float] | None = None
    """Cloud API models: list price in US dollars per million input and
    output tokens (as of :data:`PRICES_AS_OF`), or None when not known."""

    @property
    def is_demo(self) -> bool:
        return self.repo.startswith("demo://")

    @property
    def provider(self) -> str | None:
        """ "anthropic" or "openai" for a cloud API model, else None."""
        return api_provider(self.repo)

    @property
    def is_api(self) -> bool:
        """True for a cloud API model: the texts are sent to the provider."""
        return self.provider is not None

    @property
    def provider_label(self) -> str:
        """ "Anthropic" / "OpenAI", or "" for a model that runs here."""
        return API_PROVIDERS.get(self.provider or "", "")

    @property
    def api_model(self) -> str:
        """The provider's own model id (``repo`` without the prefix)."""
        return self.repo.split(":", 1)[1] if self.is_api else self.repo

    @property
    def is_local(self) -> bool:
        """True when ``repo`` names an existing local directory."""
        if self.is_demo or self.is_api:
            return False
        try:
            return Path(self.repo).expanduser().is_dir()
        except OSError:
            return False

    @property
    def size_label(self) -> str:
        if self.params_b is None:
            return ""
        if self.params_b < 1:
            return f"{round(self.params_b * 1000)}M"
        return f"{self.params_b:g}B"

    @property
    def hf_url(self) -> str | None:
        if self.is_demo or self.is_api or self.is_local:
            return None
        return f"https://huggingface.co/{self.repo}"


MODELS: tuple[ModelSpec, ...] = (
    ModelSpec("Granite-4.0-350M", "ibm-granite/granite-4.0-350m", 0.35, tier="tiny"),
    ModelSpec("Granite-4.0-1B", "ibm-granite/granite-4.0-1b", 1.0),
    ModelSpec("Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", 1.5, tier="fastest"),
    ModelSpec("Granite-4.0-Micro", "ibm-granite/granite-4.0-micro", 3.0),
    ModelSpec("Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-3B-Instruct", 3.0, tier="faster"),
    ModelSpec("Llama-3.2-3B-Instruct", "meta-llama/Llama-3.2-3B-Instruct", 3.2, gated=True),
    ModelSpec("Phi-3.5-mini-Instruct", "microsoft/Phi-3.5-mini-instruct", 3.8),
    ModelSpec("Phi-4-mini-Instruct", "microsoft/Phi-4-mini-instruct", 3.8),
    ModelSpec("Qwen3-4B-Instruct-2507", "Qwen/Qwen3-4B-Instruct-2507", 4.0),
    ModelSpec("Qwen3.5-4B", "Qwen/Qwen3.5-4B", 4.0, experimental=True),
    ModelSpec("Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct", 7.6, tier="best quality"),
    ModelSpec("Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-8B-Instruct", 8.0, gated=True),
    ModelSpec("Qwen3-8B", "Qwen/Qwen3-8B", 8.2),
    ModelSpec("Qwen3.5-9B", "Qwen/Qwen3.5-9B", 9.0, experimental=True),
)

API_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec("Claude Haiku 5.5", "anthropic:claude-haiku-5-5", tier="fastest", price=(0.10, 0.50)),
    ModelSpec("Claude Sonnet 5.5", "anthropic:claude-sonnet-5-5", price=(2.0, 10.0)),
    ModelSpec(
        "Claude Opus 5.5", "anthropic:claude-opus-5-5", tier="best quality", price=(4.0, 20.0)
    ),
    ModelSpec("GPT-6 Luna", "openai:gpt-6-luna", tier="fastest"),
    ModelSpec("GPT-6.1 Sol", "openai:gpt-6.1-sol"),
    ModelSpec("GPT-6 Astra", "openai:gpt-6-astra", tier="best quality"),
)
"""Built-in cloud API models, cheapest first within each provider. Any
other model of either provider can be entered as ``provider:model``."""

DEMO_MODEL = ModelSpec("Demo (fake outputs)", DEMO_REPO, None, tier="no GPU needed", builtin=True)

# Display labels used by version 1 (llm_batch.MODELS), mapped to repos. Used
# to import old checkpoints/evaluations, whose run ids hashed the label.
LEGACY_LABEL_TO_REPO: dict[str, str] = {
    "Granite-4.0-350M (tiny)": "ibm-granite/granite-4.0-350m",
    "Granite-4.0-1B": "ibm-granite/granite-4.0-1b",
    "Qwen2.5-1.5B-Instruct (fastest)": "Qwen/Qwen2.5-1.5B-Instruct",
    "Granite-4.0-Micro 3B": "ibm-granite/granite-4.0-micro",
    "Qwen2.5-3B-Instruct (faster)": "Qwen/Qwen2.5-3B-Instruct",
    "Llama-3.2-3B-Instruct (gated)": "meta-llama/Llama-3.2-3B-Instruct",
    "Phi-3.5-mini-Instruct": "microsoft/Phi-3.5-mini-instruct",
    "Phi-4-mini-Instruct (3.8B)": "microsoft/Phi-4-mini-instruct",
    "Qwen3-4B-Instruct-2507": "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen3.5-4B (experimental)": "Qwen/Qwen3.5-4B",
    "Qwen2.5-7B-Instruct (best quality)": "Qwen/Qwen2.5-7B-Instruct",
    "Llama-3.1-8B-Instruct (gated)": "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen3-8B": "Qwen/Qwen3-8B",
    "Qwen3.5-9B (experimental)": "Qwen/Qwen3.5-9B",
}


def available_models(include_demo: bool = False) -> list[ModelSpec]:
    """Built-in local models, smallest first; the demo model first when
    enabled. Cloud API models: :func:`api_models`."""
    models = list(MODELS)
    return [DEMO_MODEL, *models] if include_demo else models


def api_models(provider: str | None = None) -> list[ModelSpec]:
    """Built-in cloud API models, of one provider or all."""
    return [s for s in API_MODELS if provider is None or s.provider == provider]


def api_provider(repo: str) -> str | None:
    """ "anthropic" or "openai" when ``repo`` is ``provider:model``, else None."""
    prefix, sep, _ = repo.partition(":")
    return prefix if sep and prefix in API_PROVIDERS else None


def find_model(name: str) -> ModelSpec | None:
    """Look up a model by display label, repo id, or version-1 label."""
    key = name.strip()
    for spec in (*MODELS, *API_MODELS, DEMO_MODEL):
        if key in (spec.label, spec.repo):
            return spec
    repo = LEGACY_LABEL_TO_REPO.get(key)
    if repo:
        return next(s for s in MODELS if s.repo == repo)
    return None


def custom_model(repo_or_path: str) -> ModelSpec:
    """A spec for a user-supplied repo id or local model directory."""
    value = repo_or_path.strip()
    if not value:
        raise ValueError("Enter a Hugging Face repo id or a local model folder.")
    known = find_model(value)
    if known is not None:
        return known
    api = _api_model(value)
    if api is not None:
        return api
    if value.startswith("demo://"):
        return ModelSpec(label=value, repo=value, builtin=False)
    folder = _existing_dir(value)
    if folder is not None:
        # Name the resolved folder: "." or "models/x/" have no useful own name.
        return ModelSpec(label=folder.name or str(folder), repo=str(folder), builtin=False)
    # Mistakes are caught here rather than after Run as "model not found".
    repo = _repo_id(value)
    return find_model(repo) or ModelSpec(label=repo.rsplit("/", 1)[-1], repo=repo, builtin=False)


# Model ids as the providers write them: "claude-opus-5-5", "gpt-6.1-sol",
# "ft:gpt-6-luna:my-org::abc123" (a fine-tune), "org/model" on a compatible server.
_API_MODEL = re.compile(r"[A-Za-z0-9][\w.:/@-]{0,199}")


def _api_model(value: str) -> ModelSpec | None:
    """A spec for ``provider:model`` (provider in any case), None when
    ``value`` names no API provider. Raises ValueError for a bad model id."""
    prefix, sep, name = value.partition(":")
    provider = prefix.strip().lower()
    if not sep or provider not in API_PROVIDERS:
        return None
    name = name.strip()
    if not _API_MODEL.fullmatch(name):
        example = "claude-sonnet-5-5" if provider == "anthropic" else "gpt-6-luna"
        raise ValueError(
            f"Enter the {API_PROVIDERS[provider]} model id after “{provider}:”, "
            f"e.g. {provider}:{example}."
        )
    repo = f"{provider}:{name}"
    return find_model(repo) or ModelSpec(label=name, repo=repo, builtin=False)


_HF_URL = re.compile(r"(?:https?://)?(?:www\.)?(?:huggingface\.co|hf\.co)/", re.IGNORECASE)
# huggingface_hub's own rule (validate_repo_id): one optional namespace, word
# characters plus "-" and ".", not starting or ending with "-" or ".".
_REPO_ID = re.compile(r"(\b[\w\-.]+\b/)?\b[\w\-.]+\b")
_REPO_ID_MAX = 96


def _existing_dir(value: str) -> Path | None:
    try:
        path = Path(value).expanduser()
        return path.resolve() if path.is_dir() else None
    except (OSError, RuntimeError, ValueError):  # unknown ~user, invalid name...
        return None


def _looks_like_path(value: str) -> bool:
    return (
        value.startswith(("/", ".", "~"))
        or "\\" in value
        or re.match(r"[A-Za-z]:", value) is not None  # Windows drive
    )


def _repo_id(value: str) -> str:
    """Normalise what people paste (page URL, trailing "/") to a repo id.

    Raises ValueError for a folder that doesn't exist or text that can't be a
    repo id.
    """
    url = _HF_URL.match(value)
    if url:
        rest = re.split(r"[?#]", value[url.end() :], maxsplit=1)[0]
        value = "/".join([part for part in rest.split("/") if part][:2])
        if not value:
            raise ValueError("That link has no model in it - paste the model page's address.")
    elif _looks_like_path(value) or value.rstrip("/").count("/") > 1:
        raise ValueError(f"Folder not found: {value}")
    value = value.rstrip("/")
    if not _REPO_ID.fullmatch(value) or "--" in value or ".." in value or len(value) > _REPO_ID_MAX:
        raise ValueError(
            f"“{value}” isn't a Hugging Face repo id. Use the form namespace/name "
            "(e.g. Qwen/Qwen2.5-0.5B-Instruct) or give an existing local model folder."
        )
    return value


def resolve_model(name: str) -> ModelSpec:
    """Like :func:`find_model`, but falls back to :func:`custom_model`."""
    return find_model(name) or custom_model(name)


def model_revision(repo: str) -> str:
    """A short fingerprint of a local model folder's files, or "" for a hub
    repo id, the demo model, or a folder that can't be listed.

    Built from the name, size and modification time of the folder's
    top-level model files (config, weights, tokenizer, chat template), so a fine-tune
    saved again into the same folder gets a new value - and with it a new
    run id and a fresh load - while an untouched folder keeps its value.
    No file is read, so this is cheap even for a folder of large weights.

    Hub repos get "": their snapshot commit is only known once downloaded,
    and folding it in would change the run id after the first download.
    """
    if repo.startswith("demo://") or api_provider(repo):
        return ""
    folder = _existing_dir(repo)
    if folder is None:
        return ""
    files: list[tuple[str, int, int]] = []
    try:
        with os.scandir(folder) as entries:
            for entry in entries:
                name = entry.name
                if name.startswith(".") or Path(name).suffix.lower() not in _MODEL_FILE_SUFFIXES:
                    continue
                try:
                    if not entry.is_file():  # follows symlinks (HF cache snapshots)
                        continue
                    info = entry.stat()
                except OSError:
                    continue
                files.append((name, info.st_size, info.st_mtime_ns))
    except OSError:
        return ""
    payload = json.dumps(sorted(files))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
