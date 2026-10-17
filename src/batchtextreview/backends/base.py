"""Backend protocol, errors, and device / precision helpers.

A *backend* is a loaded model that turns a batch of prompts into a batch of
completions. The engine only ever talks to this protocol, so it can be tested
with :class:`~batchtextreview.backends.fake.FakeBackend` and no GPU.

Importing this module must stay cheap: torch is imported lazily inside
functions, never at module level.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

# Precision names. "auto" is only ever a *request*; it is resolved to one of
# the concrete values before it reaches a run id or a backend.
AUTO = "auto"
PRECISIONS: tuple[str, ...] = ("auto", "4bit", "fp16", "bf16", "fp32")
CONCRETE_PRECISIONS: tuple[str, ...] = ("4bit", "fp16", "bf16", "fp32", "none")
PRECISION_LABELS: dict[str, str] = {
    "auto": "Auto",
    "4bit": "4-bit (NF4)",
    "fp16": "16-bit (fp16)",
    "bf16": "16-bit (bf16)",
    "fp32": "32-bit (fp32)",
    "none": "n/a",
}


@runtime_checkable
class Backend(Protocol):
    """A loaded model.

    Optional attributes, deliberately not protocol members (so a backend
    without them still satisfies ``isinstance(x, Backend)``); read them with
    ``getattr(backend, name, None)``:

    ``last_hit_limit: list[bool]``
        Set by every ``generate()`` call: one flag per prompt of that call,
        in order, True where the completion was cut off at
        ``max_new_tokens`` rather than finished (so a label may be
        incomplete). Empty after a call that raised. A backend that can't
        tell doesn't have the attribute.
    ``last_row_errors: list[str | None]``
        Set by every ``generate()`` call of a backend that answers each
        prompt separately (the cloud APIs): an error message for each prompt
        that failed on its own - its output is then "" - and None for the
        others. ``generate()`` raises only when the whole call failed.
    ``last_usage: tuple[int, int]``
        (input tokens, output tokens) the last ``generate()`` call used, as
        the provider counted them.
    """

    model_id: str
    precision: str

    def generate(self, prompts: Sequence[str], max_new_tokens: int) -> list[str]:
        """Greedy completions, one per prompt, in order. Raises on failure."""
        ...

    def close(self) -> None:
        """Release memory (VRAM). The backend is unusable afterwards."""
        ...


class BackendError(RuntimeError):
    """Base class for backend failures that carry a user-facing hint."""

    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.hint = hint

    def __str__(self) -> str:
        base = super().__str__()
        return f"{base} {self.hint}" if self.hint else base


class ModelLoadError(BackendError):
    """The model could not be loaded (download, licence, memory, deps...)."""


class GenerationError(BackendError):
    """A generation call failed for a reason other than one row's content."""

    def __init__(self, message: str, hint: str | None = None, *, fatal: bool = False) -> None:
        super().__init__(message, hint)
        self.fatal = fatal
        """No later call can succeed either (a rejected API key, a model the
        key can't use): the engine stops the run instead of retrying."""


@dataclass(frozen=True)
class DeviceInfo:
    kind: str
    """"cuda", "mps", "cpu", or "none" when torch is not installed or fails to
    import (``name`` then says which, with the import error)."""
    name: str
    total_memory_gb: float | None = None
    bitsandbytes: bool = False
    torch_version: str | None = None
    transformers_version: str | None = None
    cuda_version: str | None = None

    @property
    def has_torch(self) -> bool:
        return self.kind != "none"

    @property
    def summary(self) -> str:
        """One-line description, e.g. "NVIDIA GeForce RTX 4060 - 8.0 GB"."""
        if self.total_memory_gb:
            return f"{self.name} - {self.total_memory_gb:.1f} GB"
        return self.name


_MAX_ERROR_CHARS = 200


def _torch_load_failure(exc: BaseException) -> str:
    """DeviceInfo.name for a torch install that exists but fails to import."""
    lines = str(exc).strip().splitlines()
    detail = f"{type(exc).__name__}: {lines[0]}" if lines else type(exc).__name__
    if len(detail) > _MAX_ERROR_CHARS:
        detail = detail[: _MAX_ERROR_CHARS - 3] + "..."
    return f"PyTorch is installed but failed to load ({detail})"


@functools.lru_cache(maxsize=1)
def device_info() -> DeviceInfo:
    """Describe the compute device. Never raises; cached for the process."""
    try:
        import torch
    except Exception as exc:
        if isinstance(exc, ModuleNotFoundError) and exc.name == "torch":
            return DeviceInfo(kind="none", name="PyTorch not installed")
        # Installed but unusable - typically a DLL that fails to load on
        # Windows. Keep the real error: "not installed" would send the user
        # off to reinstall for nothing.
        return DeviceInfo(kind="none", name=_torch_load_failure(exc))

    try:
        import transformers

        tf_version: str | None = transformers.__version__
    except Exception:
        tf_version = None

    try:
        import bitsandbytes  # noqa: F401

        bnb = True
    except Exception:
        bnb = False

    torch_version: str | None = getattr(torch, "__version__", None)
    # The CUDA version torch was *built* for (None for CPU-only wheels) - kept
    # on the CPU fallback too, where it tells "CPU-only torch" apart from
    # "CUDA torch but no usable GPU/driver".
    try:
        cuda_build: str | None = getattr(torch.version, "cuda", None)
    except Exception:
        cuda_build = None

    try:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return DeviceInfo(
                kind="cuda",
                name=torch.cuda.get_device_name(0),
                total_memory_gb=props.total_memory / 1024**3,
                bitsandbytes=bnb,
                torch_version=torch_version,
                transformers_version=tf_version,
                cuda_version=cuda_build,
            )
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return DeviceInfo(
                kind="mps",
                name="Apple GPU (MPS)",
                bitsandbytes=False,
                torch_version=torch_version,
                transformers_version=tf_version,
            )
    except Exception:
        pass
    return DeviceInfo(
        kind="cpu",
        name="CPU (no CUDA GPU detected)",
        bitsandbytes=False,
        torch_version=torch_version,
        transformers_version=tf_version,
        cuda_version=cuda_build,
    )


def resolve_precision(requested: str, device: DeviceInfo | None = None) -> str:
    """Turn a requested precision (possibly "auto") into a concrete one.

    auto -> 4bit on CUDA with bitsandbytes, fp16 on CUDA without it or on
    Apple MPS, fp32 on CPU. An explicit request is returned unchanged
    (the backend raises a clear error if the device can't honour it).
    """
    requested = (requested or AUTO).strip().lower() or AUTO
    if requested not in PRECISIONS:
        raise ValueError(f"Unknown precision {requested!r}; expected one of {PRECISIONS}.")
    if requested != AUTO:
        return requested
    device = device or device_info()
    if device.kind == "cuda":
        return "4bit" if device.bitsandbytes else "fp16"
    if device.kind == "mps":
        return "fp16"
    return "fp32"
