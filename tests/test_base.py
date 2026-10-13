"""Tests for batchtextreview.backends.base (protocol, errors, device, precision)."""

from __future__ import annotations

import importlib.abc
import subprocess
import sys
import textwrap
import types
from collections.abc import Iterator, Sequence

import pytest

from batchtextreview.backends import base
from batchtextreview.backends.base import (
    AUTO,
    CONCRETE_PRECISIONS,
    PRECISION_LABELS,
    PRECISIONS,
    Backend,
    BackendError,
    DeviceInfo,
    GenerationError,
    ModelLoadError,
    device_info,
    resolve_precision,
)

GIB = 1024**3

CUDA_BNB = DeviceInfo(kind="cuda", name="RTX 4060", total_memory_gb=8.0, bitsandbytes=True)
CUDA_NO_BNB = DeviceInfo(kind="cuda", name="RTX 4060", total_memory_gb=8.0, bitsandbytes=False)
MPS = DeviceInfo(kind="mps", name="Apple GPU (MPS)")
CPU = DeviceInfo(kind="cpu", name="CPU")
CPU_WITH_BNB = DeviceInfo(kind="cpu", name="CPU", bitsandbytes=True)
NO_TORCH = DeviceInfo(kind="none", name="PyTorch not installed")
ALL_DEVICES = [CUDA_BNB, CUDA_NO_BNB, MPS, CPU, CPU_WITH_BNB, NO_TORCH]

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------


def test_precision_constants_are_consistent() -> None:
    assert AUTO in PRECISIONS
    assert AUTO not in CONCRETE_PRECISIONS
    assert set(PRECISIONS) - {AUTO} <= set(CONCRETE_PRECISIONS)
    assert set(PRECISION_LABELS) == set(PRECISIONS) | set(CONCRETE_PRECISIONS)
    assert all(PRECISION_LABELS.values())


# --------------------------------------------------------------------------
# resolve_precision
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("device", "expected"),
    [
        (CUDA_BNB, "4bit"),
        (CUDA_NO_BNB, "fp16"),
        (MPS, "fp16"),
        (CPU, "fp32"),
        # bitsandbytes on a CPU-only box must never mean 4-bit (v1 defect 4).
        (CPU_WITH_BNB, "fp32"),
        (NO_TORCH, "fp32"),
        (DeviceInfo(kind="mps", name="M3", bitsandbytes=True), "fp16"),
    ],
    ids=["cuda+bnb", "cuda", "mps", "cpu", "cpu+bnb", "no-torch", "mps+bnb"],
)
def test_auto_resolves_per_device(device: DeviceInfo, expected: str) -> None:
    assert resolve_precision("auto", device) == expected
    assert resolve_precision(AUTO, device) == expected


@pytest.mark.parametrize("requested", ["4bit", "fp16", "bf16", "fp32"])
@pytest.mark.parametrize("device", ALL_DEVICES, ids=lambda d: f"{d.kind}-bnb{d.bitsandbytes}")
def test_explicit_request_is_returned_unchanged(requested: str, device: DeviceInfo) -> None:
    assert resolve_precision(requested, device) == requested


@pytest.mark.parametrize(
    ("requested", "expected"),
    [("FP16", "fp16"), (" bf16 ", "bf16"), ("4BIT", "4bit"), ("Auto", "fp32"), (" AUTO\n", "fp32")],
)
def test_request_is_case_and_whitespace_insensitive(requested: str, expected: str) -> None:
    assert resolve_precision(requested, CPU) == expected


@pytest.mark.parametrize("requested", ["", "   ", None])
def test_blank_request_means_auto(requested: str | None) -> None:
    assert resolve_precision(requested, CUDA_BNB) == "4bit"  # type: ignore[arg-type]
    assert resolve_precision(requested, CPU) == "fp32"  # type: ignore[arg-type]


@pytest.mark.parametrize("requested", ["int8", "8bit", "fp8", "none", "float16", "4-bit"])
def test_unknown_precision_raises(requested: str) -> None:
    with pytest.raises(ValueError, match="Unknown precision"):
        resolve_precision(requested, CPU)


@pytest.mark.parametrize("requested", PRECISIONS)
@pytest.mark.parametrize("device", ALL_DEVICES, ids=lambda d: f"{d.kind}-bnb{d.bitsandbytes}")
def test_result_is_always_concrete(requested: str, device: DeviceInfo) -> None:
    assert resolve_precision(requested, device) in CONCRETE_PRECISIONS


def test_auto_without_device_uses_device_info(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(base, "device_info", lambda: CUDA_NO_BNB)
    assert resolve_precision("auto") == "fp16"
    monkeypatch.setattr(base, "device_info", lambda: CUDA_BNB)
    assert resolve_precision("auto") == "4bit"


def test_explicit_request_never_probes_the_device(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> DeviceInfo:
        raise AssertionError("device_info should not be called")

    monkeypatch.setattr(base, "device_info", boom)
    assert resolve_precision("bf16") == "bf16"


# --------------------------------------------------------------------------
# DeviceInfo
# --------------------------------------------------------------------------


def test_summary_with_memory() -> None:
    info = DeviceInfo(kind="cuda", name="NVIDIA GeForce RTX 4060", total_memory_gb=8.0)
    assert info.summary == "NVIDIA GeForce RTX 4060 - 8.0 GB"
    assert DeviceInfo(kind="cuda", name="GPU", total_memory_gb=7.996).summary == "GPU - 8.0 GB"
    assert DeviceInfo(kind="cuda", name="GPU", total_memory_gb=23.64).summary == "GPU - 23.6 GB"


@pytest.mark.parametrize("memory", [None, 0.0])
def test_summary_without_memory(memory: float | None) -> None:
    info = DeviceInfo(kind="cpu", name="CPU (no CUDA GPU detected)", total_memory_gb=memory)
    assert info.summary == "CPU (no CUDA GPU detected)"


def test_has_torch() -> None:
    assert not NO_TORCH.has_torch
    assert CPU.has_torch and MPS.has_torch and CUDA_BNB.has_torch


# --------------------------------------------------------------------------
# device_info (with a fake torch: no GPU or real torch needed)
# --------------------------------------------------------------------------


@pytest.fixture
def clear_device_cache() -> Iterator[None]:
    device_info.cache_clear()
    yield
    device_info.cache_clear()


def _fake_torch(
    *,
    cuda: bool = False,
    mps: bool = False,
    cuda_raises: bool = False,
    name: str = "NVIDIA GeForce RTX 4060 Laptop GPU",
    memory_bytes: int = 8 * GIB,
    version: str | None = "2.5.1+cu124",
    cuda_build: str | None = "12.4",
) -> types.ModuleType:
    torch = types.ModuleType("torch")
    if version is not None:
        torch.__version__ = version  # type: ignore[attr-defined]

    def is_available() -> bool:
        if cuda_raises:
            raise RuntimeError("CUDA driver initialization failed")
        return cuda

    torch.cuda = types.SimpleNamespace(  # type: ignore[attr-defined]
        is_available=is_available,
        get_device_properties=lambda i: types.SimpleNamespace(total_memory=memory_bytes),
        get_device_name=lambda i: name,
    )
    torch.version = types.SimpleNamespace(cuda=cuda_build)  # type: ignore[attr-defined]
    torch.backends = types.SimpleNamespace(  # type: ignore[attr-defined]
        mps=types.SimpleNamespace(is_available=lambda: mps)
    )
    return torch


def _install(
    monkeypatch: pytest.MonkeyPatch,
    torch: types.ModuleType | None,
    *,
    transformers: str | None = "5.0.0",
    bitsandbytes: bool = False,
) -> None:
    monkeypatch.setitem(sys.modules, "torch", torch)
    if transformers is None:
        monkeypatch.setitem(sys.modules, "transformers", None)
    else:
        tf = types.ModuleType("transformers")
        tf.__version__ = transformers  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "transformers", tf)
    bnb = types.ModuleType("bitsandbytes") if bitsandbytes else None
    monkeypatch.setitem(sys.modules, "bitsandbytes", bnb)


@pytest.mark.usefixtures("clear_device_cache")
def test_device_info_without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, None)  # a None entry makes `import torch` raise ImportError
    info = device_info()
    assert info == DeviceInfo(kind="none", name="PyTorch not installed")
    assert not info.has_torch
    assert resolve_precision("auto", info) == "fp32"


class _BrokenTorchFinder(importlib.abc.MetaPathFinder):
    """Simulates a torch install whose DLLs fail to load (common on Windows)."""

    def find_spec(self, fullname, path, target=None):  # type: ignore[no-untyped-def]
        if fullname == "torch":
            raise OSError("[WinError 126] Error loading fbgemm.dll")
        return None


@pytest.mark.usefixtures("clear_device_cache")
def test_device_info_survives_a_broken_torch_install(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr(sys, "meta_path", [_BrokenTorchFinder(), *sys.meta_path])
    info = device_info()
    assert info.kind == "none"
    assert not info.has_torch
    # The real error is kept: "not installed" would send the user off to
    # reinstall for nothing.
    assert info.name.startswith("PyTorch is installed but failed to load")
    assert "OSError: [WinError 126] Error loading fbgemm.dll" in info.name


class _TorchMissingDependencyFinder(importlib.abc.MetaPathFinder):
    """torch itself is present but one of its own imports is missing."""

    def find_spec(self, fullname, path, target=None):  # type: ignore[no-untyped-def]
        if fullname == "torch":
            raise ModuleNotFoundError(
                "No module named 'typing_extensions'", name="typing_extensions"
            )
        return None


@pytest.mark.usefixtures("clear_device_cache")
def test_device_info_missing_torch_dependency_is_not_reported_as_missing_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr(sys, "meta_path", [_TorchMissingDependencyFinder(), *sys.meta_path])
    info = device_info()
    assert info.kind == "none"
    assert "failed to load" in info.name
    assert "typing_extensions" in info.name


class _LongErrorTorchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):  # type: ignore[no-untyped-def]
        if fullname == "torch":
            raise OSError("x" * 1000 + "\nsecond line")
        return None


@pytest.mark.usefixtures("clear_device_cache")
def test_device_info_load_error_is_one_short_line(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr(sys, "meta_path", [_LongErrorTorchFinder(), *sys.meta_path])
    name = device_info().name
    assert "\n" not in name
    assert "second line" not in name
    assert len(name) < 300
    assert name.endswith("...)")


@pytest.mark.usefixtures("clear_device_cache")
def test_device_info_cuda_with_bitsandbytes(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _fake_torch(cuda=True), bitsandbytes=True)
    info = device_info()
    assert info == DeviceInfo(
        kind="cuda",
        name="NVIDIA GeForce RTX 4060 Laptop GPU",
        total_memory_gb=8.0,
        bitsandbytes=True,
        torch_version="2.5.1+cu124",
        transformers_version="5.0.0",
        cuda_version="12.4",
    )
    assert info.summary == "NVIDIA GeForce RTX 4060 Laptop GPU - 8.0 GB"
    assert resolve_precision("auto", info) == "4bit"


@pytest.mark.usefixtures("clear_device_cache")
def test_device_info_cuda_without_bitsandbytes(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _fake_torch(cuda=True, memory_bytes=12 * GIB), bitsandbytes=False)
    info = device_info()
    assert info.kind == "cuda"
    assert info.bitsandbytes is False
    assert info.total_memory_gb == pytest.approx(12.0)
    assert resolve_precision("auto", info) == "fp16"


@pytest.mark.usefixtures("clear_device_cache")
def test_device_info_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    # Even if bitsandbytes imports, it is not usable on Apple GPUs.
    _install(monkeypatch, _fake_torch(mps=True), bitsandbytes=True)
    info = device_info()
    assert info.kind == "mps"
    assert info.name == "Apple GPU (MPS)"
    assert info.bitsandbytes is False
    assert info.total_memory_gb is None
    assert resolve_precision("auto", info) == "fp16"


@pytest.mark.usefixtures("clear_device_cache")
def test_device_info_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _fake_torch(version="2.5.1+cpu", cuda_build=None), bitsandbytes=True)
    info = device_info()
    assert info.kind == "cpu"
    assert info.has_torch
    assert info.bitsandbytes is False
    assert info.torch_version == "2.5.1+cpu"
    assert info.cuda_version is None
    assert info.summary == info.name
    assert resolve_precision("auto", info) == "fp32"


@pytest.mark.usefixtures("clear_device_cache")
def test_device_info_cpu_reports_cuda_build(monkeypatch: pytest.MonkeyPatch) -> None:
    # A CUDA build of torch with no usable GPU: the build version is still
    # reported, which tells it apart from a CPU-only wheel.
    _install(monkeypatch, _fake_torch(cuda=False, cuda_build="12.4"))
    info = device_info()
    assert info.kind == "cpu"
    assert info.cuda_version == "12.4"


@pytest.mark.usefixtures("clear_device_cache")
def test_device_info_cuda_probe_failure_falls_back_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, _fake_torch(cuda_raises=True), bitsandbytes=True)
    info = device_info()
    assert info.kind == "cpu"
    assert resolve_precision("auto", info) == "fp32"


@pytest.mark.usefixtures("clear_device_cache")
def test_device_info_tolerates_odd_torch_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = _fake_torch(version=None)
    del torch.backends
    del torch.version
    _install(monkeypatch, torch, transformers=None)
    info = device_info()
    assert info.kind == "cpu"
    assert info.torch_version is None
    assert info.transformers_version is None
    assert info.cuda_version is None


@pytest.mark.usefixtures("clear_device_cache")
def test_device_info_without_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _fake_torch(cuda=True), transformers=None)
    info = device_info()
    assert info.kind == "cuda"
    assert info.transformers_version is None


@pytest.mark.usefixtures("clear_device_cache")
def test_device_info_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _fake_torch(cuda=True))
    first = device_info()
    _install(monkeypatch, None)
    assert device_info() is first
    device_info.cache_clear()
    assert device_info().kind == "none"


@pytest.mark.usefixtures("clear_device_cache")
def test_device_info_real_environment_never_raises() -> None:
    info = device_info()
    assert info.kind in {"cuda", "mps", "cpu", "none"}
    assert info.name
    assert resolve_precision("auto", info) in CONCRETE_PRECISIONS


def test_importing_light_modules_does_not_import_torch() -> None:
    code = textwrap.dedent(
        """
        import sys
        import batchtextreview.backends.base
        import batchtextreview.config
        import batchtextreview.models
        import batchtextreview.presets
        import batchtextreview.prompting
        heavy = [m for m in ("torch", "transformers", "bitsandbytes") if m in sys.modules]
        print(",".join(heavy))
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=60
    )
    assert out.stdout.strip() == ""


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


def test_backend_error_with_hint() -> None:
    err = BackendError("Model download failed.", hint="Check your network.")
    assert str(err) == "Model download failed. Check your network."
    assert err.hint == "Check your network."
    assert err.args == ("Model download failed.",)


@pytest.mark.parametrize("hint", [None, ""])
def test_backend_error_without_hint(hint: str | None) -> None:
    err = BackendError("Out of memory.", hint)
    assert str(err) == "Out of memory."
    assert err.hint == hint


@pytest.mark.parametrize("cls", [ModelLoadError, GenerationError])
def test_error_hierarchy(cls: type[BackendError]) -> None:
    assert issubclass(cls, BackendError)
    assert issubclass(cls, RuntimeError)
    err = cls("Failed.", hint="Try fp16.")
    assert str(err) == "Failed. Try fp16."
    with pytest.raises(BackendError):
        raise err


# --------------------------------------------------------------------------
# Backend protocol
# --------------------------------------------------------------------------


class _EchoBackend:
    def __init__(self) -> None:
        self.model_id = "echo"
        self.precision = "none"
        self.closed = False

    def generate(self, prompts: Sequence[str], max_new_tokens: int) -> list[str]:
        return [p[:max_new_tokens] for p in prompts]

    def close(self) -> None:
        self.closed = True


class _NoClose:
    model_id = "x"
    precision = "fp32"

    def generate(self, prompts: Sequence[str], max_new_tokens: int) -> list[str]:
        return list(prompts)


def test_backend_protocol_is_runtime_checkable() -> None:
    backend = _EchoBackend()
    assert isinstance(backend, Backend)
    assert backend.generate(["hello", "hi"], 3) == ["hel", "hi"]
    backend.close()
    assert backend.closed


def test_backend_protocol_rejects_incomplete_objects() -> None:
    assert not isinstance(_NoClose(), Backend)
    assert not isinstance(object(), Backend)


def test_last_hit_limit_is_documented_but_optional() -> None:
    assert "last_hit_limit" in (Backend.__doc__ or "")
    # Not a protocol member: a backend without it still counts.
    assert "last_hit_limit" not in getattr(Backend, "__annotations__", {})
    backend = _EchoBackend()
    assert not hasattr(backend, "last_hit_limit")
    assert isinstance(backend, Backend)
