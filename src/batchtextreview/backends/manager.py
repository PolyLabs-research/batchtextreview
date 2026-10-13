"""Process-wide model holder.

At most one model lives in memory at a time - switching models frees the old
one first, which matters on an 8 GB card where two 7B models won't co-exist.
All generation goes through :meth:`ModelManager.acquire`, which serialises
access with a lock, so two browser sessions can't run the GPU at once or swap
the model out from under each other.
"""

from __future__ import annotations

import contextlib
import gc
import logging
import sys
import threading
from collections.abc import Callable, Iterator

from batchtextreview.backends.base import Backend
from batchtextreview.models import model_revision

BackendFactory = Callable[[str, str], Backend]
"""(model_id, precision) -> loaded Backend. May raise ModelLoadError."""

_log = logging.getLogger(__name__)


def default_factory(model_id: str, precision: str) -> Backend:
    """Demo model ids (``demo://...``) -> FakeBackend; anything else ->
    TransformersBackend (imported lazily so torch is only loaded when a real
    model is needed)."""
    if model_id.startswith("demo://"):
        from batchtextreview.backends.fake import demo_backend

        return demo_backend(model_id)

    from batchtextreview.backends.transformers_backend import TransformersBackend

    # An "auto" precision is resolved by the backend for the device it finds.
    return TransformersBackend(model_id, precision)


class ModelManager:
    def __init__(self, factory: BackendFactory | None = None) -> None:
        """``factory`` defaults to :func:`default_factory`."""
        self._factory: BackendFactory = factory or default_factory
        # Not re-entrant on purpose: a thread that already holds the model
        # must not swap it (or unload it) from under itself.
        self._lock = threading.Lock()
        self._backend: Backend | None = None
        self._loaded: tuple[str, str] | None = None
        self._revision = ""
        """``models.model_revision`` of the loaded model when it was loaded."""
        self._holders = 0

    @property
    def loaded(self) -> tuple[str, str] | None:
        """(model_id, precision) of the loaded model, or None."""
        return self._loaded

    def is_loaded(self, model_id: str, precision: str) -> bool:
        """True if :meth:`acquire` would reuse the loaded model: it is this
        (model_id, precision) and, for a local folder, the folder's files
        haven't changed since it was loaded."""
        return (
            self._backend is not None
            and self._loaded == (model_id, precision)
            and self._revision == model_revision(model_id)
        )

    @property
    def busy(self) -> bool:
        """True while some thread holds :meth:`acquire`."""
        return self._holders > 0

    @contextlib.contextmanager
    def acquire(self, model_id: str, precision: str) -> Iterator[Backend]:
        """Hold the generation lock and yield the backend for this model.

        Loads (or swaps to) the model if needed. Swapping closes the previous
        backend and runs ``gc.collect()`` + ``torch.cuda.empty_cache()`` (only
        if torch is already imported) *before* loading the new one. If loading
        fails, the manager is left with nothing loaded (never a stale label
        pointing at a freed model) and the exception propagates.

        A local model folder whose files changed since it was loaded (a
        fine-tune saved again into it) is loaded again, so a long-lived
        server never keeps generating with the old weights.
        """
        with self._lock:
            self._holders += 1
            try:
                yield self._ensure_loaded(model_id, precision)
            finally:
                self._holders -= 1

    def unload(self) -> bool:
        """Free the loaded model (waits for the lock). True if one was loaded."""
        with self._lock:
            return self._release()

    # ------------------------------------------------------------------
    # Internals - the caller holds self._lock.
    # ------------------------------------------------------------------

    def _ensure_loaded(self, model_id: str, precision: str) -> Backend:
        key = (model_id, precision)
        # Taken before loading: if the folder changes during the load, the
        # next acquire sees a mismatch and loads again (never the reverse).
        revision = model_revision(model_id)
        if self._backend is not None and self._loaded == key:
            if self._revision == revision:
                return self._backend
            _log.info("Model folder %s changed since it was loaded - loading it again", model_id)
        self._release()
        _log.info("Loading model %s (%s)", model_id, precision)
        try:
            backend = self._factory(model_id, precision)
        except BaseException:
            # Whatever the factory got as far as allocating is garbage now.
            _free_accelerator_memory()
            raise
        self._backend, self._loaded, self._revision = backend, key, revision
        return backend

    def _release(self) -> bool:
        """Forget, close and free the current backend. True if there was one."""
        backend = self._backend
        # Clear the state first, so that even if close() raises nothing ever
        # points at a half-freed model again (the v1 stale-label bug).
        self._backend, self._loaded, self._revision = None, None, ""
        if backend is None:
            return False
        _log.info("Unloading model %s", getattr(backend, "model_id", "?"))
        try:
            backend.close()
        except Exception:
            _log.warning("Error while closing the previous model", exc_info=True)
        del backend
        _free_accelerator_memory()
        return True


def _free_accelerator_memory() -> None:
    """gc.collect() and empty the CUDA cache if torch is already imported."""
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is None:
        return
    try:
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            torch.mps.empty_cache()
    except Exception:  # best effort only
        pass


_lock = threading.Lock()
_DEFAULT: ModelManager | None = None


def default_manager() -> ModelManager:
    """Lazily-created process-wide manager (shared by UI sessions and CLI)."""
    global _DEFAULT
    with _lock:
        if _DEFAULT is None:
            _DEFAULT = ModelManager()
        return _DEFAULT
