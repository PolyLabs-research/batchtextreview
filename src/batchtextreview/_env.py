"""Process-environment bootstrap.

Runs once, on package import, before anything imports torch or makes an
HTTPS request. Every setting uses ``setdefault`` so an explicit value in the
user's environment always wins.
"""

from __future__ import annotations

import os

_DONE = False


def bootstrap() -> None:
    """Apply environment defaults. Safe to call more than once."""
    global _DONE
    if _DONE:
        return
    _DONE = True

    # Anaconda's MKL and torch's bundled OpenMP both load libiomp on Windows;
    # without this the second load aborts the process.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    # Windows without developer mode can't symlink; the HF cache still works
    # (it copies instead) so the warning is noise.
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    # Tokenizers' own thread pool fights with our worker thread otherwise.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Validate TLS against the operating-system trust store so corporate
    # TLS inspection (e.g. Zscaler) is trusted for model downloads.
    if os.environ.get("BTR_NO_TRUSTSTORE", "").strip().lower() not in {"1", "true", "yes", "on"}:
        try:
            import truststore

            truststore.inject_into_ssl()
        except Exception:  # not installed, or unsupported platform
            pass
