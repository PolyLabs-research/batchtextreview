"""API keys for the cloud model providers (OpenAI, Anthropic).

A key is looked up, first match wins, in:

1. **This session** - entered on the Models & settings page and kept in
   memory only, until the app stops.
2. **The environment** - ``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY``.
3. **The system keyring** - saved from the Models & settings page into the
   operating system's credential store (Windows Credential Manager, the
   macOS Keychain, the Secret Service on Linux) through the optional
   ``keyring`` package.

Keys are never written to the results database or the log, and only
:func:`masked` forms of them are ever shown.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Literal

from batchtextreview.models import API_PROVIDERS

log = logging.getLogger(__name__)

KeySource = Literal["session", "env", "keyring"]

ENV_VARS: dict[str, str] = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}
KEYRING_SERVICE = "batchtextreview"
"""Service name of saved keys in the system keyring (one entry per provider)."""

_lock = threading.Lock()
_session: dict[str, str] = {}
_saved: dict[str, str | None] = {}
"""Keyring lookups already made (None: nothing saved), so pages that rerun
every second don't query the credential store each time."""


def _check(provider: str) -> str:
    if provider not in API_PROVIDERS:
        raise ValueError(
            f"Unknown API provider {provider!r}; expected one of {list(API_PROVIDERS)}."
        )
    return provider


def _clean(value: str | None) -> str:
    """Trim whitespace and one pair of surrounding quotes (``set X="sk-..."``
    in cmd keeps them)."""
    text = (value or "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return text


def get_key(provider: str) -> str | None:
    """The key to use for ``provider``, or None when none is set."""
    found = lookup(provider)
    return found[0] if found else None


def key_source(provider: str) -> KeySource | None:
    """Where :func:`get_key`'s key comes from, or None when none is set."""
    found = lookup(provider)
    return found[1] if found else None


def lookup(provider: str) -> tuple[str, KeySource] | None:
    """(key, where it came from) for ``provider``, or None."""
    _check(provider)
    with _lock:
        session = _session.get(provider)
    if session:
        return session, "session"
    env = _clean(os.environ.get(ENV_VARS[provider]))
    if env:
        return env, "env"
    saved = saved_key(provider)
    if saved:
        return saved, "keyring"
    return None


def set_session_key(provider: str, key: str | None) -> None:
    """Use ``key`` for ``provider`` until the app stops (None or "" forgets
    it). Shared by every browser session of this app."""
    _check(provider)
    value = _clean(key)
    with _lock:
        if value:
            _session[provider] = value
        else:
            _session.pop(provider, None)


def keyring_available() -> bool:
    """True when keys can be saved: ``keyring`` is installed and has a
    usable backend (not the "fail" or "null" placeholder)."""
    backend = _keyring_backend()
    return backend is not None


def _keyring_backend() -> object | None:
    try:
        import keyring
        from keyring.backends import fail
    except Exception:
        return None
    try:
        backend = keyring.get_keyring()
    except Exception:
        return None
    name = type(backend).__module__ + "." + type(backend).__name__
    if isinstance(backend, fail.Keyring) or "null" in name.lower():
        return None
    return backend


def saved_key(provider: str) -> str | None:
    """The key saved in the system keyring for ``provider``, or None."""
    _check(provider)
    with _lock:
        if provider in _saved:
            return _saved[provider]
    value: str | None = None
    if _keyring_backend() is not None:
        try:
            import keyring

            value = _clean(keyring.get_password(KEYRING_SERVICE, provider)) or None
        except Exception:
            log.warning("Couldn't read the saved %s key from the keyring", provider, exc_info=True)
    with _lock:
        _saved[provider] = value
    return value


def save_key(provider: str, key: str) -> None:
    """Save ``key`` in the system keyring. Raises RuntimeError when no
    keyring is available or saving fails."""
    _check(provider)
    value = _clean(key)
    if not value:
        raise ValueError("Enter a key to save.")
    if _keyring_backend() is None:
        raise RuntimeError(
            "No system keyring is available to save the key in. Install the keyring "
            "package (pip install keyring), or set the key as an environment variable."
        )
    try:
        import keyring

        keyring.set_password(KEYRING_SERVICE, provider, value)
    except Exception as exc:
        raise RuntimeError(f"Couldn't save the key: {exc}") from exc
    with _lock:
        _saved[provider] = value


def delete_saved_key(provider: str) -> bool:
    """Remove the key saved in the system keyring. True if one was removed."""
    _check(provider)
    removed = False
    if _keyring_backend() is not None:
        try:
            import keyring
            from keyring.errors import PasswordDeleteError

            try:
                keyring.delete_password(KEYRING_SERVICE, provider)
                removed = True
            except PasswordDeleteError:
                removed = False
        except Exception:
            log.warning("Couldn't remove the saved %s key", provider, exc_info=True)
    with _lock:
        _saved[provider] = None
    return removed


def masked(key: str) -> str:
    """A key shown safely: its first characters and last four, e.g.
    ``sk-ant-…a1b2``. Short keys show only their last two characters."""
    value = _clean(key)
    if len(value) <= 12:
        return "…" + value[-2:] if value else ""
    head = value[:7] if value.startswith("sk-ant-") else value[:3]
    return f"{head}…{value[-4:]}"


def _reset_for_tests() -> None:
    """Forget session keys and cached keyring lookups."""
    with _lock:
        _session.clear()
        _saved.clear()
