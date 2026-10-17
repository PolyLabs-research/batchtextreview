"""API keys for the cloud model providers (OpenAI, Anthropic).

A key is looked up, first match wins, in:

1. **The app** - entered on the Models & settings page and kept in memory
   only, until the app stops (shared by every browser tab of this app).
2. **The environment** - ``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY``.
3. **The system keyring** - saved from the Models & settings page into the
   operating system's credential store (Windows Credential Manager, the
   macOS Keychain, the Secret Service on Linux) through the optional
   ``keyring`` package.

Without a key, the Anthropic SDK can still sign in with
``ANTHROPIC_AUTH_TOKEN`` or a profile saved by ``ant auth login``, and an
OpenAI-compatible server (``OPENAI_BASE_URL``) may need no key at all - see
:func:`credentials_available`.

Keys are never written to the results database or the log, and only
:func:`masked` forms of them are ever shown.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from typing import Literal
from urllib.parse import urlparse

from batchtextreview.models import API_PROVIDERS

log = logging.getLogger(__name__)

KeySource = Literal["session", "env", "keyring"]

ENV_VARS: dict[str, str] = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}
BASE_URL_VARS: dict[str, str] = {"anthropic": "ANTHROPIC_BASE_URL", "openai": "OPENAI_BASE_URL"}
"""Another server to send a provider's requests to (a company gateway, an
OpenAI-compatible server); the SDKs read these themselves."""
ANTHROPIC_OTHER_CREDENTIALS = ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE")
"""Other ways the Anthropic SDK finds credentials (an OAuth token, a profile
saved by ``ant auth login``)."""
KEYRING_SERVICE = "batchtextreview"
"""Service name of saved keys in the system keyring (one entry per provider)."""
SDK_SIGN_IN_TTL_S = 30.0
"""How long :func:`credentials_available` trusts its last check for an
Anthropic sign-in without a key (it reads the SDK's profile files)."""

_lock = threading.Lock()
_session: dict[str, str] = {}
_saved: dict[str, str | None] = {}
"""Keyring lookups already made (None: nothing saved), so pages that rerun
every second don't query the credential store each time. A lookup that
failed is not kept."""
_writes = 0
"""Keyring saves and deletes so far: a lookup that started before one of
them must not overwrite what it wrote."""
_sdk_sign_in: tuple[float, bool] | None = None


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


def base_url(provider: str) -> str:
    """The server ``provider``'s requests go to instead of the provider's
    own (``OPENAI_BASE_URL`` / ``ANTHROPIC_BASE_URL``), or ""."""
    return _clean(os.environ.get(BASE_URL_VARS[_check(provider)]))


def recipient(provider: str) -> str:
    """Who receives the texts: the provider's name, or the host of the
    server set with :func:`base_url`."""
    url = base_url(provider)
    if not url:
        return API_PROVIDERS[provider]
    host = urlparse(url if "://" in url else f"https://{url}").netloc or url
    return host


def credentials_available(provider: str) -> bool:
    """True when a request to ``provider`` can authenticate: a key from
    :func:`lookup`, an OpenAI-compatible server (``OPENAI_BASE_URL``, which
    may need no key), or an Anthropic sign-in without a key
    (``ANTHROPIC_AUTH_TOKEN``, ``ANTHROPIC_PROFILE`` or a profile saved by
    ``ant auth login``)."""
    if lookup(provider):
        return True
    if provider == "openai":
        return bool(base_url("openai"))
    if any(_clean(os.environ.get(name)) for name in ANTHROPIC_OTHER_CREDENTIALS):
        return True
    return _anthropic_sdk_signed_in()


def _anthropic_sdk_signed_in() -> bool:
    """Whether the Anthropic SDK finds credentials of its own (a profile
    saved by ``ant auth login``, workload identity). Cached briefly."""
    global _sdk_sign_in
    now = time.monotonic()
    with _lock:
        cached = _sdk_sign_in
    if cached is not None and now - cached[0] < SDK_SIGN_IN_TTL_S:
        return cached[1]
    signed_in = False
    try:
        import anthropic

        client = anthropic.Anthropic(max_retries=0)
        try:
            signed_in = any(
                getattr(client, name, None) for name in ("api_key", "auth_token", "credentials")
            )
        finally:
            client.close()
    except Exception:
        signed_in = False
    with _lock:
        _sdk_sign_in = (now, signed_in)
    return signed_in


def fingerprint(provider: str) -> str:
    """A short hash of everything that decides which account and server a
    request to ``provider`` uses (key, server, other sign-in variables), so
    a model already loaded can be loaded again when any of them changes.
    Never shown or logged; not part of any run id."""
    found = lookup(provider)
    parts = [provider, found[0] if found else "", base_url(provider)]
    if provider == "anthropic":
        parts += [_clean(os.environ.get(name)) for name in ANTHROPIC_OTHER_CREDENTIALS]
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:16]


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
    secure backend (not the "fail" or "null" placeholder, nor a
    ``keyrings.alt`` file backend, which stores keys readably)."""
    return _keyring_backend() is not None


def _keyring_backend() -> object | None:
    try:
        import keyring
    except Exception:
        return None
    try:
        backend = keyring.get_keyring()
    except Exception:
        return None
    module = type(backend).__module__
    name = f"{module}.{type(backend).__name__}".lower()
    if module.startswith(("keyring.backends.fail", "keyrings.alt")) or "null" in name:
        return None
    return backend


def saved_key(provider: str) -> str | None:
    """The key saved in the system keyring for ``provider``, or None."""
    _check(provider)
    with _lock:
        if provider in _saved:
            return _saved[provider]
        started_at = _writes
    value: str | None = None
    if _keyring_backend() is not None:
        try:
            import keyring

            value = _clean(keyring.get_password(KEYRING_SERVICE, provider)) or None
        except Exception:
            # A locked keyring, say: try again next time rather than
            # remembering "nothing saved" until the app restarts.
            log.warning("Couldn't read the saved %s key from the keyring", provider, exc_info=True)
            return None
    with _lock:
        if _writes == started_at:  # no save or delete since the lookup began
            _saved[provider] = value
    return value


def save_key(provider: str, key: str) -> None:
    """Save ``key`` in the system keyring. Raises RuntimeError when no
    keyring is available or saving fails."""
    global _writes
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
        _writes += 1
        _saved[provider] = value


def delete_saved_key(provider: str) -> bool:
    """Remove the key saved in the system keyring. True if one was removed,
    False if none was saved. Raises RuntimeError when the keyring refuses."""
    global _writes
    _check(provider)
    removed = False
    if _keyring_backend() is not None:
        import keyring

        try:
            keyring.delete_password(KEYRING_SERVICE, provider)
            removed = True
        except Exception as exc:
            if type(exc).__name__ != "PasswordDeleteError":  # not just "nothing saved"
                raise RuntimeError(f"Couldn't remove the saved key: {exc}") from exc
    with _lock:
        _writes += 1
        _saved[provider] = None
    return removed


def masked(key: str) -> str:
    """A key shown safely: the last four characters, after the first ones
    only for a long key (``sk-ant-…a1b2``); the last two of a short one."""
    value = _clean(key)
    if not value:
        return ""
    if len(value) < 12:
        return "…" + value[-2:]
    if len(value) < 24:
        return "…" + value[-4:]
    head = value[:7] if value.startswith("sk-ant-") else value[:3]
    return f"{head}…{value[-4:]}"


def _reset_for_tests() -> None:
    """Forget session keys and cached lookups."""
    global _sdk_sign_in
    with _lock:
        _session.clear()
        _saved.clear()
        _sdk_sign_in = None
