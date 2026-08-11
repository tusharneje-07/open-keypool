"""Core implementation of the open_keypool library.

Provides the `KeyPool` class for managing a pool of API keys with automatic
cooldown, disablement, and rotation strategies to avoid HTTP 429 rate-limit
errors.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum

import cachetools
import httpx


class KeyState(Enum):
    """Possible states for a key in the pool.

    Attributes:
        ACTIVE: The key is healthy and available for use.
        COOLDOWN: The key is temporarily unavailable (rate-limited) and will
            automatically recover after *cooldown_seconds* elapses.
        DISABLED: The key is permanently unusable and will never auto-recover.
    """

    ACTIVE = "active"
    COOLDOWN = "cooldown"
    DISABLED = "disabled"


@dataclass(slots=True)
class _KeyRecord:
    """Internal representation of a single API key and its health metadata."""

    key: str
    state: KeyState = KeyState.ACTIVE
    cooldown_until: float | None = None
    last_used: float = field(default_factory=time.monotonic)
    failure_count: int = 0
    last_status_code: int | None = None
    last_error_code: str | None = None
    last_error_message: str | None = None


class AllKeysExhaustedError(Exception):
    """Raised when no ACTIVE key is available in the pool.

    The exception message includes the soonest recovery time in seconds if any
    key is in COOLDOWN, otherwise it reports that all keys are disabled.
    """

    pass


def mask(key: str) -> str:
    """Return a masked representation of an API key.

    Produces a string of the form ``"sk-1...b2c9"`` — the first few characters
    followed by ``...`` and the last 4 characters. Never use the raw key value
    in logs, print statements, or exception messages.

    Parameters
    ----------
    key : str
        The raw API key to mask.

    Returns
    -------
    str
        Masked key string, e.g. ``"sk-1a2b3...b2c9"``.

    Examples
    --------
    >>> mask("sk-1a2b3c4d5e6f7g8h9i0j")
    'sk-1a2b3...9i0j'
    """
    if len(key) <= 7:
        return key[:3] + "..." + key[-4:] if len(key) >= 4 else "..."
    return key[:6] + "..." + key[-4:]


_DOPPLER_CACHE: cachetools.TTLCache = cachetools.TTLCache(maxsize=64, ttl=3600)
_DOPPLER_DOWNLOAD_URL = "https://api.doppler.com/v3/configs/config/secrets"


class KeyPool:
    """A thread-safe pool of API keys with cooldown and rotation strategies.

    Manages a collection of API keys, cycling through them using a configurable
    strategy (round-robin or least-recently-used). Keys that receive HTTP 429
    responses can be placed on cooldown and will automatically recover after
    the cooldown period. Permanently invalid keys can be explicitly disabled.

    Parameters
    ----------
    keys : list[str]
        Initial list of API key strings. At least one key is required.
    max_retries : int, optional
        Maximum number of retries the caller should attempt per operation.
        Stored as ``self.max_retries`` for the caller's reference; the library
        does not perform automatic retries. Default is 3.
    cooldown_seconds : int, optional
        Number of seconds a key stays in COOLDOWN after being rate-limited
        before it becomes ACTIVE again. Default is 60.
    strategy : str, optional
        Rotation strategy. ``"round_robin"`` cycles through keys in insertion
        order. ``"lru"`` selects the key with the oldest ``last_used``
        timestamp. Default is ``"round_robin"``.

    Raises
    ------
    ValueError
        If *keys* is empty or ``None``.

    Examples
    --------
    >>> pool = KeyPool(["key-a", "key-b", "key-c"], strategy="round_robin")
    >>> pool.get_key()
    >>> pool.mark_success(pool.get_key())
    """

    def __init__(
        self,
        keys: list[str] | None = None,
        max_retries: int = 3,
        cooldown_seconds: int = 60,
        strategy: str = "round_robin",
    ) -> None:
        if not keys:
            raise ValueError("KeyPool requires at least one key (keys must be a non-empty list).")

        if strategy not in ("round_robin", "lru"):
            raise ValueError(f"Unknown strategy '{strategy}'. Use 'round_robin' or 'lru'.")

        self.max_retries: int = max_retries
        self.cooldown_seconds: int = cooldown_seconds
        self.strategy: str = strategy

        self._records: list[_KeyRecord] = [_KeyRecord(key=k) for k in keys]
        self._round_robin_index: int = 0
        self._lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_record(self, key: str) -> _KeyRecord | None:
        """Return the ``_KeyRecord`` for *key*, or ``None`` if not found."""
        for rec in self._records:
            if rec.key == key:
                return rec
        return None

    def _recover_cooldown_keys(self) -> None:
        """Flip COOLDOWN keys whose cooldown has expired back to ACTIVE."""
        now = time.monotonic()
        for rec in self._records:
            if rec.state == KeyState.COOLDOWN and rec.cooldown_until is not None and now >= rec.cooldown_until:
                rec.state = KeyState.ACTIVE
                rec.cooldown_until = None

    def _active_records(self) -> list[_KeyRecord]:
        """Return all currently-ACTIVE records."""
        return [r for r in self._records if r.state == KeyState.ACTIVE]

    @classmethod
    def from_doppler(
        cls,
        token: str,
        project: str,
        config: str,
        key_prefix: str | None = None,
        max_retries: int = 3,
        cooldown_seconds: int = 60,
        strategy: str = "round_robin",
        force_refresh: bool = False,
    ) -> KeyPool:
        """Create a ``KeyPool`` by fetching API keys from Doppler.

        Calls the Doppler secrets-download REST endpoint and populates the
        pool with secret values that match *key_prefix* (if given). Results
        are cached in an in-memory, module-level ``TTLCache`` (1 hour TTL)
        keyed by ``(project, config, key_prefix)`` so that repeated calls
        within the same process avoid redundant network requests.

        The cache is purely in-memory and empties naturally on every fresh
        process start — it is never persisted to disk.

        Parameters
        ----------
        token : str
            Doppler service-token for bearer authentication
            (e.g. ``"dp.st.YOUR_SERVICE_TOKEN"``).
        project : str
            Doppler project name.
        config : str
            Doppler config/environment name (e.g. ``"dev"``, ``"prd"``).
        key_prefix : str | None, optional
            If provided, only secrets whose name starts with this string are
            included as keys. ``None`` (default) includes all secrets.
        max_retries : int, optional
            Passed through to the ``KeyPool`` constructor. Default is 3.
        cooldown_seconds : int, optional
            Passed through to the ``KeyPool`` constructor. Default is 60.
        strategy : str, optional
            Passed through to the ``KeyPool`` constructor.
            Default is ``"round_robin"``.
        force_refresh : bool, optional
            If ``True``, bypass the cache and re-fetch from Doppler even when
            a valid cache entry exists. Default is ``False``.

        Returns
        -------
        KeyPool
            A new ``KeyPool`` instance populated with the fetched keys.

        Raises
        ------
        RuntimeError
            If the Doppler API call fails (non-2xx status) or returns zero
            keys — the cache is **not** populated on failure.

        Examples
        --------
        >>> import os
        >>> pool = KeyPool.from_doppler(
        ...     token=os.getenv("DOPPLER_TOKEN", "dp.st.YOUR_SERVICE_TOKEN"),
        ...     project="refactor-ai",
        ...     config="dev",
        ...     key_prefix="API_KEY_",
        ... )
        """
        cache_key = (project, config, key_prefix)

        if not force_refresh:
            cached = _DOPPLER_CACHE.get(cache_key)
            if cached is not None:
                return cls(
                    keys=list(cached),
                    max_retries=max_retries,
                    cooldown_seconds=cooldown_seconds,
                    strategy=strategy,
                )

        try:
            response = httpx.get(
                _DOPPLER_DOWNLOAD_URL,
                params={"project": project, "config": config},
                headers={"Authorization": f"Bearer {token}"},
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"Doppler API request failed: {exc.__class__.__name__}"
            ) from exc

        secrets: dict[str, str] = data.get("secrets", {})
        fetched_keys: list[str] = []
        for name, secret_data in secrets.items():
            if isinstance(secret_data, dict):
                raw = secret_data.get("raw") or secret_data.get("computed", "")
            else:
                raw = str(secret_data)
            if key_prefix is None or name.startswith(key_prefix):
                fetched_keys.append(raw)

        if not fetched_keys:
            raise RuntimeError(
                f"Doppler returned zero keys for project='{project}', "
                f"config='{config}', key_prefix={key_prefix!r}"
            )

        _DOPPLER_CACHE[cache_key] = tuple(fetched_keys)

        return cls(
            keys=fetched_keys,
            max_retries=max_retries,
            cooldown_seconds=cooldown_seconds,
            strategy=strategy,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_key(self, key: str) -> None:
        """Add a new ACTIVE key to the pool at runtime.

        If the key is already present in the pool, this is a no-op.

        Parameters
        ----------
        key : str
            The API key string to add.

        Examples
        --------
        >>> pool = KeyPool(["key-a"])
        >>> pool.add_key("key-b")
        """
        with self._lock:
            if self._find_record(key) is None:
                self._records.append(_KeyRecord(key=key))

    def remove_key(self, key: str) -> None:
        """Remove a key from the pool regardless of its current state.

        If the key is not in the pool, this is a no-op.

        Parameters
        ----------
        key : str
            The API key string to remove.

        Examples
        --------
        >>> pool = KeyPool(["key-a", "key-b"])
        >>> pool.remove_key("key-b")
        """
        with self._lock:
            rec = self._find_record(key)
            if rec is not None:
                self._records.remove(rec)

    def get_key(self) -> str:
        """Return the next available ACTIVE key according to the pool's strategy.

        Before selecting, any COOLDOWN key whose cooldown period has expired
        is automatically flipped back to ACTIVE.

        **Round-robin** (``strategy="round_robin"``): iterates through keys
        in insertion order, maintaining an internal cursor that wraps around.

        **LRU** (``strategy="lru"``): picks the ACTIVE key with the oldest
        ``last_used`` timestamp and updates it to now upon selection.

        Returns
        -------
        str
            An ACTIVE API key.

        Raises
        ------
        AllKeysExhaustedError
            If no ACTIVE key exists in the pool. The message includes the
            soonest recovery time in seconds when at least one key is in
            COOLDOWN, otherwise it says all keys are disabled.

        Examples
        --------
        >>> pool = KeyPool(["key-a", "key-b"])
        >>> key = pool.get_key()
        >>> pool.mark_success(key)
        """
        with self._lock:
            self._recover_cooldown_keys()
            active = self._active_records()

            if not active:
                cooldown_records = [r for r in self._records if r.state == KeyState.COOLDOWN]
                if cooldown_records:
                    now = time.monotonic()
                    soonest = min(
                        (r.cooldown_until - now for r in cooldown_records if r.cooldown_until is not None),
                        default=None,
                    )
                    if soonest is not None and soonest > 0:
                        raise AllKeysExhaustedError(
                            f"No active keys available. Recovery in {soonest:.1f}s "
                            f"({self._cooldown_summary(cooldown_records, now)})."
                        )
                    else:
                        raise AllKeysExhaustedError(
                            "No active keys available. All keys are in cooldown or disabled."
                        )
                raise AllKeysExhaustedError("No active keys available. All keys are disabled.")

            if self.strategy == "round_robin":
                self._round_robin_index %= len(active)
                rec = active[self._round_robin_index]
                self._round_robin_index += 1
            else:  # lru
                rec = min(active, key=lambda r: r.last_used)
                rec.last_used = time.monotonic()

        return rec.key

    def handle_response(
        self,
        key: str,
        status_code: int,
        headers: dict[str, str] | None = None,
        body: dict | str | None = None,
    ) -> KeyState:
        """Feed an HTTP response to the pool — it decides what to do with the key.

        Introspects the status code and response body and automatically:

        - On **2xx**: marks the key as successful (``mark_success``).
        - On **429** or **413**, or when the response body contains
          ``error.code == "rate_limit_exceeded"``: places the key on
          COOLDOWN using ``Retry-After`` if present, otherwise the pool's
          ``cooldown_seconds``.
        - On **401** or **403**: permanently disables the key
          (``mark_invalid``).
        - On **5xx**: places the key on COOLDOWN (transient server error).

        All relevant details (status code, error code, error message) are
        stored on the key record and surfaced in ``status()``.

        Parameters
        ----------
        key : str
            The API key that was used for the request.
        status_code : int
            HTTP status code from the response.
        headers : dict[str, str] | None, optional
            Response headers (used to extract ``Retry-After``).
        body : dict | str | None, optional
            Parsed JSON body (``dict``) or raw response text (``str``).

        Returns
        -------
        KeyState
            The new state of the key after processing.

        Examples
        --------
        >>> pool = KeyPool(["key-a", "key-b"])
        >>> k = pool.get_key()
        >>> # Successful call:
        >>> pool.handle_response(k, 200, body={"choices": [...]})
        <KeyState.ACTIVE: 'active'>
        >>> # Rate-limit (Groq-style in-body):
        >>> pool.handle_response(k, 200, body={"error": {"code": "rate_limit_exceeded", "message": "TPM limit"}})
        <KeyState.COOLDOWN: 'cooldown'>
        >>> # Re-raise to get a fresh key on cooldown:
        >>> k2 = pool.get_key()
        """
        headers = headers or {}

        # ── parse body for error details ──
        error_code = str(status_code)
        error_message = ""
        if isinstance(body, dict):
            err = body.get("error", {})
            if isinstance(err, dict):
                if err.get("code") == "rate_limit_exceeded":
                    error_code = "rate_limit_exceeded"
                error_message = err.get("message", "")
            elif isinstance(err, str):
                error_message = err
        elif isinstance(body, str):
            error_message = body[:200]

        with self._lock:
            rec = self._find_record(key)
            if rec is None:
                return KeyState.DISABLED  # key doesn't exist, nothing to do

            rec.last_status_code = status_code

            # ── 2xx success ──
            if 200 <= status_code < 300:
                rec.failure_count = 0
                rec.state = KeyState.ACTIVE
                rec.last_error_code = None
                rec.last_error_message = None
                return KeyState.ACTIVE

            # ── rate-limit (429, 413, or rate_limit_exceeded in body) ──
            if status_code in (429, 413) or error_code == "rate_limit_exceeded":
                ra = headers.get("Retry-After")
                try:
                    retry_after = float(ra) if ra else None
                except (ValueError, TypeError):
                    retry_after = None
                rec.state = KeyState.COOLDOWN
                rec.cooldown_until = time.monotonic() + (
                    retry_after if retry_after is not None else self.cooldown_seconds
                )
                rec.failure_count += 1
                rec.last_error_code = error_code
                rec.last_error_message = error_message
                return KeyState.COOLDOWN

            # ── auth failure (401, 403) ──
            if status_code in (401, 403):
                rec.state = KeyState.DISABLED
                rec.last_error_code = error_code
                rec.last_error_message = error_message
                return KeyState.DISABLED

            # ── server error (5xx) — transient, put on cooldown ──
            if 500 <= status_code < 600:
                rec.state = KeyState.COOLDOWN
                rec.cooldown_until = time.monotonic() + self.cooldown_seconds
                rec.failure_count += 1
                rec.last_error_code = error_code
                rec.last_error_message = error_message
                return KeyState.COOLDOWN

            # ── unknown status — also cooldown ──
            rec.state = KeyState.COOLDOWN
            rec.cooldown_until = time.monotonic() + self.cooldown_seconds
            rec.failure_count += 1
            rec.last_error_code = error_code
            rec.last_error_message = error_message
            return KeyState.COOLDOWN

    def mark_rate_limited(
        self,
        key: str,
        retry_after: float | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> None:
        """Mark a key as rate-limited (COOLDOWN).

        The key will remain in COOLDOWN for *retry_after* seconds (or the
        pool's ``cooldown_seconds`` if *retry_after* is ``None``). Its
        ``failure_count`` is incremented.

        Parameters
        ----------
        key : str
            The API key string.
        retry_after : float | None, optional
            Custom cooldown duration in seconds. If ``None``, defaults to
            ``self.cooldown_seconds``.
        error_code : str | None, optional
            Machine-readable code for the last rate-limit error (e.g.
            ``"rate_limit_exceeded"``, ``"413"``). Stored and surfaced in
            ``status()``.
        error_message : str | None, optional
            Human-readable description of the last rate-limit error.
            Stored and surfaced in ``status()``.

        Examples
        --------
        >>> pool = KeyPool(["key-a"])
        >>> pool.mark_rate_limited("key-a", retry_after=30)
        >>> pool.mark_rate_limited("key-a", error_code="rate_limit_exceeded",
        ...                        error_message="TPM limit 8000 exceeded")
        """
        with self._lock:
            rec = self._find_record(key)
            if rec is None:
                return
            rec.state = KeyState.COOLDOWN
            rec.cooldown_until = time.monotonic() + (retry_after if retry_after is not None else self.cooldown_seconds)
            rec.failure_count += 1
            rec.last_error_code = error_code
            rec.last_error_message = error_message

    def mark_invalid(
        self,
        key: str,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> None:
        """Permanently disable a key (DISABLED).

        Disabled keys never auto-recover. Use this when a key returns an
        authentication error (e.g. HTTP 401) rather than a rate-limit error.

        Parameters
        ----------
        key : str
            The API key string.
        error_code : str | None, optional
            Machine-readable error code (e.g. ``"401"``, ``"invalid_api_key"``).
        error_message : str | None, optional
            Human-readable error description.

        Examples
        --------
        >>> pool = KeyPool(["key-a"])
        >>> pool.mark_invalid("key-a")
        >>> pool.mark_invalid("key-a", error_code="401", error_message="Invalid API key")
        """
        with self._lock:
            rec = self._find_record(key)
            if rec is None:
                return
            rec.state = KeyState.DISABLED
            rec.last_error_code = error_code
            rec.last_error_message = error_message

    def mark_success(self, key: str) -> None:
        """Reset a key's failure count to 0 and keep it ACTIVE.

        Call this after a successful API response to indicate the key is
        healthy and reset any transient failure tracking.

        Parameters
        ----------
        key : str
            The API key string.

        Examples
        --------
        >>> pool = KeyPool(["key-a"])
        >>> k = pool.get_key()
        >>> pool.mark_success(k)
        """
        with self._lock:
            rec = self._find_record(key)
            if rec is None:
                return
            rec.failure_count = 0
            rec.state = KeyState.ACTIVE
            rec.last_status_code = None
            rec.last_error_code = None
            rec.last_error_message = None

    def status(self) -> dict[str, dict]:
        """Return a snapshot of every key's state without exposing raw keys.

        Every key value in the returned dictionary is passed through ``mask()``
        so the caller can safely log or print the result.

        Returns
        -------
        dict[str, dict]
            A mapping of ``{masked_key: {"state": str, "failure_count": int,
            "cooldown_remaining": float | None, "last_status_code": int | None,
            "last_error_code": str | None, "last_error_message": str | None}}``
            for every key in the pool.

        Examples
        --------
        >>> pool = KeyPool(["sk-abcdef1234567890"])
        >>> pool.status()
        {'sk-abc...7890': {'state': 'active', 'failure_count': 0, 'cooldown_remaining': None, 'last_status_code': None, 'last_error_code': None, 'last_error_message': None}}
        """
        with self._lock:
            result: dict[str, dict] = {}
            now = time.monotonic()
            for rec in self._records:
                if rec.state == KeyState.COOLDOWN and rec.cooldown_until is not None:
                    remaining = max(0.0, rec.cooldown_until - now)
                else:
                    remaining = None
                result[mask(rec.key)] = {
                    "state": rec.state.value,
                    "failure_count": rec.failure_count,
                    "cooldown_remaining": round(remaining, 1) if remaining is not None else None,
                    "last_status_code": rec.last_status_code,
                    "last_error_code": rec.last_error_code,
                    "last_error_message": rec.last_error_message,
                }
            return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _cooldown_summary(self, cooldown_records: list[_KeyRecord], now: float) -> str:
        """Return a brief summary of cooldown keys for error messages."""
        parts: list[str] = []
        for rec in cooldown_records:
            if rec.cooldown_until is not None:
                parts.append(f"{mask(rec.key)} in {max(0, rec.cooldown_until - now):.1f}s")
        return ", ".join(parts) if parts else "unknown"
