"""Core implementation of the open_keypool library.

Provides the `KeyPool` class for managing a pool of API keys with automatic
cooldown, disablement, and rotation strategies to avoid HTTP 429 rate-limit
errors.
"""

from __future__ import annotations

import asyncio
import inspect
import json as _json
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import cachetools
import httpx


def _parse_duration_str(val: str) -> float | None:
    """Parse duration string like '6s', '250ms', '1m' to seconds."""
    val = str(val).strip()
    if not val:
        return None
    try:
        if val.endswith("ms"):
            return float(val[:-2]) / 1000.0
        elif val.endswith("s"):
            return float(val[:-1])
        elif val.endswith("m"):
            return float(val[:-1]) * 60.0
        return float(val)
    except (ValueError, TypeError):
        return None


def _process_response_metadata(
    provider: str | None,
    status_code: int,
    headers: dict[str, str] | None,
    body: dict | str | None,
) -> tuple[bool, bool, float | None, str, str]:
    """Inspect response status, headers, and body according to provider presets.

    Returns:
        (is_rate_limited, is_disabled, retry_after, error_code, error_message)
    """
    headers = headers or {}
    h_lower = {str(k).lower(): str(v) for k, v in headers.items()}

    error_code = str(status_code)
    error_message = ""
    error_type = ""
    error_status = ""

    if isinstance(body, dict):
        err = body.get("error", {})
        if isinstance(err, dict):
            error_code = str(err.get("code") or error_code)
            error_message = str(err.get("message") or "")
            error_type = str(err.get("type") or "")
            error_status = str(err.get("status") or "")
        elif isinstance(err, str):
            error_message = err
        if "status" in body and isinstance(body["status"], str):
            error_status = body["status"]
    elif isinstance(body, str):
        error_message = body[:200]

    ra_header = h_lower.get("retry-after")
    ra_sec: float | None = None
    if ra_header:
        try:
            ra_sec = float(ra_header)
        except (ValueError, TypeError):
            ra_sec = None

    prov = provider.lower() if provider else None
    is_rate_limited = False
    is_disabled = False

    if prov == "openai":
        if status_code in (429, 413) or error_code in ("rate_limit_exceeded", "insufficient_quota") or "rate_limit" in error_type.lower() or "rate limit" in error_message.lower():
            is_rate_limited = True
            ra_ms = h_lower.get("retry-after-ms")
            if ra_ms:
                try:
                    ra_sec = float(ra_ms) / 1000.0
                except (ValueError, TypeError):
                    pass
        elif status_code in (401, 403) or error_code in ("invalid_api_key", "account_deactivated"):
            is_disabled = True

    elif prov == "groq":
        if status_code in (429, 413) or error_code == "rate_limit_exceeded" or "rate limit" in error_message.lower():
            is_rate_limited = True
            req_reset = h_lower.get("x-ratelimit-reset-requests")
            tok_reset = h_lower.get("x-ratelimit-reset-tokens")
            if req_reset:
                ra_sec = _parse_duration_str(req_reset) or ra_sec
            if tok_reset:
                t_sec = _parse_duration_str(tok_reset)
                if t_sec is not None:
                    ra_sec = max(ra_sec or 0.0, t_sec)
        elif status_code in (401, 403):
            is_disabled = True

    elif prov == "gemini":
        if status_code == 429 or error_status == "RESOURCE_EXHAUSTED" or "RESOURCE_EXHAUSTED" in error_message or "RESOURCE_EXHAUSTED" in error_code:
            is_rate_limited = True
        elif status_code in (401, 403) or "API_KEY_INVALID" in error_message or "PERMISSION_DENIED" in error_status:
            is_disabled = True

    elif prov == "together":
        if status_code in (429, 413) or error_type == "rate_limit_error" or error_code == "rate_limit_exceeded" or "rate limit" in error_message.lower():
            is_rate_limited = True
        elif status_code in (401, 403):
            is_disabled = True

    # Generic check if provider preset didn't explicitly trigger
    if not is_rate_limited and not is_disabled:
        if 200 <= status_code < 300:
            if error_code == "rate_limit_exceeded":
                is_rate_limited = True
            else:
                return False, False, None, error_code, error_message
        elif status_code in (429, 413) or error_code == "rate_limit_exceeded" or "rate limit" in error_message.lower() or "resource_exhausted" in error_message.lower():
            is_rate_limited = True
        elif status_code in (401, 403):
            is_disabled = True
        elif 500 <= status_code < 600:
            is_rate_limited = True
        else:
            is_rate_limited = True

    return is_rate_limited, is_disabled, ra_sec, error_code, error_message


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
        provider: str | None = None,
    ) -> None:
        if not keys:
            raise ValueError("KeyPool requires at least one key (keys must be a non-empty list).")

        if strategy not in ("round_robin", "lru"):
            raise ValueError(f"Unknown strategy '{strategy}'. Use 'round_robin' or 'lru'.")

        self.max_retries: int = max_retries
        self.cooldown_seconds: int = cooldown_seconds
        self.strategy: str = strategy
        self.provider: str | None = provider.lower() if provider else None

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
        provider: str | None = None,
        force_refresh: bool = False,
    ) -> KeyPool:
        """Create a ``KeyPool`` by fetching API keys from Doppler."""
        cache_key = (token, project, config, key_prefix)

        if not force_refresh:
            cached = _DOPPLER_CACHE.get(cache_key)
            if cached is not None:
                return cls(
                    keys=list(cached),
                    max_retries=max_retries,
                    cooldown_seconds=cooldown_seconds,
                    strategy=strategy,
                    provider=provider,
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
            provider=provider,
        )

    @classmethod
    def from_env(
        cls,
        suffix: str,
        env_file: str | None = None,
        max_retries: int = 3,
        cooldown_seconds: int = 60,
        strategy: str = "round_robin",
        provider: str | None = None,
    ) -> KeyPool:
        """Create a ``KeyPool`` from environment variables matching a suffix."""
        if not suffix:
            raise ValueError("suffix must be a non-empty string.")

        try:
            from dotenv import load_dotenv
        except ImportError as exc:
            raise ImportError(
                "python-dotenv is required to use KeyPool.from_env(). "
                "Install it with `pip install open-keypool[env]`."
            ) from exc

        load_dotenv(env_file)

        import os

        matched: list[str] = []
        for name, value in os.environ.items():
            if name.endswith(suffix) and value.strip():
                matched.append(value.strip())

        if not matched:
            raise RuntimeError(
                f"No environment variables ending with '{suffix}' found "
                f"(env_file={env_file!r})."
            )

        return cls(
            keys=matched,
            max_retries=max_retries,
            cooldown_seconds=cooldown_seconds,
            strategy=strategy,
            provider=provider,
        )

    @classmethod
    def from_json(
        cls,
        path: str,
        suffix: str | None = None,
        max_retries: int = 3,
        cooldown_seconds: int = 60,
        strategy: str = "round_robin",
        provider: str | None = None,
    ) -> KeyPool:
        """Create a ``KeyPool`` from a JSON file."""
        import json as _json
        import os

        if not os.path.isfile(path):
            raise FileNotFoundError(f"JSON file not found: {path}")

        with open(path, "r", encoding="utf-8") as fh:
            try:
                data = _json.load(fh)
            except _json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}: {exc}") from exc

        if not isinstance(data, dict):
            raise ValueError(
                f"Expected a JSON object at top level in {path}, got {type(data).__name__}."
            )

        if not data:
            raise RuntimeError(f"JSON object in {path} is empty.")

        matched: list[str] = []
        for name, value in data.items():
            if not isinstance(value, str):
                raise ValueError(
                    f"All values must be strings in {path}. "
                    f"Key '{name}' has type {type(value).__name__}."
                )
            if suffix is None or name.endswith(suffix):
                if value.strip():
                    matched.append(value.strip())

        if not matched:
            raise RuntimeError(
                f"No entries ending with '{suffix}' found in {path}."
            )

        return cls(
            keys=matched,
            max_retries=max_retries,
            cooldown_seconds=cooldown_seconds,
            strategy=strategy,
            provider=provider,
        )

    @classmethod
    def from_aws_secrets(
        cls,
        secret_name: str,
        region_name: str = "us-east-1",
        key_prefix: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        max_retries: int = 3,
        cooldown_seconds: int = 60,
        strategy: str = "round_robin",
        provider: str | None = None,
    ) -> KeyPool:
        """Create a ``KeyPool`` by fetching secrets from AWS Secrets Manager."""
        try:
            import boto3
        except ImportError as exc:
            raise ImportError(
                "boto3 is required to use KeyPool.from_aws_secrets(). "
                "Install it with `pip install boto3` or `pip install open-keypool[aws]`."
            ) from exc

        client_kwargs: dict[str, Any] = {"region_name": region_name}
        if aws_access_key_id:
            client_kwargs["aws_access_key_id"] = aws_access_key_id
        if aws_secret_access_key:
            client_kwargs["aws_secret_access_key"] = aws_secret_access_key
        if aws_session_token:
            client_kwargs["aws_session_token"] = aws_session_token

        client = boto3.client("secretsmanager", **client_kwargs)
        try:
            response = client.get_secret_value(SecretId=secret_name)
        except Exception as exc:
            raise RuntimeError(f"AWS Secrets Manager request failed for '{secret_name}': {exc}") from exc

        secret_string = response.get("SecretString", "")
        fetched_keys: list[str] = []
        try:
            data = _json.loads(secret_string)
            if isinstance(data, dict):
                for k, v in data.items():
                    if (key_prefix is None or k.startswith(key_prefix)) and str(v).strip():
                        fetched_keys.append(str(v).strip())
            elif isinstance(data, str) and data.strip():
                fetched_keys.append(data.strip())
        except Exception:
            for line in secret_string.splitlines():
                line_s = line.strip()
                if line_s and (key_prefix is None or line_s.startswith(key_prefix)):
                    fetched_keys.append(line_s)

        if not fetched_keys:
            raise RuntimeError(
                f"AWS Secrets Manager returned zero matching keys for secret_name='{secret_name}', key_prefix={key_prefix!r}"
            )

        return cls(
            keys=fetched_keys,
            max_retries=max_retries,
            cooldown_seconds=cooldown_seconds,
            strategy=strategy,
            provider=provider,
        )

    @classmethod
    def from_gcp_secrets(
        cls,
        secret_id: str,
        project_id: str | None = None,
        version_id: str = "latest",
        key_prefix: str | None = None,
        max_retries: int = 3,
        cooldown_seconds: int = 60,
        strategy: str = "round_robin",
        provider: str | None = None,
    ) -> KeyPool:
        """Create a ``KeyPool`` by fetching secrets from GCP Secret Manager."""
        try:
            from google.cloud import secretmanager
        except ImportError as exc:
            raise ImportError(
                "google-cloud-secret-manager is required to use KeyPool.from_gcp_secrets(). "
                "Install it with `pip install google-cloud-secret-manager` or `pip install open-keypool[gcp]`."
            ) from exc

        client = secretmanager.SecretManagerServiceClient()
        if secret_id.startswith("projects/"):
            name = secret_id
        else:
            if not project_id:
                raise ValueError("project_id is required when secret_id is not a fully qualified resource name.")
            name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"

        try:
            response = client.access_secret_version(request={"name": name})
        except Exception as exc:
            raise RuntimeError(f"GCP Secret Manager request failed for '{name}': {exc}") from exc

        payload_str = response.payload.data.decode("UTF-8")
        fetched_keys: list[str] = []
        try:
            data = _json.loads(payload_str)
            if isinstance(data, dict):
                for k, v in data.items():
                    if (key_prefix is None or k.startswith(key_prefix)) and str(v).strip():
                        fetched_keys.append(str(v).strip())
            elif isinstance(data, str) and data.strip():
                fetched_keys.append(data.strip())
        except Exception:
            for line in payload_str.splitlines():
                line_s = line.strip()
                if line_s and (key_prefix is None or line_s.startswith(key_prefix)):
                    fetched_keys.append(line_s)

        if not fetched_keys:
            raise RuntimeError(
                f"GCP Secret Manager returned zero matching keys for name='{name}', key_prefix={key_prefix!r}"
            )

        return cls(
            keys=fetched_keys,
            max_retries=max_retries,
            cooldown_seconds=cooldown_seconds,
            strategy=strategy,
            provider=provider,
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
    def handle_response(
        self,
        key: str,
        status_code: int,
        headers: dict[str, str] | None = None,
        body: dict | str | None = None,
    ) -> KeyState:
        """Feed an HTTP response to the pool — it decides what to do with the key."""
        is_rl, is_dis, ra_sec, err_code, err_msg = _process_response_metadata(
            self.provider, status_code, headers, body
        )

        with self._lock:
            rec = self._find_record(key)
            if rec is None:
                return KeyState.DISABLED

            rec.last_status_code = status_code

            if not is_rl and not is_dis:
                rec.failure_count = 0
                rec.state = KeyState.ACTIVE
                rec.last_error_code = None
                rec.last_error_message = None
                return KeyState.ACTIVE

            if is_dis:
                rec.state = KeyState.DISABLED
                rec.last_error_code = err_code
                rec.last_error_message = err_msg
                return KeyState.DISABLED

            rec.state = KeyState.COOLDOWN
            rec.cooldown_until = time.monotonic() + (
                ra_sec if ra_sec is not None else self.cooldown_seconds
            )
            rec.failure_count += 1
            rec.last_error_code = err_code
            rec.last_error_message = err_msg
            return KeyState.COOLDOWN

    def call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute *fn* with an ACTIVE API key from the pool, handling rotation automatically."""
        last_exc: Exception | None = None
        for _ in range(self.max_retries):
            key = self.get_key()
            try:
                res = fn(key, *args, **kwargs)
            except Exception as exc:
                resp = getattr(exc, "response", None)
                if resp is not None and hasattr(resp, "status_code"):
                    status_code = resp.status_code
                    headers = dict(resp.headers) if hasattr(resp, "headers") else {}
                    body = None
                    if hasattr(resp, "json"):
                        try:
                            body = resp.json()
                        except Exception:
                            body = getattr(resp, "text", str(exc))
                    else:
                        body = getattr(resp, "text", str(exc))
                    state = self.handle_response(key, status_code, headers, body)
                    if state == KeyState.ACTIVE:
                        return resp
                    last_exc = exc
                    continue
                raise exc

            if hasattr(res, "status_code"):
                headers = dict(res.headers) if hasattr(res, "headers") else {}
                body = None
                if hasattr(res, "json"):
                    try:
                        body = res.json()
                    except Exception:
                        body = getattr(res, "text", None)
                elif hasattr(res, "text"):
                    body = res.text
                state = self.handle_response(key, res.status_code, headers, body)
                if state == KeyState.ACTIVE:
                    return res
                continue

            self.mark_success(key)
            return res

        if last_exc:
            raise AllKeysExhaustedError(
                f"All retries ({self.max_retries}) exhausted. Last exception: {last_exc}"
            ) from last_exc
        raise AllKeysExhaustedError(
            f"All retries ({self.max_retries}) exhausted without a successful response."
        )

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
                masked_key = mask(rec.key)
                if masked_key in result:
                    suffix_num = 2
                    while f"{masked_key}#{suffix_num}" in result:
                        suffix_num += 1
                    masked_key = f"{masked_key}#{suffix_num}"

                result[masked_key] = {
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


class AsyncKeyPool:
    """An asyncio-compatible pool of API keys with cooldown and rotation strategies."""

    def __init__(
        self,
        keys: list[str] | None = None,
        max_retries: int = 3,
        cooldown_seconds: int = 60,
        strategy: str = "round_robin",
        provider: str | None = None,
    ) -> None:
        if not keys:
            raise ValueError("AsyncKeyPool requires at least one key (keys must be a non-empty list).")

        if strategy not in ("round_robin", "lru"):
            raise ValueError(f"Unknown strategy '{strategy}'. Use 'round_robin' or 'lru'.")

        self.max_retries: int = max_retries
        self.cooldown_seconds: int = cooldown_seconds
        self.strategy: str = strategy
        self.provider: str | None = provider.lower() if provider else None

        self._records: list[_KeyRecord] = [_KeyRecord(key=k) for k in keys]
        self._round_robin_index: int = 0
        self._lock: asyncio.Lock = asyncio.Lock()

    def _find_record(self, key: str) -> _KeyRecord | None:
        for rec in self._records:
            if rec.key == key:
                return rec
        return None

    def _recover_cooldown_keys(self) -> None:
        now = time.monotonic()
        for rec in self._records:
            if rec.state == KeyState.COOLDOWN and rec.cooldown_until is not None and now >= rec.cooldown_until:
                rec.state = KeyState.ACTIVE
                rec.cooldown_until = None

    def _active_records(self) -> list[_KeyRecord]:
        return [r for r in self._records if r.state == KeyState.ACTIVE]

    async def get_key(self) -> str:
        async with self._lock:
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

    async def async_get_key(self) -> str:
        return await self.get_key()

    async def handle_response(
        self,
        key: str,
        status_code: int,
        headers: dict[str, str] | None = None,
        body: dict | str | None = None,
    ) -> KeyState:
        is_rl, is_dis, ra_sec, err_code, err_msg = _process_response_metadata(
            self.provider, status_code, headers, body
        )

        async with self._lock:
            rec = self._find_record(key)
            if rec is None:
                return KeyState.DISABLED

            rec.last_status_code = status_code

            if not is_rl and not is_dis:
                rec.failure_count = 0
                rec.state = KeyState.ACTIVE
                rec.last_error_code = None
                rec.last_error_message = None
                return KeyState.ACTIVE

            if is_dis:
                rec.state = KeyState.DISABLED
                rec.last_error_code = err_code
                rec.last_error_message = err_msg
                return KeyState.DISABLED

            rec.state = KeyState.COOLDOWN
            rec.cooldown_until = time.monotonic() + (
                ra_sec if ra_sec is not None else self.cooldown_seconds
            )
            rec.failure_count += 1
            rec.last_error_code = err_code
            rec.last_error_message = err_msg
            return KeyState.COOLDOWN

    async def mark_rate_limited(
        self,
        key: str,
        retry_after: float | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> None:
        async with self._lock:
            rec = self._find_record(key)
            if rec is None:
                return
            rec.state = KeyState.COOLDOWN
            rec.cooldown_until = time.monotonic() + (retry_after if retry_after is not None else self.cooldown_seconds)
            rec.failure_count += 1
            rec.last_error_code = error_code
            rec.last_error_message = error_message

    async def mark_invalid(
        self,
        key: str,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> None:
        async with self._lock:
            rec = self._find_record(key)
            if rec is None:
                return
            rec.state = KeyState.DISABLED
            rec.last_error_code = error_code
            rec.last_error_message = error_message

    async def mark_success(self, key: str) -> None:
        async with self._lock:
            rec = self._find_record(key)
            if rec is None:
                return
            rec.failure_count = 0
            rec.state = KeyState.ACTIVE
            rec.last_status_code = None
            rec.last_error_code = None
            rec.last_error_message = None

    async def status(self) -> dict[str, dict]:
        async with self._lock:
            result: dict[str, dict] = {}
            now = time.monotonic()
            for rec in self._records:
                if rec.state == KeyState.COOLDOWN and rec.cooldown_until is not None:
                    remaining = max(0.0, rec.cooldown_until - now)
                else:
                    remaining = None
                masked_key = mask(rec.key)
                if masked_key in result:
                    suffix_num = 2
                    while f"{masked_key}#{suffix_num}" in result:
                        suffix_num += 1
                    masked_key = f"{masked_key}#{suffix_num}"

                result[masked_key] = {
                    "state": rec.state.value,
                    "failure_count": rec.failure_count,
                    "cooldown_remaining": round(remaining, 1) if remaining is not None else None,
                    "last_status_code": rec.last_status_code,
                    "last_error_code": rec.last_error_code,
                    "last_error_message": rec.last_error_message,
                }
            return result

    async def add_key(self, key: str) -> None:
        async with self._lock:
            if self._find_record(key) is None:
                self._records.append(_KeyRecord(key=key))

    async def remove_key(self, key: str) -> None:
        async with self._lock:
            rec = self._find_record(key)
            if rec is not None:
                self._records.remove(rec)

    async def call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        last_exc: Exception | None = None
        for _ in range(self.max_retries):
            key = await self.get_key()
            try:
                if inspect.iscoroutinefunction(fn):
                    res = await fn(key, *args, **kwargs)
                else:
                    res = fn(key, *args, **kwargs)
                    if inspect.isawaitable(res):
                        res = await res
            except Exception as exc:
                resp = getattr(exc, "response", None)
                if resp is not None and hasattr(resp, "status_code"):
                    status_code = resp.status_code
                    headers = dict(resp.headers) if hasattr(resp, "headers") else {}
                    body = None
                    if hasattr(resp, "json"):
                        try:
                            body = resp.json()
                        except Exception:
                            body = getattr(resp, "text", str(exc))
                    else:
                        body = getattr(resp, "text", str(exc))
                    state = await self.handle_response(key, status_code, headers, body)
                    if state == KeyState.ACTIVE:
                        return resp
                    last_exc = exc
                    continue
                raise exc

            if hasattr(res, "status_code"):
                headers = dict(res.headers) if hasattr(res, "headers") else {}
                body = None
                if hasattr(res, "json"):
                    try:
                        body = res.json()
                    except Exception:
                        body = getattr(res, "text", None)
                elif hasattr(res, "text"):
                    body = res.text
                state = await self.handle_response(key, res.status_code, headers, body)
                if state == KeyState.ACTIVE:
                    return res
                continue

            await self.mark_success(key)
            return res

        if last_exc:
            raise AllKeysExhaustedError(
                f"All retries ({self.max_retries}) exhausted. Last exception: {last_exc}"
            ) from last_exc
        raise AllKeysExhaustedError(
            f"All retries ({self.max_retries}) exhausted without a successful response."
        )

    @classmethod
    async def from_doppler(
        cls,
        token: str,
        project: str,
        config: str,
        key_prefix: str | None = None,
        max_retries: int = 3,
        cooldown_seconds: int = 60,
        strategy: str = "round_robin",
        provider: str | None = None,
        force_refresh: bool = False,
    ) -> AsyncKeyPool:
        cache_key = (token, project, config, key_prefix)

        if not force_refresh:
            cached = _DOPPLER_CACHE.get(cache_key)
            if cached is not None:
                return cls(
                    keys=list(cached),
                    max_retries=max_retries,
                    cooldown_seconds=cooldown_seconds,
                    strategy=strategy,
                    provider=provider,
                )

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
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
            provider=provider,
        )

    @classmethod
    def from_env(
        cls,
        suffix: str,
        env_file: str | None = None,
        max_retries: int = 3,
        cooldown_seconds: int = 60,
        strategy: str = "round_robin",
        provider: str | None = None,
    ) -> AsyncKeyPool:
        sync_pool = KeyPool.from_env(
            suffix=suffix,
            env_file=env_file,
            max_retries=max_retries,
            cooldown_seconds=cooldown_seconds,
            strategy=strategy,
            provider=provider,
        )
        return cls(
            keys=[r.key for r in sync_pool._records],
            max_retries=max_retries,
            cooldown_seconds=cooldown_seconds,
            strategy=strategy,
            provider=provider,
        )

    @classmethod
    def from_json(
        cls,
        path: str,
        suffix: str | None = None,
        max_retries: int = 3,
        cooldown_seconds: int = 60,
        strategy: str = "round_robin",
        provider: str | None = None,
    ) -> AsyncKeyPool:
        sync_pool = KeyPool.from_json(
            path=path,
            suffix=suffix,
            max_retries=max_retries,
            cooldown_seconds=cooldown_seconds,
            strategy=strategy,
            provider=provider,
        )
        return cls(
            keys=[r.key for r in sync_pool._records],
            max_retries=max_retries,
            cooldown_seconds=cooldown_seconds,
            strategy=strategy,
            provider=provider,
        )

    @classmethod
    def from_aws_secrets(
        cls,
        secret_name: str,
        region_name: str = "us-east-1",
        key_prefix: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        max_retries: int = 3,
        cooldown_seconds: int = 60,
        strategy: str = "round_robin",
        provider: str | None = None,
    ) -> AsyncKeyPool:
        sync_pool = KeyPool.from_aws_secrets(
            secret_name=secret_name,
            region_name=region_name,
            key_prefix=key_prefix,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            max_retries=max_retries,
            cooldown_seconds=cooldown_seconds,
            strategy=strategy,
            provider=provider,
        )
        return cls(
            keys=[r.key for r in sync_pool._records],
            max_retries=max_retries,
            cooldown_seconds=cooldown_seconds,
            strategy=strategy,
            provider=provider,
        )

    @classmethod
    def from_gcp_secrets(
        cls,
        secret_id: str,
        project_id: str | None = None,
        version_id: str = "latest",
        key_prefix: str | None = None,
        max_retries: int = 3,
        cooldown_seconds: int = 60,
        strategy: str = "round_robin",
        provider: str | None = None,
    ) -> AsyncKeyPool:
        sync_pool = KeyPool.from_gcp_secrets(
            secret_id=secret_id,
            project_id=project_id,
            version_id=version_id,
            key_prefix=key_prefix,
            max_retries=max_retries,
            cooldown_seconds=cooldown_seconds,
            strategy=strategy,
            provider=provider,
        )
        return cls(
            keys=[r.key for r in sync_pool._records],
            max_retries=max_retries,
            cooldown_seconds=cooldown_seconds,
            strategy=strategy,
            provider=provider,
        )

    def _cooldown_summary(self, cooldown_records: list[_KeyRecord], now: float) -> str:
        parts: list[str] = []
        for rec in cooldown_records:
            if rec.cooldown_until is not None:
                parts.append(f"{mask(rec.key)} in {max(0, rec.cooldown_until - now):.1f}s")
        return ", ".join(parts) if parts else "unknown"
