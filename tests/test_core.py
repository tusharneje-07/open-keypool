"""Tests for the open_keypool.core module."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

import open_keypool.core as core
from open_keypool.core import (
    AllKeysExhaustedError,
    KeyPool,
    KeyState,
    _DOPPLER_CACHE,
    _KeyRecord,
    mask,
)


# ---------------------------------------------------------------------------
# mask
# ---------------------------------------------------------------------------


def test_mask_short_key():
    assert mask("abc") == "..."
    assert mask("abcd") == "abc...abcd"
    assert mask("abcdefg") == "abc...defg"


def test_mask_normal_key():
    result = mask("sk-1a2b3c4d5e6f7g8h9i0j")
    assert "..." in result
    assert result.startswith("sk-1a2")
    assert result.endswith("9i0j")
    assert len(result) < len("sk-1a2b3c4d5e6f7g8h9i0j")


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


def test_empty_key_list_raises_value_error():
    with pytest.raises(ValueError, match="at least one key"):
        KeyPool(keys=[])

    with pytest.raises(ValueError, match="at least one key"):
        KeyPool(keys=None)


def test_invalid_strategy_raises_value_error():
    with pytest.raises(ValueError, match="Unknown strategy"):
        KeyPool(keys=["k1"], strategy="random")


# ---------------------------------------------------------------------------
# get_key – round_robin
# ---------------------------------------------------------------------------


def test_round_robin_cycles_correctly():
    pool = KeyPool(keys=["a", "b", "c"], strategy="round_robin")
    assert pool.get_key() == "a"
    assert pool.get_key() == "b"
    assert pool.get_key() == "c"
    assert pool.get_key() == "a"


def test_round_robin_skips_non_active():
    pool = KeyPool(keys=["a", "b", "c"], strategy="round_robin")
    pool.mark_rate_limited("b", retry_after=999)
    keys = [pool.get_key() for _ in range(4)]
    assert "b" not in keys
    assert keys == ["a", "c", "a", "c"]


# ---------------------------------------------------------------------------
# get_key – lru
# ---------------------------------------------------------------------------


def test_lru_picks_least_recently_used():
    pool = KeyPool(keys=["a", "b", "c"], strategy="lru")
    t0 = time.monotonic()

    rec_a = pool._find_record("a")
    rec_b = pool._find_record("b")
    rec_c = pool._find_record("c")
    assert rec_a is not None and rec_b is not None and rec_c is not None

    rec_a.last_used = t0
    rec_b.last_used = t0 + 10
    rec_c.last_used = t0 + 5

    assert pool.get_key() == "a"


def test_lru_updates_last_used_on_selection():
    pool = KeyPool(keys=["a", "b"], strategy="lru")
    first = pool.get_key()
    assert first == "a"

    with pool._lock:
        rec_a = pool._find_record("a")
        rec_b = pool._find_record("b")
        assert rec_a is not None and rec_b is not None
        assert rec_a.last_used > rec_b.last_used

    second = pool.get_key()
    assert second == "b"


# ---------------------------------------------------------------------------
# mark_rate_limited / cooldown recovery
# ---------------------------------------------------------------------------


def test_mark_rate_limited_moves_to_cooldown(monkeypatch):
    fixed_time = 100.0
    monkeypatch.setattr(time, "monotonic", lambda: fixed_time)

    pool = KeyPool(keys=["a", "b"], cooldown_seconds=60)
    pool.mark_rate_limited("a")

    with pool._lock:
        rec = pool._find_record("a")
        assert rec is not None
        assert rec.state == KeyState.COOLDOWN
        assert rec.cooldown_until == 160.0
        assert rec.failure_count == 1

    keys = [pool.get_key() for _ in range(3)]
    assert "a" not in keys


def test_cooldown_key_recovers_after_time(monkeypatch):
    t = 100.0
    monkeypatch.setattr(time, "monotonic", lambda: t)

    pool = KeyPool(keys=["a"], cooldown_seconds=10)
    pool.mark_rate_limited("a")

    with pytest.raises(AllKeysExhaustedError):
        pool.get_key()

    monkeypatch.setattr(time, "monotonic", lambda: 111.0)
    assert pool.get_key() == "a"


def test_mark_rate_limited_uses_custom_retry_after(monkeypatch):
    fixed_time = 100.0
    monkeypatch.setattr(time, "monotonic", lambda: fixed_time)
    pool = KeyPool(keys=["a"], cooldown_seconds=60)
    pool.mark_rate_limited("a", retry_after=5)

    with pool._lock:
        rec = pool._find_record("a")
        assert rec is not None
        assert rec.cooldown_until == 105.0


def test_mark_rate_limited_nonexistent_key_noop():
    pool = KeyPool(keys=["a"])
    pool.mark_rate_limited("nonexistent")
    assert len(pool._records) == 1


# ---------------------------------------------------------------------------
# mark_invalid
# ---------------------------------------------------------------------------


def test_mark_invalid_permanently_disables_key():
    pool = KeyPool(keys=["a", "b"])
    pool.mark_invalid("a")

    with pool._lock:
        rec = pool._find_record("a")
        assert rec is not None
        assert rec.state == KeyState.DISABLED

    assert pool.get_key() == "b"
    assert pool.get_key() == "b"


def test_mark_invalid_does_not_auto_recover(monkeypatch):
    t = 100.0
    monkeypatch.setattr(time, "monotonic", lambda: t)
    pool = KeyPool(keys=["a"])
    pool.mark_invalid("a")
    monkeypatch.setattr(time, "monotonic", lambda: 9999.0)
    with pytest.raises(AllKeysExhaustedError):
        pool.get_key()


def test_mark_invalid_nonexistent_key_noop():
    pool = KeyPool(keys=["a"])
    pool.mark_invalid("nonexistent")
    assert len(pool._records) == 1


# ---------------------------------------------------------------------------
# mark_success
# ---------------------------------------------------------------------------


def test_mark_success_resets_failure_count():
    pool = KeyPool(keys=["a"])
    pool.mark_rate_limited("a")
    pool.mark_success("a")

    with pool._lock:
        rec = pool._find_record("a")
        assert rec is not None
        assert rec.failure_count == 0
        assert rec.state == KeyState.ACTIVE


def test_mark_success_nonexistent_key_noop():
    pool = KeyPool(keys=["a"])
    pool.mark_success("nonexistent")
    assert len(pool._records) == 1


# ---------------------------------------------------------------------------
# AllKeysExhaustedError messages
# ---------------------------------------------------------------------------


def test_all_keys_exhausted_with_cooldown_recovery_message(monkeypatch):
    t = 500.0
    monkeypatch.setattr(time, "monotonic", lambda: t)
    pool = KeyPool(keys=["a", "b"], cooldown_seconds=30)
    pool.mark_rate_limited("a", retry_after=10)
    pool.mark_rate_limited("b", retry_after=5)

    with pytest.raises(AllKeysExhaustedError, match="Recovery in"):
        pool.get_key()


def test_all_keys_exhausted_all_disabled_message():
    pool = KeyPool(keys=["a", "b"])
    pool.mark_invalid("a")
    pool.mark_invalid("b")

    with pytest.raises(AllKeysExhaustedError, match="All keys are disabled"):
        pool.get_key()


# ---------------------------------------------------------------------------
# add_key / remove_key
# ---------------------------------------------------------------------------


def test_add_key_adds_active_key():
    pool = KeyPool(keys=["a"])
    pool.add_key("b")
    assert len(pool._records) == 2
    keys = {pool.get_key() for _ in range(10)}
    assert "b" in keys


def test_add_key_duplicate_noop():
    pool = KeyPool(keys=["a"])
    pool.add_key("a")
    assert len(pool._records) == 1


def test_remove_key_deletes_any_state():
    pool = KeyPool(keys=["a", "b"])
    pool.mark_rate_limited("a")
    pool.remove_key("a")
    assert len(pool._records) == 1
    assert pool._find_record("a") is None


def test_remove_key_nonexistent_noop():
    pool = KeyPool(keys=["a"])
    pool.remove_key("nonexistent")
    assert len(pool._records) == 1


def test_add_remove_immediately_affects_get_key():
    pool = KeyPool(keys=["a", "b", "c"])
    pool.remove_key("b")
    assert {pool.get_key(), pool.get_key()} == {"a", "c"}

    pool.add_key("d")
    all_keys = {pool.get_key() for _ in range(10)}
    assert "d" in all_keys


# ---------------------------------------------------------------------------
# status() never exposes raw keys
# ---------------------------------------------------------------------------


def test_status_never_contains_raw_key():
    keys = ["sk-abcdef1234567890", "sk-fedcba0987654321"]
    pool = KeyPool(keys=keys)

    status = pool.status()
    assert len(status) == 2

    for masked_key, info in status.items():
        for raw in keys:
            assert raw not in masked_key
        assert "..." in masked_key
        assert isinstance(info["state"], str)
        assert isinstance(info["failure_count"], int)
        assert "cooldown_remaining" in info

    status_str = str(status)
    for raw in keys:
        assert raw not in status_str


def test_status_during_cooldown_shows_remaining(monkeypatch):
    t = 100.0
    monkeypatch.setattr(time, "monotonic", lambda: t)
    pool = KeyPool(keys=["sk-key1", "sk-key2"], cooldown_seconds=30)
    pool.mark_rate_limited("sk-key1", retry_after=10)

    status = pool.status()
    for masked, info in status.items():
        if info["state"] == "cooldown":
            assert info["cooldown_remaining"] is not None
            assert info["cooldown_remaining"] > 0


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


def test_get_key_is_thread_safe():
    pool = KeyPool(keys=[f"key-{i}" for i in range(200)])
    import threading

    errors: list[Exception] = []
    results: list[str] = []

    def worker():
        try:
            for _ in range(100):
                results.append(pool.get_key())
                pool.mark_success(results[-1])
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
    assert len(results) == 1000


# ---------------------------------------------------------------------------
# from_doppler – mocked HTTP tests
# ---------------------------------------------------------------------------

FAKE_TOKEN = "dp.st.fake-token"
FAKE_PROJECT = "test-proj"
FAKE_CONFIG = "dev"
DOPPLER_RESPONSE = {
    "secrets": {
        "API_KEY_1": {"raw": "dp-secret-key-a"},
        "API_KEY_2": {"raw": "dp-secret-key-b"},
        "OTHER_SECRET": {"raw": "dp-other"},
    }
}


def _clear_doppler_cache():
    _DOPPLER_CACHE.clear()


@pytest.fixture(autouse=True)
def _auto_clear_cache():
    """Clear the Doppler TTL cache before every test."""
    _DOPPLER_CACHE.clear()


def test_from_doppler_fetches_keys_via_httpx(monkeypatch):
    import httpx

    call_count = 0

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return DOPPLER_RESPONSE

    original_get = httpx.get

    def fake_get(url, **kwargs):
        nonlocal call_count
        call_count += 1
        return FakeResponse()

    monkeypatch.setattr(httpx, "get", fake_get)

    pool = KeyPool.from_doppler(
        token=FAKE_TOKEN, project=FAKE_PROJECT, config=FAKE_CONFIG,
        key_prefix="API_KEY_",
    )
    assert call_count == 1
    keys = {pool.get_key() for _ in range(10)}
    assert keys == {"dp-secret-key-a", "dp-secret-key-b"}


def test_from_doppler_cache_hit_avoids_second_http_call(monkeypatch):
    import httpx

    call_count = 0

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return DOPPLER_RESPONSE

    original_get = httpx.get

    def fake_get(url, **kwargs):
        nonlocal call_count
        call_count += 1
        return FakeResponse()

    monkeypatch.setattr(httpx, "get", fake_get)

    p1 = KeyPool.from_doppler(
        token=FAKE_TOKEN, project=FAKE_PROJECT, config=FAKE_CONFIG,
        key_prefix="API_KEY_",
    )
    assert call_count == 1

    p2 = KeyPool.from_doppler(
        token=FAKE_TOKEN, project=FAKE_PROJECT, config=FAKE_CONFIG,
        key_prefix="API_KEY_",
    )
    assert call_count == 1


def test_from_doppler_cache_expiry_triggers_fresh_call(monkeypatch):
    import cachetools
    import httpx

    short_cache = cachetools.TTLCache(maxsize=64, ttl=0)

    monkeypatch.setattr(core, "_DOPPLER_CACHE", short_cache)

    call_count = 0

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return DOPPLER_RESPONSE

    def fake_get(url, **kwargs):
        nonlocal call_count
        call_count += 1
        return FakeResponse()

    monkeypatch.setattr(httpx, "get", fake_get)

    KeyPool.from_doppler(
        token=FAKE_TOKEN, project=FAKE_PROJECT, config=FAKE_CONFIG,
        key_prefix="API_KEY_",
    )
    assert call_count == 1

    KeyPool.from_doppler(
        token=FAKE_TOKEN, project=FAKE_PROJECT, config=FAKE_CONFIG,
        key_prefix="API_KEY_",
    )
    assert call_count == 2


def test_from_doppler_force_refresh_bypasses_cache(monkeypatch):
    import httpx

    call_count = 0

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return DOPPLER_RESPONSE

    def fake_get(url, **kwargs):
        nonlocal call_count
        call_count += 1
        return FakeResponse()

    monkeypatch.setattr(httpx, "get", fake_get)

    KeyPool.from_doppler(
        token=FAKE_TOKEN, project=FAKE_PROJECT, config=FAKE_CONFIG,
        key_prefix="API_KEY_", force_refresh=False,
    )
    assert call_count == 1

    KeyPool.from_doppler(
        token=FAKE_TOKEN, project=FAKE_PROJECT, config=FAKE_CONFIG,
        key_prefix="API_KEY_", force_refresh=True,
    )
    assert call_count == 2


def test_from_doppler_failed_fetch_does_not_populate_cache(monkeypatch):
    import httpx

    class FakeErrorResponse:
        status_code = 500

        def raise_for_status(self):
            raise httpx.HTTPStatusError("error", request=None, response=self)  # noqa: ASYNC100

        def json(self):
            raise ValueError("not json")

    def fake_get(url, **kwargs):
        return FakeErrorResponse()

    monkeypatch.setattr(httpx, "get", fake_get)

    with pytest.raises(RuntimeError, match="Doppler API request failed"):
        KeyPool.from_doppler(
            token=FAKE_TOKEN, project=FAKE_PROJECT, config=FAKE_CONFIG,
            key_prefix="API_KEY_",
        )

    assert _DOPPLER_CACHE.get((FAKE_PROJECT, FAKE_CONFIG, "API_KEY_")) is None


def test_from_doppler_failed_fetch_subsequent_retries(monkeypatch):
    import httpx

    call_count = 0

    def fake_get_fail(url, **kwargs):
        nonlocal call_count
        call_count += 1
        raise httpx.HTTPStatusError("error", request=None, response=None)  # noqa: ASYNC100

    monkeypatch.setattr(httpx, "get", fake_get_fail)

    with pytest.raises(RuntimeError, match="Doppler API request failed"):
        KeyPool.from_doppler(
            token=FAKE_TOKEN, project=FAKE_PROJECT, config=FAKE_CONFIG,
            key_prefix="API_KEY_",
        )
    assert call_count == 1

    with pytest.raises(RuntimeError, match="Doppler API request failed"):
        KeyPool.from_doppler(
            token=FAKE_TOKEN, project=FAKE_PROJECT, config=FAKE_CONFIG,
            key_prefix="API_KEY_",
        )
    assert call_count == 2


def test_from_doppler_zero_keys_raises_runtime_error(monkeypatch):
    import httpx

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"secrets": {}}

    monkeypatch.setattr(httpx, "get", lambda url, **kwargs: FakeResponse())

    with pytest.raises(RuntimeError, match="zero keys"):
        KeyPool.from_doppler(
            token=FAKE_TOKEN, project=FAKE_PROJECT, config=FAKE_CONFIG,
            key_prefix="API_KEY_",
        )

    assert _DOPPLER_CACHE.get((FAKE_PROJECT, FAKE_CONFIG, "API_KEY_")) is None


def test_from_doppler_different_project_config_pairs_are_independent(monkeypatch):
    import httpx

    call_count = 0

    class FakeResponseA:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"secrets": {"K1": {"raw": "key-alpha"}}}

    class FakeResponseB:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"secrets": {"K2": {"raw": "key-beta"}}}

    def fake_get(url, **kwargs):
        nonlocal call_count
        call_count += 1
        params = kwargs.get("params", {})
        if params.get("project") == "proj-A":
            return FakeResponseA()
        return FakeResponseB()

    monkeypatch.setattr(httpx, "get", fake_get)

    pA = KeyPool.from_doppler(
        token=FAKE_TOKEN, project="proj-A", config="dev",
    )
    assert call_count == 1
    assert pA.get_key() == "key-alpha"

    pB = KeyPool.from_doppler(
        token=FAKE_TOKEN, project="proj-B", config="dev",
    )
    assert call_count == 2
    assert pB.get_key() == "key-beta"

    pA2 = KeyPool.from_doppler(
        token=FAKE_TOKEN, project="proj-A", config="dev",
    )
    assert call_count == 2
    assert pA2.get_key() == "key-alpha"

    pB2 = KeyPool.from_doppler(
        token=FAKE_TOKEN, project="proj-B", config="dev",
    )
    assert call_count == 2
    assert pB2.get_key() == "key-beta"


# ---------------------------------------------------------------------------
# Error message never contains raw key
# ---------------------------------------------------------------------------


def test_exception_message_never_contains_raw_key(monkeypatch):
    t = 100.0
    monkeypatch.setattr(time, "monotonic", lambda: t)
    pool = KeyPool(keys=["sk-raw-secret-key-12345"], cooldown_seconds=30)
    pool.mark_rate_limited("sk-raw-secret-key-12345", retry_after=10)

    with pytest.raises(AllKeysExhaustedError) as excinfo:
        pool.get_key()

    msg = str(excinfo.value)
    assert "sk-raw-secret-key-12345" not in msg
    assert "..." in msg
