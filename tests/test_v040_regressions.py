import asyncio
import logging

import httpx
import pytest

from open_keypool import AllKeysExhaustedError, AsyncKeyPool, KeyPool, KeyState
from open_keypool.core import mask


def test_client_error_stays_active_and_is_returned_without_rotating():
    pool = KeyPool(["key-a", "key-b"], max_retries=3)
    state = pool.handle_response("key-a", 400, body={"error": {"message": "bad request"}})

    assert state == KeyState.ACTIVE
    assert pool.status()[mask("key-a")]["state"] == "active"
    assert pool.status()[mask("key-a")]["failure_count"] == 0


@pytest.mark.parametrize("status", [400, 404, 405, 409, 410, 415, 422])
def test_client_errors_do_not_cool_down(status):
    pool = KeyPool(["key-a", "key-b"])
    assert pool.handle_response("key-a", status, body={"error": {"message": "caller error"}}) == KeyState.ACTIVE
    assert pool.status()[mask("key-a")]["state"] == "active"
    assert pool.status()[mask("key-a")]["failure_count"] == 0


@pytest.mark.parametrize("status", [408, 500, 502, 503, 529])
def test_transient_errors_use_short_cooldown(status, monkeypatch):
    monkeypatch.setattr("open_keypool.core.time.monotonic", lambda: 100.0)
    pool = KeyPool(["key-a"], cooldown_seconds=100, transient_cooldown_seconds=5)
    assert pool.handle_response("key-a", status) == KeyState.COOLDOWN
    assert pool.status()[mask("key-a")]["cooldown_remaining"] == 5.0


def test_call_returns_client_error_without_exhausting_pool():
    pool = KeyPool(["key-a", "key-b"], max_retries=4)
    response = pool.call(lambda key: httpx.Response(422, json={"error": "invalid model"}))
    assert response.status_code == 422
    assert all(item["state"] == "active" for item in pool.status().values())


@pytest.mark.anyio
async def test_async_client_error_stays_active():
    pool = AsyncKeyPool(["key-a", "key-b"])
    assert await pool.handle_response("key-a", 400) == KeyState.ACTIVE
    assert (await pool.status())[mask("key-a")]["failure_count"] == 0


def test_raw_key_is_hidden_from_record_and_logs(caplog):
    raw = "sk-raw-secret-key-12345"
    pool = KeyPool([raw])
    with caplog.at_level(logging.INFO, logger="open_keypool"):
        pool.handle_response(raw, 429, body={"error": {"message": raw}})
    assert raw not in repr(pool._records[0])
    assert raw not in str(pool.status())
    assert raw not in caplog.text


def test_wait_for_recovery_does_not_consume_retry(monkeypatch):
    pool = KeyPool(["key-a"], wait_for_recovery=True, max_wait_seconds=30)
    monkeypatch.setattr("open_keypool.core.time.monotonic", lambda: 100.0)
    pool._records[0].state = KeyState.COOLDOWN
    pool._records[0].cooldown_until = 102.0
    sleeps = []
    monkeypatch.setattr("open_keypool.core.time.sleep", lambda value: sleeps.append(value))
    calls = [0]
    def get_key():
        calls[0] += 1
        if calls[0] == 1:
            raise AllKeysExhaustedError("wait")
        return "key-a"
    monkeypatch.setattr(pool, "get_key", get_key)
    assert pool.call(lambda key: "ok") == "ok"
    assert sleeps == [pytest.approx(2)]
