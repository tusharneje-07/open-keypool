# open-keypool

Minimal Python library for pooling and rotating API keys to avoid HTTP 429 rate-limit errors. Provide a list of keys (or pull them from Doppler), choose a rotation strategy (round-robin or least-recently-used), and the pool handles cooldown on rate-limit responses and permanent disablement on invalid keys — all thread-safe.

## Install

```bash
# From TestPyPI (until published on PyPI):
pip install --index-url https://test.pypi.org/simple/ open-keypool
```

## Quickstart

### Local keys array

```python
from open_keypool import KeyPool, AllKeysExhaustedError, KeyState

pool = KeyPool(keys=["sk-key1", "sk-key2", "sk-key3"], strategy="round_robin")

for attempt in range(pool.max_retries):
    key = pool.get_key()
    response = call_your_api(key)

    # Feed the response — the pool decides success / cooldown / disable
    new_state = pool.handle_response(
        key, response.status_code,
        headers=dict(response.headers),
        body=response.json(),
    )

    if new_state == KeyState.ACTIVE:
        break  # success
    elif new_state == KeyState.COOLDOWN:
        continue  # key is rate-limited, rotate to next
    elif new_state == KeyState.DISABLED:
        continue  # key is invalid, rotate to next
```

### Handle response auto-dispatching

`pool.handle_response(key, status_code, headers, body)` introspects the HTTP response and automatically:

| Status | Action |
|---|---|
| **2xx** | Marks success — clears errors, resets failure count |
| **429**, **413**, or `"rate_limit_exceeded"` in body | Marks cooldown, reads `Retry-After` header |
| **401**, **403** | Permanently disables the key |
| **5xx** | Places on cooldown (transient) |

Returns `KeyState` so you can branch on the result.

### Multi-key Doppler pool with status tracking

```python
import os
from open_keypool import KeyPool

DOPPLER_TOKEN = os.getenv("DOPPLER_TOKEN", "dp.st.YOUR_SERVICE_TOKEN")

pool = KeyPool.from_doppler(
    token=DOPPLER_TOKEN,
    project="refactor-ai",
    config="dev",
    key_prefix="GROQ_",
    strategy="round_robin",
)

# Every key's state, error history, and cooldown — safely masked
for masked_key, info in pool.status().items():
    print(f"{masked_key}  state={info['state']}  "
          f"http={info.get('last_status_code')}  "
          f"err=[{info.get('last_error_code')}]  "
          f"failures={info['failure_count']}")
```

## Constructor parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `keys` | `list[str]` | *required* | Initial API key strings (non-empty). |
| `max_retries` | `int` | `3` | Max retry count reference for the caller's loop. |
| `cooldown_seconds` | `int` | `60` | How long a rate-limited key stays in cooldown. |
| `strategy` | `str` | `"round_robin"` | Rotation strategy: `"round_robin"` or `"lru"`. |

## Doppler caching

`KeyPool.from_doppler()` uses an in-memory TTL cache with a 1-hour expiration. On the first call within a process, keys are fetched from Doppler and cached. Subsequent calls within the same hour serve keys from memory without touching the network. After one hour (if the process is still running), the cache entry expires and the next call fetches fresh keys automatically. The cache is never persisted across process restarts — every fresh process starts with an empty cache.

Pass `force_refresh=True` to bypass the cache and re-fetch immediately (useful after rotating keys in Doppler when you don't want to wait out the TTL).

## Full API reference

[docs/index.html](docs/index.html) — self-contained HTML page with quickstart + class/method documentation generated from docstrings.
