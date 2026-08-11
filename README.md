# open-keypool

Minimal Python library for pooling and rotating API keys to avoid HTTP 429 rate-limit errors. Provide a list of keys (or pull them from Doppler), choose a rotation strategy (round-robin or least-recently-used), and the pool handles cooldown on rate-limit responses and permanent disablement on invalid keys — all thread-safe.

## Install

```bash
pip install open-keypool
```

## Quickstart

### Local keys array

```python
from open_keypool import KeyPool, AllKeysExhaustedError

pool = KeyPool(keys=["sk-key1", "sk-key2", "sk-key3"], strategy="round_robin")

for attempt in range(pool.max_retries):
    key = pool.get_key()
    response = call_your_api(key)
    if response.status_code == 429:
        retry_after = float(response.headers.get("Retry-After", 0))
        pool.mark_rate_limited(key, retry_after=retry_after or None)
    elif response.status_code in (401, 403):
        pool.mark_invalid(key)
    else:
        pool.mark_success(key)
        break
```

### Doppler

```python
import os
from open_keypool import KeyPool

DOPPLER_TOKEN = os.getenv("DOPPLER_TOKEN", "dp.st.YOUR_SERVICE_TOKEN")
PROJECT_NAME = "refactor-ai"
CONFIG_NAME = "dev"

pool = KeyPool.from_doppler(
    token=DOPPLER_TOKEN,
    project=PROJECT_NAME,
    config=CONFIG_NAME,
    key_prefix="MY_APP_",
    strategy="lru",
)
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

Full API reference: [docs/index.html](docs/index.html)
