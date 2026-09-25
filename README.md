# open-keypool

Minimal Python library for pooling and rotating API keys to avoid HTTP 429 rate-limit errors. Provide a list of keys (or pull them from Doppler, `.env`, or JSON), choose a rotation strategy (round-robin or least-recently-used), and the pool handles cooldown on rate-limit responses and permanent disablement on invalid keys — all thread-safe.

## Install

```bash
pip install open-keypool
```

## Quickstart

### High-level pool.call() (Recommended)

`pool.call(fn, *args, **kwargs)` runs the `get_key → fn → handle_response` loop automatically:

```python
import httpx
from open_keypool import KeyPool

pool = KeyPool(keys=["sk-key1", "sk-key2", "sk-key3"], provider="groq")

# Collapses execution down to 3 lines with automatic key rotation and retries:
def fetch_chat(key):
    return httpx.get("https://api.groq.com/v1/models", headers={"Authorization": f"Bearer {key}"})

response = pool.call(fetch_chat)
```

### Async Usage (`AsyncKeyPool`)

```python
import httpx
import asyncio
from open_keypool import AsyncKeyPool

async def main():
    pool = AsyncKeyPool(keys=["sk-key1", "sk-key2"], provider="openai")

    async def fetch_models(key):
        async with httpx.AsyncClient() as client:
            return await client.get("https://api.openai.com/v1/models", headers={"Authorization": f"Bearer {key}"})

    response = await pool.call(fetch_models)

asyncio.run(main())
```

### Provider Presets

Pre-configure how response rate-limit headers and error structures are handled (`"groq"`, `"openai"`, `"gemini"`, `"together"`):

```python
pool = KeyPool(keys=["sk-groq-1", "sk-groq-2"], provider="groq")
# Automatically handles Groq rate limit headers (x-ratelimit-reset-requests, etc.)
```

### Manual rotation loop

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

### Load keys from Doppler

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

### Load keys from `.env` file

```python
from open_keypool import KeyPool

pool = KeyPool.from_env(suffix="GROQ_KEY")
```

### Load keys from JSON file

```python
from open_keypool import KeyPool

pool = KeyPool.from_json("keys.json", suffix="GROQ_KEY")
```

### Load keys from AWS Secrets Manager or GCP Secret Manager

```python
from open_keypool import KeyPool

# AWS Secrets Manager (requires open-keypool[aws] or boto3)
aws_pool = KeyPool.from_aws_secrets("my-app-secrets", key_prefix="GROQ_")

# GCP Secret Manager (requires open-keypool[gcp] or google-cloud-secret-manager)
gcp_pool = KeyPool.from_gcp_secrets("my-app-secrets", project_id="my-project", key_prefix="GROQ_")
```

## Constructor parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `keys` | `list[str]` | *required* | Initial API key strings (non-empty). |
| `max_retries` | `int` | `3` | Max retry count reference for the caller's loop. |
| `cooldown_seconds` | `int` | `60` | How long a rate-limited key stays in cooldown. |
| `strategy` | `str` | `"round_robin"` | Rotation strategy: `"round_robin"` or `"lru"`. |
| `provider` | `str` | `None` | Provider preset: `"groq"`, `"openai"`, `"gemini"`, `"together"`. |

## Doppler caching

`KeyPool.from_doppler()` uses an in-memory TTL cache with a 1-hour expiration. On the first call within a process, keys are fetched from Doppler and cached. Subsequent calls within the same hour serve keys from memory without touching the network. After one hour (if the process is still running), the cache entry expires and the next call fetches fresh keys automatically. The cache is never persisted across process restarts — every fresh process starts with an empty cache.

Pass `force_refresh=True` to bypass the cache and re-fetch immediately (useful after rotating keys in Doppler when you don't want to wait out the TTL).

## Full API reference

Read [Docs](https://tusharneje.in/projects/open-keypool/).

## Contributing

Contributions are welcome. If you have an idea, find a bug, or want to improve `open-keypool`, feel free to contribute.

### How to Contribute

1. Fork the repository.
2. Create a new branch for your changes.
3. Make your changes and add appropriate tests.
4. Run the test suite and make sure all tests pass.
5. Commit your changes with a clear message.
6. Open a Pull Request describing what you changed and why.

Please keep contributions focused on the core goal of `open-keypool`: **simple and reliable API key pooling and rotation**.

For larger changes or new features, open an issue first so the approach can be discussed before implementation.

## License

`open-keypool` is released under the [MIT License](LICENSE).

See the `LICENSE` file for the full license text.