"""open_keypool — a minimal Python library for pooling and rotating API keys.

Avoids HTTP 429 rate-limit errors by cycling through a pool of keys with
cooldown and disablement support. Provide a list of keys (or pull them from
Doppler), choose a rotation strategy (round-robin or least-recently-used),
and the pool handles cooldown on rate-limit responses and permanent
disablement on invalid keys — all thread-safe.

Install
-------
.. code-block:: bash

    pip install open-keypool

Quickstart — Local keys array
-----------------------------
.. code-block:: python

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

Quickstart — Doppler
--------------------
.. code-block:: python

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
"""

from open_keypool.core import AllKeysExhaustedError, KeyPool, KeyState

__all__ = ["KeyPool", "AllKeysExhaustedError", "KeyState"]
