"""open_keypool — a minimal Python library for pooling and rotating API keys.

Avoids HTTP 429 rate-limit errors by cycling through a pool of keys with
cooldown and disablement support. Provide a list of keys (or pull them from
Doppler, AWS, GCP, .env, or JSON), choose a rotation strategy (round-robin or
least-recently-used), and the pool handles cooldown on rate-limit responses and
permanent disablement on invalid keys — all thread-safe and async-compatible.

Install
-------
.. code-block:: bash

    pip install open-keypool

Quickstart — High-level pool.call()
-----------------------------------
.. code-block:: python

    import httpx
    from open_keypool import KeyPool

    pool = KeyPool(keys=["sk-key1", "sk-key2"], provider="groq")

    def fetch_models(key):
        return httpx.get("https://api.groq.com/v1/models", headers={"Authorization": f"Bearer {key}"})

    response = pool.call(fetch_models)

Quickstart — AsyncKeyPool
-------------------------
.. code-block:: python

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
"""

from open_keypool.core import AllKeysExhaustedError, AsyncKeyPool, KeyPool, KeyState

__all__ = ["KeyPool", "AsyncKeyPool", "AllKeysExhaustedError", "KeyState"]
