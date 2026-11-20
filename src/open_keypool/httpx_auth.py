"""httpx authentication adapters for open-keypool."""

from __future__ import annotations

from typing import Any, Generator, AsyncGenerator

import httpx

from .core import AsyncKeyPool, KeyPool, KeyState


class KeyPoolAuth(httpx.Auth):
    """Rotate keys from a pool while sending requests through httpx.

    ``scheme=None`` sends the raw key, which is useful for providers such as
    Google that expect an ``x-goog-api-key`` header.
    """

    requires_response_body = True

    def __init__(
        self,
        pool: KeyPool | AsyncKeyPool,
        header: str = "Authorization",
        scheme: str | None = "Bearer",
    ) -> None:
        self.pool = pool
        self.header = header
        self.scheme = scheme

    def _value(self, key: str) -> str:
        return key if self.scheme is None else f"{self.scheme} {key}"

    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, None]:
        if not isinstance(self.pool, KeyPool):
            raise TypeError("KeyPoolAuth.auth_flow requires a synchronous KeyPool")
        for _ in range(self.pool.max_retries):
            key = self.pool.get_key()
            request.headers[self.header] = self._value(key)
            response = yield request
            body = _response_body(response)
            state = self.pool.handle_response(key, response.status_code, dict(response.headers), body)
            if state == KeyState.ACTIVE:
                return

    async def async_auth_flow(
        self, request: httpx.Request
    ) -> AsyncGenerator[httpx.Request, httpx.Response]:
        if not isinstance(self.pool, AsyncKeyPool):
            raise TypeError("KeyPoolAuth.async_auth_flow requires an AsyncKeyPool")
        for _ in range(self.pool.max_retries):
            key = await self.pool.get_key()
            request.headers[self.header] = self._value(key)
            response = yield request
            await response.aread()
            body = _response_body(response)
            state = await self.pool.handle_response(key, response.status_code, dict(response.headers), body)
            if state == KeyState.ACTIVE:
                return


def _response_body(response: httpx.Response) -> Any:
    try:
        return response.json()
    except (ValueError, TypeError):
        return response.text
