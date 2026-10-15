"""Shared httpx plumbing with errors a user can act on."""
from __future__ import annotations

from typing import Any

import httpx

from ..errors import ModelError, ModelUnavailable


def transport_error(exc: httpx.TransportError, timeout: float) -> ModelUnavailable:
    if isinstance(exc, httpx.TimeoutException):
        return ModelUnavailable(
            f"model backend timed out after {timeout:.0f}s; the model may still be loading or the machine is low "
            "on memory (raise the timeout, close other apps, or use a smaller model)")
    return ModelUnavailable(f"cannot reach model backend: {exc!r}")


class HTTPProvider:
    """Mixin owning an ``httpx.AsyncClient``."""

    def __init__(self, base_url: str, timeout: float, client: httpx.AsyncClient | None) -> None:
        self._timeout = timeout
        self._owns_client = client is None
        self._http = client or httpx.AsyncClient(base_url=base_url, timeout=timeout)

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            resp = await self._http.post(path, json=payload)
            resp.raise_for_status()
            return resp.json()
        except httpx.TransportError as exc:
            raise transport_error(exc, self._timeout) from exc
        except httpx.HTTPStatusError as exc:
            raise ModelError(f"backend returned {exc.response.status_code}: {exc.response.text[:300]}") from exc

    async def _get(self, path: str) -> dict[str, Any]:
        try:
            resp = await self._http.get(path, timeout=10.0)
            resp.raise_for_status()
            return resp.json()
        except httpx.TransportError as exc:
            raise transport_error(exc, self._timeout) from exc
        except httpx.HTTPStatusError as exc:
            raise ModelError(f"backend returned {exc.response.status_code}") from exc

    async def aclose(self) -> None:
        if self._owns_client:
            await self._http.aclose()
