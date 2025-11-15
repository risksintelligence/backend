"""Shared HTTP client helpers for external data providers."""
from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

import httpx


class ExternalAPIError(RuntimeError):
    pass


class BaseAPIClient:
    """Wraps httpx.AsyncClient with convenience helpers."""

    def __init__(self, base_url: str, *, timeout: float = 10.0, http_client: Optional[httpx.AsyncClient] = None) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._http_client = http_client
        self._owns_client = http_client is None

    async def _client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=self._timeout, base_url=self._base_url)
        return self._http_client

    async def close(self) -> None:
        if self._owns_client and self._http_client is not None:
            await self._http_client.aclose()

    async def _request(self, method: str, path: str, *, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        client = await self._client()
        url = f"{self._base_url}{path}"
        try:
            response = await client.request(method, url, params=params)
        except httpx.HTTPError as exc:  # pragma: no cover - network failure paths
            raise ExternalAPIError(str(exc)) from exc
        if response.status_code >= 400:
            raise ExternalAPIError(f"{response.status_code} {response.text}")
        try:
            return response.json()
        except ValueError as exc:  # pragma: no cover
            raise ExternalAPIError("Invalid JSON response") from exc
