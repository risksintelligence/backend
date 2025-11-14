"""Stampede lock helpers.

Abstracts locking so we can plug Redis-based SETNX or in-process locks. Tests rely on
a deterministic `InMemoryLockManager`."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict


class StampedeLockManager:
    """Basic interface for distributed locks."""

    @asynccontextmanager
    async def lock(self, key: str) -> AsyncIterator[None]:  # pragma: no cover - interface
        raise NotImplementedError


class InMemoryLockManager(StampedeLockManager):
    """Per-process asyncio lock map used for tests + local dev."""

    def __init__(self) -> None:
        self._locks: Dict[str, asyncio.Lock] = {}
        self._global = asyncio.Lock()

    async def _get_lock(self, key: str) -> asyncio.Lock:
        async with self._global:
            lock = self._locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[key] = lock
            return lock

    @asynccontextmanager
    async def lock(self, key: str) -> AsyncIterator[None]:
        lock = await self._get_lock(key)
        await lock.acquire()
        try:
            yield
        finally:
            lock.release()
