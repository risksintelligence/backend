"""Intelligent multi-tier cache with SWR and stampede protection."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, List, Optional, Sequence

from .cache_layers import CacheEntry, CacheLayer
from .circuit_breaker import CircuitBreaker, CircuitBreakerOpen
from .locks import StampedeLockManager


LoadFn = Callable[[], Awaitable["LoadResult"]]


@dataclass
class LoadResult:
    value: Any
    soft_ttl: timedelta
    hard_ttl: timedelta
    metadata: dict[str, Any] | None = None


@dataclass
class CacheHit:
    value: Any
    stale: bool
    source: str


@dataclass
class CacheTier:
    name: str
    layer: CacheLayer
    writable: bool = True


class IntelligentCache:
    """Implements documented multi-tier caching strategy."""

    def __init__(
        self,
        tiers: Sequence[CacheTier],
        lock_manager: StampedeLockManager,
        breaker: CircuitBreaker,
        refresh_executor: Optional[Callable[[Awaitable[None]], None]] = None,
    ) -> None:
        if not tiers:
            raise ValueError("At least one cache tier is required")
        self._tiers = list(tiers)
        self._lock_manager = lock_manager
        self._breaker = breaker
        self._refresh_executor = refresh_executor or (lambda task: asyncio.create_task(task))

    async def get(self, key: str, loader: LoadFn) -> CacheHit:
        now = datetime.now(timezone.utc)
        for idx, tier in enumerate(self._tiers):
            entry = await tier.layer.get(key)
            if not entry:
                continue
            if entry.is_hard_expired(now):
                continue
            stale = entry.is_soft_expired(now)
            if stale:
                self._schedule_refresh(key, loader)
            await self._promote(key, entry, idx)
            return CacheHit(entry.value, stale, tier.name)
        # cache miss across all tiers
        entry = await self._refresh(key, loader)
        return CacheHit(entry.value, False, "origin")

    async def _refresh(self, key: str, loader: LoadFn) -> CacheEntry:
        async with self._lock_manager.lock(key):
            # Another coroutine might have repopulated while we waited
            existing = await self._tiers[0].layer.get(key)
            if existing and not existing.is_hard_expired() and not existing.is_soft_expired():
                return existing
            self._breaker.guard()
            result = await loader()
            entry = CacheEntry(
                value=result.value,
                stored_at=datetime.now(timezone.utc),
                soft_ttl=result.soft_ttl,
                hard_ttl=result.hard_ttl,
                metadata=result.metadata or {},
            )
            await self._write_all(key, entry)
            self._breaker.record_success()
            return entry

    def _schedule_refresh(self, key: str, loader: LoadFn) -> None:
        async def refresh_task() -> None:
            try:
                await self._refresh(key, loader)
            except CircuitBreakerOpen:
                # Already logged elsewhere
                pass
        self._refresh_executor(refresh_task())

    async def _write_all(self, key: str, entry: CacheEntry) -> None:
        for tier in self._tiers:
            if tier.writable:
                await tier.layer.set(key, entry)

    async def _promote(self, key: str, entry: CacheEntry, tier_index: int) -> None:
        if tier_index == 0:
            return
        for idx in range(tier_index):
            tier = self._tiers[idx]
            if tier.writable:
                await tier.layer.set(key, entry)


async def build_default_cache(
    l1: CacheLayer,
    l2: CacheLayer,
    l3: CacheLayer,
    l4: CacheLayer,
    lock_manager: StampedeLockManager,
    breaker: CircuitBreaker,
) -> IntelligentCache:
    tiers: List[CacheTier] = [
        CacheTier("memory", l1, True),
        CacheTier("redis", l2, True),
        CacheTier("postgres", l3, False),
        CacheTier("file", l4, True),
    ]
    return IntelligentCache(tiers, lock_manager, breaker)
