import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from src.core.cache_layers import CacheEntry, DictCacheLayer
from src.core.cache_service import (
    CacheTier,
    IntelligentCache,
    LoadResult,
)
from src.core.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerOpen
from src.core.locks import InMemoryLockManager


def _entry(value: str, age_seconds: int = 0, soft: int = 60, hard: int = 3600) -> CacheEntry:
    stored_at = datetime.now(timezone.utc) - timedelta(seconds=age_seconds)
    return CacheEntry(
        value=value,
        stored_at=stored_at,
        soft_ttl=timedelta(seconds=soft),
        hard_ttl=timedelta(seconds=hard),
    )


async def _build_cache(refresh_executor=None) -> tuple[IntelligentCache, DictCacheLayer, DictCacheLayer, DictCacheLayer, DictCacheLayer]:
    l1 = DictCacheLayer()
    l2 = DictCacheLayer()
    l3 = DictCacheLayer()
    l4 = DictCacheLayer()
    tiers = [
        CacheTier("memory", l1, True),
        CacheTier("redis", l2, True),
        CacheTier("postgres", l3, False),
        CacheTier("file", l4, True),
    ]
    breaker = CircuitBreaker(CircuitBreakerConfig(failure_threshold=2, reset_timeout=timedelta(seconds=5)))
    cache = IntelligentCache(tiers, InMemoryLockManager(), breaker, refresh_executor=refresh_executor)
    return cache, l1, l2, l3, l4


@pytest.mark.asyncio
async def test_hit_in_memory_layer_returns_fresh_result():
    cache, l1, *_ = await _build_cache()
    await l1.set("geri", _entry("fresh"))

    hit = await cache.get("geri", loader=lambda: asyncio.sleep(0))

    assert hit.value == "fresh"
    assert hit.stale is False
    assert hit.source == "memory"


@pytest.mark.asyncio
async def test_soft_expired_entry_triggers_background_refresh():
    tasks: list[asyncio.Task] = []

    def executor(coro):
        tasks.append(asyncio.create_task(coro))

    loader_called = 0

    async def loader():
        nonlocal loader_called
        loader_called += 1
        return LoadResult(value="new", soft_ttl=timedelta(seconds=60), hard_ttl=timedelta(seconds=300))

    cache, l1, l2, *_ = await _build_cache(refresh_executor=executor)
    await l2.set("geri", _entry("old", age_seconds=120, soft=60, hard=600))

    hit = await cache.get("geri", loader)
    assert hit.value == "old"
    assert hit.stale is True
    assert loader_called == 0  # refresh scheduled async

    # Execute scheduled refresh
    await asyncio.gather(*tasks)
    assert loader_called == 1
    assert (await l1.get("geri")).value == "new"


@pytest.mark.asyncio
async def test_cache_miss_invokes_loader_once_and_promotes():
    loader_called = 0

    async def loader():
        nonlocal loader_called
        loader_called += 1
        return LoadResult(value="fresh", soft_ttl=timedelta(seconds=120), hard_ttl=timedelta(seconds=600))

    cache, l1, l2, *_ = await _build_cache()
    hit = await cache.get("geri", loader)

    assert hit.value == "fresh"
    assert loader_called == 1
    assert (await l1.get("geri")).value == "fresh"
    assert (await l2.get("geri")).value == "fresh"


@pytest.mark.asyncio
async def test_circuit_breaker_prevents_loader_when_open():
    async def loader():
        raise AssertionError("loader should not run when breaker open")

    cache, *_ = await _build_cache()
    cache._breaker.record_failure()
    cache._breaker.record_failure()
    with pytest.raises(CircuitBreakerOpen):
        await cache.get("missing", loader)


@pytest.mark.asyncio
async def test_stampede_lock_allows_single_loader_execution():
    loader_called = 0

    async def loader():
        nonlocal loader_called
        loader_called += 1
        await asyncio.sleep(0.01)
        return LoadResult(value="value", soft_ttl=timedelta(seconds=60), hard_ttl=timedelta(seconds=600))

    cache, *_ = await _build_cache()

    results = await asyncio.gather(*(cache.get("geri", loader) for _ in range(5)))
    assert all(result.value == "value" for result in results)
    assert loader_called == 1
