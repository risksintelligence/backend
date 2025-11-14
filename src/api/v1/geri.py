"""FastAPI router exposing deterministic GERI endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from src.core.cache_layers import DictCacheLayer
from src.core.cache_service import CacheHit, IntelligentCache, LoadResult, build_default_cache
from src.core.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from src.core.locks import InMemoryLockManager
from src.services.geri_service import GERISnapshotService, seed_demo_data
from src.services.research_service import get_research_service
from src.services.transparency_service import get_transparency_service
from time import perf_counter

router = APIRouter(prefix="/api/v1/analytics/geri", tags=["geri"])

_GLOBAL_SERVICE: GERISnapshotService | None = None
_GLOBAL_CACHE: IntelligentCache | None = None


def get_geri_service() -> GERISnapshotService:
    global _GLOBAL_SERVICE
    if _GLOBAL_SERVICE is None:
        service = GERISnapshotService()
        seed_demo_data(service)
        _GLOBAL_SERVICE = service
    return _GLOBAL_SERVICE


async def get_cache() -> IntelligentCache:
    global _GLOBAL_CACHE
    if _GLOBAL_CACHE is None:
        try:
            memory = DictCacheLayer()
            redis = DictCacheLayer()
            postgres = DictCacheLayer()
            file_cache = DictCacheLayer()
            breaker = CircuitBreaker(CircuitBreakerConfig(failure_threshold=3, reset_timeout=60))
            _GLOBAL_CACHE = await build_default_cache(memory, redis, postgres, file_cache, InMemoryLockManager(), breaker)
        except Exception as e:
            # Fall back to memory-only cache if there are connection issues
            from src.core.logging import get_logger
            logger = get_logger(__name__)
            logger.warning(f"Failed to initialize full cache, falling back to memory-only: {e}")
            memory = DictCacheLayer()
            breaker = CircuitBreaker(CircuitBreakerConfig(failure_threshold=3, reset_timeout=60))
            from src.core.cache_service import IntelligentCache, CacheTier
            _GLOBAL_CACHE = IntelligentCache([CacheTier("memory", memory, True)], InMemoryLockManager(), breaker)
    return _GLOBAL_CACHE


@router.get("")
async def get_current_geri(
    cache: IntelligentCache = Depends(get_cache),
    service: GERISnapshotService = Depends(get_geri_service),
    transparency=Depends(get_transparency_service),
    research=Depends(get_research_service),
):
    async def loader() -> LoadResult:
        try:
            payload, soft_ttl, hard_ttl = service.build_payload()
        except ValueError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return LoadResult(value=payload, soft_ttl=soft_ttl, hard_ttl=hard_ttl)

    start = perf_counter()
    hit: CacheHit = await cache.get("geri_current", loader)
    duration_ms = (perf_counter() - start) * 1000
    payload = hit.value.copy()
    payload["stale"] = hit.stale
    payload["cache_source"] = hit.source
    transparency.record_cache_query(hit.source, duration_ms, hit.stale)
    transparency.record_geri_snapshot(payload)
    research.record_snapshot(payload)
    return payload
