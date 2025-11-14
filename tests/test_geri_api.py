import asyncio
from datetime import datetime, timedelta, timezone

import httpx
from fastapi import FastAPI

from src.analytics.normalization import NormalizationParams
from src.api.v1 import geri as geri_router
from src.core.cache_layers import DictCacheLayer
from src.core.cache_service import build_default_cache
from src.core.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from src.core.locks import InMemoryLockManager
from src.services.geri_service import GERISnapshotService, IndicatorRecord


async def _build_cache():
    memory = DictCacheLayer()
    redis = DictCacheLayer()
    postgres = DictCacheLayer()
    file_cache = DictCacheLayer()
    breaker = CircuitBreaker(CircuitBreakerConfig(3, 60))
    return await build_default_cache(memory, redis, postgres, file_cache, InMemoryLockManager(), breaker)


def create_app(service: GERISnapshotService):
    app = FastAPI()
    app.include_router(geri_router.router)

    async def override_service():
        return service

    async def override_cache():
        return await _build_cache()

    app.dependency_overrides[geri_router.get_geri_service] = override_service
    app.dependency_overrides[geri_router.get_cache] = override_cache
    return app


def _seed_service(service: GERISnapshotService):
    now = datetime.now(timezone.utc)
    service.ingest(
        "VIX",
        IndicatorRecord(
            value=25,
            params=NormalizationParams(20, 5, "positive"),
            provider="Yahoo",
            series_id="^VIX",
            observed_at=now,
            soft_ttl=timedelta(minutes=15),
            hard_ttl=timedelta(hours=24),
        ),
    )
    service.ingest(
        "YC_10Y_2Y",
        IndicatorRecord(
            value=0.2,
            params=NormalizationParams(0.5, 0.2, "negative"),
            provider="FRED",
            series_id="DGS10,DGS2",
            observed_at=now,
            soft_ttl=timedelta(hours=1),
            hard_ttl=timedelta(days=3),
        ),
    )
    service.ingest(
        "CREDIT_SPREAD",
        IndicatorRecord(
            value=1.5,
            params=NormalizationParams(1.0, 0.5, "positive"),
            provider="FRED",
            series_id="BAA10YM",
            observed_at=now,
            soft_ttl=timedelta(hours=1),
            hard_ttl=timedelta(days=3),
        ),
    )
    service.ingest(
        "FREIGHT_SHIPPING",
        IndicatorRecord(
            value=110,
            params=NormalizationParams(100, 10, "positive"),
            provider="EIA",
            series_id="FREIGHT",
            observed_at=now,
            soft_ttl=timedelta(hours=1),
            hard_ttl=timedelta(days=2),
        ),
    )
    service.ingest(
        "PMI_MANUFACTURING",
        IndicatorRecord(
            value=50,
            params=NormalizationParams(52, 3, "negative"),
            provider="ISM",
            series_id="NAPM",
            observed_at=now,
            soft_ttl=timedelta(days=1),
            hard_ttl=timedelta(days=45),
        ),
    )
    service.ingest(
        "OIL_WTI",
        IndicatorRecord(
            value=80,
            params=NormalizationParams(70, 5, "positive"),
            provider="EIA",
            series_id="DCOILWTICO",
            observed_at=now,
            soft_ttl=timedelta(hours=1),
            hard_ttl=timedelta(days=3),
        ),
    )
    service.ingest(
        "CPI_INDEX",
        IndicatorRecord(
            value=3,
            params=NormalizationParams(2, 0.5, "positive"),
            provider="FRED",
            series_id="CPIAUCSL",
            observed_at=now,
            soft_ttl=timedelta(days=1),
            hard_ttl=timedelta(days=60),
        ),
    )
    service.ingest(
        "UNEMPLOYMENT",
        IndicatorRecord(
            value=5,
            params=NormalizationParams(4, 0.5, "positive"),
            provider="FRED",
            series_id="UNRATE",
            observed_at=now,
            soft_ttl=timedelta(days=1),
            hard_ttl=timedelta(days=60),
        ),
    )


def test_geri_api_returns_payload():
    service = GERISnapshotService()
    _seed_service(service)
    app = create_app(service)

    async def exercise():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.get("/api/v1/analytics/geri")
        assert response.status_code == 200
        data = response.json()
        assert data["index_name"] == "geri_v1.0"
        assert "components" in data
        assert "data_freshness" in data
        assert data["cache_source"] in {"memory", "redis", "postgres", "file", "origin"}

    asyncio.run(exercise())
