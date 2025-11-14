import asyncio

import httpx
from fastapi import FastAPI

from backend.src.api.v1.transparency import router as transparency_router
from backend.src.services.transparency_service import TransparencyService, get_transparency_service


def test_transparency_endpoint_returns_expected_payload():
    app = FastAPI()
    app.include_router(transparency_router)

    service = TransparencyService()
    service.record_cache_query("redis", 5.0, False)
    service.record_geri_snapshot({
        "data_freshness": {
            "VIX": {"observed_at": "2025-01-01T00:00:00Z", "stale": False},
        }
    })

    app.dependency_overrides[get_transparency_service] = lambda: service

    async def exercise():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.get("/api/v1/transparency")
        assert response.status_code == 200
        payload = response.json()
        assert "generatedAt" in payload
        assert len(payload["cacheLayers"]) > 0
        assert len(payload["mlModels"]) > 0
        assert payload["cacheHitRatio"] >= 0

    asyncio.run(exercise())
