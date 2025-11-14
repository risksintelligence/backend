import asyncio

import httpx
from fastapi import FastAPI

from src.api.v1.ai import router as ai_router


def create_app() -> FastAPI:
    app = FastAPI()
    app.include_router(ai_router)
    return app


def test_ai_endpoints_return_payloads():
    app = create_app()

    async def exercise():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            reg = await client.get("/api/v1/ai/regime/current")
            forecast = await client.get("/api/v1/ai/geri/forecast?horizon_hours=24")
            anomalies = await client.get("/api/v1/ai/anomalies")
        assert reg.status_code == 200
        assert forecast.status_code == 200
        assert anomalies.status_code == 200
        assert "regime" in reg.json()
        assert "delta_geri_prediction" in forecast.json()
        assert "anomalies" in anomalies.json()

    asyncio.run(exercise())
