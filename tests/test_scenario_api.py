import asyncio

import httpx
from fastapi import FastAPI

from backend.src.api.v1.scenario import router as scenario_router


def test_scenario_simulation_alert_and_runs():
    app = FastAPI()
    app.include_router(scenario_router)

    async def exercise():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            alert = await client.post(
                "/api/v1/alerts/subscribe",
                json={"channel": "email", "address": "test@example.com", "conditions": [{"type": "geri_band", "band": "High"}]},
            )
            sim = await client.post(
                "/api/v1/scenario/simulate",
                json={"shocks": [{"series": "OIL_WTI", "delta_percent": 10}], "horizon_hours": 24},
            )
            runs = await client.get("/api/v1/scenario/runs")
            subs = await client.get("/api/v1/alerts/subscriptions")
            deliveries = await client.get("/api/v1/alerts/deliveries")
            export = await client.get("/api/v1/scenario/runs/export")
        assert sim.status_code == 200
        assert "scenario" in sim.json()
        assert alert.status_code == 200
        assert runs.status_code == 200
        runs_payload = runs.json()
        assert runs_payload["count"] >= 1
        assert "ml_context" in runs_payload["runs"][0]
        assert subs.status_code == 200
        assert subs.json()["count"] >= 1
        assert deliveries.status_code == 200
        assert len(deliveries.json()["deliveries"]) >= 1
        assert export.status_code == 200
        assert "csv" in export.json()

    asyncio.run(exercise())
