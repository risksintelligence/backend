import asyncio
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI

from backend.src.api.v1.research import router as research_router
from backend.src.services.research_service import ResearchService, get_research_service


def test_research_history_and_methodology_endpoints():
    app = FastAPI()
    app.include_router(research_router)

    service = ResearchService()
    service.record_snapshot({
        "value": 60.0,
        "band": "Moderate",
        "components": {"financial": 62, "supply_chain": 58, "macro": 50},
        "stale": False,
        "snapshot_ts_utc": datetime.now(timezone.utc).isoformat(),
    })

    app.dependency_overrides[get_research_service] = lambda: service

    async def exercise():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            history = await client.get("/api/v1/research/history")
            method = await client.get("/api/v1/research/methodology")
            review_resp = await client.post(
                "/api/v1/research/reviews",
                json={"name": "Reviewer", "decision": "approve", "comments": "Looks good"},
            )
            reviews_list = await client.get("/api/v1/research/reviews")
        assert history.status_code == 200
        payload = history.json()
        assert payload["count"] >= 1
        assert payload["data"][0]["value"] == 60.0
        assert method.status_code == 200
        assert method.json()["current_version"] == "geri_v1.0"
        assert review_resp.status_code == 200
        assert reviews_list.json()["count"] >= 1

    asyncio.run(exercise())
