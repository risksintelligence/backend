import httpx
import asyncio
from fastapi import FastAPI

from src.api.v1.admin import router as admin_router

def test_admin_alerts_endpoint():
    app = FastAPI()
    app.include_router(admin_router)

    async def exercise():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            alerts = await client.get("/api/v1/admin/alerts")
        assert alerts.status_code == 200

    asyncio.run(exercise())
