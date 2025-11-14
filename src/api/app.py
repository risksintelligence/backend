from fastapi import FastAPI

from backend.src.api.v1.ai import router as ai_router
from backend.src.api.v1.geri import router as geri_router
from backend.src.api.v1.research import router as research_router
from backend.src.api.v1.scenario import router as scenario_router
from backend.src.api.v1.scenario_alerts import router as scenario_alerts_router
from backend.src.api.v1.subscription import router as subscription_router
from backend.src.api.v1.transparency import router as transparency_router
from backend.src.api.v1.admin import router as admin_router
from backend.src.api.v1.alerts import router as alerts_router
from backend.src.api.v1.monitoring import router as monitoring_router
from backend.src.api.v1.auth import router as auth_router
from backend.src.api.v1.backup import router as backup_router
from backend.src.api.v1.ml_admin import router as ml_admin_router
from backend.src.api.v1.scenario_sharing import router as scenario_sharing_router
from backend.src.api.v1.cron_admin import router as cron_admin_router
from backend.src.api.v1.advanced_exports import router as advanced_exports_router

app = FastAPI(title="RiskSX Intelligence System API")
app.include_router(geri_router)
app.include_router(ai_router)
app.include_router(transparency_router)
app.include_router(research_router)
app.include_router(scenario_router)
app.include_router(scenario_alerts_router)
app.include_router(subscription_router)
app.include_router(admin_router)
app.include_router(alerts_router)
app.include_router(monitoring_router)
app.include_router(auth_router)
app.include_router(backup_router)
app.include_router(ml_admin_router)
app.include_router(scenario_sharing_router)
app.include_router(cron_admin_router)
app.include_router(advanced_exports_router)


@app.get("/healthz")
async def health_check():
    return {"status": "ok"}
