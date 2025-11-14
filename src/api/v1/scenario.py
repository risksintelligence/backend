from typing import List

from fastapi import APIRouter, Depends

from backend.src.services.scenario_service import AlertService, ScenarioService, get_alert_service, get_scenario_service
from backend.src.api.middleware.auth import require_auth, limit_scenarios, limit_exports
from backend.src.services.auth_service import User

router = APIRouter(prefix="/api/v1", tags=["scenario"])


@router.post("/scenario/simulate")
async def simulate_scenario(
    request: dict,
    user: User = Depends(limit_scenarios),  # Check usage limits and track
    service: ScenarioService = Depends(get_scenario_service),
    alert_service: AlertService = Depends(get_alert_service),
):
    shocks: List[dict] = request.get("shocks", [])
    horizon = int(request.get("horizon_hours", 24))
    result = await service.simulate(shocks, horizon)
    payload = {
        "event": "scenario_run",
        "baseline": result.baseline,
        "scenario": result.scenario,
        "delta": result.delta,
    }
    await alert_service.deliver_alerts(payload)
    return {
        "baseline": result.baseline,
        "scenario": result.scenario,
        "band": result.band,
        "delta": result.delta,
        "explanation": result.explanation,
    }


@router.post("/alerts/subscribe")
async def subscribe_alert(
    request: dict,
    alert_service: AlertService = Depends(get_alert_service),
):
    channel = request.get("channel")
    address = request.get("address")
    conditions = request.get("conditions", [])
    subscription = alert_service.subscribe(channel, address, conditions)
    return subscription


@router.get("/scenario/runs")
async def list_runs(service: ScenarioService = Depends(get_scenario_service), limit: int = 20):
    runs = service.list_runs(limit)
    return {"count": len(runs), "runs": runs}


@router.get("/alerts/subscriptions")
async def list_subscriptions(alert_service: AlertService = Depends(get_alert_service)):
    subs = alert_service.list_subscriptions()
    return {"count": len(subs), "subscriptions": subs}


@router.get("/alerts/deliveries")
async def list_deliveries(alert_service: AlertService = Depends(get_alert_service)):
    return {"deliveries": alert_service.deliveries()}


@router.get("/scenario/runs/export")
async def export_runs(
    limit: int = 50, 
    format: str = "csv",
    user: User = Depends(limit_exports),  # Check export limits
    service: ScenarioService = Depends(get_scenario_service)
):
    import csv
    from io import StringIO
    runs = service.list_runs(limit)
    
    if format.lower() == "csv":
        buffer = StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["id", "timestamp", "baseline", "scenario", "band", "delta", "horizon_hours", "shocks"])
        for run in runs:
            shocks_str = "; ".join([f"{s['series']}:{s['delta_percent']}%" for s in run.get("shocks", [])])
            writer.writerow([
                run["id"], 
                run["timestamp"], 
                run["baseline"], 
                run["scenario"], 
                run["band"], 
                run.get("delta", run["scenario"] - run["baseline"]),
                run["horizon_hours"],
                shocks_str
            ])
        return {"csv": buffer.getvalue()}
    else:
        return {"runs": runs}
