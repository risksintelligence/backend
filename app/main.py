from datetime import datetime
from typing import Dict, List

from fastapi import FastAPI

from app.services.impact import load_snapshot

app = FastAPI(title="RRIO GRII API", version="0.2.0")

sample_drivers: List[Dict[str, str]] = [
    {"component": "VIX", "contribution": "2.1", "direction": "up"},
    {"component": "PMI", "contribution": "-1.4", "direction": "down"},
]


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok", "checked_at": datetime.utcnow().isoformat() + "Z"}


@app.get("/api/v1/analytics/geri")
def current_griscore() -> Dict[str, object]:
    return {
        "score": 61.2,
        "band": "high",
        "color": "#FFAB00",
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "drivers": sample_drivers,
    }


@app.get("/api/v1/impact/ras")
def ras_snapshot() -> Dict[str, object]:
    snapshot = load_snapshot()
    return snapshot.to_dict()
