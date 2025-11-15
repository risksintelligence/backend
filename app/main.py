from fastapi import FastAPI
from datetime import datetime
from typing import List, Dict

app = FastAPI(title="RRIO GRII API", version="0.1.0")

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
    return {
        "composite": 0.68,
        "components": {
            "policy": 0.20,
            "analyses": 0.15,
            "labs": 0.12,
            "media": 0.11,
            "community": 0.10,
        },
        "calculated_at": datetime.utcnow().isoformat() + "Z",
    }
