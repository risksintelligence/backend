from datetime import datetime
from typing import Dict

from fastapi import FastAPI

from app.services.ingestion import ingest_local_series
from app.services.geri import compute_griscore
from app.services.impact import load_snapshot
from app.ml.regime import classify_regime
from app.ml.forecast import forecast_delta
from app.ml.anomaly import detect_anomalies
from app.services.transparency import get_data_freshness, get_update_log
from app.api import submissions as submissions_router

app = FastAPI(title="RRIO GRII API", version="0.4.0")


@app.on_event("startup")
def load_data_cache() -> None:
    app.state.observations = ingest_local_series()


def _get_observations() -> dict:
    observations = getattr(app.state, "observations", None)
    if observations is None:
        observations = ingest_local_series()
        app.state.observations = observations
    return observations


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok", "checked_at": datetime.utcnow().isoformat() + "Z"}


@app.get("/api/v1/analytics/geri")
def current_griscore() -> Dict[str, object]:
    observations = _get_observations()
    result = compute_griscore(observations)
    result["drivers"] = [
        {"component": comp, "contribution": round(value, 3)}
        for comp, value in result["contributions"].items()
    ]
    result["color"] = _band_color(result["band"])
    return result


@app.get("/api/v1/ai/regime/current")
def current_regime() -> Dict[str, object]:
    observations = _get_observations()
    probabilities = classify_regime(observations)
    return {
        "regime": max(probabilities, key=probabilities.get),
        "probabilities": probabilities,
    }


@app.get("/api/v1/ai/forecast/next-24h")
def next_day_forecast() -> Dict[str, float]:
    observations = _get_observations()
    return forecast_delta(observations)


@app.get("/api/v1/anomalies/latest")
def anomaly_feed() -> Dict[str, float]:
    observations = _get_observations()
    return detect_anomalies(observations)


@app.get("/api/v1/impact/ras")
def ras_snapshot() -> Dict[str, object]:
    snapshot = load_snapshot()
    return snapshot.to_dict()

@app.get("/api/v1/transparency/data-freshness")
def data_freshness() -> dict:
    return {"freshness": get_data_freshness()}


@app.get("/api/v1/transparency/update-log")
def update_log() -> dict:
    return {"entries": get_update_log()}

app.include_router(submissions_router.router)


def _band_color(band: str) -> str:
    mapping = {
        "minimal": "#00C853",
        "low": "#64DD17",
        "moderate": "#FFD600",
        "high": "#FFAB00",
        "critical": "#D50000",
    }
    return mapping.get(band, "#64DD17")
