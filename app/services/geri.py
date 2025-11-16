from datetime import datetime
from statistics import mean, pstdev
from typing import Dict, List

from app.services.ingestion import Observation

WEIGHTS = {
    "VIX": 0.3,
    "PMI": 0.3,
    "CREDIT_SPREAD": 0.4,
}


def z_scores(observations: List[Observation]) -> List[float]:
    values = [obs.value for obs in observations]
    if not values:
        return []
    avg = mean(values)
    deviation = pstdev(values) or 1
    return [(value - avg) / deviation for value in values]


def compute_griscore(observations: Dict[str, List[Observation]]) -> Dict[str, float]:
    contributions = {}
    composite = 0.0
    for series_id, obs_list in observations.items():
        scores = z_scores(obs_list)
        latest_z = scores[-1] if scores else 0.0
        weight = WEIGHTS.get(series_id, 0)
        contributions[series_id] = latest_z * weight
        composite += latest_z * weight
    scaled = max(0.0, min(100.0, 50 + composite * 10))
    return {
        "score": round(scaled, 2),
        "band": _band(scaled),
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "contributions": contributions,
    }


def _band(score: float) -> str:
    if score < 20:
        return "minimal"
    if score < 40:
        return "low"
    if score < 60:
        return "moderate"
    if score < 80:
        return "high"
    return "critical"
