from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

from app.data.registry import SERIES_REGISTRY
from app.data.sources import get_source
from app.db import SessionLocal
from app.models import ObservationModel


@dataclass
class Observation:
    series_id: str
    observed_at: datetime
    value: float


def ingest_local_series() -> Dict[str, List[Observation]]:
    observations: Dict[str, List[Observation]] = {}
    for key, metadata in SERIES_REGISTRY.items():
        fetch_func = get_source(metadata.provider)
        raw_points = fetch_func(metadata.id)
        obs_list = []
        for point in raw_points:
            obs_list.append(
                Observation(
                    series_id=key,
                    observed_at=datetime.fromisoformat(point["timestamp"]),
                    value=float(point["value"]),
                )
            )
        obs_list.sort(key=lambda o: o.observed_at)
        observations[key] = obs_list
        _persist_observations(key, obs_list)
    return observations


def _persist_observations(series_id: str, obs_list: List[Observation]) -> None:
    db = SessionLocal()
    for obs in obs_list:
        db.merge(
            ObservationModel(series_id=series_id, observed_at=obs.observed_at, value=obs.value)
        )
    db.commit()
    db.close()
