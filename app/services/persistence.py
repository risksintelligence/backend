from typing import Dict, List

from app.services.ingestion import Observation

_storage: Dict[str, List[Observation]] = {}


def save_observations(series_id: str, observations: List[Observation]) -> None:
    _storage[series_id] = observations


def get_observations(series_id: str) -> List[Observation]:
    return _storage.get(series_id, [])
