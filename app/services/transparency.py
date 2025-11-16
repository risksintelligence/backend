import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from app.core.cache import FileCache
from app.core.config import get_settings
from app.services.ingestion import ingest_local_series

settings = get_settings()
TRANSPARENCY_FILE = settings.data_dir / "transparency.json"
cache = FileCache("freshness")

DEFAULT_DATA = {
    "update_log": [
        {"date": "2024-11-12", "description": "Added RRIO automation scripts"},
        {"date": "2024-11-10", "description": "Updated GRII weights"},
    ]
}


def _load_data() -> Dict[str, List[Dict[str, str]]]:
    if not TRANSPARENCY_FILE.exists():
        TRANSPARENCY_FILE.parent.mkdir(parents=True, exist_ok=True)
        TRANSPARENCY_FILE.write_text(json.dumps(DEFAULT_DATA, indent=2))
    return json.loads(TRANSPARENCY_FILE.read_text())


def _write_data(data: Dict[str, List[Dict[str, str]]]) -> None:
    TRANSPARENCY_FILE.write_text(json.dumps(data, indent=2))


def get_data_freshness() -> List[Dict[str, str]]:
    cached = cache.get("freshness")
    if cached:
        return cached
    observations = ingest_local_series()
    freshness = []
    for component, obs_list in observations.items():
        if not obs_list:
            continue
        last = obs_list[-1]
        status = 'fresh'
        age_days = (datetime.utcnow() - last.observed_at).days
        if age_days > 45:
            status = 'stale'
        elif age_days > 7:
            status = 'warning'
        freshness.append({
            "component": component,
            "status": status,
            "last_updated": last.observed_at.date().isoformat(),
        })
    cache.set("freshness", freshness)
    return freshness


def get_update_log() -> List[Dict[str, str]]:
    return _load_data().get("update_log", [])


def record_update(description: str) -> None:
    data = _load_data()
    log = data.setdefault("update_log", [])
    log.insert(0, {"date": datetime.utcnow().date().isoformat(), "description": description})
    _write_data(data)
