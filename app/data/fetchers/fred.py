from typing import List, Dict

import httpx

from app.core.cache import FileCache
from app.core.config import get_settings

cache = FileCache("fred")


def fetch_fred_series(series_id: str, limit: int = 30) -> List[Dict[str, str]]:
    cached = cache.get(series_id)
    if cached:
        return cached
    settings = get_settings()
    api_key = settings.fred_api_key
    if not api_key:
        raise RuntimeError("FRED API key missing; set RRIO_FRED_API_KEY")
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "desc",
        "limit": limit,
    }
    response = httpx.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()
    observations = [
        {"timestamp": obs["date"], "value": obs["value"]}
        for obs in data.get("observations", [])
        if obs.get("value") not in (".", None)
    ]
    observations.reverse()
    cache.set(series_id, observations)
    return observations


# Alias for backward compatibility
fetch_series = fetch_fred_series
