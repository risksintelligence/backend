from typing import List, Dict
import httpx
from app.core.cache import FileCache
from app.core.config import get_settings

cache = FileCache("eia")


def fetch_eia_series(series_id: str, limit: int = 30) -> List[Dict[str, str]]:
    """Fetch data from EIA (Energy Information Administration) API."""
    cached = cache.get(series_id)
    if cached:
        return cached
    
    settings = get_settings()
    api_key = settings.eia_api_key
    if not api_key:
        raise RuntimeError("EIA API key missing; set RIS_EIA_API_KEY")
    
    # EIA API v2 format
    url = f"https://api.eia.gov/v2/petroleum/pri/gnd/data/"
    params = {
        "api_key": api_key,
        "frequency": "weekly",
        "data[0]": "value",
        "facets[product][]": "EPD2DXL0",  # Diesel fuel
        "facets[area][]": "NUS",  # National US
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "length": limit
    }
    
    try:
        response = httpx.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        observations = []
        if "response" in data and "data" in data["response"]:
            for item in data["response"]["data"]:
                if item.get("value") is not None:
                    observations.append({
                        "timestamp": item["period"],
                        "value": str(item["value"])
                    })
        
        # Sort chronologically (oldest first)
        observations.reverse()
        cache.set(series_id, observations)
        return observations
        
    except Exception as e:
        # No fallback - only real data allowed per architecture requirements
        raise RuntimeError(f"EIA API fetch failed: {e}")


# Alias for backward compatibility
fetch_series = fetch_eia_series