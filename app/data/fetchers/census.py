from typing import List, Dict
import httpx
from app.core.cache import FileCache
from app.core.config import get_settings

cache = FileCache("census")


def fetch_census_series(series_id: str, limit: int = 30) -> List[Dict[str, str]]:
    """Fetch data from US Census Bureau API."""
    cached = cache.get(series_id)
    if cached:
        return cached
    
    settings = get_settings()
    api_key = settings.census_api_key
    if not api_key:
        raise RuntimeError("Census API key missing; set RIS_CENSUS_API_KEY")
    
    # Census API typically focuses on trade data (WTO Statistics alternative)
    # For demonstration, we'll fetch international trade data
    year = "2024"
    url = f"https://api.census.gov/data/{year}/timeseries/intltrade/exports/hs"
    params = {
        "key": api_key,
        "get": "E_COMMODITY,time_slot",
        "COMMODITY": "TOTAL",  # Total exports
        "time_slot": "2024-01,2024-02,2024-03,2024-04,2024-05,2024-06,2024-07,2024-08,2024-09,2024-10"
    }
    
    try:
        response = httpx.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        observations = []
        if data and len(data) > 1:  # Skip header row
            for row in data[1:]:
                if len(row) >= 2 and row[0] and row[1]:
                    # Convert export value to string
                    observations.append({
                        "timestamp": row[1],  # time_slot 
                        "value": str(row[0])  # E_COMMODITY
                    })
        
        # Sort chronologically
        observations.sort(key=lambda x: x["timestamp"])
        cache.set(series_id, observations)
        return observations
        
    except Exception as e:
        # No fallback - only real data allowed per architecture requirements
        raise RuntimeError(f"Census API fetch failed: {e}")


# Alias for backward compatibility
fetch_series = fetch_census_series