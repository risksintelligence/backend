from typing import List, Dict
import httpx
import json
from app.core.cache import FileCache
from app.core.config import get_settings

cache = FileCache("bls")


def fetch_bls_series(series_id: str, limit: int = 30) -> List[Dict[str, str]]:
    """Fetch data from Bureau of Labor Statistics (BLS) API."""
    cached = cache.get(series_id)
    if cached:
        return cached
    
    settings = get_settings()
    api_key = settings.bls_api_key
    if not api_key:
        raise RuntimeError("BLS API key missing; set RIS_BLS_API_KEY")
    
    # BLS API v2 format
    url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
    
    # Map our internal series_id to BLS series IDs
    bls_series_mapping = {
        "UNEMPLOYMENT": "LNS14000000",  # Unemployment rate
        "CPI": "CUUR0000SA0",  # Consumer Price Index - All Urban Consumers
        "PPI": "WPSFD49207",  # Producer Price Index - Final Demand
    }
    
    bls_series_id = bls_series_mapping.get(series_id, series_id)
    
    payload = {
        "seriesid": [bls_series_id],
        "startyear": "2023",
        "endyear": "2024",
        "registrationkey": api_key
    }
    
    try:
        response = httpx.post(
            url, 
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        observations = []
        if "Results" in data and "series" in data["Results"] and data["Results"]["series"]:
            series_data = data["Results"]["series"][0]
            if "data" in series_data:
                for item in series_data["data"]:
                    if item.get("value") not in [None, "", "."]:
                        # Create timestamp from year and period
                        year = item.get("year", "")
                        period = item.get("period", "")
                        
                        # Handle monthly data (M01, M02, etc.) and quarterly (Q01, etc.)
                        if period.startswith("M"):
                            month = period[1:]
                            timestamp = f"{year}-{month.zfill(2)}-01"
                        elif period.startswith("Q"):
                            quarter = period[1:]
                            quarter_months = {"01": "01", "02": "04", "03": "07", "04": "10"}
                            month = quarter_months.get(quarter, "01")
                            timestamp = f"{year}-{month}-01"
                        else:
                            timestamp = f"{year}-01-01"
                        
                        observations.append({
                            "timestamp": timestamp,
                            "value": str(item["value"])
                        })
        
        # Sort chronologically
        observations.sort(key=lambda x: x["timestamp"])
        # Limit results
        observations = observations[-limit:] if len(observations) > limit else observations
        
        cache.set(series_id, observations)
        return observations
        
    except Exception as e:
        # No fallback - only real data allowed per architecture requirements
        raise RuntimeError(f"BLS API fetch failed: {e}")


# Alias for backward compatibility
fetch_series = fetch_bls_series