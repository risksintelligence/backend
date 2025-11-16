from typing import List, Dict
import httpx
from app.core.cache import FileCache
from app.core.config import get_settings

cache = FileCache("bea")


def fetch_bea_series(series_id: str, limit: int = 30) -> List[Dict[str, str]]:
    """Fetch data from Bureau of Economic Analysis (BEA) API."""
    cached = cache.get(series_id)
    if cached:
        return cached
    
    settings = get_settings()
    api_key = settings.bea_api_key
    if not api_key:
        raise RuntimeError("BEA API key missing; set RIS_BEA_API_KEY")
    
    # BEA API for GDP or other economic indicators
    # Using NIPA (National Income and Product Accounts) dataset
    url = "https://apps.bea.gov/api/data/"
    params = {
        "UserID": api_key,
        "method": "GetData",
        "datasetname": "NIPA",
        "TableName": "T20305",  # Real GDP by industry
        "Frequency": "Q",  # Quarterly
        "Year": "2023,2024",
        "ResultFormat": "json"
    }
    
    try:
        response = httpx.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        observations = []
        if "BEAAPI" in data and "Results" in data["BEAAPI"] and "Data" in data["BEAAPI"]["Results"]:
            for item in data["BEAAPI"]["Results"]["Data"]:
                if item.get("DataValue") not in [None, "", "..."]:
                    # Create date from TimePeriod (e.g., "2024Q3" -> "2024-07-01")
                    time_period = item.get("TimePeriod", "")
                    if "Q" in time_period:
                        year, quarter = time_period.split("Q")
                        quarter_months = {"1": "01", "2": "04", "3": "07", "4": "10"}
                        month = quarter_months.get(quarter, "01")
                        timestamp = f"{year}-{month}-01"
                    else:
                        timestamp = time_period
                    
                    observations.append({
                        "timestamp": timestamp,
                        "value": str(item["DataValue"])
                    })
        
        # Sort chronologically
        observations.sort(key=lambda x: x["timestamp"])
        # Limit results
        observations = observations[-limit:] if len(observations) > limit else observations
        
        cache.set(series_id, observations)
        return observations
        
    except Exception as e:
        # No fallback - only real data allowed per architecture requirements  
        raise RuntimeError(f"BEA API fetch failed: {e}")


# Alias for backward compatibility
fetch_series = fetch_bea_series