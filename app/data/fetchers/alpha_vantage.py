from typing import List, Dict
import httpx
from app.core.cache import FileCache
from app.core.config import get_settings

cache = FileCache("alpha_vantage")


def fetch_alpha_vantage_series(series_id: str, limit: int = 30) -> List[Dict[str, str]]:
    """Fetch data from Alpha Vantage API."""
    cached = cache.get(series_id)
    if cached:
        return cached
    
    settings = get_settings()
    api_key = settings.alpha_vantage_api_key
    if not api_key:
        raise RuntimeError("Alpha Vantage API key missing; set RIS_ALPHA_VANTAGE_API_KEY")
    
    # Alpha Vantage API format
    url = "https://www.alphavantage.co/query"
    
    # Map our internal series_id to Alpha Vantage functions and symbols
    av_mapping = {
        "VIX": {"function": "TIME_SERIES_DAILY", "symbol": "VIX"},
        "SPY": {"function": "TIME_SERIES_DAILY", "symbol": "SPY"},
        "TREASURY_10Y": {"function": "TIME_SERIES_DAILY", "symbol": "TNX"},
        "DXY": {"function": "TIME_SERIES_DAILY", "symbol": "DXY"},
    }
    
    if series_id not in av_mapping:
        raise RuntimeError(f"Alpha Vantage series {series_id} not configured")
    
    mapping = av_mapping[series_id]
    params = {
        "function": mapping["function"],
        "symbol": mapping["symbol"],
        "apikey": api_key,
        "outputsize": "compact"  # Last 100 data points
    }
    
    try:
        response = httpx.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        observations = []
        
        # Handle daily time series data
        if "Time Series (Daily)" in data:
            time_series = data["Time Series (Daily)"]
            for date, values in time_series.items():
                # Use close price
                close_price = values.get("4. close", "")
                if close_price:
                    observations.append({
                        "timestamp": date,
                        "value": str(close_price)
                    })
        
        # Sort chronologically (oldest first)
        observations.sort(key=lambda x: x["timestamp"])
        # Limit results
        observations = observations[-limit:] if len(observations) > limit else observations
        
        cache.set(series_id, observations)
        return observations
        
    except Exception as e:
        # No fallback - only real data allowed per architecture requirements
        raise RuntimeError(f"Alpha Vantage API fetch failed: {e}")


# Alias for backward compatibility
fetch_series = fetch_alpha_vantage_series