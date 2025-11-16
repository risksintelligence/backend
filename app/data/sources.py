"""Data source mapping and provider functions."""

from typing import Callable, List, Dict
from app.data.fetchers.fred import fetch_fred_series
from app.data.fetchers.local import fetch_local_series


def get_source(provider: str) -> Callable[[str], List[Dict[str, str]]]:
    """Get the appropriate data fetcher for a provider."""
    sources = {
        "fred": fetch_fred_series,
        "local": fetch_local_series,
    }
    
    if provider not in sources:
        raise ValueError(f"Unknown data provider: {provider}")
    
    return sources[provider]