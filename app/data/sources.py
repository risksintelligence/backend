"""Data source mapping and provider functions."""

from typing import Callable, List, Dict
from app.data.fetchers.fred import fetch_fred_series
from app.data.fetchers.local import fetch_local_series
from app.data.fetchers.eia import fetch_eia_series
from app.data.fetchers.census import fetch_census_series
from app.data.fetchers.bea import fetch_bea_series
from app.data.fetchers.bls import fetch_bls_series
from app.data.fetchers.alpha_vantage import fetch_alpha_vantage_series


def get_source(provider: str) -> Callable[[str], List[Dict[str, str]]]:
    """Get the appropriate data fetcher for a provider."""
    sources = {
        "fred": fetch_fred_series,
        "local": fetch_local_series,
        "eia": fetch_eia_series,
        "census": fetch_census_series,
        "bea": fetch_bea_series,
        "bls": fetch_bls_series,
        "alpha_vantage": fetch_alpha_vantage_series,
    }
    
    if provider not in sources:
        raise ValueError(f"Unknown data provider: {provider}")
    
    return sources[provider]