"""
Federal Reserve Economic Data (FRED) API Integration
"""
import aiohttp
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import logging
from src.core.config import get_settings

logger = logging.getLogger(__name__)

FRED_BASE_URL = "https://api.stlouisfed.org/fred"


class FREDClient:
    """Async client for FRED API with rate limiting and error handling."""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limit_delay = 1.0  # 1 second between requests
        self.last_request_time = 0
        self.settings = get_settings()
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def _rate_limit(self):
        """Enforce rate limiting between requests."""
        now = asyncio.get_event_loop().time()
        time_since_last = now - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)
        
        self.last_request_time = asyncio.get_event_loop().time()
    
    async def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Optional[Dict]:
        """Make request to FRED API with error handling."""
        if not self.session:
            raise RuntimeError("Client not initialized. Use 'async with' context.")
        
        if not self.settings.fred_api_key:
            logger.error("FRED_API_KEY not configured")
            return None
        
        await self._rate_limit()
        
        # Add API key and format
        params.update({
            "api_key": self.settings.fred_api_key,
            "file_type": "json"
        })
        
        url = f"{FRED_BASE_URL}/{endpoint}"
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    logger.error(f"FRED API error {response.status}: {await response.text()}")
                    return None
        
        except asyncio.TimeoutError:
            logger.error(f"FRED API timeout for {endpoint}")
            return None
        except Exception as e:
            logger.error(f"FRED API error for {endpoint}: {e}")
            return None
    
    async def get_series(
        self, 
        series_id: str, 
        limit: int = 100,
        start_date: Optional[str] = None
    ) -> Optional[Dict]:
        """Get time series data for a specific series."""
        
        params = {
            "series_id": series_id,
            "limit": limit
        }
        
        if start_date:
            params["start_date"] = start_date
        
        data = await self._make_request("series/observations", params)
        
        if data and "observations" in data:
            observations = data["observations"]
            
            # Get the most recent non-null observation
            latest_obs = None
            for obs in reversed(observations):
                if obs.get("value") and obs["value"] != ".":
                    try:
                        latest_obs = {
                            "value": float(obs["value"]),
                            "date": obs["date"],
                            "series_id": series_id
                        }
                        break
                    except (ValueError, TypeError):
                        continue
            
            if latest_obs:
                # Get series metadata
                series_info = await self.get_series_info(series_id)
                if series_info:
                    latest_obs.update({
                        "title": series_info.get("title", series_id),
                        "units": series_info.get("units", ""),
                        "frequency": series_info.get("frequency", ""),
                        "last_updated": datetime.utcnow().isoformat()
                    })
                
                return latest_obs
        
        return None
    
    async def get_series_info(self, series_id: str) -> Optional[Dict]:
        """Get metadata for a series."""
        
        params = {"series_id": series_id}
        data = await self._make_request("series", params)
        
        if data and "seriess" in data and data["seriess"]:
            series_info = data["seriess"][0]
            return {
                "id": series_info.get("id"),
                "title": series_info.get("title"),
                "units": series_info.get("units"),
                "frequency": series_info.get("frequency"),
                "seasonal_adjustment": series_info.get("seasonal_adjustment"),
                "last_updated": series_info.get("last_updated")
            }
        
        return None
    
    async def search_series(self, search_text: str, limit: int = 10) -> List[Dict]:
        """Search for series by text."""
        
        params = {
            "search_text": search_text,
            "limit": limit
        }
        
        data = await self._make_request("series/search", params)
        
        if data and "seriess" in data:
            return [
                {
                    "id": series.get("id"),
                    "title": series.get("title"),
                    "units": series.get("units"),
                    "frequency": series.get("frequency")
                }
                for series in data["seriess"]
            ]
        
        return []


# Convenience functions for common series
async def get_gdp() -> Optional[Dict]:
    """Get latest GDP data."""
    async with FREDClient() as client:
        return await client.get_series("GDP")


async def get_unemployment_rate() -> Optional[Dict]:
    """Get latest unemployment rate."""
    async with FREDClient() as client:
        return await client.get_series("UNRATE")


async def get_inflation_rate() -> Optional[Dict]:
    """Get latest CPI inflation data."""
    async with FREDClient() as client:
        return await client.get_series("CPIAUCSL")


async def get_fed_funds_rate() -> Optional[Dict]:
    """Get latest federal funds rate."""
    async with FREDClient() as client:
        return await client.get_series("FEDFUNDS")


async def get_key_indicators() -> Dict[str, Any]:
    """Get all key economic indicators."""
    
    async with FREDClient() as client:
        # Fetch multiple indicators concurrently
        results = await asyncio.gather(
            client.get_series("GDP"),
            client.get_series("UNRATE"),
            client.get_series("CPIAUCSL"),
            client.get_series("FEDFUNDS"),
            client.get_series("PAYEMS"),  # Non-farm payrolls
            client.get_series("HOUST"),   # Housing starts
            return_exceptions=True
        )
        
        indicators = {}
        series_names = ["gdp", "unemployment", "inflation", "fed_funds", "payrolls", "housing"]
        
        for i, result in enumerate(results):
            if isinstance(result, dict) and result:
                indicators[series_names[i]] = result
        
        return {
            "indicators": indicators,
            "count": len(indicators),
            "source": "fred",
            "last_updated": datetime.utcnow().isoformat()
        }


async def get_market_overview() -> Dict[str, Any]:
    """Get comprehensive market overview from FRED data."""
    async with FREDClient() as client:
        # Fetch key market indicators concurrently
        results = await asyncio.gather(
            client.get_series("SP500"),      # S&P 500
            client.get_series("VIXCLS"),     # VIX Volatility Index
            client.get_series("DGS10"),      # 10-Year Treasury Rate
            client.get_series("DGS2"),       # 2-Year Treasury Rate
            client.get_series("DEXUSEU"),    # USD/EUR Exchange Rate
            return_exceptions=True
        )
        
        indicators = {}
        series_names = ["sp500", "vix", "treasury_10y", "treasury_2y", "usd_eur"]
        
        for i, result in enumerate(results):
            if isinstance(result, dict) and result:
                indicators[series_names[i]] = result
        
        return {
            "market_indicators": indicators,
            "count": len(indicators),
            "source": "fred_market",
            "last_updated": datetime.utcnow().isoformat()
        }


async def get_sp500_data() -> Optional[Dict]:
    """Get S&P 500 index data."""
    async with FREDClient() as client:
        data = await client.get_series("SP500")
        if data:
            return {
                "symbol": "SP500",
                "data": data,
                "source": "fred",
                "last_updated": datetime.utcnow().isoformat()
            }
        return None


async def get_vix_data() -> Optional[Dict]:
    """Get VIX volatility index data."""
    async with FREDClient() as client:
        data = await client.get_series("VIXCLS")
        if data:
            return {
                "symbol": "VIX",
                "data": data,
                "source": "fred",
                "last_updated": datetime.utcnow().isoformat()
            }
        return None


async def health_check(timeout: int = 5) -> bool:
    """Check if FRED API is accessible."""
    try:
        async with FREDClient() as client:
            # Try to get a simple series info
            result = await client.get_series_info("GDP")
            return result is not None
    except Exception:
        return False