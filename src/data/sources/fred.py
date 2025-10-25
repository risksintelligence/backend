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
        self.rate_limit_delay = 2.0  # 2 seconds between requests for production rate limiting
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
        
        # Default to recent data (last year) if no start date specified
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        
        params = {
            "series_id": series_id,
            "limit": limit,
            "sort_order": "desc",  # Most recent first
            "start_date": start_date
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
    
    async def get_series_historical(
        self, 
        series_id: str, 
        limit: int = 10,
        start_date: Optional[str] = None
    ) -> Optional[List[Dict]]:
        """Get multiple historical observations for a series."""
        
        # Default to recent data (last year) if no start date specified
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        
        params = {
            "series_id": series_id,
            "limit": limit,
            "sort_order": "desc",  # Most recent first
            "start_date": start_date
        }
        
        data = await self._make_request("series/observations", params)
        
        if data and "observations" in data:
            observations = data["observations"]
            
            # Get all non-null observations
            valid_obs = []
            for obs in reversed(observations):  # Most recent first
                if obs.get("value") and obs["value"] != ".":
                    try:
                        valid_obs.append({
                            "value": float(obs["value"]),
                            "date": obs["date"],
                            "series_id": series_id
                        })
                    except (ValueError, TypeError):
                        continue
            
            if valid_obs:
                # Get series metadata for first observation
                series_info = await self.get_series_info(series_id)
                if series_info:
                    title = series_info.get("title", series_id)
                    units = series_info.get("units", "")
                    frequency = series_info.get("frequency", "")
                    
                    # Add metadata to all observations
                    for obs in valid_obs:
                        obs.update({
                            "title": title,
                            "units": units,
                            "frequency": frequency,
                            "last_updated": datetime.utcnow().isoformat()
                        })
                
                return valid_obs[:limit]  # Return requested number of observations
        
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
    
    async def get_multiple_series(self, series_ids: List[str], limit: int = 100) -> Dict[str, Any]:
        """Get multiple FRED series data concurrently."""
        # Fetch multiple series concurrently
        results = await asyncio.gather(
            *[self.get_series(series_id, limit=limit) for series_id in series_ids],
            return_exceptions=True
        )
        
        series_data = {}
        for i, result in enumerate(results):
            if isinstance(result, dict) and result:
                series_data[series_ids[i]] = result
        
        return {
            "series_data": series_data,
            "count": len(series_data),
            "source": "fred_multiple_series",
            "last_updated": datetime.utcnow().isoformat()
        }


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
        # Fetch indicators sequentially to avoid rate limits
        series_ids = ["GDP", "UNRATE", "CPIAUCSL", "FEDFUNDS", "PAYEMS", "HOUST"]
        results = []
        
        for series_id in series_ids:
            try:
                result = await client.get_series(series_id)
                results.append(result)
            except Exception as e:
                results.append(e)
        
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
        # Fetch key market indicators sequentially to avoid rate limits
        results = []
        series_ids = ["WILL5000INDFC", "VIXCLS", "DGS10", "DGS2", "DEXUSEU"]
        
        for series_id in series_ids:
            try:
                result = await client.get_series(series_id)
                results.append(result)
            except Exception as e:
                results.append(e)
        
        indicators = {}
        series_names = ["stock_market", "vix", "treasury_10y", "treasury_2y", "usd_eur"]
        
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
        data = await client.get_series("WILL5000INDFC")
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


async def get_multiple_series(series_ids: List[str], limit: int = 100) -> Dict[str, Any]:
    """Get multiple FRED series data sequentially to avoid rate limits."""
    async with FREDClient() as client:
        # Fetch multiple series sequentially
        results = []
        for series_id in series_ids:
            try:
                result = await client.get_series(series_id, limit=limit)
                results.append(result)
            except Exception as e:
                results.append(e)
        
        series_data = {}
        for i, result in enumerate(results):
            if isinstance(result, dict) and result:
                series_data[series_ids[i]] = result
        
        return {
            "series_data": series_data,
            "count": len(series_data),
            "source": "fred_multiple_series",
            "last_updated": datetime.utcnow().isoformat()
        }


async def get_market_volatility_indicators() -> Optional[Dict]:
    """Get market volatility indicators from FRED."""
    async with FREDClient() as client:
        results = []
        series_ids = ["VIXCLS", "WILL5000PR", "NASDAQCOM"]
        
        for series_id in series_ids:
            try:
                result = await client.get_series(series_id)
                results.append(result)
            except Exception as e:
                results.append(e)
        
        indicators = {}
        series_names = ["vix", "wilshire_5000", "nasdaq"]
        
        for i, result in enumerate(results):
            if isinstance(result, dict) and result:
                indicators[series_names[i]] = result
        
        if indicators:
            return {
                "volatility_indicators": indicators,
                "source": "fred_volatility",
                "last_updated": datetime.utcnow().isoformat()
            }
        return None


async def get_economic_uncertainty_data() -> Optional[Dict]:
    """Get economic uncertainty indicators from FRED."""
    async with FREDClient() as client:
        results = []
        series_ids = ["USEPUINDXD", "UNRATE", "CPIAUCSL"]
        
        for series_id in series_ids:
            try:
                result = await client.get_series(series_id)
                results.append(result)
            except Exception as e:
                results.append(e)
        
        indicators = {}
        series_names = ["policy_uncertainty", "unemployment", "inflation"]
        
        for i, result in enumerate(results):
            if isinstance(result, dict) and result:
                indicators[series_names[i]] = result
        
        if indicators:
            return {
                "uncertainty_indicators": indicators,
                "source": "fred_uncertainty",
                "last_updated": datetime.utcnow().isoformat()
            }
        return None


async def get_treasury_yields() -> Optional[Dict]:
    """Get Treasury yield data from FRED."""
    async with FREDClient() as client:
        results = []
        series_ids = ["DGS3MO", "DGS2", "DGS10", "DGS30"]
        
        for series_id in series_ids:
            try:
                result = await client.get_series(series_id)
                results.append(result)
            except Exception as e:
                results.append(e)
        
        yields = {}
        series_names = ["3_month", "2_year", "10_year", "30_year"]
        
        for i, result in enumerate(results):
            if isinstance(result, dict) and result:
                yields[series_names[i]] = result
        
        if yields:
            return {
                "treasury_yields": yields,
                "source": "fred_yields",
                "last_updated": datetime.utcnow().isoformat()
            }
        return None


async def get_economic_stability_indicators() -> Optional[Dict]:
    """Get economic stability indicators from FRED."""
    async with FREDClient() as client:
        results = []
        series_ids = ["GDP", "PAYEMS", "INDPRO", "HOUST"]
        
        for series_id in series_ids:
            try:
                result = await client.get_series(series_id)
                results.append(result)
            except Exception as e:
                results.append(e)
        
        indicators = {}
        series_names = ["gdp", "employment", "production", "housing"]
        
        for i, result in enumerate(results):
            if isinstance(result, dict) and result:
                indicators[series_names[i]] = result
        
        if indicators:
            return {
                "stability_indicators": indicators,
                "source": "fred_stability",
                "last_updated": datetime.utcnow().isoformat()
            }
        return None


async def get_recent_indicators(limit: int = 100) -> Optional[Dict]:
    """Get recent economic indicators from FRED."""
    async with FREDClient() as client:
        # Get recent data from key economic series sequentially
        series_ids = ["GDP", "UNRATE", "CPIAUCSL", "FEDFUNDS", "PAYEMS", "INDPRO"]
        results = []
        
        for series_id in series_ids:
            try:
                result = await client.get_series(series_id, limit=min(limit//len(series_ids), 50))
                results.append(result)
            except Exception as e:
                results.append(e)
        
        indicators = {}
        for i, result in enumerate(results):
            if isinstance(result, dict) and result:
                indicators[series_ids[i]] = result
        
        if indicators:
            return {
                "recent_indicators": indicators,
                "series_count": len(indicators),
                "source": "fred_recent",
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