"""
U.S. Census Bureau API Integration
"""
import aiohttp
import asyncio
import os
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

CENSUS_API_KEY = os.getenv("CENSUS_API_KEY")
CENSUS_BASE_URL = "https://api.census.gov/data"

if not CENSUS_API_KEY:
    logger.warning("CENSUS_API_KEY not found in environment variables")


class CensusClient:
    """Async client for Census API with rate limiting and error handling."""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limit_delay = 1.0  # 1 second between requests
        self.last_request_time = 0
    
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
    
    async def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Optional[List]:
        """Make request to Census API with error handling."""
        if not self.session:
            raise RuntimeError("Client not initialized. Use 'async with' context.")
        
        if not CENSUS_API_KEY:
            logger.error("CENSUS_API_KEY not configured")
            return None
        
        await self._rate_limit()
        
        # Add API key
        params["key"] = CENSUS_API_KEY
        
        url = f"{CENSUS_BASE_URL}/{endpoint}"
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    logger.error(f"Census API error {response.status}: {await response.text()}")
                    return None
        
        except asyncio.TimeoutError:
            logger.error(f"Census API timeout for {endpoint}")
            return None
        except Exception as e:
            logger.error(f"Census API error for {endpoint}: {e}")
            return None
    
    async def get_population_estimates(self, year: str = "2021") -> Optional[Dict]:
        """Get population estimates."""
        
        params = {
            "get": "NAME,POP_2021",
            "for": "us:1"
        }
        
        data = await self._make_request(f"{year}/pep/population", params)
        
        if data and len(data) > 1:
            # First row is headers, second row is data
            headers = data[0]
            values = data[1]
            
            result = {}
            for i, header in enumerate(headers):
                if i < len(values):
                    result[header.lower()] = values[i]
            
            return {
                "population": int(result.get("pop_2021", 0)),
                "name": result.get("name", "United States"),
                "year": year,
                "source": "census",
                "last_updated": datetime.utcnow().isoformat()
            }
        
        return None
    
    async def get_economic_indicators(self) -> Optional[Dict]:
        """Get economic indicators from American Community Survey."""
        
        # Get median household income (most recent available)
        params = {
            "get": "NAME,B19013_001E",  # Median household income
            "for": "us:1"
        }
        
        data = await self._make_request("2019/acs/acs1", params)
        
        if data and len(data) > 1:
            income_value = data[1][1]
            
            return {
                "median_household_income": int(income_value) if income_value else None,
                "currency": "USD",
                "year": "2019",
                "source": "census_acs",
                "last_updated": datetime.utcnow().isoformat()
            }
        
        return None


async def get_population_data() -> Optional[Dict]:
    """Get latest population estimates."""
    async with CensusClient() as client:
        return await client.get_population_estimates()


async def get_household_income() -> Optional[Dict]:
    """Get median household income data."""
    async with CensusClient() as client:
        return await client.get_economic_indicators()


async def health_check(timeout: int = 5) -> bool:
    """Check if Census API is accessible."""
    try:
        async with CensusClient() as client:
            # Try to get population data
            result = await client.get_population_estimates()
            return result is not None
    except Exception:
        return False