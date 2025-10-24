"""
Bureau of Economic Analysis (BEA) API Integration
"""
import aiohttp
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging
from src.core.config import get_settings

logger = logging.getLogger(__name__)

BEA_BASE_URL = "https://apps.bea.gov/api/data"


class BEAClient:
    """Async client for BEA API with rate limiting and error handling."""
    
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
    
    async def _make_request(self, params: Dict[str, Any]) -> Optional[Dict]:
        """Make request to BEA API with error handling."""
        if not self.session:
            raise RuntimeError("Client not initialized. Use 'async with' context.")
        
        if not self.settings.bea_api_key:
            logger.error("BEA_API_KEY not configured")
            return None
        
        await self._rate_limit()
        
        # Add API key and format
        params.update({
            "UserID": self.settings.bea_api_key,
            "ResultFormat": "JSON"
        })
        
        try:
            async with self.session.get(BEA_BASE_URL, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    logger.error(f"BEA API error {response.status}: {await response.text()}")
                    return None
        
        except asyncio.TimeoutError:
            logger.error("BEA API timeout")
            return None
        except Exception as e:
            logger.error(f"BEA API error: {e}")
            return None
    
    async def get_gdp_data(self) -> Optional[Dict]:
        """Get latest GDP data from NIPA tables."""
        
        params = {
            "Method": "GetData",
            "DatasetName": "NIPA",
            "TableName": "T10101",  # Gross Domestic Product
            "Frequency": "Q",       # Quarterly
            "Year": "2023,2024"
        }
        
        data = await self._make_request(params)
        
        if data and "BEAAPI" in data and "Results" in data["BEAAPI"]:
            results = data["BEAAPI"]["Results"]
            if "Data" in results and results["Data"]:
                # Get the most recent data point
                latest_data = results["Data"][-1]
                
                return {
                    "value": float(latest_data.get("DataValue", 0).replace(",", "")),
                    "units": "billions_of_dollars",
                    "frequency": "quarterly",
                    "time_period": latest_data.get("TimePeriod"),
                    "line_description": latest_data.get("LineDescription"),
                    "source": "bea_nipa",
                    "last_updated": datetime.utcnow().isoformat()
                }
        
        return None
    
    async def get_personal_income(self) -> Optional[Dict]:
        """Get personal income data."""
        
        params = {
            "Method": "GetData",
            "DatasetName": "NIPA",
            "TableName": "T20100",  # Personal Income
            "Frequency": "M",       # Monthly
            "Year": "2024"
        }
        
        data = await self._make_request(params)
        
        if data and "BEAAPI" in data and "Results" in data["BEAAPI"]:
            results = data["BEAAPI"]["Results"]
            if "Data" in results and results["Data"]:
                # Get the most recent data point
                latest_data = results["Data"][-1]
                
                return {
                    "value": float(latest_data.get("DataValue", 0).replace(",", "")),
                    "units": "billions_of_dollars",
                    "frequency": "monthly",
                    "time_period": latest_data.get("TimePeriod"),
                    "line_description": latest_data.get("LineDescription"),
                    "source": "bea_nipa",
                    "last_updated": datetime.utcnow().isoformat()
                }
        
        return None
    
    async def get_trade_balance(self) -> Optional[Dict]:
        """Get international trade balance."""
        
        params = {
            "Method": "GetData",
            "DatasetName": "IntlServTrade",
            "TypeOfService": "AllServiceTypes",
            "TradeDirection": "Exports",
            "Affiliation": "AllAffiliations",
            "AreaOrCountry": "AllCountries",
            "Year": "2023,2024"
        }
        
        data = await self._make_request(params)
        
        if data and "BEAAPI" in data and "Results" in data["BEAAPI"]:
            results = data["BEAAPI"]["Results"]
            if "Data" in results and results["Data"]:
                # Sum recent data
                total_value = sum(
                    float(item.get("DataValue", 0).replace(",", ""))
                    for item in results["Data"][-12:]  # Last 12 data points
                )
                
                return {
                    "value": total_value,
                    "units": "millions_of_dollars",
                    "frequency": "monthly",
                    "description": "International Services Trade",
                    "source": "bea_trade",
                    "last_updated": datetime.utcnow().isoformat()
                }
        
        return None


async def get_economic_accounts() -> Dict[str, Any]:
    """Get key economic accounts data."""
    
    async with BEAClient() as client:
        # Fetch multiple indicators concurrently
        results = await asyncio.gather(
            client.get_gdp_data(),
            client.get_personal_income(),
            client.get_trade_balance(),
            return_exceptions=True
        )
        
        indicators = {}
        indicator_names = ["gdp", "personal_income", "trade_balance"]
        
        for i, result in enumerate(results):
            if isinstance(result, dict) and result:
                indicators[indicator_names[i]] = result
        
        return {
            "indicators": indicators,
            "count": len(indicators),
            "source": "bea",
            "last_updated": datetime.utcnow().isoformat()
        }


async def health_check(timeout: int = 5) -> bool:
    """Check if BEA API is accessible."""
    try:
        async with BEAClient() as client:
            # Try to get GDP data
            result = await client.get_gdp_data()
            return result is not None
    except Exception:
        return False