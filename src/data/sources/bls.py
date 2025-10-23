"""
Bureau of Labor Statistics (BLS) API Integration
"""
import aiohttp
import asyncio
import os
import json
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

BLS_API_KEY = os.getenv("BLS_API_KEY")
BLS_BASE_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data"

if not BLS_API_KEY:
    logger.warning("BLS_API_KEY not found in environment variables")


class BLSClient:
    """Async client for BLS API with rate limiting and error handling."""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limit_delay = 2.0  # 2 seconds between requests (BLS has stricter limits)
        self.last_request_time = 0
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=15),
            headers={"Content-Type": "application/json"}
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
    
    async def _make_request(self, series_ids: List[str], start_year: str, end_year: str) -> Optional[Dict]:
        """Make request to BLS API with error handling."""
        if not self.session:
            raise RuntimeError("Client not initialized. Use 'async with' context.")
        
        if not BLS_API_KEY:
            logger.error("BLS_API_KEY not configured")
            return None
        
        await self._rate_limit()
        
        # Prepare request payload
        payload = {
            "seriesid": series_ids,
            "startyear": start_year,
            "endyear": end_year,
            "registrationkey": BLS_API_KEY
        }
        
        try:
            async with self.session.post(BLS_BASE_URL, data=json.dumps(payload)) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    logger.error(f"BLS API error {response.status}: {await response.text()}")
                    return None
        
        except asyncio.TimeoutError:
            logger.error("BLS API timeout")
            return None
        except Exception as e:
            logger.error(f"BLS API error: {e}")
            return None
    
    async def get_employment_data(self) -> Optional[Dict]:
        """Get employment statistics."""
        
        current_year = datetime.now().year
        
        # Key employment series
        series_ids = [
            "LNS14000000",  # Unemployment rate
            "CES0000000001",  # Total nonfarm payrolls
            "LNS11300000"   # Labor force participation rate
        ]
        
        data = await self._make_request(
            series_ids, 
            str(current_year - 1), 
            str(current_year)
        )
        
        if data and "Results" in data and "status" in data and data["status"] == "REQUEST_SUCCEEDED":
            series_data = data["Results"]["series"]
            
            result = {}
            
            for series in series_data:
                series_id = series["seriesID"]
                if series["data"]:
                    latest = series["data"][0]  # Most recent data point
                    
                    if series_id == "LNS14000000":
                        result["unemployment_rate"] = {
                            "value": float(latest["value"]),
                            "units": "percent",
                            "period": f"{latest['year']}-{latest['periodName']}",
                            "series_id": series_id
                        }
                    elif series_id == "CES0000000001":
                        result["nonfarm_payrolls"] = {
                            "value": float(latest["value"]) * 1000,  # Convert to actual number
                            "units": "thousands_of_jobs",
                            "period": f"{latest['year']}-{latest['periodName']}",
                            "series_id": series_id
                        }
                    elif series_id == "LNS11300000":
                        result["labor_participation"] = {
                            "value": float(latest["value"]),
                            "units": "percent",
                            "period": f"{latest['year']}-{latest['periodName']}",
                            "series_id": series_id
                        }
            
            if result:
                result.update({
                    "source": "bls",
                    "last_updated": datetime.utcnow().isoformat()
                })
                
                return result
        
        return None
    
    async def get_inflation_data(self) -> Optional[Dict]:
        """Get Consumer Price Index data."""
        
        current_year = datetime.now().year
        
        # CPI series
        series_ids = [
            "CUUR0000SA0",  # CPI-U All items
            "CUUR0000SAF1",  # CPI-U Food
            "CUUR0000SAH1"   # CPI-U Housing
        ]
        
        data = await self._make_request(
            series_ids,
            str(current_year - 1),
            str(current_year)
        )
        
        if data and "Results" in data and "status" in data and data["status"] == "REQUEST_SUCCEEDED":
            series_data = data["Results"]["series"]
            
            result = {}
            
            for series in series_data:
                series_id = series["seriesID"]
                if len(series["data"]) >= 2:
                    current = series["data"][0]
                    previous = series["data"][1]
                    
                    # Calculate year-over-year change
                    current_val = float(current["value"])
                    previous_val = float(previous["value"])
                    yoy_change = ((current_val - previous_val) / previous_val) * 100
                    
                    if series_id == "CUUR0000SA0":
                        result["cpi_all_items"] = {
                            "index_value": current_val,
                            "yoy_change_percent": round(yoy_change, 2),
                            "period": f"{current['year']}-{current['periodName']}",
                            "series_id": series_id
                        }
                    elif series_id == "CUUR0000SAF1":
                        result["cpi_food"] = {
                            "index_value": current_val,
                            "yoy_change_percent": round(yoy_change, 2),
                            "period": f"{current['year']}-{current['periodName']}",
                            "series_id": series_id
                        }
                    elif series_id == "CUUR0000SAH1":
                        result["cpi_housing"] = {
                            "index_value": current_val,
                            "yoy_change_percent": round(yoy_change, 2),
                            "period": f"{current['year']}-{current['periodName']}",
                            "series_id": series_id
                        }
            
            if result:
                result.update({
                    "source": "bls_cpi",
                    "last_updated": datetime.utcnow().isoformat()
                })
                
                return result
        
        return None


async def get_labor_statistics() -> Dict[str, Any]:
    """Get comprehensive labor statistics."""
    
    async with BLSClient() as client:
        # Fetch employment and inflation data
        results = await asyncio.gather(
            client.get_employment_data(),
            client.get_inflation_data(),
            return_exceptions=True
        )
        
        indicators = {}
        
        for result in results:
            if isinstance(result, dict) and result:
                indicators.update(result)
        
        return {
            "indicators": indicators,
            "count": len(indicators),
            "source": "bls",
            "last_updated": datetime.utcnow().isoformat()
        }


async def health_check(timeout: int = 5) -> bool:
    """Check if BLS API is accessible."""
    try:
        async with BLSClient() as client:
            # Try to get employment data
            result = await client.get_employment_data()
            return result is not None
    except Exception:
        return False