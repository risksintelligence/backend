"""
Bureau of Labor Statistics (BLS) data fetcher.
"""
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import asyncio

import aiohttp
import pandas as pd

from src.core.config import settings
from src.cache.cache_manager import CacheManager
from src.data.sources.bls import BlsConnector

logger = logging.getLogger(__name__)


class BlsDataFetcher:
    """
    Fetches labor and employment data from Bureau of Labor Statistics API.
    """
    
    def __init__(self):
        """Initialize BLS data fetcher."""
        self.cache_manager = CacheManager()
        self.bls_connector = BlsConnector(self.cache_manager)
        
        # Key BLS series for risk analysis
        self.key_series = {
            "unemployment": {
                "LNS14000000": "Unemployment Rate",
                "LNS11300000": "Labor Force Participation Rate",
                "LNS12300000": "Employment-Population Ratio"
            },
            "employment": {
                "CES0000000001": "Total Nonfarm Employment",
                "CES0500000001": "Total Private Employment", 
                "CES9000000001": "Government Employment"
            },
            "inflation": {
                "CUUR0000SA0": "Consumer Price Index - All Urban Consumers",
                "CUUR0000SA0L1E": "CPI - All items less food and energy",
                "CUUR0000SAF": "CPI - Food",
                "CUUR0000SAE": "CPI - Energy"
            },
            "productivity": {
                "PRS85006092": "Business Sector: Labor Productivity",
                "PRS85006112": "Business Sector: Unit Labor Costs",
                "PRS85006062": "Business Sector: Output"
            },
            "wages": {
                "CES0500000003": "Average Hourly Earnings - Private",
                "CES0500000011": "Average Weekly Hours - Private",
                "LES1252881600": "Employment Cost Index"
            }
        }
    
    async def fetch_latest_data(self) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch latest labor and employment data from BLS.
        
        Returns:
            List of employment data records or None if failed
        """
        logger.info("Starting BLS data fetch")
        
        try:
            all_data = []
            
            for category, series_dict in self.key_series.items():
                logger.info(f"Fetching BLS {category} data")
                
                # Fetch data for this category
                category_data = await self._fetch_category_data(category, series_dict)
                if category_data:
                    all_data.extend(category_data)
                
                # Rate limiting - BLS allows 25 queries per 10 seconds for unregistered users
                await asyncio.sleep(0.5)  # Wait 500ms between requests
            
            if all_data:
                # Cache the aggregated data
                cache_key = f"bls:latest_fetch:{datetime.now().strftime('%Y%m%d')}"
                await self._cache_data(cache_key, all_data)
                
                logger.info(f"BLS data fetch completed: {len(all_data)} records")
                return all_data
            else:
                logger.warning("No BLS data retrieved")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching BLS data: {str(e)}")
            
            # Try to get cached data as fallback
            fallback_data = await self._get_fallback_data()
            if fallback_data:
                logger.info("Using fallback BLS data")
                return fallback_data
            
            return None
    
    async def _fetch_category_data(
        self, 
        category: str, 
        series_dict: Dict[str, str]
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch data for a specific BLS data category.
        
        Args:
            category: Category name (unemployment, employment, etc.)
            series_dict: Dictionary of series IDs and descriptions
            
        Returns:
            List of data records or None if failed
        """
        try:
            all_series_data = []
            
            # Fetch data for each series in this category
            for series_id, description in series_dict.items():
                try:
                    # Get latest 24 months of data
                    series_data = await self.bls_connector.get_series_data(
                        series_id=series_id,
                        start_year=datetime.now().year - 1,
                        end_year=datetime.now().year
                    )
                    
                    if series_data:
                        # Transform data for consistent format
                        for record in series_data:
                            transformed_record = {
                                "source": "bls",
                                "category": category,
                                "series_id": series_id,
                                "series_description": description,
                                "year": record.get("year"),
                                "period": record.get("period"),
                                "period_name": record.get("periodName"),
                                "value": self._parse_numeric_value(record.get("value")),
                                "footnotes": record.get("footnotes", []),
                                "last_updated": datetime.now().isoformat(),
                                "date": self._create_date_string(record.get("year"), record.get("period"))
                            }
                            all_series_data.append(transformed_record)
                    
                    # Rate limiting between series
                    await asyncio.sleep(0.2)
                    
                except Exception as e:
                    logger.warning(f"Error fetching BLS series {series_id}: {str(e)}")
                    continue
            
            return all_series_data if all_series_data else None
            
        except Exception as e:
            logger.error(f"Error fetching BLS category {category}: {str(e)}")
            return None
    
    def _parse_numeric_value(self, value: Any) -> Optional[float]:
        """
        Parse BLS numeric value, handling special cases.
        
        Args:
            value: Value from BLS API
            
        Returns:
            Float value or None if unparseable
        """
        if value is None or value == "":
            return None
        
        try:
            # Remove any non-numeric characters except decimal point and negative sign
            clean_value = str(value).replace(",", "").strip()
            if clean_value in ["-", "(D)", "N/A", "NA"]:
                return None
            return float(clean_value)
        except (ValueError, TypeError):
            return None
    
    def _create_date_string(self, year: str, period: str) -> Optional[str]:
        """
        Create ISO date string from BLS year and period.
        
        Args:
            year: Year string
            period: Period string (M01-M12, Q01-Q04, etc.)
            
        Returns:
            ISO date string or None
        """
        try:
            if not year or not period:
                return None
            
            year_int = int(year)
            
            # Handle monthly data (M01-M12)
            if period.startswith("M"):
                month = int(period[1:])
                if 1 <= month <= 12:
                    return f"{year_int}-{month:02d}-01"
            
            # Handle quarterly data (Q01-Q04)
            elif period.startswith("Q"):
                quarter = int(period[1:])
                if 1 <= quarter <= 4:
                    month = (quarter - 1) * 3 + 1
                    return f"{year_int}-{month:02d}-01"
            
            # Handle annual data (A01)
            elif period == "A01":
                return f"{year_int}-01-01"
            
            return None
            
        except (ValueError, TypeError):
            return None
    
    async def _cache_data(self, cache_key: str, data: List[Dict[str, Any]]) -> bool:
        """
        Cache fetched data.
        
        Args:
            cache_key: Cache key
            data: Data to cache
            
        Returns:
            True if cached successfully
        """
        try:
            # Cache for 8 hours (BLS data updates daily)
            return self.cache_manager.set(
                cache_key, 
                data, 
                ttl=8 * 3600,
                persist_to_postgres=True
            )
        except Exception as e:
            logger.error(f"Error caching BLS data: {str(e)}")
            return False
    
    async def _get_fallback_data(self) -> Optional[List[Dict[str, Any]]]:
        """
        Get fallback data from cache or fallback handler.
        
        Returns:
            Fallback data or None
        """
        try:
            # Try to get recent cached data
            today = datetime.now()
            for days_ago in range(1, 8):  # Try last 7 days
                cache_date = (today - timedelta(days=days_ago)).strftime('%Y%m%d')
                cache_key = f"bls:latest_fetch:{cache_date}"
                
                cached_data = self.cache_manager.get(cache_key)
                if cached_data:
                    logger.info(f"Using BLS fallback data from {cache_date}")
                    return cached_data
            
            # Try fallback handler
            fallback_data = self.cache_manager.fallback_handler.get_fallback_data("bls")
            if fallback_data:
                return fallback_data.get("data")
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting BLS fallback data: {str(e)}")
            return None
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on BLS data source.
        
        Returns:
            Health status dictionary
        """
        try:
            # Check if we can reach BLS API
            health_result = await self.bls_connector.health_check()
            
            # Check cache availability
            cache_available = self.cache_manager.exists("bls:latest_fetch")
            
            # Check fallback data
            fallback_available = bool(await self._get_fallback_data())
            
            overall_healthy = (
                health_result.get("api_available", False) or 
                cache_available or 
                fallback_available
            )
            
            return {
                "overall_healthy": overall_healthy,
                "api_available": health_result.get("api_available", False),
                "cache_available": cache_available,
                "fallback_available": fallback_available,
                "last_successful_fetch": health_result.get("last_successful_fetch"),
                "error_count": health_result.get("error_count", 0),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"BLS health check failed: {str(e)}")
            return {
                "overall_healthy": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


async def main():
    """Test function for BLS data fetcher."""
    fetcher = BlsDataFetcher()
    
    print("Testing BLS data fetcher...")
    
    # Test health check
    health = await fetcher.health_check()
    print(f"Health check: {health}")
    
    # Test data fetch
    data = await fetcher.fetch_latest_data()
    if data:
        print(f"Fetched {len(data)} BLS records")
        print(f"Sample record: {data[0] if data else 'None'}")
    else:
        print("No data retrieved")


if __name__ == "__main__":
    asyncio.run(main())