"""
Bureau of Economic Analysis (BEA) data fetcher.
"""
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import asyncio

import aiohttp
import pandas as pd

from src.core.config import settings
from src.cache.cache_manager import CacheManager
from src.data.sources.bea import BeaConnector

logger = logging.getLogger(__name__)


class BeaDataFetcher:
    """
    Fetches economic data from Bureau of Economic Analysis API.
    """
    
    def __init__(self):
        """Initialize BEA data fetcher."""
        self.cache_manager = CacheManager()
        self.bea_connector = BeaConnector(self.cache_manager)
        
        # Key BEA datasets for risk analysis
        self.key_datasets = {
            "gdp": {
                "table_name": "T20305",
                "line_codes": ["1", "2", "3"],  # GDP, Personal Consumption, Investment
                "frequency": "Q"  # Quarterly
            },
            "trade": {
                "table_name": "T40101", 
                "line_codes": ["1", "2", "3"],  # Total Trade, Exports, Imports
                "frequency": "M"  # Monthly
            },
            "personal_income": {
                "table_name": "T20600",
                "line_codes": ["1", "2"],  # Personal Income, Disposable Income
                "frequency": "M"
            }
        }
    
    async def fetch_latest_data(self) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch latest economic data from BEA.
        
        Returns:
            List of economic data records or None if failed
        """
        logger.info("Starting BEA data fetch")
        
        try:
            all_data = []
            
            for dataset_name, config in self.key_datasets.items():
                logger.info(f"Fetching BEA {dataset_name} data")
                
                # Fetch data for this dataset
                dataset_data = await self._fetch_dataset(dataset_name, config)
                if dataset_data:
                    all_data.extend(dataset_data)
                
                # Rate limiting - BEA allows 100 requests per minute
                await asyncio.sleep(0.6)  # Wait 600ms between requests
            
            if all_data:
                # Cache the aggregated data
                cache_key = f"bea:latest_fetch:{datetime.now().strftime('%Y%m%d')}"
                await self._cache_data(cache_key, all_data)
                
                logger.info(f"BEA data fetch completed: {len(all_data)} records")
                return all_data
            else:
                logger.warning("No BEA data retrieved")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching BEA data: {str(e)}")
            
            # Try to get cached data as fallback
            fallback_data = await self._get_fallback_data()
            if fallback_data:
                logger.info("Using fallback BEA data")
                return fallback_data
            
            return None
    
    async def _fetch_dataset(
        self, 
        dataset_name: str, 
        config: Dict[str, Any]
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch data for a specific BEA dataset.
        
        Args:
            dataset_name: Name of the dataset
            config: Dataset configuration
            
        Returns:
            List of data records or None if failed
        """
        try:
            # Use BEA connector to fetch data
            data = await self.bea_connector.get_nipa_data(
                table_name=config["table_name"],
                line_codes=config["line_codes"],
                frequency=config["frequency"],
                start_year=datetime.now().year - 2,  # Last 2 years
                end_year=datetime.now().year
            )
            
            if data:
                # Transform data for consistent format
                transformed_data = []
                for record in data:
                    transformed_record = {
                        "source": "bea",
                        "dataset": dataset_name,
                        "table_name": config["table_name"],
                        "line_code": record.get("LineCode"),
                        "line_description": record.get("LineDescription"),
                        "time_period": record.get("TimePeriod"),
                        "value": self._parse_numeric_value(record.get("DataValue")),
                        "units": record.get("UNIT_MULT", "Billions of dollars"),
                        "last_updated": datetime.now().isoformat(),
                        "frequency": config["frequency"]
                    }
                    transformed_data.append(transformed_record)
                
                return transformed_data
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching BEA dataset {dataset_name}: {str(e)}")
            return None
    
    def _parse_numeric_value(self, value: str) -> Optional[float]:
        """
        Parse BEA numeric value, handling special cases.
        
        Args:
            value: String value from BEA API
            
        Returns:
            Float value or None if unparseable
        """
        if not value or value in ["(NA)", "(D)", "(L)", "*"]:
            return None
        
        try:
            # Remove commas and convert to float
            clean_value = str(value).replace(",", "")
            return float(clean_value)
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
            # Cache for 12 hours (BEA data updates less frequently)
            return self.cache_manager.set(
                cache_key, 
                data, 
                ttl=12 * 3600,
                persist_to_postgres=True
            )
        except Exception as e:
            logger.error(f"Error caching BEA data: {str(e)}")
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
                cache_key = f"bea:latest_fetch:{cache_date}"
                
                cached_data = self.cache_manager.get(cache_key)
                if cached_data:
                    logger.info(f"Using BEA fallback data from {cache_date}")
                    return cached_data
            
            # Try fallback handler
            fallback_data = self.cache_manager.fallback_handler.get_fallback_data("bea")
            if fallback_data:
                return fallback_data.get("data")
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting BEA fallback data: {str(e)}")
            return None
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on BEA data source.
        
        Returns:
            Health status dictionary
        """
        try:
            # Check if we can reach BEA API
            health_result = await self.bea_connector.health_check()
            
            # Check cache availability
            cache_available = self.cache_manager.exists("bea:latest_fetch")
            
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
            logger.error(f"BEA health check failed: {str(e)}")
            return {
                "overall_healthy": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


async def main():
    """Test function for BEA data fetcher."""
    fetcher = BeaDataFetcher()
    
    print("Testing BEA data fetcher...")
    
    # Test health check
    health = await fetcher.health_check()
    print(f"Health check: {health}")
    
    # Test data fetch
    data = await fetcher.fetch_latest_data()
    if data:
        print(f"Fetched {len(data)} BEA records")
        print(f"Sample record: {data[0] if data else 'None'}")
    else:
        print("No data retrieved")


if __name__ == "__main__":
    asyncio.run(main())