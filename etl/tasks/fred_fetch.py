"""
FRED data extraction task for ETL pipeline.
"""
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import pandas as pd

from src.data.sources.fred import FREDConnector
from src.cache.cache_manager import CacheManager
from src.core.database import get_db
from src.core.config import settings

logger = logging.getLogger(__name__)


class FREDDataFetcher:
    """
    Fetches economic data from FRED API for ETL pipeline.
    """
    
    def __init__(self):
        self.fred_connector = FREDConnector()
        self.cache_manager = CacheManager()
        
    def fetch_all_indicators(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Fetch all key economic indicators from FRED.
        
        Args:
            force_refresh: Force refresh of all data ignoring cache
            
        Returns:
            Dictionary containing all indicator data
        """
        logger.info("Starting FRED data fetch process")
        
        results = {
            "timestamp": datetime.utcnow(),
            "status": "success",
            "indicators": {},
            "errors": []
        }
        
        try:
            logger.info("Fetching all key economic indicators")
            all_indicators = self.fred_connector.get_key_indicators(
                use_cache=not force_refresh
            )
            
            if all_indicators:
                results["indicators"] = all_indicators
                total_indicators = sum(len(category_data) for category_data in all_indicators.values())
                logger.info(f"Successfully fetched {total_indicators} indicators across {len(all_indicators)} categories")
            else:
                logger.warning("No data returned from FRED connector")
                
        except Exception as e:
            error_msg = f"Error fetching FRED indicators: {str(e)}"
            logger.error(error_msg)
            results["errors"].append(error_msg)
            results["status"] = "failed"
        
        # Store results in cache for ETL monitoring
        cache_key = f"etl:fred_fetch:{datetime.utcnow().strftime('%Y%m%d_%H')}"
        self.cache_manager.set(cache_key, results, ttl=3600)
        
        logger.info(f"FRED fetch completed with status: {results['status']}")
        return results
    
    def fetch_specific_indicators(self, indicators: list, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Fetch specific FRED indicators.
        
        Args:
            indicators: List of FRED series IDs to fetch
            force_refresh: Force refresh ignoring cache
            
        Returns:
            Dictionary containing indicator data
        """
        logger.info(f"Fetching specific indicators: {indicators}")
        
        results = {
            "timestamp": datetime.utcnow(),
            "status": "success",
            "data": {},
            "errors": []
        }
        
        for indicator in indicators:
            try:
                data = self.fred_connector.get_series(
                    indicator,
                    use_cache=not force_refresh
                )
                
                if data is not None:
                    results["data"][indicator] = data
                    logger.info(f"Successfully fetched {indicator}")
                else:
                    logger.warning(f"No data returned for indicator: {indicator}")
                    
            except Exception as e:
                error_msg = f"Error fetching {indicator}: {str(e)}"
                logger.error(error_msg)
                results["errors"].append(error_msg)
                results["status"] = "partial_success" if results["data"] else "failed"
        
        return results
    
    def validate_data_freshness(self) -> Dict[str, Any]:
        """
        Validate that cached data is fresh and complete.
        
        Returns:
            Validation report
        """
        logger.info("Validating FRED data freshness")
        
        validation_report = {
            "timestamp": datetime.utcnow(),
            "overall_status": "healthy",
            "indicators": {},
            "issues": []
        }
        
        # Check key indicators for freshness
        key_indicators = [
            "FEDFUNDS", "UNRATE", "CPIAUCSL", "GDP", "GS10", "GS2"
        ]
        
        for indicator in key_indicators:
            try:
                cached_data = self.cache_manager.get(f"fred:{indicator}")
                
                if cached_data is None:
                    validation_report["indicators"][indicator] = "missing"
                    validation_report["issues"].append(f"No cached data for {indicator}")
                    validation_report["overall_status"] = "degraded"
                else:
                    # Check data age
                    if isinstance(cached_data, dict) and "timestamp" in cached_data:
                        data_age = datetime.utcnow() - datetime.fromisoformat(cached_data["timestamp"])
                        if data_age > timedelta(hours=24):
                            validation_report["indicators"][indicator] = "stale"
                            validation_report["issues"].append(f"Stale data for {indicator}: {data_age}")
                            validation_report["overall_status"] = "degraded"
                        else:
                            validation_report["indicators"][indicator] = "fresh"
                    else:
                        validation_report["indicators"][indicator] = "format_issue"
                        validation_report["issues"].append(f"Data format issue for {indicator}")
                        
            except Exception as e:
                validation_report["indicators"][indicator] = "error"
                validation_report["issues"].append(f"Error checking {indicator}: {str(e)}")
                validation_report["overall_status"] = "degraded"
        
        if len(validation_report["issues"]) > len(key_indicators) // 2:
            validation_report["overall_status"] = "unhealthy"
        
        logger.info(f"Data validation completed with status: {validation_report['overall_status']}")
        return validation_report
    
    def get_etl_metrics(self) -> Dict[str, Any]:
        """
        Get metrics about ETL performance and data quality.
        
        Returns:
            ETL metrics report
        """
        metrics = {
            "timestamp": datetime.utcnow(),
            "data_sources": {
                "fred": {
                    "status": "operational",
                    "last_update": None,
                    "total_indicators": 0,
                    "cached_indicators": 0,
                    "cache_hit_rate": 0.0
                }
            },
            "pipeline_health": "healthy"
        }
        
        try:
            # Check FRED connector status
            test_data = self.fred_connector.get_series("FEDFUNDS", use_cache=True)
            if test_data is not None:
                metrics["data_sources"]["fred"]["status"] = "operational"
            else:
                metrics["data_sources"]["fred"]["status"] = "degraded"
                metrics["pipeline_health"] = "degraded"
                
            # Get cache statistics (simplified)
            cache_stats = self._get_cache_statistics()
            metrics["data_sources"]["fred"].update(cache_stats)
            
        except Exception as e:
            logger.error(f"Error getting ETL metrics: {str(e)}")
            metrics["data_sources"]["fred"]["status"] = "error"
            metrics["pipeline_health"] = "unhealthy"
        
        return metrics
    
    def _get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache statistics for FRED data."""
        stats = {
            "total_indicators": 0,
            "cached_indicators": 0,
            "cache_hit_rate": 0.0,
            "last_update": None
        }
        
        try:
            # This is a simplified implementation
            # In production, we'd track these metrics in Redis or database
            key_indicators = ["FEDFUNDS", "UNRATE", "CPIAUCSL", "GDP"]
            cached_count = 0
            
            for indicator in key_indicators:
                if self.cache_manager.get(f"fred:{indicator}") is not None:
                    cached_count += 1
            
            stats["total_indicators"] = len(key_indicators)
            stats["cached_indicators"] = cached_count
            stats["cache_hit_rate"] = cached_count / len(key_indicators) if key_indicators else 0.0
            
        except Exception as e:
            logger.error(f"Error getting cache statistics: {str(e)}")
        
        return stats


def run_fred_etl_task(force_refresh: bool = False) -> Dict[str, Any]:
    """
    Main ETL task function for FRED data.
    
    Args:
        force_refresh: Whether to force refresh all data
        
    Returns:
        Task execution results
    """
    logger.info("Starting FRED ETL task")
    
    fetcher = FREDDataFetcher()
    
    try:
        # Fetch all economic indicators
        results = fetcher.fetch_all_indicators(force_refresh=force_refresh)
        
        # Validate data quality
        validation = fetcher.validate_data_freshness()
        results["validation"] = validation
        
        # Get metrics
        metrics = fetcher.get_etl_metrics()
        results["metrics"] = metrics
        
        logger.info(f"FRED ETL task completed successfully: {results['status']}")
        return results
        
    except Exception as e:
        error_msg = f"FRED ETL task failed: {str(e)}"
        logger.error(error_msg)
        return {
            "timestamp": datetime.utcnow(),
            "status": "failed",
            "error": error_msg
        }


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the ETL task
    result = run_fred_etl_task(force_refresh=True)
    print(f"ETL Task Result: {result}")