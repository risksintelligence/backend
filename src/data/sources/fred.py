"""
FRED (Federal Reserve Economic Data) API connector with caching.
"""
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import hashlib

import pandas as pd
import requests
from fredapi import Fred

from src.core.config import settings
from src.cache.cache_manager import CacheManager
from src.monitoring.metrics_collector import metrics_collector

logger = logging.getLogger(__name__)


class FREDConnector:
    """
    FRED API connector with comprehensive caching and fallback handling.
    """
    
    def __init__(self, cache_manager: Optional[CacheManager] = None):
        """
        Initialize FRED connector.
        
        Args:
            cache_manager: Cache manager instance for data caching
        """
        self.cache_manager = cache_manager or CacheManager()
        self._fred_client = None
        self._initialize_client()
        
        # Key economic indicators for risk assessment
        self.key_indicators = {
            "interest_rates": {
                "FEDFUNDS": "Federal Funds Rate",
                "DGS10": "10-Year Treasury Rate",
                "DGS2": "2-Year Treasury Rate",
                "TB3MS": "3-Month Treasury Rate",
            },
            "inflation": {
                "CPIAUCSL": "Consumer Price Index",
                "CPILFESL": "Core CPI",
                "PCEPI": "PCE Price Index",
                "DFEDTARL": "Fed Target Rate Lower",
            },
            "employment": {
                "UNRATE": "Unemployment Rate",
                "PAYEMS": "Non-Farm Payrolls",
                "CIVPART": "Labor Force Participation Rate",
                "EMRATIO": "Employment-Population Ratio",
            },
            "gdp": {
                "GDP": "Gross Domestic Product",
                "GDPC1": "Real GDP",
                "GDPPOT": "Real Potential GDP",
                "NYGDPMKTPCDWLD": "World GDP Per Capita",
            },
            "financial_stress": {
                "NFCI": "National Financial Conditions Index",
                "ANFCI": "Adjusted NFCI",
                "STLFSI4": "Financial Stress Index",
                "BAMLH0A0HYM2": "High Yield Bond Spread",
            },
            "credit": {
                "TOTALSL": "Total Consumer Loans",
                "DRBLACBS": "Business Loans",
                "DRCCLACBS": "Credit Card Loans",
                "MORTGAGE30US": "30-Year Mortgage Rate",
            }
        }
    
    def _initialize_client(self) -> None:
        """Initialize FRED API client."""
        try:
            if not settings.fred_api_key:
                logger.warning("FRED API key not provided, using demo mode")
                self._fred_client = None
                self._setup_fallback_data()
                return
            
            self._fred_client = Fred(api_key=settings.fred_api_key)
            
            # Test connection
            test_data = self._fred_client.get_series("GDP", limit=1)
            if test_data is not None and not test_data.empty:
                logger.info("FRED API client initialized successfully")
            else:
                raise Exception("FRED API test query failed")
                
        except Exception as e:
            logger.error(f"Failed to initialize FRED client: {e}")
            self._fred_client = None
            self._setup_fallback_data()
            self.cache_manager.register_data_source_failure("fred", str(e))
    
    def _setup_fallback_data(self) -> None:
        """Setup fallback data for when FRED API is unavailable."""
        fallback_data = {
            "FEDFUNDS": {"value": 5.25, "date": "2024-01-01", "description": "Federal Funds Rate"},
            "DGS10": {"value": 4.45, "date": "2024-01-01", "description": "10-Year Treasury Rate"},
            "UNRATE": {"value": 3.8, "date": "2024-01-01", "description": "Unemployment Rate"},
            "CPIAUCSL": {"value": 310.5, "date": "2024-01-01", "description": "Consumer Price Index"},
            "GDP": {"value": 25000, "date": "2023-Q4", "description": "Gross Domestic Product"},
        }
        
        for series_id, data in fallback_data.items():
            self.cache_manager.register_fallback_data(
                f"fred:{series_id}", 
                data, 
                datetime(2024, 1, 1)
            )
    
    def _generate_cache_key(
        self, 
        series_id: str, 
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate cache key for FRED data request.
        
        Args:
            series_id: FRED series identifier
            start_date: Start date for data
            end_date: End date for data
            **kwargs: Additional parameters
            
        Returns:
            Cache key string
        """
        # Create a hash of the parameters for consistent caching
        params = {
            "series_id": series_id,
            "start_date": start_date,
            "end_date": end_date,
            **kwargs
        }
        
        # Sort parameters for consistent hashing
        param_str = "&".join(f"{k}={v}" for k, v in sorted(params.items()) if v is not None)
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
        
        return f"fred:{series_id}:{param_hash}"
    
    def get_series(
        self, 
        series_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_cache: bool = True,
        **kwargs
    ) -> Optional[pd.Series]:
        """
        Get FRED time series data with caching.
        
        Args:
            series_id: FRED series identifier
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            use_cache: Whether to use cached data
            **kwargs: Additional FRED API parameters
            
        Returns:
            Pandas Series with time series data or None if unavailable
        """
        cache_key = self._generate_cache_key(series_id, start_date, end_date, **kwargs)
        
        # Try cache first
        if use_cache:
            cached_data = self.cache_manager.get(cache_key)
            if cached_data is not None:
                try:
                    # Convert back to pandas Series
                    data_dict = cached_data["data"]
                    if isinstance(data_dict, dict):
                        # Convert date strings back to datetime index
                        dates = [pd.to_datetime(date_str) for date_str in data_dict.keys()]
                        values = list(data_dict.values())
                        return pd.Series(values, index=dates, name=series_id)
                    else:
                        return pd.Series(data_dict, name=series_id)
                except Exception as e:
                    logger.warning(f"Failed to deserialize cached data for {series_id}: {e}")
        
        # Try to fetch from FRED API
        if self._fred_client is not None:
            try:
                data = self._fred_client.get_series(
                    series_id, 
                    start=start_date, 
                    end=end_date,
                    **kwargs
                )
                
                if data is not None and not data.empty:
                    # Cache the successful result
                    # Convert index to string for JSON serialization
                    data_dict = {}
                    for idx, val in data.items():
                        date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, 'strftime') else str(idx)
                        data_dict[date_str] = float(val) if pd.notna(val) else None
                    
                    cache_data = {
                        "data": data_dict,
                        "series_id": series_id,
                        "fetched_at": datetime.utcnow().isoformat(),
                        "source": "fred_api"
                    }
                    
                    self.cache_manager.set(cache_key, cache_data, ttl=3600)  # 1 hour TTL
                    
                    # Update fallback handler
                    self.cache_manager.fallback_handler.set_last_known_good(
                        f"fred:{series_id}", 
                        cache_data
                    )
                    
                    logger.info(f"Successfully fetched FRED series {series_id}")
                    return data
                    
            except Exception as e:
                logger.error(f"Error fetching FRED series {series_id}: {e}")
                self.cache_manager.register_data_source_failure("fred", str(e))
        
        # Use fallback data
        fallback_data = self.cache_manager.get(cache_key, use_fallback=True)
        if fallback_data:
            logger.warning(f"Using fallback data for FRED series {series_id}")
            if isinstance(fallback_data, dict) and "data" in fallback_data:
                return pd.Series(fallback_data["data"], name=series_id)
        
        logger.error(f"No data available for FRED series {series_id}")
        return None
    
    def get_multiple_series(
        self, 
        series_ids: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_cache: bool = True
    ) -> Dict[str, Optional[pd.Series]]:
        """
        Get multiple FRED series with caching.
        
        Args:
            series_ids: List of FRED series identifiers
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            use_cache: Whether to use cached data
            
        Returns:
            Dictionary mapping series IDs to pandas Series
        """
        start_time = datetime.now()
        results = {}
        error_count = 0
        
        for series_id in series_ids:
            try:
                results[series_id] = self.get_series(
                    series_id, 
                    start_date=start_date, 
                    end_date=end_date,
                    use_cache=use_cache
                )
            except Exception as e:
                logger.error(f"Error fetching FRED series {series_id}: {e}")
                error_count += 1
                results[series_id] = None
        
        # Record data quality metrics
        try:
            total_data_points = sum([
                len(series) if series is not None else 0 
                for series in results.values()
            ])
            
            successful_series = len([s for s in results.values() if s is not None])
            quality_score = successful_series / len(series_ids) if series_ids else 0
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            import asyncio
            asyncio.create_task(metrics_collector.record_data_quality_metrics(
                source_name="fred",
                data_points=total_data_points,
                quality_score=quality_score,
                error_count=error_count,
                latency_seconds=processing_time
            ))
        except Exception as e:
            logger.error(f"Error recording FRED metrics: {e}")
        
        return results
    
    def get_latest_values(
        self, 
        series_ids: List[str],
        use_cache: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get latest values for multiple series.
        
        Args:
            series_ids: List of FRED series identifiers
            use_cache: Whether to use cached data
            
        Returns:
            Dictionary with latest values and metadata
        """
        results = {}
        
        for series_id in series_ids:
            try:
                # Get last 30 days of data to ensure we have recent values
                end_date = datetime.now().strftime("%Y-%m-%d")
                start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
                
                series_data = self.get_series(
                    series_id, 
                    start_date=start_date,
                    end_date=end_date,
                    use_cache=use_cache
                )
                
                if series_data is not None and not series_data.empty:
                    latest_value = series_data.iloc[-1]
                    latest_date = series_data.index[-1]
                    
                    results[series_id] = {
                        "value": float(latest_value) if pd.notna(latest_value) else None,
                        "date": latest_date.strftime("%Y-%m-%d") if hasattr(latest_date, 'strftime') else str(latest_date),
                        "series_id": series_id,
                        "source": "fred",
                        "description": self._get_series_description(series_id)
                    }
                else:
                    results[series_id] = {
                        "value": None,
                        "date": None,
                        "series_id": series_id,
                        "source": "fred",
                        "error": "No data available",
                        "description": self._get_series_description(series_id)
                    }
                    
            except Exception as e:
                logger.error(f"Error getting latest value for {series_id}: {e}")
                results[series_id] = {
                    "value": None,
                    "date": None,
                    "series_id": series_id,
                    "source": "fred",
                    "error": str(e),
                    "description": self._get_series_description(series_id)
                }
        
        return results
    
    def get_key_indicators(self, use_cache: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Get latest values for all key economic indicators.
        
        Args:
            use_cache: Whether to use cached data
            
        Returns:
            Dictionary organized by indicator category
        """
        results = {}
        
        for category, indicators in self.key_indicators.items():
            series_ids = list(indicators.keys())
            latest_values = self.get_latest_values(series_ids, use_cache=use_cache)
            
            results[category] = {}
            for series_id, description in indicators.items():
                value_data = latest_values.get(series_id, {})
                value_data["description"] = description
                results[category][series_id] = value_data
        
        return results
    
    def _get_series_description(self, series_id: str) -> str:
        """
        Get description for a FRED series.
        
        Args:
            series_id: FRED series identifier
            
        Returns:
            Series description
        """
        # Search through key indicators
        for category, indicators in self.key_indicators.items():
            if series_id in indicators:
                return indicators[series_id]
        
        # Try to get from FRED API
        if self._fred_client is not None:
            try:
                info = self._fred_client.get_series_info(series_id)
                return info.get("title", "No description available")
            except Exception:
                pass
        
        return "No description available"
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on FRED connector.
        
        Returns:
            Health status information
        """
        status = {
            "service": "fred",
            "timestamp": datetime.utcnow().isoformat(),
            "api_available": False,
            "cache_available": False,
            "fallback_available": False
        }
        
        # Check API availability
        if self._fred_client is not None:
            try:
                test_data = self._fred_client.get_series("GDP", limit=1)
                status["api_available"] = test_data is not None and not test_data.empty
            except Exception as e:
                status["api_error"] = str(e)
        
        # Check cache availability
        cache_health = self.cache_manager.health_check()
        status["cache_available"] = cache_health["overall_healthy"]
        status["cache_details"] = cache_health
        
        # Check fallback availability
        fallback_data = self.cache_manager.fallback_handler.get_fallback_data("fred:GDP")
        status["fallback_available"] = fallback_data is not None
        
        # Overall health
        status["overall_healthy"] = (
            status["api_available"] or 
            status["cache_available"] or 
            status["fallback_available"]
        )
        
        return status