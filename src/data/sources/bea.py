"""
BEA (Bureau of Economic Analysis) API connector with caching.
"""
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import hashlib
import json

import pandas as pd
import requests

from src.core.config import settings
from src.cache.cache_manager import CacheManager

logger = logging.getLogger(__name__)


class BEAConnector:
    """
    BEA API connector with comprehensive caching and fallback handling.
    
    The Bureau of Economic Analysis provides key economic data including:
    - GDP and components
    - Personal income and spending
    - International trade data
    - Regional economic accounts
    """
    
    def __init__(self, cache_manager: Optional[CacheManager] = None):
        """
        Initialize BEA connector.
        
        Args:
            cache_manager: Cache manager instance for data caching
        """
        self.cache_manager = cache_manager or CacheManager()
        self.base_url = "https://apps.bea.gov/api/data/"
        self.api_key = settings.bea_api_key
        
        # Key BEA datasets and tables for risk assessment
        self.key_datasets = {
            "gdp": {
                "dataset_name": "NIPA",
                "table_name": "T10101",
                "description": "Gross Domestic Product",
                "frequency": "Q",  # Quarterly
                "key_lines": ["1", "2", "3", "4"]  # GDP, consumption, investment, etc.
            },
            "personal_income": {
                "dataset_name": "NIPA", 
                "table_name": "T20100",
                "description": "Personal Income and Outlays",
                "frequency": "M",  # Monthly
                "key_lines": ["1", "2", "3"]  # Personal income, disposable income, spending
            },
            "international_trade": {
                "dataset_name": "ITA",
                "table_name": "U70205S",
                "description": "International Trade in Goods and Services",
                "frequency": "M",
                "key_lines": ["1", "2", "3"]  # Exports, imports, balance
            },
            "regional_gdp": {
                "dataset_name": "Regional",
                "table_name": "CAGDP1",
                "description": "County GDP",
                "frequency": "A",  # Annual
                "key_lines": ["1"]  # Total GDP
            }
        }
        
        self._setup_fallback_data()
    
    def _setup_fallback_data(self) -> None:
        """Setup fallback data for when BEA API is unavailable."""
        fallback_data = {
            "gdp_current": {
                "value": 25000.0,
                "date": "2024-Q1",
                "description": "Gross Domestic Product (Billions)",
                "change_percent": 2.1
            },
            "personal_income": {
                "value": 22500.0,
                "date": "2024-01",
                "description": "Personal Income (Billions)",
                "change_percent": 3.2
            },
            "exports": {
                "value": 265.5,
                "date": "2024-01", 
                "description": "Exports of Goods and Services (Billions)",
                "change_percent": -1.5
            },
            "imports": {
                "value": 338.2,
                "date": "2024-01",
                "description": "Imports of Goods and Services (Billions)", 
                "change_percent": 2.8
            },
            "trade_balance": {
                "value": -72.7,
                "date": "2024-01",
                "description": "Trade Balance (Billions)",
                "change_percent": -15.2
            }
        }
        
        for indicator, data in fallback_data.items():
            self.cache_manager.register_fallback_data(
                f"bea:{indicator}",
                data,
                datetime(2024, 1, 1)
            )
    
    def _generate_cache_key(
        self,
        dataset: str,
        table: str,
        **kwargs
    ) -> str:
        """
        Generate cache key for BEA data request.
        
        Args:
            dataset: BEA dataset name
            table: BEA table name
            **kwargs: Additional parameters
            
        Returns:
            Cache key string
        """
        params = {
            "dataset": dataset,
            "table": table,
            **kwargs
        }
        
        param_str = "&".join(f"{k}={v}" for k, v in sorted(params.items()) if v is not None)
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
        
        return f"bea:{dataset}_{table}:{param_hash}"
    
    def _make_api_request(
        self,
        dataset: str,
        table: str,
        frequency: str = "Q",
        year: Optional[str] = None,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Make API request to BEA.
        
        Args:
            dataset: BEA dataset name
            table: BEA table name  
            frequency: Data frequency (A, Q, M)
            year: Year or year range (e.g., "2020,2021,2022")
            **kwargs: Additional API parameters
            
        Returns:
            API response data or None if failed
        """
        if not self.api_key:
            logger.warning("BEA API key not provided")
            return None
        
        # Default to last 3 years if no year specified
        if not year:
            current_year = datetime.now().year
            year = f"{current_year-2},{current_year-1},{current_year}"
        
        params = {
            "UserID": self.api_key,
            "method": "GetData",
            "datasetname": dataset,
            "TableName": table,
            "Frequency": frequency,
            "Year": year,
            "ResultFormat": "JSON",
            **kwargs
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for API errors
            if "BEAAPI" not in data:
                logger.error(f"Invalid BEA API response: {data}")
                return None
            
            if "Error" in data.get("BEAAPI", {}):
                error_info = data["BEAAPI"]["Error"]
                logger.error(f"BEA API error: {error_info}")
                return None
            
            return data
            
        except requests.RequestException as e:
            logger.error(f"BEA API request failed: {e}")
            return None
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"BEA API response parsing failed: {e}")
            return None
    
    def get_gdp_data(
        self,
        use_cache: bool = True,
        year: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get GDP data from BEA.
        
        Args:
            use_cache: Whether to use cached data
            year: Year or year range to fetch
            
        Returns:
            GDP data with latest values and trends
        """
        cache_key = self._generate_cache_key("NIPA", "T10101", year=year)
        
        # Try cache first
        if use_cache:
            cached_data = self.cache_manager.get(cache_key)
            if cached_data is not None:
                return cached_data
        
        # Try BEA API
        api_data = self._make_api_request("NIPA", "T10101", "Q", year)
        
        if api_data:
            try:
                # Process GDP data
                results = self._process_gdp_response(api_data)
                
                # Cache successful result
                self.cache_manager.set(cache_key, results, ttl=3600)
                self.cache_manager.fallback_handler.set_last_known_good(
                    "bea:gdp", results
                )
                
                return results
                
            except Exception as e:
                logger.error(f"Error processing BEA GDP data: {e}")
        
        # Use fallback data
        fallback_data = self.cache_manager.get(cache_key, use_fallback=True)
        if fallback_data:
            logger.warning("Using fallback data for BEA GDP")
            return fallback_data
        
        logger.error("No GDP data available from BEA")
        return None
    
    def get_personal_income_data(
        self,
        use_cache: bool = True,
        year: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get personal income data from BEA.
        
        Args:
            use_cache: Whether to use cached data
            year: Year or year range to fetch
            
        Returns:
            Personal income data with latest values and trends
        """
        cache_key = self._generate_cache_key("NIPA", "T20100", year=year)
        
        if use_cache:
            cached_data = self.cache_manager.get(cache_key)
            if cached_data is not None:
                return cached_data
        
        api_data = self._make_api_request("NIPA", "T20100", "M", year)
        
        if api_data:
            try:
                results = self._process_personal_income_response(api_data)
                self.cache_manager.set(cache_key, results, ttl=3600)
                return results
            except Exception as e:
                logger.error(f"Error processing BEA personal income data: {e}")
        
        fallback_data = self.cache_manager.get(cache_key, use_fallback=True)
        if fallback_data:
            logger.warning("Using fallback data for BEA personal income")
            return fallback_data
        
        return None
    
    def get_trade_data(
        self,
        use_cache: bool = True,
        year: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get international trade data from BEA.
        
        Args:
            use_cache: Whether to use cached data
            year: Year or year range to fetch
            
        Returns:
            Trade data with exports, imports, and balance
        """
        cache_key = self._generate_cache_key("ITA", "U70205S", year=year)
        
        if use_cache:
            cached_data = self.cache_manager.get(cache_key)
            if cached_data is not None:
                return cached_data
        
        api_data = self._make_api_request("ITA", "U70205S", "M", year)
        
        if api_data:
            try:
                results = self._process_trade_response(api_data)
                self.cache_manager.set(cache_key, results, ttl=3600)
                return results
            except Exception as e:
                logger.error(f"Error processing BEA trade data: {e}")
        
        fallback_data = self.cache_manager.get(cache_key, use_fallback=True)
        if fallback_data:
            logger.warning("Using fallback data for BEA trade")
            return fallback_data
        
        return None
    
    def get_key_indicators(
        self,
        use_cache: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get all key BEA economic indicators.
        
        Args:
            use_cache: Whether to use cached data
            
        Returns:
            Dictionary with all key indicators
        """
        indicators = {}
        
        # Get GDP data
        gdp_data = self.get_gdp_data(use_cache=use_cache)
        if gdp_data:
            indicators["gdp"] = gdp_data
        
        # Get personal income data
        income_data = self.get_personal_income_data(use_cache=use_cache)
        if income_data:
            indicators["personal_income"] = income_data
        
        # Get trade data
        trade_data = self.get_trade_data(use_cache=use_cache)
        if trade_data:
            indicators["trade"] = trade_data
        
        return indicators
    
    def _process_gdp_response(self, api_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process BEA GDP API response."""
        data = api_data["BEAAPI"]["Results"]["Data"]
        
        # Extract latest GDP values
        latest_data = {}
        for item in data:
            if item.get("LineDescription") == "Gross domestic product":
                latest_data["current_value"] = float(item.get("DataValue", 0))
                latest_data["date"] = item.get("TimePeriod")
                latest_data["description"] = item.get("LineDescription")
                break
        
        return {
            "source": "bea",
            "dataset": "gdp",
            "timestamp": datetime.utcnow().isoformat(),
            "data": latest_data,
            "raw_response_count": len(data)
        }
    
    def _process_personal_income_response(self, api_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process BEA personal income API response."""
        data = api_data["BEAAPI"]["Results"]["Data"]
        
        latest_data = {}
        for item in data:
            if "Personal income" in item.get("LineDescription", ""):
                latest_data["current_value"] = float(item.get("DataValue", 0))
                latest_data["date"] = item.get("TimePeriod")
                latest_data["description"] = item.get("LineDescription")
                break
        
        return {
            "source": "bea",
            "dataset": "personal_income", 
            "timestamp": datetime.utcnow().isoformat(),
            "data": latest_data,
            "raw_response_count": len(data)
        }
    
    def _process_trade_response(self, api_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process BEA trade API response."""
        data = api_data["BEAAPI"]["Results"]["Data"]
        
        trade_data = {}
        for item in data:
            line_desc = item.get("LineDescription", "").lower()
            if "exports" in line_desc:
                trade_data["exports"] = {
                    "value": float(item.get("DataValue", 0)),
                    "date": item.get("TimePeriod"),
                    "description": item.get("LineDescription")
                }
            elif "imports" in line_desc:
                trade_data["imports"] = {
                    "value": float(item.get("DataValue", 0)),
                    "date": item.get("TimePeriod"),
                    "description": item.get("LineDescription")
                }
        
        # Calculate trade balance
        if "exports" in trade_data and "imports" in trade_data:
            trade_data["balance"] = {
                "value": trade_data["exports"]["value"] - trade_data["imports"]["value"],
                "date": trade_data["exports"]["date"],
                "description": "Trade Balance"
            }
        
        return {
            "source": "bea",
            "dataset": "trade",
            "timestamp": datetime.utcnow().isoformat(),
            "data": trade_data,
            "raw_response_count": len(data)
        }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on BEA connector.
        
        Returns:
            Health status information
        """
        status = {
            "service": "bea",
            "timestamp": datetime.utcnow().isoformat(),
            "api_available": False,
            "cache_available": False,
            "fallback_available": False
        }
        
        # Check API availability
        if self.api_key:
            try:
                # Test with a simple GDP request
                test_response = self._make_api_request("NIPA", "T10101", "Q", "2023")
                status["api_available"] = test_response is not None
            except Exception as e:
                status["api_error"] = str(e)
        else:
            status["api_error"] = "No API key provided"
        
        # Check cache availability
        cache_health = self.cache_manager.health_check()
        status["cache_available"] = cache_health["overall_healthy"]
        status["cache_details"] = cache_health
        
        # Check fallback availability
        fallback_data = self.cache_manager.fallback_handler.get_fallback_data("bea:gdp")
        status["fallback_available"] = fallback_data is not None
        
        # Overall health
        status["overall_healthy"] = (
            status["api_available"] or 
            status["cache_available"] or 
            status["fallback_available"]
        )
        
        return status