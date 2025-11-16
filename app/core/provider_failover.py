"""
Real Provider Failover Strategy

Implements failover to alternative REAL data providers per architecture requirements:
- Alpha Vantage when Yahoo Finance throttles
- Cached real data when all providers fail  
- No fake/mock data fallbacks

Provider hierarchy:
- Primary: Preferred provider per SERIES_REGISTRY
- Secondary: Alternative real provider for same data
- Cache: Last known good real data with staleness warnings
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

from app.data.registry import SERIES_REGISTRY, SeriesMetadata
from app.core.unified_cache import UnifiedCache
from app.core.config import get_settings

logger = logging.getLogger(__name__)

@dataclass
class ProviderConfig:
    name: str
    series_mapping: Dict[str, str]  # internal_series_id -> provider_series_id
    rate_limit_per_minute: int
    reliability_score: float  # 0.0 - 1.0
    last_failure: Optional[datetime] = None
    failure_count: int = 0

class ProviderFailoverManager:
    """Manages failover between real data providers with no fake fallbacks."""
    
    def __init__(self):
        self.cache = UnifiedCache("provider_failover")
        self.settings = get_settings()
        
        # Define provider configurations with failover hierarchy
        self.providers = {
            "fred": ProviderConfig(
                name="fred",
                series_mapping={
                    "VIX": "VIXCLS",
                    "YIELD_CURVE": "T10Y2Y",
                    "CREDIT_SPREAD": "BAA10YM", 
                    "PMI": "NAPM",
                    "WTI_OIL": "DCOILWTICO",
                    "UNEMPLOYMENT": "UNRATE"
                },
                rate_limit_per_minute=120,
                reliability_score=0.95
            ),
            "alpha_vantage": ProviderConfig(
                name="alpha_vantage", 
                series_mapping={
                    "YIELD_CURVE": "TNX",  # 10Y Treasury as proxy
                    "WTI_OIL": "WTI"
                },
                rate_limit_per_minute=5,  # Free tier limit
                reliability_score=0.85
            ),
            "eia": ProviderConfig(
                name="eia",
                series_mapping={
                    "FREIGHT_DIESEL": "EPD2DXL0",
                    "WTI_OIL": "RWTC"  # Alternative oil series
                },
                rate_limit_per_minute=30,
                reliability_score=0.90
            ),
            "bls": ProviderConfig(
                name="bls",
                series_mapping={
                    "UNEMPLOYMENT": "LNS14000000",
                    "CPI": "CUUR0000SA0"
                },
                rate_limit_per_minute=25,
                reliability_score=0.88
            )
        }
        
        # Define failover chains for each series
        self.failover_chains = {
            "VIX": ["fred"],  # FRED has VIXCLS data
            "YIELD_CURVE": ["fred", "alpha_vantage"],
            "CREDIT_SPREAD": ["fred"],  # FRED is best source
            "WTI_OIL": ["fred", "eia", "alpha_vantage"],
            "UNEMPLOYMENT": ["bls", "fred"],
            "PMI": ["fred"],  # Specialized data
            "FREIGHT_DIESEL": ["eia"],
            "BALTIC_DRY": ["local"]  # Only available in local cache
        }
    
    def fetch_with_failover(self, series_id: str, limit: int = 30) -> List[Dict[str, str]]:
        """
        Fetch data with intelligent failover through real providers only.
        Returns cached real data if all providers fail.
        """
        if series_id not in SERIES_REGISTRY:
            raise RuntimeError(f"Series {series_id} not found in registry")
        
        original_metadata = SERIES_REGISTRY[series_id]
        failover_providers = self.failover_chains.get(series_id, [original_metadata.provider])
        
        # Try each provider in the failover chain
        for provider_name in failover_providers:
            try:
                # Skip if provider recently failed multiple times
                if self._should_skip_provider(provider_name):
                    logger.warning(f"Skipping {provider_name} due to recent failures")
                    continue
                
                # Attempt to fetch from this provider
                data = self._fetch_from_provider(series_id, provider_name, limit)
                
                if data:
                    # Reset failure count on success
                    self.providers[provider_name].failure_count = 0
                    self.providers[provider_name].last_failure = None
                    
                    logger.info(f"✅ Successfully fetched {series_id} from {provider_name}")
                    return data
                
            except Exception as e:
                # Record the failure
                self._record_provider_failure(provider_name, e)
                logger.warning(f"❌ Provider {provider_name} failed for {series_id}: {e}")
                continue
        
        # All providers failed - try cached real data
        return self._get_cached_real_data(series_id)
    
    def _fetch_from_provider(self, series_id: str, provider_name: str, limit: int) -> List[Dict[str, str]]:
        """Fetch data from a specific provider."""
        if provider_name not in self.providers:
            raise RuntimeError(f"Unknown provider: {provider_name}")
        
        provider_config = self.providers[provider_name]
        
        # Check if series is supported by this provider
        if series_id not in provider_config.series_mapping:
            raise RuntimeError(f"Series {series_id} not supported by {provider_name}")
        
        provider_series_id = provider_config.series_mapping[series_id]
        
        # Import and use the appropriate fetcher
        if provider_name == "fred":
            from app.data.fetchers.fred import fetch_fred_series
            return fetch_fred_series(provider_series_id, limit)
        elif provider_name == "alpha_vantage":
            from app.data.fetchers.alpha_vantage import fetch_alpha_vantage_series
            return fetch_alpha_vantage_series(provider_series_id, limit)
        elif provider_name == "eia":
            from app.data.fetchers.eia import fetch_eia_series
            return fetch_eia_series(provider_series_id, limit)
        elif provider_name == "bls":
            from app.data.fetchers.bls import fetch_bls_series
            return fetch_bls_series(provider_series_id, limit)
        elif provider_name == "local":
            from app.data.fetchers.local import fetch_local_series
            return fetch_local_series(series_id, limit)
        else:
            raise RuntimeError(f"Fetcher not implemented for provider: {provider_name}")
    
    def _should_skip_provider(self, provider_name: str) -> bool:
        """Check if provider should be skipped due to recent failures."""
        if provider_name not in self.providers:
            return True
        
        provider = self.providers[provider_name]
        
        # Skip if too many recent failures
        if provider.failure_count >= 3:
            # Allow retry after exponential backoff
            if provider.last_failure:
                backoff_minutes = min(60, 5 * (2 ** (provider.failure_count - 3)))
                retry_time = provider.last_failure + timedelta(minutes=backoff_minutes)
                return datetime.utcnow() < retry_time
        
        return False
    
    def _record_provider_failure(self, provider_name: str, error: Exception) -> None:
        """Record a provider failure for circuit breaker logic."""
        if provider_name in self.providers:
            provider = self.providers[provider_name]
            provider.failure_count += 1
            provider.last_failure = datetime.utcnow()
            provider.reliability_score = max(0.1, provider.reliability_score * 0.9)
            
            logger.error(f"Recorded failure for {provider_name}: {error} (count: {provider.failure_count})")
    
    def _get_cached_real_data(self, series_id: str) -> List[Dict[str, str]]:
        """
        Get cached real data as last resort failover.
        Returns last known good data with staleness warnings.
        """
        try:
            data, metadata = self.cache.get(series_id)
            
            if data and metadata:
                # Add staleness warning metadata
                age_hours = metadata.age_seconds / 3600
                
                logger.warning(f"⚠️  Using cached data for {series_id} (age: {age_hours:.1f} hours)")
                
                # Return in expected format, but with staleness indicators
                if isinstance(data, dict) and 'timestamp' in data and 'value' in data:
                    # Single observation format
                    return [{
                        "timestamp": data["timestamp"],
                        "value": data["value"], 
                        "_cache_metadata": {
                            "stale": True,
                            "age_hours": age_hours,
                            "source": metadata.source,
                            "cache_status": metadata.cache_status
                        }
                    }]
                elif isinstance(data, list):
                    # Multiple observations format
                    for obs in data:
                        if isinstance(obs, dict):
                            obs["_cache_metadata"] = {
                                "stale": True,
                                "age_hours": age_hours,
                                "source": metadata.source,
                                "cache_status": metadata.cache_status
                            }
                    return data
            
            # No cached data available - hard failure
            logger.error(f"❌ No cached data available for {series_id}")
            raise RuntimeError(f"All providers failed for {series_id} and no cached data available")
            
        except Exception as e:
            logger.error(f"Failed to retrieve cached data for {series_id}: {e}")
            raise RuntimeError(f"Complete data fetch failure for {series_id}")
    
    def get_provider_health(self) -> Dict[str, any]:
        """Get health status of all providers for monitoring."""
        health_report = {}
        
        for name, provider in self.providers.items():
            health_report[name] = {
                "reliability_score": provider.reliability_score,
                "failure_count": provider.failure_count,
                "last_failure": provider.last_failure.isoformat() if provider.last_failure else None,
                "rate_limit_per_minute": provider.rate_limit_per_minute,
                "should_skip": self._should_skip_provider(name),
                "supported_series": list(provider.series_mapping.keys())
            }
        
        return health_report
    
    def reset_provider_failures(self, provider_name: str) -> bool:
        """Reset failure count for a provider (manual recovery)."""
        if provider_name in self.providers:
            provider = self.providers[provider_name]
            provider.failure_count = 0
            provider.last_failure = None
            provider.reliability_score = min(1.0, provider.reliability_score + 0.1)
            
            logger.info(f"🔄 Reset failure count for provider {provider_name}")
            return True
        
        return False
    
    def get_failover_chain(self, series_id: str) -> List[str]:
        """Get the failover chain for a specific series."""
        return self.failover_chains.get(series_id, [])

# Global instance
failover_manager = ProviderFailoverManager()