import asyncio
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import logging
import aiohttp
from src.cache.cache_manager import IntelligentCacheManager
from src.data.sources import fred, bea, bls, census

logger = logging.getLogger(__name__)


class BackgroundRefreshWorker:
    """
    Refreshes cache in background without blocking user requests.
    Handles API failures gracefully.
    """
    
    def __init__(self, cache_manager: IntelligentCacheManager):
        self.cache = cache_manager
        self.is_running = False
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Priority data to refresh frequently
        self.priority_series = [
            # Economic indicators (every 5 min)
            {"source": "fred", "series": "GDP", "interval": 300, "priority": 1},
            {"source": "fred", "series": "UNRATE", "interval": 300, "priority": 1},
            {"source": "fred", "series": "CPIAUCSL", "interval": 300, "priority": 1},
            {"source": "fred", "series": "FEDFUNDS", "interval": 300, "priority": 1},
            
            # Market indicators (every 10 min)
            {"source": "market", "series": "VIX", "interval": 600, "priority": 2},
            {"source": "market", "series": "SP500", "interval": 600, "priority": 2},
            
            # Risk metrics (every 15 min)
            {"source": "risk", "series": "overview", "interval": 900, "priority": 3},
            {"source": "risk", "series": "factors", "interval": 900, "priority": 3},
        ]
        
        # No sample data - only real API data allowed
    
    async def start(self):
        """Start background refresh workers."""
        self.is_running = True
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10))
        
        # Start multiple workers for parallel processing
        workers = [
            asyncio.create_task(self._refresh_loop()),
            asyncio.create_task(self._health_check_loop()),
            asyncio.create_task(self._cache_warmer_loop())
        ]
        
        logger.info("Background workers started")
        
        try:
            await asyncio.gather(*workers)
        except Exception as e:
            logger.error(f"Worker error: {e}")
        finally:
            await self.cleanup()
    
    async def stop(self):
        """Stop all workers."""
        self.is_running = False
        logger.info("⏹️ Background workers stopped")
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.session:
            await self.session.close()
    
    async def _refresh_loop(self):
        """Main refresh loop - updates cache periodically."""
        
        while self.is_running:
            for series_config in self.priority_series:
                if not self.is_running:
                    break
                    
                try:
                    # Get fresh data from real API only
                    fresh_data = await self._fetch_from_api(
                        series_config["source"],
                        series_config["series"]
                    )
                    
                    if fresh_data:
                        # Update cache
                        cache_key = f"{series_config['source']}:{series_config['series']}"
                        await self.cache.set(
                            cache_key, 
                            fresh_data,
                            ttl_seconds=series_config["interval"] * 2
                        )
                        
                        logger.info(f"Refreshed: {cache_key}")
                    
                except Exception as e:
                    # API failed? No problem - cache serves stale data
                    cache_key = f"{series_config['source']}:{series_config['series']}"
                    logger.warning(
                        f"API unavailable: {cache_key} - "
                        f"Error: {e}. Serving cached data."
                    )
                
                # Rate limiting - don't hammer APIs
                await asyncio.sleep(2)
            
            # Wait before next refresh cycle
            await asyncio.sleep(60)
    
    async def _fetch_from_api(self, source: str, series: str) -> Optional[Dict]:
        """Fetch data from external API - real data only."""
        
        try:
            # Only use real API data
            if source == "fred":
                if series == "GDP":
                    return await fred.get_gdp()
                elif series == "UNRATE":
                    return await fred.get_unemployment_rate()
                elif series == "CPIAUCSL":
                    return await fred.get_inflation_rate()
                elif series == "FEDFUNDS":
                    return await fred.get_fed_funds_rate()
            
            elif source == "bea":
                if series == "NIPA":
                    accounts = await bea.get_economic_accounts()
                    if accounts and "indicators" in accounts:
                        return accounts["indicators"]
            
            elif source == "bls":
                if series == "employment":
                    return await bls.get_labor_statistics()
            
            elif source == "census":
                if series == "population":
                    return await census.get_population_data()
                elif series == "income":
                    return await census.get_household_income()
            
            elif source == "market":
                if series == "overview":
                    return await fred.get_market_overview()
                elif series == "SP500":
                    return await fred.get_sp500_data()
                elif series == "VIX":
                    return await fred.get_vix_data()
            
            elif source == "risk":
                # Route to real economic data APIs for risk indicators
                if series == "overview":
                    # Use real economic indicators as risk overview
                    indicators = await fred.get_key_indicators()
                    if indicators and "indicators" in indicators:
                        return {
                            "economic_indicators": indicators["indicators"],
                            "source": "fred_economic_data",
                            "last_updated": datetime.utcnow().isoformat()
                        }
                elif series == "factors":
                    # Use real economic factors
                    factors = await asyncio.gather(
                        fred.get_unemployment_rate(),
                        fred.get_inflation_rate(), 
                        fred.get_fed_funds_rate(),
                        return_exceptions=True
                    )
                    valid_factors = [f for f in factors if isinstance(f, dict) and f]
                    return {
                        "risk_factors": valid_factors,
                        "source": "fred_economic_factors",
                        "last_updated": datetime.utcnow().isoformat()
                    }
            
            # No real API data available for this series
            logger.error(f"No real API implementation for {source}:{series}")
            raise ValueError(f"Real data not available for {source}:{series}")
        
        except Exception as e:
            logger.error(f"API call failed for {source}:{series}: {e}")
            raise ValueError(f"Failed to fetch real data for {source}:{series} - synthetic data not allowed")
    
    async def _health_check_loop(self):
        """Monitor API health and adjust refresh intervals."""
        
        while self.is_running:
            # Check each API's health
            api_sources = ["fred", "market", "risk"]
            
            for source in api_sources:
                if not self.is_running:
                    break
                    
                try:
                    # Quick health check
                    is_healthy = await self._check_api_health(source)
                    
                    if is_healthy:
                        logger.debug(f"{source} API healthy")
                    else:
                        logger.warning(f"{source} API degraded")
                
                except Exception as e:
                    logger.error(f"{source} API health check failed: {e}")
            
            await asyncio.sleep(300)  # Check every 5 minutes
    
    async def _check_api_health(self, source: str) -> bool:
        """Quick health check for an API."""
        try:
            # Make actual API health checks
            if source == "fred":
                return await fred.health_check()
            elif source == "bea":
                return await bea.health_check()
            elif source == "bls":
                return await bls.health_check()
            elif source == "census":
                return await census.health_check()
            else:
                return False
                
        except Exception:
            return False
    
    async def _cache_warmer_loop(self):
        """Pre-warm cache with frequently accessed data."""
        
        while self.is_running:
            # On startup or periodically, warm up cache
            logger.info("Warming up cache...")
            
            # Pre-load most common queries
            common_queries = [
                {"source": "risk", "series": "overview"},
                {"source": "risk", "series": "factors"},
                {"source": "fred", "series": "GDP"},
                {"source": "fred", "series": "UNRATE"},
                {"source": "market", "series": "VIX"}
            ]
            
            for query in common_queries:
                if not self.is_running:
                    break
                    
                try:
                    # Trigger cache population
                    cache_key = f"{query['source']}:{query['series']}"
                    
                    # Check if already cached
                    cached_data = await self.cache.get(cache_key)
                    if not cached_data:
                        # Populate cache
                        fresh_data = await self._fetch_from_api(
                            query["source"], 
                            query["series"]
                        )
                        if fresh_data:
                            await self.cache.set(cache_key, fresh_data, ttl_seconds=3600)
                            logger.debug(f"Warmed cache: {cache_key}")
                    
                except Exception as e:
                    cache_key = f"{query['source']}:{query['series']}"
                    logger.error(f"Cache warm error for {cache_key}: {e}")
            
            logger.info("Cache warmed")
            
            # Warm up every hour
            await asyncio.sleep(3600)