"""
Cache manager that coordinates multiple cache layers with file-based fallback.
"""
import logging
from typing import Any, Optional, Dict, List
from datetime import datetime

from .redis_cache import RedisCache
from .postgres_cache import PostgresCache
from .file_cache import FileCache
from .fallback_handler import FallbackHandler

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Unified cache manager that coordinates multiple cache layers.
    
    Cache hierarchy:
    1. Redis (primary, fast access)
    2. PostgreSQL (persistent storage)
    3. File cache (local file-based fallback)
    4. Fallback handler (last resort, static data)
    """
    
    def __init__(self):
        """Initialize cache manager with all cache layers."""
        from src.core.config import settings
        
        # Initialize caches only if enabled and available
        self.redis_cache = None
        self.postgres_cache = None
        self.file_cache = FileCache()
        self.fallback_handler = FallbackHandler()
        
        # Initialize Redis cache if enabled
        if settings.enable_redis:
            try:
                self.redis_cache = RedisCache()
            except Exception as e:
                logger.warning(f"Redis cache initialization failed: {e}")
        
        # Initialize PostgreSQL cache if enabled
        if settings.enable_postgres_cache:
            try:
                self.postgres_cache = PostgresCache()
            except Exception as e:
                logger.warning(f"PostgreSQL cache initialization failed: {e}")
        
        # Check which caches are available
        self.redis_available = self.redis_cache is not None and self.redis_cache.is_available()
        self.postgres_available = self.postgres_cache is not None and self.postgres_cache.is_available()
        self.file_available = self.file_cache.is_available()
        
        if not self.redis_available and not self.postgres_available:
            logger.warning("Redis and PostgreSQL unavailable, using file cache as primary")
        elif not self.redis_available:
            logger.warning("Redis unavailable, using PostgreSQL and file cache")
        
        logger.info(f"Cache manager initialized - Redis: {self.redis_available}, PostgreSQL: {self.postgres_available}, File: {self.file_available}")
        
        # Initialize default cache statistics to prevent fallback errors
        self._initialize_cache_stats()
    
    def get(self, key: str, use_fallback: bool = True) -> Optional[Any]:
        """
        Get value from cache with automatic fallback.
        
        Args:
            key: Cache key
            use_fallback: Whether to use fallback if primary caches fail
            
        Returns:
            Cached value or None if not found
        """
        # Try Redis first (fastest) if available
        if self.redis_available:
            value = self.redis_cache.get(key)
            if value is not None:
                logger.debug(f"Cache hit (Redis): {key}")
                return value
        
        # Try PostgreSQL cache if available
        if self.postgres_available:
            value = self.postgres_cache.get(key)
            if value is not None:
                logger.debug(f"Cache hit (PostgreSQL): {key}")
                
                # Populate Redis cache for faster future access if Redis is available
                if self.redis_available:
                    self.redis_cache.set(key, value)
                return value
        
        # Try file cache (always available)
        if self.file_available:
            value = self.file_cache.get(key, use_fallback=False)  # File cache has its own fallback
            if value is not None:
                logger.debug(f"Cache hit (File): {key}")
                
                # Populate higher-level caches if available
                if self.redis_available:
                    self.redis_cache.set(key, value)
                if self.postgres_available:
                    self.postgres_cache.set(key, value)
                return value
        
        # Try fallback handler if enabled
        if use_fallback:
            fallback_data = self.fallback_handler.get_fallback_data(key)
            if fallback_data:
                logger.warning(f"Using fallback data for key: {key}")
                return fallback_data["data"]
        
        logger.debug(f"Cache miss: {key}")
        return None
    
    def _initialize_cache_stats(self) -> None:
        """Initialize default cache statistics."""
        try:
            from datetime import datetime
            default_stats = {
                "hits": 0,
                "misses": 0,
                "total_keys": 0,
                "redis_available": self.redis_available,
                "postgres_available": self.postgres_available,
                "file_available": self.file_available,
                "initialized_at": datetime.utcnow().isoformat()
            }
            
            # Set in file cache as it's always available
            if self.file_available:
                self.file_cache.set("cache_stats", default_stats, ttl=86400)  # 24 hours
                logger.debug("Initialized default cache statistics")
                
        except Exception as e:
            logger.warning(f"Failed to initialize cache statistics: {e}")
    
    def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None,
        persist_to_postgres: bool = True
    ) -> bool:
        """
        Set value in cache with automatic replication.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (default: 7 days)
            persist_to_postgres: Whether to also store in PostgreSQL
            
        Returns:
            True if at least one cache layer succeeded
        """
        if ttl is None:
            ttl = 7 * 24 * 3600  # 1 week default
            
        success = False
        
        # Set in Redis (primary) if available
        if self.redis_available and self.redis_cache.set(key, value, ttl):
            success = True
            logger.debug(f"Cached in Redis: {key}")
        
        # Set in PostgreSQL for persistence if available
        if persist_to_postgres and self.postgres_available and self.postgres_cache.set(key, value, ttl):
            success = True
            logger.debug(f"Cached in PostgreSQL: {key}")
        
        # Always set in file cache as reliable fallback
        if self.file_available and self.file_cache.set(key, value, ttl):
            success = True
            logger.debug(f"Cached in File: {key}")
        
        # Update fallback handler with successful data
        if success:
            source = self._extract_source_from_key(key)
            self.fallback_handler.set_last_known_good(source, value)
        
        return success
    
    def delete(self, key: str) -> bool:
        """
        Delete key from all cache layers.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if at least one cache layer succeeded
        """
        redis_success = self.redis_available and self.redis_cache.delete(key)
        postgres_success = self.postgres_available and self.postgres_cache.delete(key)
        file_success = self.file_available and self.file_cache.delete(key)
        
        return redis_success or postgres_success or file_success
    
    def exists(self, key: str) -> bool:
        """
        Check if key exists in any cache layer.
        
        Args:
            key: Cache key to check
            
        Returns:
            True if key exists in any cache layer
        """
        return (
            (self.redis_available and self.redis_cache.exists(key)) or 
            (self.postgres_available and self.postgres_cache.exists(key)) or
            (self.file_available and self.file_cache.exists(key))
        )
    
    def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate all keys matching a pattern.
        
        Args:
            pattern: Key pattern to match (basic string matching)
            
        Returns:
            Number of keys invalidated
        """
        # This is a simplified implementation
        # In production, you'd want more sophisticated pattern matching
        invalidated = 0
        
        # For now, we'll need to implement pattern matching in the future
        logger.warning(f"Pattern invalidation not fully implemented: {pattern}")
        
        return invalidated
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics.
        
        Returns:
            Dictionary with statistics from all cache layers
        """
        return {
            "redis": self.redis_cache.get_stats(),
            "postgres": self.postgres_cache.get_stats(),
            "system_health": self.fallback_handler.get_system_health(),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all cache layers.
        
        Returns:
            Health status for each cache layer
        """
        redis_healthy = self.redis_available and self.redis_cache.is_available()
        postgres_healthy = self.postgres_available and self.postgres_cache.is_available()
        file_healthy = self.file_available and self.file_cache.is_available()
        
        overall_healthy = redis_healthy or postgres_healthy or file_healthy
        
        return {
            "overall_healthy": overall_healthy,
            "redis": {
                "healthy": redis_healthy,
                "status": "connected" if redis_healthy else "disconnected"
            },
            "postgres": {
                "healthy": postgres_healthy,
                "status": "connected" if postgres_healthy else "disconnected"
            },
            "file": {
                "healthy": file_healthy,
                "status": "available" if file_healthy else "unavailable"
            },
            "degraded_mode": not redis_healthy and not postgres_healthy and file_healthy,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def cleanup(self) -> Dict[str, int]:
        """
        Cleanup expired entries and old data.
        
        Returns:
            Dictionary with cleanup statistics
        """
        postgres_cleaned = 0
        file_cleaned = 0
        
        if self.postgres_available:
            postgres_cleaned = self.postgres_cache.cleanup_expired()
        
        if self.file_available:
            file_cleaned = self.file_cache.cleanup_expired()
        
        # Cleanup old fallback data
        self.fallback_handler.cleanup_old_data()
        
        return {
            "postgres_expired_cleaned": postgres_cleaned,
            "file_expired_cleaned": file_cleaned,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def register_data_source_failure(self, source: str, error: str) -> None:
        """
        Register a failure for a data source.
        
        Args:
            source: Data source identifier
            error: Error description
        """
        self.fallback_handler.record_failure(source, error)
        logger.warning(f"Data source failure recorded: {source} - {error}")
    
    def register_fallback_data(
        self, 
        source: str, 
        data: Any, 
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Register fallback data for a source.
        
        Args:
            source: Data source identifier
            data: Fallback data
            timestamp: When this data was created
        """
        self.fallback_handler.register_fallback_data(source, data, timestamp)
    
    def get_data_source_status(self, source: str) -> Dict[str, Any]:
        """
        Get status for a specific data source.
        
        Args:
            source: Data source identifier
            
        Returns:
            Source status information
        """
        return self.fallback_handler.get_source_status(source)
    
    def _extract_source_from_key(self, key: str) -> str:
        """
        Extract data source name from cache key.
        
        Args:
            key: Cache key
            
        Returns:
            Data source name
        """
        # Assume keys are formatted as "source:endpoint:params"
        parts = key.split(":")
        return parts[0] if parts else "unknown"
    
    def flush_all(self) -> bool:
        """
        Clear all cache data from all layers.
        
        Returns:
            True if successful
        """
        redis_success = self.redis_cache.flush_all()
        postgres_success = self.postgres_cache.flush_all()
        
        logger.warning("All cache data flushed")
        return redis_success or postgres_success
    
    async def warm_redis_cache(self) -> bool:
        """
        Warm Redis cache with frequently accessed data.
        
        Returns:
            True if cache warming was successful
        """
        try:
            if not self.redis_available:
                logger.warning("Redis not available for cache warming")
                return False
            
            logger.info("Starting Redis cache warming")
            warmed_keys = 0
            
            # Get commonly accessed data from PostgreSQL cache
            if self.postgres_available:
                postgres_keys = await self._get_postgres_cache_keys()
                for key in postgres_keys[:100]:  # Warm top 100 keys
                    try:
                        value = self.postgres_cache.get(key)
                        if value is not None:
                            # Set in Redis with appropriate TTL
                            self.redis_cache.set(key, value, ttl=3600)
                            warmed_keys += 1
                    except Exception as e:
                        logger.debug(f"Error warming key {key}: {str(e)}")
                        continue
            
            # Warm with recent fallback data
            fallback_sources = ["fred", "bea", "bls", "noaa", "cisa"]
            for source in fallback_sources:
                try:
                    fallback_data = self.fallback_handler.get_fallback_data(source)
                    if fallback_data:
                        cache_key = f"{source}:fallback_data"
                        self.redis_cache.set(cache_key, fallback_data, ttl=7200)
                        warmed_keys += 1
                except Exception as e:
                    logger.debug(f"Error warming fallback for {source}: {str(e)}")
                    continue
            
            logger.info(f"Redis cache warming completed: {warmed_keys} keys warmed")
            return warmed_keys > 0
            
        except Exception as e:
            logger.error(f"Error warming Redis cache: {str(e)}")
            return False
    
    async def warm_postgres_cache(self) -> bool:
        """
        Warm PostgreSQL cache with persistent data.
        
        Returns:
            True if cache warming was successful
        """
        try:
            if not self.postgres_available:
                logger.warning("PostgreSQL not available for cache warming")
                return False
            
            logger.info("Starting PostgreSQL cache warming")
            warmed_keys = 0
            
            # Warm with file cache data
            if self.file_available:
                file_keys = await self._get_file_cache_keys()
                for key in file_keys:
                    try:
                        value = self.file_cache.get(key, use_fallback=False)
                        if value is not None:
                            # Set in PostgreSQL with longer TTL
                            self.postgres_cache.set(key, value, ttl=86400)  # 24 hours
                            warmed_keys += 1
                    except Exception as e:
                        logger.debug(f"Error warming PostgreSQL key {key}: {str(e)}")
                        continue
            
            # Warm with fallback data for persistence
            fallback_sources = ["fred", "bea", "bls", "noaa", "cisa"]
            for source in fallback_sources:
                try:
                    fallback_data = self.fallback_handler.get_fallback_data(source)
                    if fallback_data:
                        cache_key = f"{source}:persistent_fallback"
                        self.postgres_cache.set(cache_key, fallback_data, ttl=604800)  # 7 days
                        warmed_keys += 1
                except Exception as e:
                    logger.debug(f"Error warming PostgreSQL fallback for {source}: {str(e)}")
                    continue
            
            logger.info(f"PostgreSQL cache warming completed: {warmed_keys} keys warmed")
            return warmed_keys > 0
            
        except Exception as e:
            logger.error(f"Error warming PostgreSQL cache: {str(e)}")
            return False
    
    async def warm_file_cache(self) -> bool:
        """
        Warm file cache with backup data.
        
        Returns:
            True if cache warming was successful
        """
        try:
            if not self.file_available:
                logger.warning("File cache not available for warming")
                return False
            
            logger.info("Starting file cache warming")
            warmed_keys = 0
            
            # Warm with current fallback data
            fallback_sources = ["fred", "bea", "bls", "noaa", "cisa"]
            for source in fallback_sources:
                try:
                    fallback_data = self.fallback_handler.get_fallback_data(source)
                    if fallback_data:
                        cache_key = f"{source}:file_backup"
                        success = self.file_cache.set(cache_key, fallback_data, ttl=2592000)  # 30 days
                        if success:
                            warmed_keys += 1
                except Exception as e:
                    logger.debug(f"Error warming file cache for {source}: {str(e)}")
                    continue
            
            # Create sample data for testing purposes
            sample_data = {
                "system_health": {
                    "status": "operational",
                    "last_check": datetime.now().isoformat(),
                    "components": {
                        "cache": "healthy",
                        "database": "healthy",
                        "api": "healthy"
                    }
                }
            }
            
            try:
                self.file_cache.set("system:health_check", sample_data, ttl=3600)
                warmed_keys += 1
            except Exception as e:
                logger.debug(f"Error warming system health data: {str(e)}")
            
            logger.info(f"File cache warming completed: {warmed_keys} keys warmed")
            return warmed_keys > 0
            
        except Exception as e:
            logger.error(f"Error warming file cache: {str(e)}")
            return False
    
    async def _get_postgres_cache_keys(self) -> List[str]:
        """
        Get list of keys from PostgreSQL cache.
        
        Returns:
            List of cache keys
        """
        try:
            # This would need to be implemented based on your PostgreSQL cache schema
            # For now, return commonly used keys
            common_keys = [
                "fred:economic_indicators",
                "fred:gdp_data", 
                "fred:unemployment_rate",
                "bea:trade_data",
                "bls:employment_data",
                "noaa:weather_alerts",
                "cisa:vulnerabilities"
            ]
            return common_keys
            
        except Exception as e:
            logger.error(f"Error getting PostgreSQL cache keys: {str(e)}")
            return []
    
    async def _get_file_cache_keys(self) -> List[str]:
        """
        Get list of keys from file cache.
        
        Returns:
            List of cache keys
        """
        try:
            # Get keys from file cache metadata
            if hasattr(self.file_cache, 'get_all_keys'):
                return await self.file_cache.get_all_keys()
            else:
                # Return commonly used keys if method not available
                common_keys = [
                    "fred:backup_data",
                    "bea:backup_data",
                    "bls:backup_data", 
                    "noaa:backup_data",
                    "cisa:backup_data"
                ]
                return common_keys
                
        except Exception as e:
            logger.error(f"Error getting file cache keys: {str(e)}")
            return []