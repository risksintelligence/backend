"""
Redis cache implementation for primary caching.
"""
import json
import logging
from typing import Any, Optional, Union
from datetime import datetime, timedelta

import redis
from redis.exceptions import ConnectionError, TimeoutError

from src.core.config import settings

logger = logging.getLogger(__name__)


class RedisCache:
    """Redis cache implementation for high-speed data caching."""
    
    def __init__(self):
        """Initialize Redis connection."""
        self._client = None
        self._connect()
    
    def _connect(self) -> None:
        """Establish Redis connection."""
        try:
            self._client = redis.from_url(
                settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
            )
            # Test connection
            self._client.ping()
            logger.info("Redis cache connected successfully")
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._client = None
    
    def is_available(self) -> bool:
        """Check if Redis is available."""
        if not self._client:
            return False
        try:
            self._client.ping()
            return True
        except (ConnectionError, TimeoutError):
            logger.warning("Redis connection lost")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/unavailable
        """
        if not self.is_available():
            return None
        
        try:
            value = self._client.get(key)
            if value is None:
                return None
            
            # Try to deserialize JSON
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
                
        except Exception as e:
            logger.error(f"Error getting cache key {key}: {e}")
            return None
    
    def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (defaults to settings.cache_ttl)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            return False
        
        try:
            # Serialize value
            if not isinstance(value, str):
                value = json.dumps(value, default=str)
            
            ttl = ttl or settings.cache_ttl
            result = self._client.setex(key, ttl, value)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Error setting cache key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """
        Delete key from cache.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            return False
        
        try:
            result = self._client.delete(key)
            return bool(result)
        except Exception as e:
            logger.error(f"Error deleting cache key {key}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.
        
        Args:
            key: Cache key to check
            
        Returns:
            True if key exists, False otherwise
        """
        if not self.is_available():
            return False
        
        try:
            return bool(self._client.exists(key))
        except Exception as e:
            logger.error(f"Error checking cache key {key}: {e}")
            return False
    
    def get_ttl(self, key: str) -> Optional[int]:
        """
        Get time to live for a key.
        
        Args:
            key: Cache key
            
        Returns:
            TTL in seconds or None if key doesn't exist
        """
        if not self.is_available():
            return None
        
        try:
            ttl = self._client.ttl(key)
            return ttl if ttl > 0 else None
        except Exception as e:
            logger.error(f"Error getting TTL for key {key}: {e}")
            return None
    
    def flush_all(self) -> bool:
        """
        Clear all cache data.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            return False
        
        try:
            self._client.flushall()
            logger.info("Redis cache flushed")
            return True
        except Exception as e:
            logger.error(f"Error flushing cache: {e}")
            return False
    
    def get_stats(self) -> dict:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        if not self.is_available():
            return {"status": "unavailable"}
        
        try:
            info = self._client.info()
            return {
                "status": "connected",
                "used_memory": info.get("used_memory_human", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"status": "error", "error": str(e)}