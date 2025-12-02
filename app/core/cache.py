import os
import json
import hashlib
import logging
from typing import Any, Optional, Dict, Tuple, Callable, Awaitable
from datetime import datetime, timedelta
from functools import wraps
import redis

class RedisCache:
    """L1 Redis cache with TTL management and stale-while-revalidate support."""
    
    def __init__(self, namespace: str) -> None:
        from app.core.config import get_settings
        settings = get_settings()
        url = settings.redis_url
        if not url:
            logging.getLogger(__name__).warning("Redis URL missing; cache will be disabled for local development")
            self.client = None
            self.available = False
            self.namespace = namespace
            return
        try:
            # Production Redis configuration with connection pooling
            pool = redis.ConnectionPool.from_url(
                url,
                max_connections=20,
                retry_on_timeout=True,
                socket_keepalive=True,
                socket_keepalive_options={},
                health_check_interval=30,
            )
            self.client = redis.Redis(connection_pool=pool)
            self.client.ping()  # Test connection
            self.available = True
        except Exception as e:
            logging.getLogger(__name__).error(f"Redis connection failed: {e}")
            self.client = None
            self.available = False
        self.namespace = namespace

    def _key(self, key: str) -> str:
        return f"rrio:{self.namespace}:{key}"
    
    def _metadata_key(self, key: str) -> str:
        return f"rrio:{self.namespace}:meta:{key}"

    def get_with_metadata(self, key: str) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
        """Get cached value with TTL and freshness metadata."""
        if not self.available:
            return None, None
            
        try:
            # Get both data and metadata
            pipe = self.client.pipeline()
            pipe.get(self._key(key))
            pipe.hgetall(self._metadata_key(key))
            data_raw, meta_raw = pipe.execute()
            
            if not data_raw:
                return None, None
            
            data = json.loads(data_raw)
            metadata = {k.decode(): v.decode() for k, v in meta_raw.items()} if meta_raw else {}
            
            # Check TTL status
            if metadata:
                cached_at = datetime.fromisoformat(metadata.get('cached_at', ''))
                soft_ttl = int(metadata.get('soft_ttl', 900))
                hard_ttl = int(metadata.get('hard_ttl', 86400))
                
                now = datetime.utcnow()
                age_seconds = (now - cached_at).total_seconds()
                
                metadata.update({
                    'age_seconds': int(age_seconds),
                    'is_stale_soft': age_seconds > soft_ttl,
                    'is_stale_hard': age_seconds > hard_ttl,
                    'cache_status': 'l1_hit'
                })
            
            return data, metadata
            
        except Exception:
            return None, None

    def get(self, key: str) -> Optional[Any]:
        """Simple get method for backward compatibility."""
        data, _ = self.get_with_metadata(key)
        return data

    def set_with_metadata(self, key: str, value: Any, soft_ttl: int = 900, hard_ttl: int = 86400, 
                         source: str = None, source_url: str = None, derivation_flag: str = "raw") -> None:
        """Set cached value with full metadata per architecture requirements."""
        if not self.available:
            return
            
        try:
            now = datetime.utcnow()
            data_json = json.dumps(value, default=str)
            
            # Create metadata
            metadata = {
                'cached_at': now.isoformat(),
                'soft_ttl': str(soft_ttl),
                'hard_ttl': str(hard_ttl),
                'checksum': hashlib.sha256(data_json.encode()).hexdigest()[:16],
                'source': source or 'unknown',
                'source_url': source_url or '',
                'derivation_flag': derivation_flag,
                'size_bytes': str(len(data_json))
            }
            
            # Store data and metadata
            pipe = self.client.pipeline()
            pipe.set(self._key(key), data_json, ex=hard_ttl)
            pipe.hset(self._metadata_key(key), mapping=metadata)
            pipe.expire(self._metadata_key(key), hard_ttl)
            pipe.execute()
            
        except Exception:
            pass  # Fail silently for cache errors

    def set(self, key: str, value: Any, ttl: int = 900) -> None:
        """Simple set method for backward compatibility."""
        self.set_with_metadata(key, value, soft_ttl=ttl, hard_ttl=ttl*4)

    def delete(self, key: str) -> None:
        """Delete cached value and its metadata."""
        if not self.available:
            return
            
        try:
            pipe = self.client.pipeline()
            pipe.delete(self._key(key))
            pipe.delete(self._metadata_key(key))
            pipe.execute()
        except Exception:
            pass

    def get_freshness_status(self) -> Dict[str, Any]:
        """Get overall cache freshness status for monitoring."""
        if not self.available:
            return {"status": "unavailable", "cache_layer": "l1_redis"}
            
        try:
            # Get all metadata keys for this namespace
            pattern = f"rrio:{self.namespace}:meta:*"
            keys = self.client.keys(pattern)
            
            total_keys = len(keys)
            stale_soft_count = 0
            stale_hard_count = 0
            
            for key in keys[:100]:  # Sample up to 100 keys
                meta = self.client.hgetall(key)
                if meta:
                    metadata = {k.decode(): v.decode() for k, v in meta.items()}
                    cached_at = datetime.fromisoformat(metadata.get('cached_at', ''))
                    soft_ttl = int(metadata.get('soft_ttl', 900))
                    hard_ttl = int(metadata.get('hard_ttl', 86400))
                    
                    age_seconds = (datetime.utcnow() - cached_at).total_seconds()
                    if age_seconds > soft_ttl:
                        stale_soft_count += 1
                    if age_seconds > hard_ttl:
                        stale_hard_count += 1
            
            return {
                "status": "available",
                "cache_layer": "l1_redis",
                "total_keys": total_keys,
                "stale_soft_count": stale_soft_count,
                "stale_hard_count": stale_hard_count,
                "fresh_percentage": ((total_keys - stale_soft_count) / total_keys * 100) if total_keys > 0 else 0
            }
            
        except Exception as e:
            return {"status": "error", "cache_layer": "l1_redis", "error": str(e)}


class CacheConfig:
    """Configuration for cache decorators"""
    def __init__(self, key_prefix: str, ttl_seconds: int = 900, fallback_ttl_seconds: int = 86400):
        self.key_prefix = key_prefix
        self.ttl_seconds = ttl_seconds
        self.fallback_ttl_seconds = fallback_ttl_seconds


def cache_with_fallback(config: CacheConfig):
    """
    Decorator for caching async functions with fallback support.
    Uses Redis cache with stale-while-revalidate pattern.
    """
    def decorator(func: Callable[..., Awaitable[Dict[str, Any]]]) -> Callable[..., Awaitable[Dict[str, Any]]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Dict[str, Any]:
            try:
                # Create cache instance
                cache = RedisCache(config.key_prefix)
                
                # Generate cache key from function name and arguments
                key_parts = [func.__name__]
                if args:
                    key_parts.extend(str(arg) for arg in args)
                if kwargs:
                    key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = ":".join(key_parts)
                
                # Try to get from cache with metadata
                cached_data, metadata = cache.get_with_metadata(cache_key)
                
                if cached_data and metadata:
                    # Check if data is still fresh
                    is_stale_soft = metadata.get('is_stale_soft', False)
                    is_stale_hard = metadata.get('is_stale_hard', False)
                    
                    if not is_stale_hard:
                        # Data is still valid (not hard stale)
                        if not is_stale_soft:
                            # Data is fresh, return it
                            return cached_data
                        else:
                            # Data is soft stale, return it but trigger background refresh
                            # For now, just return the stale data
                            return cached_data
                
                # Cache miss or hard stale - fetch fresh data
                try:
                    fresh_data = await func(*args, **kwargs)
                    
                    # Cache the fresh data
                    cache.set_with_metadata(
                        cache_key, 
                        fresh_data, 
                        soft_ttl=config.ttl_seconds,
                        hard_ttl=config.fallback_ttl_seconds,
                        source=func.__module__,
                        derivation_flag="fresh"
                    )
                    
                    return fresh_data
                    
                except Exception as e:
                    logging.getLogger(__name__).error(f"Function {func.__name__} failed: {e}")
                    
                    # If we have stale data, return it
                    if cached_data:
                        return cached_data
                    
                    # No cached data available, re-raise the exception
                    raise
                    
            except Exception as e:
                # Cache system failed, call function directly
                logging.getLogger(__name__).warning(f"Cache system failed for {func.__name__}: {e}")
                return await func(*args, **kwargs)
        
        return wrapper
    return decorator


class FileCache:
    """Simple file-based cache for data providers that need FileCache interface."""
    
    def __init__(self, namespace: str):
        self.namespace = namespace
        self.cache_dir = f"cache/{namespace}"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def get(self, key: str, default=None):
        """Get cached value by key."""
        try:
            cache_file = os.path.join(self.cache_dir, f"{key}.json")
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    # Check if cache is expired (24 hours)
                    cached_time = datetime.fromisoformat(data.get('cached_at', ''))
                    if datetime.now() - cached_time < timedelta(hours=24):
                        return data.get('value')
            return default
        except Exception as e:
            self.logger.debug(f"FileCache get error for {key}: {e}")
            return default
    
    def set(self, key: str, value: Any, ttl_seconds: int = 86400):
        """Set cached value with TTL."""
        try:
            cache_file = os.path.join(self.cache_dir, f"{key}.json")
            data = {
                'value': value,
                'cached_at': datetime.now().isoformat(),
                'ttl_seconds': ttl_seconds
            }
            with open(cache_file, 'w') as f:
                json.dump(data, f, default=str)
        except Exception as e:
            self.logger.debug(f"FileCache set error for {key}: {e}")
    
    def delete(self, key: str):
        """Delete cached value by key."""
        try:
            cache_file = os.path.join(self.cache_dir, f"{key}.json")
            if os.path.exists(cache_file):
                os.remove(cache_file)
        except Exception as e:
            self.logger.debug(f"FileCache delete error for {key}: {e}")


