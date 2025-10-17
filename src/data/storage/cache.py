"""
Cache storage utilities for RiskX platform.
Provides Redis, memory, and file-based caching with automatic fallback mechanisms.
"""

import os
import json
import pickle
import hashlib
import logging
from typing import Optional, Dict, Any, List, Union, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import asyncio
from pathlib import Path

try:
    import redis
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger = logging.getLogger('riskx.data.storage.cache')
    logger.warning("Redis not available, using fallback caching")

from ...core.exceptions import CacheError, ConfigurationError
from ...utils.helpers import generate_cache_key, safe_divide
from ...utils.constants import CacheConfig

logger = logging.getLogger('riskx.data.storage.cache')


@dataclass
class CacheConfig:
    """Cache configuration settings."""
    redis_url: str = "redis://localhost:6379"
    redis_db: int = 0
    redis_password: Optional[str] = None
    redis_socket_timeout: int = 5
    redis_connection_timeout: int = 10
    default_ttl: int = 3600  # 1 hour
    max_memory_size: int = 100 * 1024 * 1024  # 100MB
    file_cache_dir: str = "/tmp/riskx_cache"
    compression_enabled: bool = True
    serialize_format: str = "json"  # json, pickle
    
    @classmethod
    def from_env(cls) -> 'CacheConfig':
        """Create config from environment variables."""
        return cls(
            redis_url=os.getenv('REDIS_URL', 'redis://localhost:6379'),
            redis_db=int(os.getenv('REDIS_DB', '0')),
            redis_password=os.getenv('REDIS_PASSWORD'),
            redis_socket_timeout=int(os.getenv('REDIS_SOCKET_TIMEOUT', '5')),
            redis_connection_timeout=int(os.getenv('REDIS_CONNECTION_TIMEOUT', '10')),
            default_ttl=int(os.getenv('CACHE_DEFAULT_TTL', '3600')),
            max_memory_size=int(os.getenv('CACHE_MAX_MEMORY_SIZE', str(100 * 1024 * 1024))),
            file_cache_dir=os.getenv('CACHE_FILE_DIR', '/tmp/riskx_cache'),
            compression_enabled=os.getenv('CACHE_COMPRESSION', 'true').lower() == 'true',
            serialize_format=os.getenv('CACHE_SERIALIZE_FORMAT', 'json')
        )


class CacheBackend(ABC):
    """Abstract base class for cache backends."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """Clear all cache entries."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        pass


class RedisCacheBackend(CacheBackend):
    """Redis-based cache backend."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self._client = None
        self._async_client = None
        self._is_connected = False
        self._stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'errors': 0
        }
        
        if REDIS_AVAILABLE:
            self._initialize_connection()
    
    def _initialize_connection(self):
        """Initialize Redis connection."""
        try:
            self._client = redis.from_url(
                self.config.redis_url,
                db=self.config.redis_db,
                password=self.config.redis_password,
                socket_timeout=self.config.redis_socket_timeout,
                socket_connect_timeout=self.config.redis_connection_timeout,
                decode_responses=True
            )
            
            # Test connection
            self._client.ping()
            self._is_connected = True
            logger.info("Redis cache backend initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis cache: {e}")
            self._is_connected = False
    
    async def _initialize_async_connection(self):
        """Initialize async Redis connection."""
        try:
            if not REDIS_AVAILABLE:
                return
            
            self._async_client = aioredis.from_url(
                self.config.redis_url,
                db=self.config.redis_db,
                password=self.config.redis_password,
                socket_timeout=self.config.redis_socket_timeout,
                socket_connect_timeout=self.config.redis_connection_timeout,
                decode_responses=True
            )
            
            # Test connection
            await self._async_client.ping()
            logger.info("Async Redis cache backend initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize async Redis cache: {e}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        if not self._is_connected or not self._client:
            return None
        
        try:
            value = self._client.get(key)
            if value is not None:
                self._stats['hits'] += 1
                return self._deserialize(value)
            else:
                self._stats['misses'] += 1
                return None
        
        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"Redis get error for key {key}: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis cache."""
        if not self._is_connected or not self._client:
            return False
        
        try:
            serialized_value = self._serialize(value)
            ttl = ttl or self.config.default_ttl
            
            result = self._client.setex(key, ttl, serialized_value)
            if result:
                self._stats['sets'] += 1
                return True
            return False
        
        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"Redis set error for key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from Redis cache."""
        if not self._is_connected or not self._client:
            return False
        
        try:
            result = self._client.delete(key)
            if result:
                self._stats['deletes'] += 1
                return True
            return False
        
        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"Redis delete error for key {key}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in Redis cache."""
        if not self._is_connected or not self._client:
            return False
        
        try:
            return bool(self._client.exists(key))
        
        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"Redis exists error for key {key}: {e}")
            return False
    
    def clear(self) -> bool:
        """Clear all Redis cache entries."""
        if not self._is_connected or not self._client:
            return False
        
        try:
            self._client.flushdb()
            logger.info("Redis cache cleared")
            return True
        
        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"Redis clear error: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Redis cache statistics."""
        stats = self._stats.copy()
        stats['is_connected'] = self._is_connected
        stats['hit_rate'] = safe_divide(stats['hits'], stats['hits'] + stats['misses']) * 100
        
        if self._is_connected and self._client:
            try:
                info = self._client.info('memory')
                stats['memory_used'] = info.get('used_memory', 0)
                stats['memory_peak'] = info.get('used_memory_peak', 0)
            except Exception:
                pass
        
        return stats
    
    def _serialize(self, value: Any) -> str:
        """Serialize value for storage."""
        try:
            if self.config.serialize_format == 'json':
                return json.dumps(value, default=str)
            elif self.config.serialize_format == 'pickle':
                import base64
                return base64.b64encode(pickle.dumps(value)).decode('utf-8')
            else:
                return str(value)
        except Exception as e:
            logger.error(f"Serialization error: {e}")
            return str(value)
    
    def _deserialize(self, value: str) -> Any:
        """Deserialize value from storage."""
        try:
            if self.config.serialize_format == 'json':
                return json.loads(value)
            elif self.config.serialize_format == 'pickle':
                import base64
                return pickle.loads(base64.b64decode(value.encode('utf-8')))
            else:
                return value
        except Exception as e:
            logger.error(f"Deserialization error: {e}")
            return value


class MemoryCacheBackend(CacheBackend):
    """In-memory cache backend with LRU eviction."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self._cache = {}
        self._access_times = {}
        self._current_size = 0
        self._stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'evictions': 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache."""
        if key in self._cache:
            entry = self._cache[key]
            
            # Check TTL
            if entry['expires_at'] and datetime.utcnow() > entry['expires_at']:
                self.delete(key)
                self._stats['misses'] += 1
                return None
            
            # Update access time for LRU
            self._access_times[key] = datetime.utcnow()
            self._stats['hits'] += 1
            return entry['value']
        
        self._stats['misses'] += 1
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in memory cache."""
        try:
            # Calculate value size (approximate)
            value_size = len(str(value).encode('utf-8'))
            
            # Check if we need to evict entries
            while (self._current_size + value_size > self.config.max_memory_size and 
                   len(self._cache) > 0):
                self._evict_lru()
            
            # Set expiration time
            expires_at = None
            if ttl:
                expires_at = datetime.utcnow() + timedelta(seconds=ttl)
            elif self.config.default_ttl:
                expires_at = datetime.utcnow() + timedelta(seconds=self.config.default_ttl)
            
            # Store entry
            if key in self._cache:
                self._current_size -= self._cache[key]['size']
            
            self._cache[key] = {
                'value': value,
                'expires_at': expires_at,
                'size': value_size
            }
            self._access_times[key] = datetime.utcnow()
            self._current_size += value_size
            self._stats['sets'] += 1
            
            return True
        
        except Exception as e:
            logger.error(f"Memory cache set error for key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from memory cache."""
        if key in self._cache:
            self._current_size -= self._cache[key]['size']
            del self._cache[key]
            del self._access_times[key]
            self._stats['deletes'] += 1
            return True
        return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in memory cache."""
        if key in self._cache:
            entry = self._cache[key]
            
            # Check TTL
            if entry['expires_at'] and datetime.utcnow() > entry['expires_at']:
                self.delete(key)
                return False
            
            return True
        return False
    
    def clear(self) -> bool:
        """Clear all memory cache entries."""
        self._cache.clear()
        self._access_times.clear()
        self._current_size = 0
        logger.info("Memory cache cleared")
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory cache statistics."""
        stats = self._stats.copy()
        stats['entries_count'] = len(self._cache)
        stats['current_size'] = self._current_size
        stats['max_size'] = self.config.max_memory_size
        stats['hit_rate'] = safe_divide(stats['hits'], stats['hits'] + stats['misses']) * 100
        stats['memory_utilization'] = safe_divide(self._current_size, self.config.max_memory_size) * 100
        
        return stats
    
    def _evict_lru(self):
        """Evict least recently used entry."""
        if not self._access_times:
            return
        
        # Find least recently used key
        lru_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
        self.delete(lru_key)
        self._stats['evictions'] += 1
        logger.debug(f"Evicted LRU entry: {lru_key}")


class FileCacheBackend(CacheBackend):
    """File-based cache backend for persistent storage."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache_dir = Path(config.file_cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'errors': 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from file cache."""
        try:
            file_path = self._get_file_path(key)
            
            if not file_path.exists():
                self._stats['misses'] += 1
                return None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check TTL
            if data.get('expires_at'):
                expires_at = datetime.fromisoformat(data['expires_at'])
                if datetime.utcnow() > expires_at:
                    self.delete(key)
                    self._stats['misses'] += 1
                    return None
            
            self._stats['hits'] += 1
            return data['value']
        
        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"File cache get error for key {key}: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in file cache."""
        try:
            file_path = self._get_file_path(key)
            
            # Set expiration time
            expires_at = None
            if ttl:
                expires_at = datetime.utcnow() + timedelta(seconds=ttl)
            elif self.config.default_ttl:
                expires_at = datetime.utcnow() + timedelta(seconds=self.config.default_ttl)
            
            data = {
                'value': value,
                'expires_at': expires_at.isoformat() if expires_at else None,
                'created_at': datetime.utcnow().isoformat()
            }
            
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, default=str)
            
            self._stats['sets'] += 1
            return True
        
        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"File cache set error for key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from file cache."""
        try:
            file_path = self._get_file_path(key)
            
            if file_path.exists():
                file_path.unlink()
                self._stats['deletes'] += 1
                return True
            return False
        
        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"File cache delete error for key {key}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in file cache."""
        try:
            file_path = self._get_file_path(key)
            
            if not file_path.exists():
                return False
            
            # Check TTL
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if data.get('expires_at'):
                expires_at = datetime.fromisoformat(data['expires_at'])
                if datetime.utcnow() > expires_at:
                    self.delete(key)
                    return False
            
            return True
        
        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"File cache exists error for key {key}: {e}")
            return False
    
    def clear(self) -> bool:
        """Clear all file cache entries."""
        try:
            for file_path in self.cache_dir.glob('**/*.json'):
                file_path.unlink()
            
            logger.info("File cache cleared")
            return True
        
        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"File cache clear error: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get file cache statistics."""
        stats = self._stats.copy()
        stats['hit_rate'] = safe_divide(stats['hits'], stats['hits'] + stats['misses']) * 100
        
        try:
            file_count = len(list(self.cache_dir.glob('**/*.json')))
            stats['file_count'] = file_count
            
            total_size = sum(f.stat().st_size for f in self.cache_dir.glob('**/*.json'))
            stats['total_size'] = total_size
        except Exception:
            stats['file_count'] = 0
            stats['total_size'] = 0
        
        return stats
    
    def _get_file_path(self, key: str) -> Path:
        """Get file path for cache key."""
        # Create hash of key for safe filename
        key_hash = hashlib.sha256(key.encode('utf-8')).hexdigest()
        
        # Create subdirectory structure based on hash prefix
        subdir = key_hash[:2]
        return self.cache_dir / subdir / f"{key_hash}.json"


class CacheManager:
    """Multi-tier cache manager with automatic fallback."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self._backends = []
        self._primary_backend = None
        self._stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'fallback_used': 0
        }
        
        self._initialize_backends()
    
    def _initialize_backends(self):
        """Initialize cache backends in priority order."""
        try:
            # Primary: Redis cache
            if REDIS_AVAILABLE:
                redis_backend = RedisCacheBackend(self.config)
                if redis_backend._is_connected:
                    self._backends.append(redis_backend)
                    self._primary_backend = redis_backend
                    logger.info("Redis cache backend enabled as primary")
            
            # Secondary: Memory cache
            memory_backend = MemoryCacheBackend(self.config)
            self._backends.append(memory_backend)
            
            if not self._primary_backend:
                self._primary_backend = memory_backend
                logger.info("Memory cache backend enabled as primary")
            
            # Tertiary: File cache
            file_backend = FileCacheBackend(self.config)
            self._backends.append(file_backend)
            logger.info("File cache backend enabled as fallback")
            
        except Exception as e:
            logger.error(f"Error initializing cache backends: {e}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with fallback."""
        self._stats['total_requests'] += 1
        
        for i, backend in enumerate(self._backends):
            try:
                value = backend.get(key)
                if value is not None:
                    self._stats['cache_hits'] += 1
                    
                    # Populate higher-priority backends
                    for j in range(i):
                        try:
                            self._backends[j].set(key, value)
                        except Exception:
                            pass
                    
                    return value
            
            except Exception as e:
                logger.warning(f"Cache backend {type(backend).__name__} failed: {e}")
                if i > 0:
                    self._stats['fallback_used'] += 1
                continue
        
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in all available cache backends."""
        success = False
        
        for backend in self._backends:
            try:
                if backend.set(key, value, ttl):
                    success = True
            except Exception as e:
                logger.warning(f"Cache backend {type(backend).__name__} set failed: {e}")
                continue
        
        return success
    
    def delete(self, key: str) -> bool:
        """Delete key from all cache backends."""
        success = False
        
        for backend in self._backends:
            try:
                if backend.delete(key):
                    success = True
            except Exception as e:
                logger.warning(f"Cache backend {type(backend).__name__} delete failed: {e}")
                continue
        
        return success
    
    def exists(self, key: str) -> bool:
        """Check if key exists in any cache backend."""
        for backend in self._backends:
            try:
                if backend.exists(key):
                    return True
            except Exception as e:
                logger.warning(f"Cache backend {type(backend).__name__} exists failed: {e}")
                continue
        
        return False
    
    def clear(self) -> bool:
        """Clear all cache backends."""
        success = False
        
        for backend in self._backends:
            try:
                if backend.clear():
                    success = True
            except Exception as e:
                logger.warning(f"Cache backend {type(backend).__name__} clear failed: {e}")
                continue
        
        return success
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = self._stats.copy()
        stats['hit_rate'] = safe_divide(stats['cache_hits'], stats['total_requests']) * 100
        stats['backends'] = {}
        
        for backend in self._backends:
            backend_name = type(backend).__name__
            try:
                stats['backends'][backend_name] = backend.get_stats()
            except Exception:
                stats['backends'][backend_name] = {'error': 'Failed to get stats'}
        
        return stats
    
    def cached(self, key: Optional[str] = None, ttl: Optional[int] = None):
        """Decorator for caching function results."""
        def decorator(func: Callable):
            def wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = key or generate_cache_key(func.__name__, args, kwargs)
                
                # Try to get from cache
                cached_result = self.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Execute function and cache result
                result = func(*args, **kwargs)
                self.set(cache_key, result, ttl)
                
                return result
            
            return wrapper
        return decorator


# Global cache manager instance
_cache_manager = None

def create_cache_manager(config: Optional[CacheConfig] = None) -> CacheManager:
    """Create and initialize cache manager."""
    global _cache_manager
    
    if config is None:
        config = CacheConfig.from_env()
    
    _cache_manager = CacheManager(config)
    return _cache_manager

def get_cache_instance() -> Optional[CacheManager]:
    """Get global cache manager instance."""
    return _cache_manager