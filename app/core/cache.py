import os
import json
import time
import hashlib
import logging
from pathlib import Path
from typing import Any, Optional, Dict, Tuple
from datetime import datetime, timedelta
import redis

class RedisCache:
    """L1 Redis cache with TTL management and stale-while-revalidate support."""
    
    def __init__(self, namespace: str) -> None:
        url = os.getenv('RIS_REDIS_URL')
        if not url:
            raise RuntimeError("Redis URL missing; set RIS_REDIS_URL environment variable")
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


class FileCache:
    """File-based cache implementation for reliability when Redis unavailable."""
    
    def __init__(self, namespace: str):
        self.namespace = namespace
        cache_dir = Path("cache") / namespace
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = cache_dir

    def _key_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def get(self, key: str) -> Optional[Any]:
        """Get cached value if exists and not expired."""
        try:
            key_path = self._key_path(key)
            if not key_path.exists():
                return None
            
            with open(key_path, 'r') as f:
                data = json.load(f)
            
            # Check expiration
            if data.get('expires_at', 0) < time.time():
                key_path.unlink(missing_ok=True)
                return None
            
            return data.get('value')
        except Exception:
            return None

    def set(self, key: str, value: Any, ttl: int = 900) -> None:
        """Set cached value with TTL."""
        try:
            key_path = self._key_path(key)
            data = {
                'value': value,
                'expires_at': time.time() + ttl,
                'created_at': time.time()
            }
            
            with open(key_path, 'w') as f:
                json.dump(data, f)
        except Exception:
            pass  # Fail silently for cache errors
