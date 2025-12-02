"""
Supply Chain Specific Caching Configuration

Provides optimized caching strategies for different types of supply chain data
with appropriate TTL settings, cache keys, and invalidation patterns.
"""

import logging
import hashlib
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from .unified_cache import UnifiedCache
from .cache import RedisCache

logger = logging.getLogger(__name__)

@dataclass
class CachePolicy:
    """Defines caching behavior for different data types."""
    soft_ttl: int  # Stale-while-revalidate threshold (seconds)
    hard_ttl: int  # Maximum staleness allowed (seconds) 
    compression: bool = False  # Enable compression for large datasets
    encryption: bool = False  # Enable encryption for sensitive data
    priority: str = "normal"  # Cache eviction priority (low, normal, high, critical)

class SupplyChainCache:
    """Enhanced caching specifically optimized for supply chain data patterns."""
    
    # Cache policies for different data types
    CACHE_POLICIES = {
        # Real-time feeds - require frequent updates
        'cascade_events': CachePolicy(soft_ttl=300, hard_ttl=900),  # 5min/15min
        'realtime_alerts': CachePolicy(soft_ttl=60, hard_ttl=300),  # 1min/5min
        'port_congestion': CachePolicy(soft_ttl=1800, hard_ttl=3600),  # 30min/1hour
        'geopolitical_events': CachePolicy(soft_ttl=600, hard_ttl=1800),  # 10min/30min
        
        # Market intelligence - moderate frequency
        'sp_global_risk': CachePolicy(soft_ttl=3600, hard_ttl=14400),  # 1hour/4hours
        'trade_flows': CachePolicy(soft_ttl=7200, hard_ttl=86400),  # 2hours/1day
        'supplier_assessments': CachePolicy(soft_ttl=14400, hard_ttl=86400),  # 4hours/1day
        
        # Analytics and reports - less frequent updates
        'sector_vulnerability': CachePolicy(soft_ttl=86400, hard_ttl=604800),  # 1day/1week
        'resilience_metrics': CachePolicy(soft_ttl=43200, hard_ttl=259200),  # 12hours/3days
        'timeline_analysis': CachePolicy(soft_ttl=3600, hard_ttl=14400),  # 1hour/4hours
        
        # Network topology - rarely changes
        'supply_chain_nodes': CachePolicy(soft_ttl=86400, hard_ttl=604800, priority="high"),  # 1day/1week
        'relationships': CachePolicy(soft_ttl=86400, hard_ttl=604800, priority="high"),  # 1day/1week
        'critical_paths': CachePolicy(soft_ttl=43200, hard_ttl=259200),  # 12hours/3days
        
        # External API data - depends on source refresh rates
        'acled_data': CachePolicy(soft_ttl=7200, hard_ttl=43200),  # 2hours/12hours
        'wits_data': CachePolicy(soft_ttl=86400, hard_ttl=604800),  # 1day/1week (monthly updates)
        'wto_statistics': CachePolicy(soft_ttl=86400, hard_ttl=604800),  # 1day/1week (quarterly updates)
        
        # Machine learning models and predictions
        'ml_predictions': CachePolicy(soft_ttl=1800, hard_ttl=7200),  # 30min/2hours
        'model_metadata': CachePolicy(soft_ttl=3600, hard_ttl=86400),  # 1hour/1day
        'training_results': CachePolicy(soft_ttl=86400, hard_ttl=604800, priority="high"),  # 1day/1week
    }
    
    def __init__(self, namespace: str = "supply_chain"):
        self.namespace = namespace
        self.cache = UnifiedCache(namespace)
        self.redis = RedisCache(namespace)
        
    def get_policy(self, data_type: str) -> CachePolicy:
        """Get cache policy for a specific data type."""
        return self.CACHE_POLICIES.get(data_type, CachePolicy(soft_ttl=3600, hard_ttl=86400))
    
    def generate_key(self, data_type: str, identifier: str, **kwargs) -> str:
        """Generate cache key with consistent format and optional parameters."""
        # Create base key
        key_parts = [data_type, identifier]
        
        # Add optional parameters in sorted order for consistency
        if kwargs:
            param_string = "&".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
            # Hash long parameter strings to avoid key length issues
            if len(param_string) > 100:
                param_hash = hashlib.md5(param_string.encode()).hexdigest()[:16]
                key_parts.append(f"params_{param_hash}")
            else:
                key_parts.append(param_string)
        
        return ":".join(key_parts)
    
    def get(self, data_type: str, identifier: str, **kwargs) -> Tuple[Optional[Any], Optional[Dict]]:
        """Get cached data with automatic policy application."""
        key = self.generate_key(data_type, identifier, **kwargs)
        return self.cache.get(key)
    
    def set(self, data_type: str, identifier: str, value: Any, 
            source: str, source_url: str = "", **kwargs) -> None:
        """Set cached data with automatic policy application."""
        policy = self.get_policy(data_type)
        key = self.generate_key(data_type, identifier, **kwargs)
        
        # Apply compression for large datasets
        if policy.compression and isinstance(value, (list, dict)):
            # Add compression metadata
            pass  # Implement compression if needed
        
        self.cache.set(
            key=key,
            value=value,
            source=source,
            source_url=source_url,
            soft_ttl=policy.soft_ttl,
            hard_ttl=policy.hard_ttl
        )
    
    def invalidate(self, data_type: str, identifier: str = "*", **kwargs) -> int:
        """Invalidate cached data by pattern."""
        if identifier == "*":
            pattern = f"{self.namespace}:{data_type}:*"
        else:
            key = self.generate_key(data_type, identifier, **kwargs)
            pattern = f"{self.namespace}:{key}"
        
        return self._invalidate_pattern(pattern)
    
    def _invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching a pattern."""
        if not self.redis.available:
            return 0
        
        try:
            # Get all keys matching pattern
            keys = self.redis.client.keys(pattern)
            if keys:
                # Delete keys and their metadata
                meta_keys = [key.replace(f"{self.namespace}:", f"{self.namespace}:meta:") 
                           for key in keys]
                all_keys = keys + meta_keys
                return self.redis.client.delete(*all_keys)
            return 0
        except Exception as e:
            logger.error(f"Cache invalidation failed: {e}")
            return 0
    
    def get_stale_keys(self, data_type: Optional[str] = None) -> List[str]:
        """Get stale keys for background refresh, optionally filtered by data type."""
        if not self.redis.available:
            return []
        
        try:
            # Get all metadata keys
            if data_type:
                pattern = f"{self.namespace}:meta:{data_type}:*"
            else:
                pattern = f"{self.namespace}:meta:*"
            
            meta_keys = self.redis.client.keys(pattern)
            stale_keys = []
            
            for meta_key in meta_keys:
                try:
                    metadata = self.redis.client.hgetall(meta_key)
                    if not metadata:
                        continue
                    
                    cached_at = datetime.fromisoformat(metadata.get(b'cached_at', '').decode())
                    soft_ttl = int(metadata.get(b'soft_ttl', 0))
                    
                    # Check if soft TTL exceeded
                    if datetime.utcnow() > cached_at + timedelta(seconds=soft_ttl):
                        # Extract original key from metadata key
                        original_key = meta_key.decode().replace(f"{self.namespace}:meta:", "")
                        stale_keys.append(original_key)
                        
                except Exception as e:
                    logger.debug(f"Error checking staleness for {meta_key}: {e}")
                    continue
            
            return stale_keys
            
        except Exception as e:
            logger.error(f"Failed to get stale keys: {e}")
            return []
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        if not self.redis.available:
            return {"status": "redis_unavailable"}
        
        try:
            info = self.redis.client.info()
            
            # Get namespace-specific statistics
            namespace_keys = self.redis.client.keys(f"{self.namespace}:*")
            meta_keys = [key for key in namespace_keys if b":meta:" in key]
            data_keys = [key for key in namespace_keys if b":meta:" not in key]
            
            # Analyze cache policies usage
            policy_usage = {}
            for key in data_keys:
                try:
                    key_str = key.decode()
                    data_type = key_str.split(":")[2] if ":" in key_str else "unknown"
                    policy_usage[data_type] = policy_usage.get(data_type, 0) + 1
                except:
                    continue
            
            return {
                "status": "healthy",
                "redis_info": {
                    "used_memory": info.get("used_memory_human"),
                    "connected_clients": info.get("connected_clients"),
                    "total_commands_processed": info.get("total_commands_processed"),
                    "keyspace_hits": info.get("keyspace_hits", 0),
                    "keyspace_misses": info.get("keyspace_misses", 0),
                },
                "namespace_stats": {
                    "total_keys": len(namespace_keys),
                    "data_keys": len(data_keys),
                    "metadata_keys": len(meta_keys),
                    "policy_usage": policy_usage,
                },
                "hit_rate": self._calculate_hit_rate(info),
                "stale_keys_count": len(self.get_stale_keys()),
            }
            
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {"status": "error", "error": str(e)}
    
    def _calculate_hit_rate(self, info: Dict) -> float:
        """Calculate cache hit rate percentage."""
        hits = info.get("keyspace_hits", 0)
        misses = info.get("keyspace_misses", 0)
        total = hits + misses
        return (hits / total * 100) if total > 0 else 0.0
    
    def optimize_cache(self) -> Dict[str, Any]:
        """Perform cache optimization and cleanup."""
        if not self.redis.available:
            return {"status": "redis_unavailable"}
        
        try:
            # Remove expired keys
            expired_removed = self._cleanup_expired_keys()
            
            # Identify and log frequently accessed data
            frequent_keys = self._analyze_access_patterns()
            
            # Suggest policy adjustments
            suggestions = self._suggest_policy_optimizations()
            
            return {
                "status": "completed",
                "expired_keys_removed": expired_removed,
                "frequent_access_patterns": frequent_keys,
                "optimization_suggestions": suggestions,
            }
            
        except Exception as e:
            logger.error(f"Cache optimization failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def _cleanup_expired_keys(self) -> int:
        """Remove expired keys and orphaned metadata."""
        # Redis handles TTL expiration automatically, but we clean up orphaned metadata
        removed = 0
        try:
            meta_keys = self.redis.client.keys(f"{self.namespace}:meta:*")
            for meta_key in meta_keys:
                # Check if corresponding data key exists
                data_key = meta_key.decode().replace(":meta:", ":")
                if not self.redis.client.exists(data_key):
                    self.redis.client.delete(meta_key)
                    removed += 1
        except Exception as e:
            logger.debug(f"Cleanup error: {e}")
        
        return removed
    
    def _analyze_access_patterns(self) -> List[str]:
        """Analyze which cache keys are accessed most frequently."""
        # This would require additional tracking; simplified for now
        return []
    
    def _suggest_policy_optimizations(self) -> List[str]:
        """Suggest cache policy optimizations based on usage patterns."""
        suggestions = []
        
        try:
            stale_keys = self.get_stale_keys()
            if len(stale_keys) > 10:
                suggestions.append("Consider increasing soft_ttl for frequently stale data types")
            
            stats = self.get_cache_stats()
            hit_rate = stats.get("hit_rate", 0)
            if hit_rate < 70:
                suggestions.append("Low hit rate detected - consider increasing TTL values")
            
            # Add more optimization suggestions based on patterns
            
        except Exception as e:
            logger.debug(f"Analysis error: {e}")
        
        return suggestions


# Singleton instance for application-wide use
supply_chain_cache = SupplyChainCache()

# Convenience functions for common operations
def get_supply_chain_cache() -> SupplyChainCache:
    """Get the global supply chain cache instance."""
    return supply_chain_cache

def cache_cascade_event(event_id: str, data: Any, source: str = "cascade_service") -> None:
    """Cache cascade event data."""
    supply_chain_cache.set("cascade_events", event_id, data, source)

def get_cached_cascade_event(event_id: str) -> Tuple[Optional[Any], Optional[Dict]]:
    """Get cached cascade event data."""
    return supply_chain_cache.get("cascade_events", event_id)

def cache_sector_vulnerability(sector: str, data: Any, source: str = "vulnerability_service") -> None:
    """Cache sector vulnerability assessment."""
    supply_chain_cache.set("sector_vulnerability", sector, data, source)

def get_cached_sector_vulnerability(sector: str) -> Tuple[Optional[Any], Optional[Dict]]:
    """Get cached sector vulnerability assessment."""
    return supply_chain_cache.get("sector_vulnerability", sector)

def cache_sp_global_data(entity_id: str, data: Any, source: str = "sp_global") -> None:
    """Cache S&P Global intelligence data."""
    supply_chain_cache.set("sp_global_risk", entity_id, data, source)

def get_cached_sp_global_data(entity_id: str) -> Tuple[Optional[Any], Optional[Dict]]:
    """Get cached S&P Global intelligence data."""
    return supply_chain_cache.get("sp_global_risk", entity_id)

def invalidate_all_cache() -> int:
    """Invalidate all supply chain cache data."""
    return supply_chain_cache._invalidate_pattern(f"{supply_chain_cache.namespace}:*")