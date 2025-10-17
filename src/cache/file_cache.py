"""
File-based cache system for when Redis/PostgreSQL are unavailable.
Provides persistent local storage with weekly refresh and fallback capabilities.
"""
import json
import os
import logging
from datetime import datetime, timedelta
from typing import Any, Optional, Dict
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)


class FileCache:
    """
    File-based cache implementation with weekly refresh and fallback support.
    
    Features:
    - Local JSON file storage
    - Weekly automatic refresh
    - Fallback to last known good data
    - TTL support
    - Graceful degradation
    """
    
    def __init__(self, cache_dir: str = "data/cache"):
        """
        Initialize file cache.
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Separate directories for different cache types
        self.data_dir = self.cache_dir / "data"
        self.metadata_dir = self.cache_dir / "metadata"
        self.fallback_dir = self.cache_dir / "fallback"
        
        for dir_path in [self.data_dir, self.metadata_dir, self.fallback_dir]:
            dir_path.mkdir(exist_ok=True)
        
        logger.info(f"File cache initialized at: {self.cache_dir}")
    
    def _get_file_path(self, key: str, cache_type: str = "data") -> Path:
        """Get file path for a cache key."""
        # Create safe filename from key
        safe_key = hashlib.md5(key.encode()).hexdigest()
        
        if cache_type == "data":
            return self.data_dir / f"{safe_key}.json"
        elif cache_type == "metadata":
            return self.metadata_dir / f"{safe_key}_meta.json"
        elif cache_type == "fallback":
            return self.fallback_dir / f"{safe_key}_fallback.json"
        else:
            raise ValueError(f"Unknown cache type: {cache_type}")
    
    def _save_metadata(self, key: str, metadata: Dict[str, Any]) -> None:
        """Save metadata for a cache entry."""
        meta_path = self._get_file_path(key, "metadata")
        try:
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, default=str)
        except Exception as e:
            logger.error(f"Failed to save metadata for {key}: {e}")
    
    def _load_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """Load metadata for a cache entry."""
        meta_path = self._get_file_path(key, "metadata")
        try:
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load metadata for {key}: {e}")
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in file cache with optional TTL.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (default: 7 days)
            
        Returns:
            True if successful, False otherwise
        """
        if ttl is None:
            ttl = 7 * 24 * 3600  # 1 week default
        
        data_path = self._get_file_path(key, "data")
        
        try:
            # Prepare data
            cache_data = {
                "key": key,
                "value": value,
                "timestamp": datetime.utcnow().isoformat(),
                "ttl": ttl
            }
            
            # Save data
            with open(data_path, 'w') as f:
                json.dump(cache_data, f, default=str, indent=2)
            
            # Save metadata
            metadata = {
                "key": key,
                "created": datetime.utcnow().isoformat(),
                "expires": (datetime.utcnow() + timedelta(seconds=ttl)).isoformat(),
                "size": data_path.stat().st_size if data_path.exists() else 0,
                "source": self._extract_source_from_key(key)
            }
            self._save_metadata(key, metadata)
            
            # Also save as fallback data
            self._save_fallback(key, value)
            
            logger.debug(f"Cached data for key: {key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache data for {key}: {e}")
            return False
    
    def get(self, key: str, use_fallback: bool = True) -> Optional[Any]:
        """
        Get value from file cache with fallback support.
        
        Args:
            key: Cache key
            use_fallback: Whether to use fallback data if main cache is expired
            
        Returns:
            Cached value or None if not found
        """
        data_path = self._get_file_path(key, "data")
        
        try:
            # Try to load from main cache
            if data_path.exists():
                with open(data_path, 'r') as f:
                    cache_data = json.load(f)
                
                # Check if data is still valid
                timestamp = datetime.fromisoformat(cache_data["timestamp"])
                ttl = cache_data.get("ttl", 7 * 24 * 3600)
                expires = timestamp + timedelta(seconds=ttl)
                
                if datetime.utcnow() <= expires:
                    logger.debug(f"Cache hit for key: {key}")
                    return cache_data["value"]
                else:
                    logger.debug(f"Cache expired for key: {key}")
                    
                    # If fallback is enabled, try fallback data
                    if use_fallback:
                        fallback_value = self._get_fallback(key)
                        if fallback_value is not None:
                            logger.warning(f"Using fallback data for key: {key}")
                            return fallback_value
            
            # Try fallback if main cache doesn't exist
            if use_fallback:
                fallback_value = self._get_fallback(key)
                if fallback_value is not None:
                    logger.warning(f"No main cache, using fallback for key: {key}")
                    return fallback_value
            
            logger.debug(f"Cache miss for key: {key}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to get cached data for {key}: {e}")
            
            # Try fallback on error
            if use_fallback:
                fallback_value = self._get_fallback(key)
                if fallback_value is not None:
                    logger.warning(f"Error in main cache, using fallback for key: {key}")
                    return fallback_value
            
            return None
    
    def _save_fallback(self, key: str, value: Any) -> None:
        """Save fallback data for a key."""
        fallback_path = self._get_file_path(key, "fallback")
        
        try:
            fallback_data = {
                "key": key,
                "value": value,
                "timestamp": datetime.utcnow().isoformat(),
                "source": self._extract_source_from_key(key)
            }
            
            with open(fallback_path, 'w') as f:
                json.dump(fallback_data, f, default=str, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save fallback data for {key}: {e}")
    
    def _get_fallback(self, key: str, max_age_days: int = 30) -> Optional[Any]:
        """Get fallback data for a key."""
        fallback_path = self._get_file_path(key, "fallback")
        
        try:
            if fallback_path.exists():
                with open(fallback_path, 'r') as f:
                    fallback_data = json.load(f)
                
                # Check age
                timestamp = datetime.fromisoformat(fallback_data["timestamp"])
                age = datetime.utcnow() - timestamp
                
                if age.days <= max_age_days:
                    return fallback_data["value"]
                else:
                    logger.debug(f"Fallback data too old for key: {key} (age: {age.days} days)")
                    
        except Exception as e:
            logger.error(f"Failed to get fallback data for {key}: {e}")
        
        return None
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        data_path = self._get_file_path(key, "data")
        fallback_path = self._get_file_path(key, "fallback")
        return data_path.exists() or fallback_path.exists()
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        success = True
        
        for cache_type in ["data", "metadata", "fallback"]:
            file_path = self._get_file_path(key, cache_type)
            try:
                if file_path.exists():
                    file_path.unlink()
            except Exception as e:
                logger.error(f"Failed to delete {cache_type} file for {key}: {e}")
                success = False
        
        return success
    
    def cleanup_expired(self) -> int:
        """Clean up expired cache entries."""
        cleaned = 0
        
        # Clean up expired main cache files
        for data_file in self.data_dir.glob("*.json"):
            try:
                with open(data_file, 'r') as f:
                    cache_data = json.load(f)
                
                timestamp = datetime.fromisoformat(cache_data["timestamp"])
                ttl = cache_data.get("ttl", 7 * 24 * 3600)
                expires = timestamp + timedelta(seconds=ttl)
                
                if datetime.utcnow() > expires:
                    data_file.unlink()
                    
                    # Also remove metadata
                    key = cache_data["key"]
                    meta_path = self._get_file_path(key, "metadata")
                    if meta_path.exists():
                        meta_path.unlink()
                    
                    cleaned += 1
                    
            except Exception as e:
                logger.error(f"Error cleaning up {data_file}: {e}")
        
        # Clean up old fallback files (older than 30 days)
        cutoff = datetime.utcnow() - timedelta(days=30)
        for fallback_file in self.fallback_dir.glob("*.json"):
            try:
                with open(fallback_file, 'r') as f:
                    fallback_data = json.load(f)
                
                timestamp = datetime.fromisoformat(fallback_data["timestamp"])
                if timestamp < cutoff:
                    fallback_file.unlink()
                    cleaned += 1
                    
            except Exception as e:
                logger.error(f"Error cleaning up fallback {fallback_file}: {e}")
        
        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} expired cache entries")
        
        return cleaned
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            data_files = list(self.data_dir.glob("*.json"))
            fallback_files = list(self.fallback_dir.glob("*.json"))
            
            total_size = sum(f.stat().st_size for f in data_files + fallback_files)
            
            return {
                "cache_entries": len(data_files),
                "fallback_entries": len(fallback_files),
                "total_size_bytes": total_size,
                "cache_dir": str(self.cache_dir),
                "available": True
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"available": False, "error": str(e)}
    
    def is_available(self) -> bool:
        """Check if cache is available."""
        try:
            # Test write access
            test_file = self.cache_dir / "test.tmp"
            test_file.write_text("test")
            test_file.unlink()
            return True
        except Exception:
            return False
    
    def flush_all(self) -> bool:
        """Clear all cache data."""
        try:
            # Remove all files in all directories
            for dir_path in [self.data_dir, self.metadata_dir, self.fallback_dir]:
                for file_path in dir_path.glob("*.json"):
                    file_path.unlink()
            
            logger.warning("All file cache data flushed")
            return True
        except Exception as e:
            logger.error(f"Failed to flush file cache: {e}")
            return False
    
    def _extract_source_from_key(self, key: str) -> str:
        """Extract data source name from cache key."""
        parts = key.split(":")
        return parts[0] if parts else "unknown"
    
    def get_cache_age(self, key: str) -> Optional[timedelta]:
        """Get age of cached data."""
        metadata = self._load_metadata(key)
        if metadata and "created" in metadata:
            created = datetime.fromisoformat(metadata["created"])
            return datetime.utcnow() - created
        return None
    
    def refresh_if_needed(self, key: str, refresh_callback) -> bool:
        """
        Refresh cache entry if it's older than a week.
        
        Args:
            key: Cache key to check
            refresh_callback: Function to call to get fresh data
            
        Returns:
            True if refresh was attempted, False if not needed
        """
        age = self.get_cache_age(key)
        if age is None or age.days >= 7:
            try:
                fresh_data = refresh_callback()
                if fresh_data is not None:
                    self.set(key, fresh_data)
                    logger.info(f"Refreshed cache for key: {key}")
                    return True
            except Exception as e:
                logger.error(f"Failed to refresh cache for {key}: {e}")
        
        return False