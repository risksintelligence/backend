"""
PostgreSQL cache implementation for fallback caching and historical data.
"""
import json
import logging
from typing import Any, Optional, Dict
from datetime import datetime, timedelta

from sqlalchemy import create_engine, Column, String, DateTime, Text, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

from src.core.config import settings

logger = logging.getLogger(__name__)

Base = declarative_base()


class CacheEntry(Base):
    """Cache entry model for PostgreSQL storage."""
    
    __tablename__ = "cache_entries"
    
    key = Column(String(255), primary_key=True)
    value = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
    access_count = Column(Integer, default=0)
    last_accessed = Column(DateTime, default=datetime.utcnow)


class PostgresCache:
    """PostgreSQL-based cache for fallback and historical data."""
    
    def __init__(self):
        """Initialize PostgreSQL cache connection."""
        self._engine = None
        self._session_factory = None
        self._connect()
    
    def _connect(self) -> None:
        """Establish PostgreSQL connection."""
        try:
            # Use effective database URL which handles environment overrides properly
            database_url = settings.effective_database_url
            
            # Skip connection if using default SQLite for development
            if "sqlite" in database_url:
                logger.info("Using SQLite database - skipping PostgreSQL cache connection")
                return
            
            # Different connect_args for SQLite vs PostgreSQL
            connect_args = {"connect_timeout": 10}
            
            self._engine = create_engine(
                database_url,
                pool_pre_ping=True,
                pool_recycle=3600,
                connect_args=connect_args
            )
            self._session_factory = sessionmaker(bind=self._engine)
            
            # Create tables if they don't exist
            Base.metadata.create_all(self._engine)
            
            logger.info("PostgreSQL cache connected successfully")
        except SQLAlchemyError as e:
            logger.error(f"Failed to connect to PostgreSQL cache: {e}")
            self._engine = None
            self._session_factory = None
    
    def is_available(self) -> bool:
        """Check if PostgreSQL cache is available."""
        if not self._engine or not self._session_factory:
            return False
        
        try:
            with self._session_factory() as session:
                session.execute("SELECT 1")
                return True
        except SQLAlchemyError:
            logger.warning("PostgreSQL cache connection lost")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from PostgreSQL cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        if not self.is_available():
            return None
        
        try:
            with self._session_factory() as session:
                entry = session.query(CacheEntry).filter(
                    CacheEntry.key == key
                ).first()
                
                if not entry:
                    return None
                
                # Check expiration
                if entry.expires_at and entry.expires_at < datetime.utcnow():
                    # Clean up expired entry
                    session.delete(entry)
                    session.commit()
                    return None
                
                # Update access statistics
                entry.access_count += 1
                entry.last_accessed = datetime.utcnow()
                session.commit()
                
                # Deserialize value
                try:
                    return json.loads(entry.value)
                except (json.JSONDecodeError, TypeError):
                    return entry.value
                    
        except SQLAlchemyError as e:
            logger.error(f"Error getting cache key {key} from PostgreSQL: {e}")
            return None
    
    def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set value in PostgreSQL cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            return False
        
        try:
            # Serialize value
            if not isinstance(value, str):
                value = json.dumps(value, default=str)
            
            expires_at = None
            if ttl:
                expires_at = datetime.utcnow() + timedelta(seconds=ttl)
            
            with self._session_factory() as session:
                # Check if entry exists
                entry = session.query(CacheEntry).filter(
                    CacheEntry.key == key
                ).first()
                
                if entry:
                    # Update existing entry
                    entry.value = value
                    entry.expires_at = expires_at
                    entry.last_accessed = datetime.utcnow()
                else:
                    # Create new entry
                    entry = CacheEntry(
                        key=key,
                        value=value,
                        expires_at=expires_at
                    )
                    session.add(entry)
                
                session.commit()
                return True
                
        except SQLAlchemyError as e:
            logger.error(f"Error setting cache key {key} in PostgreSQL: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """
        Delete key from PostgreSQL cache.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            return False
        
        try:
            with self._session_factory() as session:
                deleted = session.query(CacheEntry).filter(
                    CacheEntry.key == key
                ).delete()
                session.commit()
                return deleted > 0
                
        except SQLAlchemyError as e:
            logger.error(f"Error deleting cache key {key} from PostgreSQL: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """
        Check if key exists in PostgreSQL cache.
        
        Args:
            key: Cache key to check
            
        Returns:
            True if key exists and not expired, False otherwise
        """
        if not self.is_available():
            return False
        
        try:
            with self._session_factory() as session:
                entry = session.query(CacheEntry).filter(
                    CacheEntry.key == key
                ).first()
                
                if not entry:
                    return False
                
                # Check expiration
                if entry.expires_at and entry.expires_at < datetime.utcnow():
                    return False
                
                return True
                
        except SQLAlchemyError as e:
            logger.error(f"Error checking cache key {key} in PostgreSQL: {e}")
            return False
    
    def cleanup_expired(self) -> int:
        """
        Clean up expired cache entries.
        
        Returns:
            Number of entries cleaned up
        """
        if not self.is_available():
            return 0
        
        try:
            with self._session_factory() as session:
                deleted = session.query(CacheEntry).filter(
                    CacheEntry.expires_at < datetime.utcnow()
                ).delete()
                session.commit()
                
                if deleted > 0:
                    logger.info(f"Cleaned up {deleted} expired cache entries")
                
                return deleted
                
        except SQLAlchemyError as e:
            logger.error(f"Error cleaning up expired cache entries: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get PostgreSQL cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        if not self.is_available():
            return {"status": "unavailable"}
        
        try:
            with self._session_factory() as session:
                total_entries = session.query(CacheEntry).count()
                
                expired_entries = session.query(CacheEntry).filter(
                    CacheEntry.expires_at < datetime.utcnow()
                ).count()
                
                return {
                    "status": "connected",
                    "total_entries": total_entries,
                    "expired_entries": expired_entries,
                    "active_entries": total_entries - expired_entries,
                }
                
        except SQLAlchemyError as e:
            logger.error(f"Error getting PostgreSQL cache stats: {e}")
            return {"status": "error", "error": str(e)}
    
    def flush_all(self) -> bool:
        """
        Clear all cache data.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            return False
        
        try:
            with self._session_factory() as session:
                deleted = session.query(CacheEntry).delete()
                session.commit()
                logger.info(f"PostgreSQL cache flushed: {deleted} entries removed")
                return True
                
        except SQLAlchemyError as e:
            logger.error(f"Error flushing PostgreSQL cache: {e}")
            return False