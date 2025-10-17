"""
Fallback handler for graceful degradation when data sources are unavailable.
"""
import logging
from typing import Any, Optional, Dict, List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class FallbackHandler:
    """Handles graceful degradation when primary data sources fail."""
    
    def __init__(self):
        """Initialize fallback handler."""
        self._fallback_data = {}
        self._last_known_good = {}
        self._service_status = {}
    
    def register_fallback_data(
        self, 
        source: str, 
        data: Any, 
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Register fallback data for a specific source.
        
        Args:
            source: Data source identifier
            data: Fallback data to use when source is unavailable
            timestamp: When this data was last updated
        """
        timestamp = timestamp or datetime.utcnow()
        
        self._fallback_data[source] = {
            "data": data,
            "timestamp": timestamp,
            "is_fallback": True
        }
        
        logger.info(f"Registered fallback data for source: {source}")
    
    def set_last_known_good(
        self, 
        source: str, 
        data: Any, 
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Set last known good data for a source.
        
        Args:
            source: Data source identifier
            data: Last successful data from source
            timestamp: When this data was retrieved
        """
        timestamp = timestamp or datetime.utcnow()
        
        self._last_known_good[source] = {
            "data": data,
            "timestamp": timestamp,
            "is_fallback": False
        }
        
        # Update service status to healthy
        self._service_status[source] = {
            "status": "healthy",
            "last_success": timestamp,
            "failure_count": 0
        }
    
    def record_failure(self, source: str, error: str) -> None:
        """
        Record a failure for a data source.
        
        Args:
            source: Data source identifier
            error: Error description
        """
        if source not in self._service_status:
            self._service_status[source] = {
                "status": "healthy",
                "failure_count": 0
            }
        
        status = self._service_status[source]
        status["failure_count"] += 1
        status["last_failure"] = datetime.utcnow()
        status["last_error"] = error
        
        # Mark as degraded after 3 failures
        if status["failure_count"] >= 3:
            status["status"] = "degraded"
            logger.warning(f"Source {source} marked as degraded after {status['failure_count']} failures")
        
        # Mark as failed after 10 failures
        if status["failure_count"] >= 10:
            status["status"] = "failed"
            logger.error(f"Source {source} marked as failed after {status['failure_count']} failures")
    
    def get_fallback_data(
        self, 
        source: str, 
        max_age_hours: int = 24
    ) -> Optional[Dict[str, Any]]:
        """
        Get fallback data for a source.
        
        Args:
            source: Data source identifier
            max_age_hours: Maximum age of fallback data in hours
            
        Returns:
            Fallback data with metadata or None if not available
        """
        # Try last known good data first
        if source in self._last_known_good:
            data_info = self._last_known_good[source]
            age = datetime.utcnow() - data_info["timestamp"]
            
            if age.total_seconds() / 3600 <= max_age_hours:
                logger.info(f"Using last known good data for {source} (age: {age})")
                return data_info
        
        # Fall back to registered fallback data
        if source in self._fallback_data:
            data_info = self._fallback_data[source]
            age = datetime.utcnow() - data_info["timestamp"]
            
            logger.warning(f"Using fallback data for {source} (age: {age})")
            return data_info
        
        logger.error(f"No fallback data available for source: {source}")
        return None
    
    def is_source_healthy(self, source: str) -> bool:
        """
        Check if a data source is healthy.
        
        Args:
            source: Data source identifier
            
        Returns:
            True if source is healthy, False otherwise
        """
        if source not in self._service_status:
            return True  # Assume healthy if no failures recorded
        
        return self._service_status[source]["status"] == "healthy"
    
    def get_source_status(self, source: str) -> Dict[str, Any]:
        """
        Get detailed status for a data source.
        
        Args:
            source: Data source identifier
            
        Returns:
            Dictionary with source status information
        """
        if source not in self._service_status:
            return {
                "status": "unknown",
                "failure_count": 0,
                "has_fallback": source in self._fallback_data,
                "has_last_known_good": source in self._last_known_good
            }
        
        status = self._service_status[source].copy()
        status["has_fallback"] = source in self._fallback_data
        status["has_last_known_good"] = source in self._last_known_good
        
        # Add data freshness information
        if source in self._last_known_good:
            age = datetime.utcnow() - self._last_known_good[source]["timestamp"]
            status["last_known_good_age_hours"] = age.total_seconds() / 3600
        
        return status
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get overall system health status.
        
        Returns:
            Dictionary with system health information
        """
        total_sources = len(self._service_status)
        if total_sources == 0:
            return {
                "overall_status": "unknown",
                "total_sources": 0,
                "healthy_sources": 0,
                "degraded_sources": 0,
                "failed_sources": 0
            }
        
        healthy = sum(1 for s in self._service_status.values() if s["status"] == "healthy")
        degraded = sum(1 for s in self._service_status.values() if s["status"] == "degraded")
        failed = sum(1 for s in self._service_status.values() if s["status"] == "failed")
        
        # Determine overall status
        if failed > total_sources * 0.5:
            overall_status = "critical"
        elif degraded + failed > total_sources * 0.3:
            overall_status = "degraded"
        elif degraded + failed > 0:
            overall_status = "warning"
        else:
            overall_status = "healthy"
        
        return {
            "overall_status": overall_status,
            "total_sources": total_sources,
            "healthy_sources": healthy,
            "degraded_sources": degraded,
            "failed_sources": failed,
            "health_percentage": (healthy / total_sources) * 100
        }
    
    def reset_source_status(self, source: str) -> None:
        """
        Reset failure count for a data source.
        
        Args:
            source: Data source identifier
        """
        if source in self._service_status:
            self._service_status[source] = {
                "status": "healthy",
                "failure_count": 0,
                "last_success": datetime.utcnow()
            }
            logger.info(f"Reset status for source: {source}")
    
    def cleanup_old_data(self, max_age_days: int = 7) -> None:
        """
        Clean up old fallback and last known good data.
        
        Args:
            max_age_days: Maximum age of data to keep in days
        """
        cutoff = datetime.utcnow() - timedelta(days=max_age_days)
        cleaned_count = 0
        
        # Clean up old fallback data
        for source in list(self._fallback_data.keys()):
            if self._fallback_data[source]["timestamp"] < cutoff:
                del self._fallback_data[source]
                cleaned_count += 1
        
        # Clean up old last known good data
        for source in list(self._last_known_good.keys()):
            if self._last_known_good[source]["timestamp"] < cutoff:
                del self._last_known_good[source]
                cleaned_count += 1
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old fallback data entries")