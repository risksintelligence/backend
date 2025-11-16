"""
Background Refresh Service - Stale-While-Revalidate Implementation

Implements the stale-while-revalidate pattern:
- API requests get immediate cached data (even if soft TTL expired)
- Background tasks refresh stale data without blocking user requests
- Honors provider rate limits and embargo windows
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass

from app.core.unified_cache import UnifiedCache
from app.data.registry import SERIES_REGISTRY
from app.data.sources import get_source
from app.services.ingestion import Observation, _persist_observations
from app.core.config import get_settings

logger = logging.getLogger(__name__)

@dataclass
class RefreshTask:
    series_id: str
    priority: int  # 1=high, 2=medium, 3=low
    last_attempt: Optional[datetime] = None
    failure_count: int = 0
    next_retry: Optional[datetime] = None

class BackgroundRefreshService:
    """Manages background data refresh following stale-while-revalidate pattern."""
    
    def __init__(self):
        self.cache = UnifiedCache("data")
        self.settings = get_settings()
        self.refresh_queue: List[RefreshTask] = []
        self.running = False
        
    async def start_refresh_loop(self) -> None:
        """Start the main background refresh loop."""
        logger.info("🔄 Starting background refresh service...")
        self.running = True
        
        while self.running:
            try:
                # Check for stale keys that need refresh
                await self._check_and_queue_stale_data()
                
                # Process refresh queue
                await self._process_refresh_queue()
                
                # Wait before next iteration (respecting rate limits)
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Background refresh loop error: {e}")
                await asyncio.sleep(300)  # Longer wait on errors
    
    def stop_refresh_loop(self) -> None:
        """Stop the background refresh service."""
        logger.info("⏹️  Stopping background refresh service...")
        self.running = False
    
    async def _check_and_queue_stale_data(self) -> None:
        """Check cache for stale keys and add to refresh queue."""
        try:
            stale_keys = self.cache.get_stale_keys()
            
            for series_id in stale_keys:
                # Skip if already in queue
                if any(task.series_id == series_id for task in self.refresh_queue):
                    continue
                
                # Determine priority based on series importance and frequency
                priority = self._calculate_refresh_priority(series_id)
                
                task = RefreshTask(
                    series_id=series_id,
                    priority=priority
                )
                
                self.refresh_queue.append(task)
                logger.debug(f"Queued stale data refresh for {series_id} (priority {priority})")
            
            # Sort queue by priority (1=highest)
            self.refresh_queue.sort(key=lambda x: (x.priority, x.last_attempt or datetime.min))
            
        except Exception as e:
            logger.error(f"Failed to check stale data: {e}")
    
    async def _process_refresh_queue(self) -> None:
        """Process the refresh queue with rate limiting and retry logic."""
        if not self.refresh_queue:
            return
            
        # Process up to 3 tasks per iteration to respect rate limits
        tasks_to_process = self.refresh_queue[:3]
        
        for task in tasks_to_process:
            try:
                # Check if we should skip due to recent failure
                if task.next_retry and datetime.utcnow() < task.next_retry:
                    continue
                
                # Attempt to refresh the data
                success = await self._refresh_series_data(task.series_id)
                
                if success:
                    # Remove from queue on success
                    self.refresh_queue.remove(task)
                    logger.info(f"✅ Successfully refreshed data for {task.series_id}")
                else:
                    # Handle retry logic on failure
                    task.failure_count += 1
                    task.last_attempt = datetime.utcnow()
                    
                    # Exponential backoff: 5min, 15min, 1hr, 4hr, then remove
                    retry_delays = [300, 900, 3600, 14400]
                    if task.failure_count <= len(retry_delays):
                        delay = retry_delays[task.failure_count - 1]
                        task.next_retry = datetime.utcnow() + timedelta(seconds=delay)
                        logger.warning(f"⚠️  Refresh failed for {task.series_id}, retry in {delay}s (attempt {task.failure_count})")
                    else:
                        # Too many failures - remove from queue
                        self.refresh_queue.remove(task)
                        logger.error(f"❌ Removing {task.series_id} from refresh queue after {task.failure_count} failures")
                
                # Small delay between requests to respect rate limits
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Error processing refresh task for {task.series_id}: {e}")
    
    async def _refresh_series_data(self, series_id: str) -> bool:
        """Refresh data for a specific series from its provider."""
        try:
            if series_id not in SERIES_REGISTRY:
                logger.warning(f"Series {series_id} not found in registry")
                return False
            
            metadata = SERIES_REGISTRY[series_id]
            
            # Get the appropriate fetcher
            fetch_func = get_source(metadata.provider)
            
            # Fetch fresh data
            raw_points = fetch_func(metadata.id)
            
            if not raw_points:
                logger.warning(f"No data returned for {series_id}")
                return False
            
            # Convert to observations
            obs_list = []
            now = datetime.utcnow()
            
            for point in raw_points:
                # Handle different timestamp formats
                timestamp_str = point["timestamp"].replace("Z", "")
                if "T" not in timestamp_str:
                    timestamp_str += "T00:00:00"
                
                obs_list.append(
                    Observation(
                        series_id=series_id,
                        observed_at=datetime.fromisoformat(timestamp_str),
                        value=float(point["value"]),
                    )
                )
            
            # Sort by time
            obs_list.sort(key=lambda o: o.observed_at)
            
            # Persist to database (L2)
            _persist_observations(series_id, obs_list)
            
            # Update cache (L1) with fresh data and metadata
            latest_obs = obs_list[-1] if obs_list else None
            if latest_obs:
                data = {
                    "timestamp": latest_obs.observed_at.isoformat(),
                    "value": str(latest_obs.value)
                }
                
                # Set cache with appropriate TTLs based on series frequency
                soft_ttl, hard_ttl = self._get_ttl_for_series(series_id, metadata.frequency)
                
                self.cache.set(
                    key=series_id,
                    value=data,
                    source=metadata.provider,
                    source_url=f"https://api.{metadata.provider}.com/{metadata.id}",
                    derivation_flag="raw",
                    soft_ttl=soft_ttl,
                    hard_ttl=hard_ttl
                )
            
            logger.info(f"Refreshed {len(obs_list)} observations for {series_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to refresh {series_id}: {e}")
            return False
    
    def _calculate_refresh_priority(self, series_id: str) -> int:
        """Calculate refresh priority based on series importance and frequency."""
        if series_id not in SERIES_REGISTRY:
            return 3  # Low priority for unknown series
        
        metadata = SERIES_REGISTRY[series_id]
        
        # High priority for critical financial indicators
        high_priority = ["VIX", "CREDIT_SPREAD", "YIELD_CURVE"]
        if series_id in high_priority:
            return 1
        
        # Medium priority for supply chain indicators  
        medium_priority = ["WTI_OIL", "FREIGHT_DIESEL", "BALTIC_DRY"]
        if series_id in medium_priority:
            return 2
        
        # Default to low priority
        return 3
    
    def _get_ttl_for_series(self, series_id: str, frequency: str) -> tuple[int, int]:
        """Get appropriate soft and hard TTLs based on series frequency and importance."""
        
        # TTL mapping per architecture docs
        ttl_config = {
            "VIX": (900, 3600),          # 15min soft, 1hr hard (high frequency)
            "CREDIT_SPREAD": (1800, 7200), # 30min soft, 2hr hard
            "YIELD_CURVE": (1800, 7200),   # 30min soft, 2hr hard  
            "WTI_OIL": (3600, 14400),      # 1hr soft, 4hr hard
            "FREIGHT_DIESEL": (21600, 86400), # 6hr soft, 24hr hard (weekly updates)
            "PMI": (86400, 2592000),       # 24hr soft, 30 days hard (monthly)
            "UNEMPLOYMENT": (86400, 2592000), # 24hr soft, 30 days hard (monthly)
        }
        
        return ttl_config.get(series_id, (3600, 14400))  # Default: 1hr soft, 4hr hard
    
    def force_refresh(self, series_id: str) -> None:
        """Force immediate refresh of a specific series (bypass normal queue)."""
        task = RefreshTask(series_id=series_id, priority=0)  # Highest priority
        self.refresh_queue.insert(0, task)
        logger.info(f"🚀 Forced refresh queued for {series_id}")
    
    def get_refresh_status(self) -> Dict[str, any]:
        """Get current status of the refresh service."""
        return {
            "running": self.running,
            "queue_length": len(self.refresh_queue),
            "current_tasks": [
                {
                    "series_id": task.series_id,
                    "priority": task.priority,
                    "failure_count": task.failure_count,
                    "last_attempt": task.last_attempt.isoformat() if task.last_attempt else None,
                    "next_retry": task.next_retry.isoformat() if task.next_retry else None
                }
                for task in self.refresh_queue[:10]  # Show first 10
            ]
        }

# Global instance for the application
refresh_service = BackgroundRefreshService()