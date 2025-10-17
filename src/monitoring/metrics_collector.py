"""
Production Metrics Collection System for RiskX

Comprehensive monitoring system that tracks API performance, data quality,
ML model performance, and system health metrics for production deployment.
"""

import time
import psutil
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import asyncio
from collections import defaultdict, deque

from src.core.config import get_settings
from src.cache.cache_manager import CacheManager

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class APIMetrics:
    """API performance metrics"""
    endpoint: str
    method: str
    status_code: int
    response_time: float
    timestamp: datetime
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class SystemMetrics:
    """System resource metrics"""
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    active_connections: int
    cache_hit_rate: float
    timestamp: datetime


@dataclass
class DataQualityMetrics:
    """Data quality and freshness metrics"""
    source_name: str
    last_update: datetime
    data_points: int
    quality_score: float
    error_count: int
    latency_seconds: float
    timestamp: datetime


@dataclass
class MLModelMetrics:
    """ML model performance metrics"""
    model_name: str
    prediction_count: int
    avg_confidence: float
    error_rate: float
    avg_response_time: float
    last_training: Optional[datetime]
    timestamp: datetime


class MetricsCollector:
    """
    Comprehensive metrics collection system for production monitoring.
    
    Collects and stores metrics for:
    - API performance and usage
    - System resource utilization
    - Data source quality and freshness
    - ML model performance
    - Business intelligence and usage patterns
    """
    
    def __init__(self):
        self.cache = CacheManager()
        self.metrics_dir = Path("logs/metrics")
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory metric storage for fast access
        self.api_metrics: deque = deque(maxlen=10000)
        self.system_metrics: deque = deque(maxlen=1000)
        self.data_quality_metrics: deque = deque(maxlen=1000)
        self.ml_metrics: deque = deque(maxlen=1000)
        
        # Performance counters
        self.request_counter = defaultdict(int)
        self.error_counter = defaultdict(int)
        self.response_times = defaultdict(list)
        
        logger.info("MetricsCollector initialized")
    
    async def record_api_request(self, 
                                endpoint: str,
                                method: str, 
                                status_code: int,
                                response_time: float,
                                user_agent: Optional[str] = None,
                                ip_address: Optional[str] = None,
                                error_message: Optional[str] = None):
        """Record API request metrics"""
        try:
            metric = APIMetrics(
                endpoint=endpoint,
                method=method,
                status_code=status_code,
                response_time=response_time,
                timestamp=datetime.now(),
                user_agent=user_agent,
                ip_address=ip_address,
                error_message=error_message
            )
            
            # Store in memory
            self.api_metrics.append(metric)
            
            # Update counters
            self.request_counter[f"{method}:{endpoint}"] += 1
            if status_code >= 400:
                self.error_counter[f"{method}:{endpoint}"] += 1
            
            # Track response times (keep last 100 for rolling average)
            if len(self.response_times[endpoint]) >= 100:
                self.response_times[endpoint].pop(0)
            self.response_times[endpoint].append(response_time)
            
            # Cache recent metrics
            cache_key = f"api_metrics:recent"
            recent_metrics = await self.cache.get(cache_key) or []
            recent_metrics.append(asdict(metric))
            
            # Keep only last 100 API calls in cache
            if len(recent_metrics) > 100:
                recent_metrics = recent_metrics[-100:]
            
            await self.cache.set(cache_key, recent_metrics, ttl=3600)
            
        except Exception as e:
            logger.error(f"Error recording API metrics: {e}")
    
    async def collect_system_metrics(self):
        """Collect system resource metrics"""
        try:
            # CPU and memory usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network connections
            connections = len(psutil.net_connections())
            
            # Cache hit rate
            cache_hit_rate = await self._calculate_cache_hit_rate()
            
            metric = SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_usage_percent=disk.percent,
                active_connections=connections,
                cache_hit_rate=cache_hit_rate,
                timestamp=datetime.now()
            )
            
            self.system_metrics.append(metric)
            
            # Cache system metrics
            await self.cache.set("system_metrics:current", asdict(metric), ttl=60)
            
            return metric
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return None
    
    async def record_data_quality_metrics(self,
                                        source_name: str,
                                        data_points: int,
                                        quality_score: float,
                                        error_count: int = 0,
                                        latency_seconds: float = 0.0):
        """Record data source quality metrics"""
        try:
            metric = DataQualityMetrics(
                source_name=source_name,
                last_update=datetime.now(),
                data_points=data_points,
                quality_score=quality_score,
                error_count=error_count,
                latency_seconds=latency_seconds,
                timestamp=datetime.now()
            )
            
            self.data_quality_metrics.append(metric)
            
            # Cache data quality metrics
            cache_key = f"data_quality:{source_name}"
            await self.cache.set(cache_key, asdict(metric), ttl=1800)
            
        except Exception as e:
            logger.error(f"Error recording data quality metrics: {e}")
    
    async def record_ml_model_metrics(self,
                                    model_name: str,
                                    prediction_count: int,
                                    avg_confidence: float,
                                    error_rate: float,
                                    avg_response_time: float,
                                    last_training: Optional[datetime] = None):
        """Record ML model performance metrics"""
        try:
            metric = MLModelMetrics(
                model_name=model_name,
                prediction_count=prediction_count,
                avg_confidence=avg_confidence,
                error_rate=error_rate,
                avg_response_time=avg_response_time,
                last_training=last_training,
                timestamp=datetime.now()
            )
            
            self.ml_metrics.append(metric)
            
            # Cache ML metrics
            cache_key = f"ml_metrics:{model_name}"
            await self.cache.set(cache_key, asdict(metric), ttl=600)
            
        except Exception as e:
            logger.error(f"Error recording ML model metrics: {e}")
    
    async def get_api_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get API performance summary for the last N hours"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_metrics = [m for m in self.api_metrics if m.timestamp > cutoff_time]
            
            if not recent_metrics:
                return {"message": "No recent API metrics available"}
            
            # Calculate statistics
            total_requests = len(recent_metrics)
            error_requests = len([m for m in recent_metrics if m.status_code >= 400])
            
            response_times = [m.response_time for m in recent_metrics]
            avg_response_time = sum(response_times) / len(response_times)
            p95_response_time = sorted(response_times)[int(len(response_times) * 0.95)]
            
            # Top endpoints
            endpoint_counts = defaultdict(int)
            for metric in recent_metrics:
                endpoint_counts[metric.endpoint] += 1
            
            top_endpoints = sorted(endpoint_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            return {
                "period_hours": hours,
                "total_requests": total_requests,
                "error_rate": error_requests / total_requests if total_requests > 0 else 0,
                "avg_response_time_ms": round(avg_response_time * 1000, 2),
                "p95_response_time_ms": round(p95_response_time * 1000, 2),
                "top_endpoints": top_endpoints,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting API performance summary: {e}")
            return {"error": str(e)}
    
    async def get_system_health_status(self) -> Dict[str, Any]:
        """Get current system health status"""
        try:
            current_metrics = await self.collect_system_metrics()
            if not current_metrics:
                return {"status": "unknown", "message": "Could not collect system metrics"}
            
            # Determine health status
            health_score = 100
            issues = []
            
            if current_metrics.cpu_percent > 80:
                health_score -= 30
                issues.append(f"High CPU usage: {current_metrics.cpu_percent}%")
            
            if current_metrics.memory_percent > 85:
                health_score -= 25
                issues.append(f"High memory usage: {current_metrics.memory_percent}%")
            
            if current_metrics.disk_usage_percent > 90:
                health_score -= 20
                issues.append(f"High disk usage: {current_metrics.disk_usage_percent}%")
            
            if current_metrics.cache_hit_rate < 0.7:
                health_score -= 15
                issues.append(f"Low cache hit rate: {current_metrics.cache_hit_rate:.2%}")
            
            # Determine status
            if health_score >= 90:
                status = "healthy"
            elif health_score >= 70:
                status = "warning"
            elif health_score >= 50:
                status = "degraded"
            else:
                status = "critical"
            
            return {
                "status": status,
                "health_score": health_score,
                "metrics": asdict(current_metrics),
                "issues": issues,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting system health status: {e}")
            return {"status": "error", "error": str(e)}
    
    async def get_data_quality_report(self) -> Dict[str, Any]:
        """Get data quality report for all sources"""
        try:
            report = {}
            
            # Group metrics by source
            source_metrics = defaultdict(list)
            for metric in self.data_quality_metrics:
                source_metrics[metric.source_name].append(metric)
            
            for source_name, metrics in source_metrics.items():
                latest_metric = max(metrics, key=lambda x: x.timestamp)
                
                # Calculate source health
                hours_since_update = (datetime.now() - latest_metric.last_update).total_seconds() / 3600
                freshness_score = max(0, 100 - (hours_since_update * 5))  # Degrade 5% per hour
                
                report[source_name] = {
                    "quality_score": latest_metric.quality_score,
                    "freshness_score": freshness_score,
                    "data_points": latest_metric.data_points,
                    "error_count": latest_metric.error_count,
                    "last_update": latest_metric.last_update.isoformat(),
                    "latency_seconds": latest_metric.latency_seconds,
                    "status": "healthy" if freshness_score > 80 and latest_metric.quality_score > 0.8 else "degraded"
                }
            
            return {
                "sources": report,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating data quality report: {e}")
            return {"error": str(e)}
    
    async def collect_system_metrics(self):
        """Collect current system metrics"""
        try:
            # Get system resource usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Get cache hit rate
            cache_hit_rate = await self._calculate_cache_hit_rate()
            
            # Get active connections (approximate)
            active_connections = len(psutil.net_connections())
            
            metric = SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_usage_percent=disk.percent,
                active_connections=active_connections,
                cache_hit_rate=cache_hit_rate,
                timestamp=datetime.now()
            )
            
            self.system_metrics.append(metric)
            
            # Cache system metrics
            cache_key = "system_metrics:latest"
            self.cache.set(cache_key, asdict(metric), ttl=300)
            
            logger.debug(f"System metrics collected: CPU={cpu_percent}%, Memory={memory.percent}%")
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    async def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        try:
            # This is a simplified implementation
            # In a real system, you'd track hits and misses
            cache_stats = self.cache.get("cache_stats") or {"hits": 0, "misses": 0}
            total = cache_stats["hits"] + cache_stats["misses"]
            
            if total == 0:
                return 0.0
            
            return cache_stats["hits"] / total
            
        except Exception:
            return 0.0
    
    async def export_metrics_to_file(self):
        """Export current metrics to JSON files for persistence"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Export API metrics
            if self.api_metrics:
                api_file = self.metrics_dir / f"api_metrics_{timestamp}.json"
                with open(api_file, 'w') as f:
                    json.dump([asdict(m) for m in self.api_metrics], f, default=str, indent=2)
            
            # Export system metrics  
            if self.system_metrics:
                system_file = self.metrics_dir / f"system_metrics_{timestamp}.json"
                with open(system_file, 'w') as f:
                    json.dump([asdict(m) for m in self.system_metrics], f, default=str, indent=2)
            
            logger.info(f"Metrics exported to {self.metrics_dir}")
            
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")


# Global metrics collector instance
metrics_collector = MetricsCollector()


async def start_background_metrics_collection():
    """Start background task for continuous metrics collection"""
    try:
        async def metrics_loop():
            logger.info("Background metrics collection started")
            while True:
                try:
                    await metrics_collector.collect_system_metrics()
                    await asyncio.sleep(60)  # Collect every minute
                except Exception as e:
                    logger.error(f"Error collecting system metrics: {e}")
                    await asyncio.sleep(60)
        
        # Start the background task
        asyncio.create_task(metrics_loop())
        logger.info("Metrics collection background task started")
        
    except Exception as e:
        logger.error(f"Failed to start background metrics collection: {e}")