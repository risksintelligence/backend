"""
Performance monitoring and health checking utilities for RiskX platform.
Provides system monitoring, alerting, and health checks for comprehensive observability.
"""

import os
import sys
import time
import psutil
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import logging

from ..core.config import get_settings
from ..core.exceptions import RiskXBaseException, SystemError
from ..core.logging import performance_logger
from .constants import NetworkConfig, CacheConfig, ModelConfig

logger = logging.getLogger('riskx.utils.monitoring')


@dataclass
class SystemMetrics:
    """System performance metrics snapshot."""
    
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    disk_usage_percent: float
    disk_free_gb: float
    network_bytes_sent: int
    network_bytes_recv: int
    process_count: int
    load_average: Tuple[float, float, float]  # 1min, 5min, 15min
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class ApplicationMetrics:
    """Application-specific performance metrics."""
    
    timestamp: datetime
    active_requests: int
    total_requests: int
    error_rate: float
    average_response_time_ms: float
    cache_hit_rate: float
    database_connection_count: int
    model_prediction_count: int
    api_endpoint_stats: Dict[str, Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class HealthStatus:
    """Health check status for a component."""
    
    component: str
    status: str  # healthy, degraded, unhealthy
    response_time_ms: float
    error_message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    last_check: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        if self.last_check:
            data['last_check'] = self.last_check.isoformat()
        return data


class SystemMonitor:
    """System resource monitoring and alerting."""
    
    def __init__(self):
        self.settings = get_settings()
        self.metrics_history: deque = deque(maxlen=1000)  # Keep last 1000 metrics
        self.alert_thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_usage_percent': 90.0,
            'error_rate': 0.05,  # 5%
            'response_time_ms': 5000.0
        }
        self.alert_callbacks: List[Callable] = []
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get system metrics as dictionary (alias for get_system_metrics)."""
        try:
            metrics = self.get_system_metrics()
            return {
                'cpu_percent': metrics.cpu_percent,
                'memory_percent': metrics.memory_percent,
                'disk_percent': metrics.disk_usage_percent,
                'load_avg': metrics.load_average,
                'timestamp': metrics.timestamp.isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return {
                'cpu_percent': 0,
                'memory_percent': 0,
                'disk_percent': 0,
                'load_avg': [0, 0, 0],
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def get_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_mb = memory.available / 1024 / 1024
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_usage_percent = disk.percent
            disk_free_gb = disk.free / 1024 / 1024 / 1024
            
            # Network metrics
            net_io = psutil.net_io_counters()
            network_bytes_sent = net_io.bytes_sent
            network_bytes_recv = net_io.bytes_recv
            
            # Process metrics
            process_count = len(psutil.pids())
            
            # Load average (Unix-like systems)
            try:
                load_avg = os.getloadavg()
            except (OSError, AttributeError):
                load_avg = (0.0, 0.0, 0.0)  # Windows fallback
            
            return SystemMetrics(
                timestamp=datetime.utcnow(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_available_mb=memory_available_mb,
                disk_usage_percent=disk_usage_percent,
                disk_free_gb=disk_free_gb,
                network_bytes_sent=network_bytes_sent,
                network_bytes_recv=network_bytes_recv,
                process_count=process_count,
                load_average=load_avg
            )
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            raise SystemError(f"Failed to collect system metrics: {e}")
    
    def add_alert_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Add callback function for alerts."""
        self.alert_callbacks.append(callback)
    
    def check_alerts(self, metrics: SystemMetrics) -> List[Dict[str, Any]]:
        """Check metrics against alert thresholds."""
        alerts = []
        
        # CPU alert
        if metrics.cpu_percent > self.alert_thresholds['cpu_percent']:
            alerts.append({
                'type': 'cpu_high',
                'severity': 'warning',
                'message': f"High CPU usage: {metrics.cpu_percent:.1f}%",
                'value': metrics.cpu_percent,
                'threshold': self.alert_thresholds['cpu_percent']
            })
        
        # Memory alert
        if metrics.memory_percent > self.alert_thresholds['memory_percent']:
            alerts.append({
                'type': 'memory_high',
                'severity': 'warning',
                'message': f"High memory usage: {metrics.memory_percent:.1f}%",
                'value': metrics.memory_percent,
                'threshold': self.alert_thresholds['memory_percent']
            })
        
        # Disk alert
        if metrics.disk_usage_percent > self.alert_thresholds['disk_usage_percent']:
            alerts.append({
                'type': 'disk_full',
                'severity': 'critical',
                'message': f"Low disk space: {metrics.disk_usage_percent:.1f}% used",
                'value': metrics.disk_usage_percent,
                'threshold': self.alert_thresholds['disk_usage_percent']
            })
        
        # Trigger alert callbacks
        for alert in alerts:
            for callback in self.alert_callbacks:
                try:
                    callback(alert['type'], alert)
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}")
        
        return alerts
    
    def monitor_continuously(self, interval_seconds: int = 60):
        """Start continuous monitoring in background."""
        def monitor_loop():
            while True:
                try:
                    metrics = self.get_system_metrics()
                    self.metrics_history.append(metrics)
                    
                    # Check for alerts
                    alerts = self.check_alerts(metrics)
                    if alerts:
                        logger.warning(f"System alerts triggered: {len(alerts)}")
                    
                    # Log metrics
                    performance_logger.logger.info(
                        "System metrics collected",
                        extra={
                            'operation': 'system_monitoring',
                            'cpu_percent': metrics.cpu_percent,
                            'memory_percent': metrics.memory_percent,
                            'disk_usage_percent': metrics.disk_usage_percent
                        }
                    )
                    
                    time.sleep(interval_seconds)
                    
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(interval_seconds)
        
        import threading
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        logger.info(f"Started continuous system monitoring (interval: {interval_seconds}s)")


class ApplicationMonitor:
    """Application-specific performance monitoring."""
    
    def __init__(self):
        self.request_stats = defaultdict(lambda: {
            'count': 0,
            'total_time': 0.0,
            'error_count': 0,
            'last_access': None
        })
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'total_requests': 0
        }
        self.model_stats = {
            'predictions': 0,
            'total_inference_time': 0.0,
            'error_count': 0
        }
        self.start_time = datetime.utcnow()
    
    def record_request(self, endpoint: str, method: str, 
                      status_code: int, response_time_ms: float):
        """Record API request metrics."""
        key = f"{method} {endpoint}"
        stats = self.request_stats[key]
        
        stats['count'] += 1
        stats['total_time'] += response_time_ms
        stats['last_access'] = datetime.utcnow()
        
        if status_code >= 400:
            stats['error_count'] += 1
        
        # Log request
        performance_logger.log_api_request(
            method=method,
            path=endpoint,
            status_code=status_code,
            duration_ms=response_time_ms
        )
    
    def record_cache_hit(self, hit: bool):
        """Record cache hit/miss."""
        if hit:
            self.cache_stats['hits'] += 1
        else:
            self.cache_stats['misses'] += 1
        self.cache_stats['total_requests'] += 1
    
    def record_model_prediction(self, model_name: str, inference_time_ms: float, 
                               success: bool = True):
        """Record ML model prediction metrics."""
        self.model_stats['predictions'] += 1
        self.model_stats['total_inference_time'] += inference_time_ms
        
        if not success:
            self.model_stats['error_count'] += 1
        
        # Log model inference
        performance_logger.log_model_inference(
            model_name=model_name,
            duration_ms=inference_time_ms,
            input_size=1,  # Simplified
            prediction_count=1
        )
    
    def get_application_metrics(self) -> ApplicationMetrics:
        """Get current application metrics."""
        # Calculate aggregate statistics
        total_requests = sum(stats['count'] for stats in self.request_stats.values())
        total_errors = sum(stats['error_count'] for stats in self.request_stats.values())
        total_response_time = sum(stats['total_time'] for stats in self.request_stats.values())
        
        error_rate = total_errors / total_requests if total_requests > 0 else 0.0
        avg_response_time = total_response_time / total_requests if total_requests > 0 else 0.0
        
        # Cache hit rate
        cache_total = self.cache_stats['total_requests']
        cache_hit_rate = self.cache_stats['hits'] / cache_total if cache_total > 0 else 0.0
        
        # Prepare endpoint stats
        endpoint_stats = {}
        for endpoint, stats in self.request_stats.items():
            avg_time = stats['total_time'] / stats['count'] if stats['count'] > 0 else 0.0
            endpoint_error_rate = stats['error_count'] / stats['count'] if stats['count'] > 0 else 0.0
            
            endpoint_stats[endpoint] = {
                'request_count': stats['count'],
                'average_response_time_ms': avg_time,
                'error_rate': endpoint_error_rate,
                'last_access': stats['last_access'].isoformat() if stats['last_access'] else None
            }
        
        return ApplicationMetrics(
            timestamp=datetime.utcnow(),
            active_requests=0,  # Would need request tracking
            total_requests=total_requests,
            error_rate=error_rate,
            average_response_time_ms=avg_response_time,
            cache_hit_rate=cache_hit_rate,
            database_connection_count=0,  # Would need DB monitoring
            model_prediction_count=self.model_stats['predictions'],
            api_endpoint_stats=endpoint_stats
        )
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for dashboard."""
        metrics = self.get_application_metrics()
        uptime = datetime.utcnow() - self.start_time
        
        return {
            'uptime_seconds': uptime.total_seconds(),
            'uptime_formatted': str(uptime).split('.')[0],  # Remove microseconds
            'total_requests': metrics.total_requests,
            'error_rate_percent': metrics.error_rate * 100,
            'average_response_time_ms': metrics.average_response_time_ms,
            'cache_hit_rate_percent': metrics.cache_hit_rate * 100,
            'model_predictions': metrics.model_prediction_count,
            'top_endpoints': self._get_top_endpoints(5)
        }
    
    def _get_top_endpoints(self, limit: int) -> List[Dict[str, Any]]:
        """Get top endpoints by request count."""
        sorted_endpoints = sorted(
            self.request_stats.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )
        
        return [
            {
                'endpoint': endpoint,
                'request_count': stats['count'],
                'average_response_time_ms': stats['total_time'] / stats['count'] if stats['count'] > 0 else 0
            }
            for endpoint, stats in sorted_endpoints[:limit]
        ]


class HealthChecker:
    """Component health checking and status aggregation."""
    
    def __init__(self):
        self.health_checks: Dict[str, Callable] = {}
        self.last_results: Dict[str, HealthStatus] = {}
        self.check_interval = 60  # seconds
    
    def register_check(self, name: str, check_func: Callable[[], Tuple[bool, float, Optional[str]]]):
        """
        Register a health check function.
        check_func should return (is_healthy, response_time_ms, error_message)
        """
        self.health_checks[name] = check_func
        logger.info(f"Registered health check: {name}")
    
    async def run_check(self, name: str) -> HealthStatus:
        """Run a single health check."""
        if name not in self.health_checks:
            return HealthStatus(
                component=name,
                status="unhealthy",
                response_time_ms=0.0,
                error_message="Health check not found"
            )
        
        start_time = time.time()
        
        try:
            check_func = self.health_checks[name]
            
            if asyncio.iscoroutinefunction(check_func):
                is_healthy, response_time_ms, error_msg = await check_func()
            else:
                is_healthy, response_time_ms, error_msg = check_func()
            
            status = "healthy" if is_healthy else "unhealthy"
            
            result = HealthStatus(
                component=name,
                status=status,
                response_time_ms=response_time_ms,
                error_message=error_msg,
                last_check=datetime.utcnow()
            )
            
            self.last_results[name] = result
            return result
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            result = HealthStatus(
                component=name,
                status="unhealthy",
                response_time_ms=execution_time,
                error_message=str(e),
                last_check=datetime.utcnow()
            )
            
            self.last_results[name] = result
            logger.error(f"Health check failed for {name}: {e}")
            return result
    
    async def run_all_checks(self) -> Dict[str, HealthStatus]:
        """Run all registered health checks."""
        if not self.health_checks:
            return {}
        
        # Run checks concurrently
        tasks = [
            self.run_check(name) for name in self.health_checks.keys()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        health_status = {}
        for i, (name, result) in enumerate(zip(self.health_checks.keys(), results)):
            if isinstance(result, Exception):
                health_status[name] = HealthStatus(
                    component=name,
                    status="unhealthy",
                    response_time_ms=0.0,
                    error_message=str(result)
                )
            else:
                health_status[name] = result
        
        return health_status
    
    def get_overall_status(self) -> Dict[str, Any]:
        """Get overall system health status."""
        if not self.last_results:
            return {
                'status': 'unknown',
                'components': {},
                'summary': 'No health checks configured'
            }
        
        healthy_count = sum(1 for status in self.last_results.values() if status.status == 'healthy')
        total_count = len(self.last_results)
        
        if healthy_count == total_count:
            overall_status = 'healthy'
        elif healthy_count > 0:
            overall_status = 'degraded'
        else:
            overall_status = 'unhealthy'
        
        return {
            'status': overall_status,
            'healthy_components': healthy_count,
            'total_components': total_count,
            'components': {name: status.to_dict() for name, status in self.last_results.items()},
            'last_check': max(
                (status.last_check for status in self.last_results.values() if status.last_check),
                default=None
            )
        }


# Built-in health checks
def database_health_check() -> Tuple[bool, float, Optional[str]]:
    """Basic database connectivity health check."""
    start_time = time.time()
    
    try:
        # Import here to avoid circular dependencies
        from ..core.database import get_database_url
        
        db_url = get_database_url()
        
        if db_url.startswith('sqlite'):
            # For SQLite, just check if file exists and is accessible
            import sqlite3
            db_path = db_url.replace('sqlite:///', '')
            
            conn = sqlite3.connect(db_path, timeout=5)
            cursor = conn.cursor()
            cursor.execute('SELECT 1')
            conn.close()
            
        else:
            # For other databases, would need specific connection logic
            pass
        
        response_time = (time.time() - start_time) * 1000
        return True, response_time, None
        
    except Exception as e:
        response_time = (time.time() - start_time) * 1000
        return False, response_time, str(e)


def cache_health_check() -> Tuple[bool, float, Optional[str]]:
    """Basic cache system health check."""
    start_time = time.time()
    
    try:
        # Test file cache (primary fallback)
        import tempfile
        import os
        
        test_file = os.path.join(tempfile.gettempdir(), 'riskx_cache_test')
        with open(test_file, 'w') as f:
            f.write('test')
        
        # Read back
        with open(test_file, 'r') as f:
            content = f.read()
        
        # Cleanup
        os.remove(test_file)
        
        if content != 'test':
            raise Exception("Cache write/read test failed")
        
        response_time = (time.time() - start_time) * 1000
        return True, response_time, None
        
    except Exception as e:
        response_time = (time.time() - start_time) * 1000
        return False, response_time, str(e)


def external_api_health_check() -> Tuple[bool, float, Optional[str]]:
    """Check external API connectivity."""
    start_time = time.time()
    
    try:
        import requests
        
        # Test FRED API (primary data source)
        response = requests.get(
            'https://api.stlouisfed.org/fred/series',
            params={'api_key': 'test', 'series_id': 'GDP'},
            timeout=10
        )
        
        # Even with invalid API key, we should get a response
        if response.status_code in [200, 400, 403]:
            response_time = (time.time() - start_time) * 1000
            return True, response_time, None
        else:
            response_time = (time.time() - start_time) * 1000
            return False, response_time, f"Unexpected status code: {response.status_code}"
        
    except Exception as e:
        response_time = (time.time() - start_time) * 1000
        return False, response_time, str(e)


# Global monitoring instances
system_monitor = SystemMonitor()
app_monitor = ApplicationMonitor()
health_checker = HealthChecker()

# Register default health checks
health_checker.register_check('database', database_health_check)
health_checker.register_check('cache', cache_health_check)
health_checker.register_check('external_apis', external_api_health_check)


def setup_monitoring():
    """Initialize monitoring system with default configuration."""
    logger.info("Setting up monitoring system")
    
    # Setup alert callback
    def log_alert(alert_type: str, alert_data: Dict[str, Any]):
        logger.warning(f"ALERT [{alert_type}]: {alert_data['message']}")
    
    system_monitor.add_alert_callback(log_alert)
    
    # Start continuous monitoring
    system_monitor.monitor_continuously(interval_seconds=60)
    
    logger.info("Monitoring system initialized")


def get_monitoring_summary() -> Dict[str, Any]:
    """Get comprehensive monitoring summary for dashboards."""
    try:
        system_metrics = system_monitor.get_system_metrics()
        app_metrics = app_monitor.get_application_metrics()
        health_status = health_checker.get_overall_status()
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'system': system_metrics.to_dict(),
            'application': app_metrics.to_dict(),
            'health': health_status,
            'summary': app_monitor.get_summary_stats()
        }
        
    except Exception as e:
        logger.error(f"Error getting monitoring summary: {e}")
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'error': str(e),
            'status': 'monitoring_error'
        }