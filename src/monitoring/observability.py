"""Enhanced observability and monitoring for production RiskSX Intelligence System."""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from prometheus_client import Counter, Gauge, Histogram, generate_latest
from backend.src.monitoring.metrics import *

# Import backup metrics
from backend.src.monitoring.backup_metrics import (
    BACKUP_SUCCESS_TOTAL,
    BACKUP_ERROR_TOTAL, 
    BACKUP_DURATION_SECONDS,
    BACKUP_FILE_SIZE_BYTES,
    BACKUP_LAST_SUCCESS_TIMESTAMP,
    BACKUP_STORAGE_USED_BYTES,
    BACKUP_STORAGE_AVAILABLE_BYTES,
    BACKUP_UPLOAD_SUCCESS_TOTAL,
    BACKUP_UPLOAD_ERROR_TOTAL,
    BACKUP_UPLOAD_DURATION_SECONDS,
    BACKUP_RECOVERY_SUCCESS_TOTAL,
    BACKUP_RECOVERY_ERROR_TOTAL,
    BACKUP_RECOVERY_DURATION_SECONDS,
    BACKUP_SYSTEM_HEALTH,
    BACKUP_RETENTION_DAYS,
    BACKUP_ALERT_TRIGGERED_TOTAL,
    BackupMetricsCollector
)

logger = logging.getLogger(__name__)


# Additional production metrics
DATA_FRESHNESS_HOURS = Gauge(
    "ris_data_freshness_hours",
    "Hours since last data update per series",
    labelnames=("series_id",)
)

GERI_VALUE = Gauge(
    "ris_geri_current_value",
    "Current GERI index value"
)

GERI_BAND = Gauge(
    "ris_geri_band_numeric",
    "Current GERI band as numeric (0=Critical, 1=High, 2=Medium, 3=Low)"
)

SYSTEM_HEALTH = Gauge(
    "ris_system_health",
    "System health metrics",
    labelnames=("metric_type",)
)

DATABASE_CONNECTIONS = Gauge(
    "ris_database_connections",
    "Active database connections"
)

CACHE_OPERATIONS = Counter(
    "ris_cache_operations_total",
    "Cache operations by type",
    labelnames=("operation", "tier")
)

ML_PREDICTIONS = Counter(
    "ris_ml_predictions_total",
    "ML model predictions made",
    labelnames=("model_type", "model_version")
)

ML_ANOMALY_SCORE = Gauge(
    "ris_ml_anomaly_score",
    "Latest ML anomaly detection score"
)

ALERT_TRIGGERS = Counter(
    "ris_alert_triggers_total",
    "Alert conditions triggered",
    labelnames=("alert_type", "severity")
)

JOB_DURATION = Histogram(
    "ris_job_duration_seconds",
    "Duration of background jobs",
    labelnames=("job_name",)
)

JOB_SUCCESS = Counter(
    "ris_job_success_total",
    "Successful job executions",
    labelnames=("job_name",)
)

JOB_FAILURES = Counter(
    "ris_job_failures_total",
    "Failed job executions",
    labelnames=("job_name",)
)


@dataclass
class HealthMetrics:
    """Container for system health metrics."""
    geri_value: Optional[float]
    geri_band: Optional[str]
    data_series_fresh: int
    data_series_stale: int
    database_connections: int
    cpu_percent: float
    memory_percent: float
    ml_models_active: int
    alert_subscriptions: int
    recent_api_requests: int
    timestamp: str


class ObservabilityService:
    """Service for collecting and exposing observability metrics."""
    
    def __init__(self):
        self._last_collection = datetime.utcnow()
        self._collection_interval = 60  # seconds
        
    async def collect_all_metrics(self) -> HealthMetrics:
        """Collect comprehensive system metrics."""
        try:
            from backend.src.services.admin_service import get_admin_service
            admin_service = get_admin_service()
            
            # Get system health and data quality
            health_data = await admin_service.get_system_health()
            quality_data = await admin_service.get_data_quality_metrics()
            
            # Extract GERI metrics
            latest_geri = health_data.get("data", {}).get("latest_geri", {})
            geri_value = latest_geri.get("value")
            geri_band = latest_geri.get("band")
            
            # Update Prometheus metrics
            if geri_value is not None:
                GERI_VALUE.set(geri_value)
                
                # Convert band to numeric for alerting
                band_numeric = {
                    "Critical": 0, "High": 1, "Medium": 2, "Low": 3
                }.get(geri_band, -1)
                GERI_BAND.set(band_numeric)
            
            # Data freshness metrics
            freshness_data = quality_data.get("data_freshness", [])
            fresh_count = 0
            stale_count = 0
            
            for series in freshness_data:
                series_id = series.get("series_id", "unknown")
                hours_stale = series.get("hours_stale", 0)
                
                DATA_FRESHNESS_HOURS.labels(series_id=series_id).set(hours_stale)
                
                if series.get("status") == "fresh":
                    fresh_count += 1
                else:
                    stale_count += 1
            
            # System health metrics
            system_info = health_data.get("system", {})
            db_info = health_data.get("database", {})
            activity = health_data.get("activity", {})
            
            cpu_percent = system_info.get("cpu_percent", 0)
            memory_percent = system_info.get("memory_percent", 0)
            db_connections = db_info.get("connections", 0)
            
            SYSTEM_HEALTH.labels(metric_type="cpu_percent").set(cpu_percent)
            SYSTEM_HEALTH.labels(metric_type="memory_percent").set(memory_percent)
            SYSTEM_HEALTH.labels(metric_type="disk_percent").set(system_info.get("disk_percent", 0))
            DATABASE_CONNECTIONS.set(db_connections)
            
            # ML model metrics
            ml_models = health_data.get("ml_models", [])
            ML_MODEL_ACCURACY.labels(model_type="regime", metric_type="count").set(len(ml_models))
            
            # Activity metrics
            api_requests = activity.get("api_requests_24h", 0)
            active_alerts = activity.get("active_alerts", 0)
            
            # Create health metrics summary
            metrics = HealthMetrics(
                geri_value=geri_value,
                geri_band=geri_band,
                data_series_fresh=fresh_count,
                data_series_stale=stale_count,
                database_connections=db_connections,
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                ml_models_active=len(ml_models),
                alert_subscriptions=active_alerts,
                recent_api_requests=api_requests,
                timestamp=datetime.utcnow().isoformat()
            )
            
            self._last_collection = datetime.utcnow()
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            # Return default metrics on error
            return HealthMetrics(
                geri_value=None,
                geri_band=None,
                data_series_fresh=0,
                data_series_stale=0,
                database_connections=0,
                cpu_percent=0,
                memory_percent=0,
                ml_models_active=0,
                alert_subscriptions=0,
                recent_api_requests=0,
                timestamp=datetime.utcnow().isoformat()
            )
    
    def record_job_execution(self, job_name: str, duration: float, success: bool):
        """Record background job execution metrics."""
        JOB_DURATION.labels(job_name=job_name).observe(duration)
        
        if success:
            JOB_SUCCESS.labels(job_name=job_name).inc()
        else:
            JOB_FAILURES.labels(job_name=job_name).inc()
    
    def record_ml_prediction(self, model_type: str, model_version: str, anomaly_score: Optional[float] = None):
        """Record ML model prediction metrics."""
        ML_PREDICTIONS.labels(model_type=model_type, model_version=model_version).inc()
        
        if anomaly_score is not None:
            ML_ANOMALY_SCORE.set(anomaly_score)
    
    def record_alert_trigger(self, alert_type: str, severity: str):
        """Record alert trigger metrics."""
        ALERT_TRIGGERS.labels(alert_type=alert_type, severity=severity).inc()
    
    def record_cache_operation(self, operation: str, tier: str):
        """Record cache operation metrics."""
        CACHE_OPERATIONS.labels(operation=operation, tier=tier).inc()
    
    def should_collect_metrics(self) -> bool:
        """Check if metrics should be collected based on interval."""
        return (datetime.utcnow() - self._last_collection).total_seconds() > self._collection_interval
    
    async def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of key metrics for dashboards."""
        try:
            metrics = await self.collect_all_metrics()
            
            return {
                "timestamp": metrics.timestamp,
                "geri": {
                    "value": metrics.geri_value,
                    "band": metrics.geri_band,
                    "status": "healthy" if metrics.geri_value and metrics.geri_value > 20 else "warning"
                },
                "data_quality": {
                    "fresh_series": metrics.data_series_fresh,
                    "stale_series": metrics.data_series_stale,
                    "freshness_ratio": metrics.data_series_fresh / (metrics.data_series_fresh + metrics.data_series_stale) if (metrics.data_series_fresh + metrics.data_series_stale) > 0 else 0
                },
                "system": {
                    "cpu_percent": metrics.cpu_percent,
                    "memory_percent": metrics.memory_percent,
                    "database_connections": metrics.database_connections,
                    "status": "healthy" if metrics.cpu_percent < 80 and metrics.memory_percent < 80 else "warning"
                },
                "ml": {
                    "active_models": metrics.ml_models_active,
                    "status": "operational"
                },
                "activity": {
                    "alert_subscriptions": metrics.alert_subscriptions,
                    "recent_requests": metrics.recent_api_requests
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting metrics summary: {e}")
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "status": "error",
                "error": str(e)
            }


# Global observability service instance
_observability_service: Optional[ObservabilityService] = None


def get_observability_service() -> ObservabilityService:
    """Get or create the global observability service."""
    global _observability_service
    if _observability_service is None:
        _observability_service = ObservabilityService()
    return _observability_service


async def collect_metrics_job():
    """Background job for metric collection."""
    observability = get_observability_service()
    
    start_time = time.time()
    success = True
    
    try:
        await observability.collect_all_metrics()
        logger.info("Metrics collection completed successfully")
        
    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        success = False
    
    finally:
        duration = time.time() - start_time
        observability.record_job_execution("metrics_collection", duration, success)


def get_prometheus_metrics() -> str:
    """Get all Prometheus metrics in the standard format."""
    return generate_latest().decode('utf-8')