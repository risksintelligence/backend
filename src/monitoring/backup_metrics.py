"""
Backup and Recovery Monitoring Metrics
Prometheus metrics for backup system observability
"""

from prometheus_client import Counter, Gauge, Histogram
from typing import Dict, Any

# Backup execution metrics
BACKUP_SUCCESS_TOTAL = Counter(
    'ris_backup_success_total',
    'Total number of successful backups',
    labelnames=['backup_type']
)

BACKUP_ERROR_TOTAL = Counter(
    'ris_backup_error_total', 
    'Total number of failed backups',
    labelnames=['backup_type', 'error_type']
)

BACKUP_DURATION_SECONDS = Histogram(
    'ris_backup_duration_seconds',
    'Duration of backup operations in seconds',
    labelnames=['backup_type'],
    buckets=[30, 60, 120, 300, 600, 1200, 1800]  # 30s to 30min
)

# Backup file metrics
BACKUP_FILE_SIZE_BYTES = Gauge(
    'ris_backup_file_size_bytes',
    'Size of backup files in bytes',
    labelnames=['backup_type', 'compressed']
)

BACKUP_LAST_SUCCESS_TIMESTAMP = Gauge(
    'ris_backup_last_success_timestamp_seconds',
    'Timestamp of last successful backup',
    labelnames=['backup_type']
)

# Storage metrics
BACKUP_STORAGE_USED_BYTES = Gauge(
    'ris_backup_storage_used_bytes',
    'Total backup storage used in bytes',
    labelnames=['storage_type']  # 'local', 's3'
)

BACKUP_STORAGE_AVAILABLE_BYTES = Gauge(
    'ris_backup_storage_available_bytes',
    'Available backup storage in bytes',
    labelnames=['storage_type']
)

# Upload metrics  
BACKUP_UPLOAD_SUCCESS_TOTAL = Counter(
    'ris_backup_upload_success_total',
    'Total number of successful backup uploads',
    labelnames=['storage_provider']
)

BACKUP_UPLOAD_ERROR_TOTAL = Counter(
    'ris_backup_upload_error_total',
    'Total number of failed backup uploads', 
    labelnames=['storage_provider', 'error_type']
)

BACKUP_UPLOAD_DURATION_SECONDS = Histogram(
    'ris_backup_upload_duration_seconds',
    'Duration of backup upload operations in seconds',
    labelnames=['storage_provider'],
    buckets=[10, 30, 60, 120, 300, 600, 1800]  # 10s to 30min
)

# Recovery metrics
BACKUP_RECOVERY_SUCCESS_TOTAL = Counter(
    'ris_backup_recovery_success_total',
    'Total number of successful recovery operations',
    labelnames=['recovery_type']
)

BACKUP_RECOVERY_ERROR_TOTAL = Counter(
    'ris_backup_recovery_error_total',
    'Total number of failed recovery operations',
    labelnames=['recovery_type', 'error_type'] 
)

BACKUP_RECOVERY_DURATION_SECONDS = Histogram(
    'ris_backup_recovery_duration_seconds',
    'Duration of recovery operations in seconds',
    labelnames=['recovery_type'],
    buckets=[60, 300, 600, 1800, 3600, 7200]  # 1min to 2hrs
)

# Health metrics
BACKUP_SYSTEM_HEALTH = Gauge(
    'ris_backup_system_health',
    'Backup system health status (1=healthy, 0=unhealthy)',
    labelnames=['component']  # 'postgres', 'redis', 's3', 'filesystem'
)

BACKUP_RETENTION_DAYS = Gauge(
    'ris_backup_retention_days',
    'Backup retention period in days',
    labelnames=['backup_type', 'storage_type']
)

# Alert metrics
BACKUP_ALERT_TRIGGERED_TOTAL = Counter(
    'ris_backup_alert_triggered_total',
    'Total number of backup-related alerts triggered',
    labelnames=['alert_type']  # 'failure', 'storage_low', 'old_backup'
)

class BackupMetricsCollector:
    """Collector for backup system metrics."""
    
    @staticmethod
    def record_backup_success(backup_type: str, duration: float, file_size: int, compressed: bool = False):
        """Record successful backup metrics."""
        BACKUP_SUCCESS_TOTAL.labels(backup_type=backup_type).inc()
        BACKUP_DURATION_SECONDS.labels(backup_type=backup_type).observe(duration)
        BACKUP_FILE_SIZE_BYTES.labels(backup_type=backup_type, compressed=str(compressed)).set(file_size)
        BACKUP_LAST_SUCCESS_TIMESTAMP.labels(backup_type=backup_type).set_to_current_time()
    
    @staticmethod
    def record_backup_failure(backup_type: str, duration: float, error_type: str):
        """Record backup failure metrics."""
        BACKUP_ERROR_TOTAL.labels(backup_type=backup_type, error_type=error_type).inc()
        BACKUP_DURATION_SECONDS.labels(backup_type=backup_type).observe(duration)
    
    @staticmethod
    def record_upload_success(provider: str, duration: float):
        """Record successful upload metrics."""
        BACKUP_UPLOAD_SUCCESS_TOTAL.labels(storage_provider=provider).inc()
        BACKUP_UPLOAD_DURATION_SECONDS.labels(storage_provider=provider).observe(duration)
    
    @staticmethod
    def record_upload_failure(provider: str, duration: float, error_type: str):
        """Record upload failure metrics.""" 
        BACKUP_UPLOAD_ERROR_TOTAL.labels(storage_provider=provider, error_type=error_type).inc()
        BACKUP_UPLOAD_DURATION_SECONDS.labels(storage_provider=provider).observe(duration)
    
    @staticmethod
    def record_recovery_success(recovery_type: str, duration: float):
        """Record successful recovery metrics."""
        BACKUP_RECOVERY_SUCCESS_TOTAL.labels(recovery_type=recovery_type).inc()
        BACKUP_RECOVERY_DURATION_SECONDS.labels(recovery_type=recovery_type).observe(duration)
    
    @staticmethod
    def record_recovery_failure(recovery_type: str, duration: float, error_type: str):
        """Record recovery failure metrics."""
        BACKUP_RECOVERY_ERROR_TOTAL.labels(recovery_type=recovery_type, error_type=error_type).inc()
        BACKUP_RECOVERY_DURATION_SECONDS.labels(recovery_type=recovery_type).observe(duration)
    
    @staticmethod
    def set_system_health(component: str, healthy: bool):
        """Set backup system health status."""
        BACKUP_SYSTEM_HEALTH.labels(component=component).set(1 if healthy else 0)
    
    @staticmethod
    def set_storage_metrics(storage_type: str, used_bytes: int, available_bytes: int):
        """Set storage usage metrics."""
        BACKUP_STORAGE_USED_BYTES.labels(storage_type=storage_type).set(used_bytes)
        BACKUP_STORAGE_AVAILABLE_BYTES.labels(storage_type=storage_type).set(available_bytes)
    
    @staticmethod
    def trigger_backup_alert(alert_type: str):
        """Record backup alert trigger."""
        BACKUP_ALERT_TRIGGERED_TOTAL.labels(alert_type=alert_type).inc()