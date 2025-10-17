"""
System monitoring and operational data models.
"""
from datetime import datetime
from typing import Optional

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Index, Boolean

from src.core.database import Base


class DataSourceStatus(Base):
    """Status tracking for external data sources."""
    
    __tablename__ = "data_source_status"
    
    id = Column(Integer, primary_key=True, index=True)
    source_name = Column(String(50), nullable=False)  # fred, census, noaa, etc.
    endpoint = Column(String(200), nullable=True)  # specific API endpoint
    
    # Status information
    status = Column(String(20), nullable=False)  # healthy, degraded, failed
    last_success = Column(DateTime, nullable=True)
    last_failure = Column(DateTime, nullable=True)
    last_checked = Column(DateTime, default=datetime.utcnow)
    
    # Performance metrics
    success_count = Column(Integer, default=0)
    failure_count = Column(Integer, default=0)
    avg_response_time = Column(Float, nullable=True)  # seconds
    
    # Current error information
    last_error = Column(Text, nullable=True)
    error_count_24h = Column(Integer, default=0)
    
    # Data freshness
    last_data_update = Column(DateTime, nullable=True)
    expected_update_frequency = Column(String(20), nullable=True)  # daily, hourly, etc.
    data_delay_minutes = Column(Integer, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index("idx_source_status", "source_name", "status"),
        Index("idx_last_checked", "last_checked", "status"),
    )
    
    def __repr__(self):
        return f"<DataSourceStatus(source='{self.source_name}', status='{self.status}')>"


class SystemMetrics(Base):
    """System performance and health metrics."""
    
    __tablename__ = "system_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    metric_name = Column(String(100), nullable=False)
    metric_category = Column(String(50), nullable=False)  # performance, cache, api, etc.
    
    # Metric values
    value = Column(Float, nullable=False)
    unit = Column(String(20), nullable=True)  # seconds, bytes, count, percentage
    
    # Context
    component = Column(String(50), nullable=True)  # api, cache, database, etc.
    environment = Column(String(20), nullable=True)  # development, staging, production
    
    # Temporal
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Additional data
    additional_data = Column(Text, nullable=True)  # JSON with additional metric data
    
    __table_args__ = (
        Index("idx_metric_time", "metric_name", "timestamp"),
        Index("idx_category_component", "metric_category", "component"),
    )
    
    def __repr__(self):
        return f"<SystemMetrics(name='{self.metric_name}', value={self.value}, unit='{self.unit}')>"


class AuditLog(Base):
    """Audit log for system operations and data changes."""
    
    __tablename__ = "audit_log"
    
    id = Column(Integer, primary_key=True, index=True)
    operation_type = Column(String(50), nullable=False)  # create, update, delete, api_call, etc.
    resource_type = Column(String(50), nullable=False)  # risk_score, prediction, data_source, etc.
    resource_id = Column(String(100), nullable=True)
    
    # Operation details
    operation_description = Column(Text, nullable=True)
    old_values = Column(Text, nullable=True)  # JSON of old values for updates
    new_values = Column(Text, nullable=True)  # JSON of new values
    
    # User/system context
    user_id = Column(String(100), nullable=True)  # If user-initiated
    source_ip = Column(String(45), nullable=True)  # IP address
    user_agent = Column(String(500), nullable=True)
    
    # System context
    component = Column(String(50), nullable=True)  # Which system component
    correlation_id = Column(String(100), nullable=True)  # For tracing related operations
    
    # Result
    success = Column(Boolean, nullable=False)
    error_message = Column(Text, nullable=True)
    
    # Temporal
    timestamp = Column(DateTime, default=datetime.utcnow)
    duration_ms = Column(Float, nullable=True)  # Operation duration
    
    __table_args__ = (
        Index("idx_operation_time", "operation_type", "timestamp"),
        Index("idx_resource", "resource_type", "resource_id"),
        Index("idx_user_operations", "user_id", "timestamp"),
    )
    
    def __repr__(self):
        return f"<AuditLog(operation='{self.operation_type}', resource='{self.resource_type}', success={self.success})>"


class CacheMetrics(Base):
    """Cache performance and utilization metrics."""
    
    __tablename__ = "cache_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    cache_type = Column(String(20), nullable=False)  # redis, postgres
    
    # Usage metrics
    total_keys = Column(Integer, nullable=True)
    active_keys = Column(Integer, nullable=True)
    expired_keys = Column(Integer, nullable=True)
    
    # Performance metrics
    hit_rate = Column(Float, nullable=True)  # 0-1 cache hit rate
    miss_rate = Column(Float, nullable=True)  # 0-1 cache miss rate
    avg_get_time_ms = Column(Float, nullable=True)
    avg_set_time_ms = Column(Float, nullable=True)
    
    # Memory/storage usage
    memory_used_bytes = Column(Integer, nullable=True)
    memory_available_bytes = Column(Integer, nullable=True)
    storage_used_bytes = Column(Integer, nullable=True)
    
    # Connection metrics
    active_connections = Column(Integer, nullable=True)
    total_commands = Column(Integer, nullable=True)
    
    # Time period
    measurement_start = Column(DateTime, nullable=False)
    measurement_end = Column(DateTime, nullable=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index("idx_cache_time", "cache_type", "measurement_end"),
        Index("idx_performance", "hit_rate", "measurement_end"),
    )
    
    def __repr__(self):
        return f"<CacheMetrics(type='{self.cache_type}', hit_rate={self.hit_rate}, keys={self.total_keys})>"


class AlertLog(Base):
    """System alerts and notifications."""
    
    __tablename__ = "alert_log"
    
    id = Column(Integer, primary_key=True, index=True)
    alert_type = Column(String(50), nullable=False)  # data_quality, system_health, risk_threshold, etc.
    severity = Column(String(20), nullable=False)  # info, warning, error, critical
    
    # Alert content
    title = Column(String(200), nullable=False)
    message = Column(Text, nullable=False)
    details = Column(Text, nullable=True)  # JSON with additional details
    
    # Source information
    component = Column(String(50), nullable=True)  # Which component generated the alert
    source_entity = Column(String(100), nullable=True)  # Specific entity (series_id, model_name, etc.)
    
    # Alert lifecycle
    created_at = Column(DateTime, default=datetime.utcnow)
    acknowledged_at = Column(DateTime, nullable=True)
    resolved_at = Column(DateTime, nullable=True)
    
    # Status
    status = Column(String(20), default="active")  # active, acknowledged, resolved, suppressed
    is_active = Column(Boolean, default=True)
    
    # Notification tracking
    notifications_sent = Column(Integer, default=0)
    last_notification = Column(DateTime, nullable=True)
    
    # Resolution
    resolution_notes = Column(Text, nullable=True)
    resolved_by = Column(String(100), nullable=True)
    
    __table_args__ = (
        Index("idx_alert_severity", "alert_type", "severity", "created_at"),
        Index("idx_active_alerts", "is_active", "status", "created_at"),
        Index("idx_component_alerts", "component", "created_at"),
    )
    
    def __repr__(self):
        return f"<AlertLog(type='{self.alert_type}', severity='{self.severity}', title='{self.title[:50]}')>"