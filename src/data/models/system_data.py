from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()


class CacheEntry(Base):
    """Cache entries for multi-layer caching system."""
    __tablename__ = "cache_entries"

    id = Column(Integer, primary_key=True, index=True)
    cache_key = Column(String(255), unique=True, index=True, nullable=False)
    data = Column(Text, nullable=False)
    cached_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=True)
    cache_tier = Column(String(50), default="L2", nullable=False)  # L1=Redis, L2=PostgreSQL, L3=File
    data_type = Column(String(100), nullable=False)
    access_count = Column(Integer, default=0)
    last_accessed = Column(DateTime, default=datetime.utcnow)


class SystemMetric(Base):
    """System performance and health metrics."""
    __tablename__ = "system_metrics"

    id = Column(Integer, primary_key=True, index=True)
    metric_name = Column(String(100), nullable=False, index=True)
    metric_value = Column(String(255), nullable=False)
    metric_type = Column(String(50), nullable=False)  # counter, gauge, histogram
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    component = Column(String(100), nullable=False)  # api, cache, database, external_api
    environment = Column(String(50), default="production")
    metric_metadata = Column(JSON, nullable=True)


class ApiHealthCheck(Base):
    """External API health check results."""
    __tablename__ = "api_health_checks"

    id = Column(Integer, primary_key=True, index=True)
    api_name = Column(String(100), nullable=False, index=True)
    status = Column(String(50), nullable=False)  # healthy, degraded, down
    response_time_ms = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)
    checked_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    endpoint_url = Column(String(500), nullable=True)
    status_code = Column(Integer, nullable=True)


class DataSourceMonitor(Base):
    """Monitor data source freshness and availability."""
    __tablename__ = "data_source_monitors"

    id = Column(Integer, primary_key=True, index=True)
    source_name = Column(String(100), nullable=False, index=True)
    series_id = Column(String(100), nullable=True, index=True)
    last_update_attempt = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_successful_update = Column(DateTime, nullable=True)
    status = Column(String(50), default="unknown")  # fresh, stale, error, unavailable
    error_count = Column(Integer, default=0)
    success_count = Column(Integer, default=0)
    data_freshness_hours = Column(Integer, nullable=True)
    is_critical = Column(Boolean, default=False)


class User(Base):
    """User accounts for authentication."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, index=True, nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_login = Column(DateTime, nullable=True)
    failed_login_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime, nullable=True)


class AuditLog(Base):
    """Audit trail for system activities."""
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=True, index=True)
    action = Column(String(100), nullable=False, index=True)
    resource = Column(String(255), nullable=False)
    resource_id = Column(String(100), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)
    details = Column(JSON, nullable=True)
    status = Column(String(50), default="success")  # success, failure, warning