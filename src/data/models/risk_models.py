"""
Risk assessment database models
"""
from sqlalchemy import Column, Integer, Float, String, DateTime, Text, Boolean, JSON, Index
from sqlalchemy.sql import func
from src.core.database import Base


class RiskScore(Base):
    """Historical risk scores with time series tracking."""
    
    __tablename__ = "risk_scores"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    overall_score = Column(Float, nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    trend = Column(String(20), nullable=False)  # 'rising', 'falling', 'stable'
    
    # Component scores
    economic_score = Column(Float, nullable=False)
    market_score = Column(Float, nullable=False)
    geopolitical_score = Column(Float, nullable=False)
    technical_score = Column(Float, nullable=False)
    
    # Metadata
    data_sources = Column(JSON)  # List of sources used
    calculation_method = Column(String(50), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_risk_scores_timestamp', 'timestamp'),
        Index('idx_risk_scores_overall', 'overall_score'),
        Index('idx_risk_scores_trend', 'trend'),
    )


class RiskFactor(Base):
    """Individual risk factors with categories and weights."""
    
    __tablename__ = "risk_factors"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False, unique=True, index=True)
    category = Column(String(50), nullable=False, index=True)  # 'economic', 'market', 'geopolitical', 'technical'
    description = Column(Text)
    
    # Current values
    current_value = Column(Float, nullable=False)
    current_score = Column(Float, nullable=False)  # 0-100 normalized score
    impact_level = Column(String(20), nullable=False)  # 'low', 'moderate', 'high', 'critical'
    
    # Configuration
    weight = Column(Float, default=1.0, nullable=False)  # Weighting in overall calculation
    threshold_low = Column(Float)  # Low threshold value
    threshold_high = Column(Float)  # High threshold value
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Data source information
    data_source = Column(String(50), nullable=False)  # 'fred', 'bea', 'bls', etc.
    series_id = Column(String(100))  # External API series identifier
    update_frequency = Column(String(20))  # 'daily', 'weekly', 'monthly', 'quarterly'
    
    # Timestamps
    last_updated = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_risk_factors_category', 'category'),
        Index('idx_risk_factors_active', 'is_active'),
        Index('idx_risk_factors_source', 'data_source'),
    )


class EconomicIndicator(Base):
    """Economic indicators time series data."""
    
    __tablename__ = "economic_indicators"
    
    id = Column(Integer, primary_key=True, index=True)
    series_id = Column(String(100), nullable=False, index=True)
    source = Column(String(50), nullable=False, index=True)
    name = Column(String(200), nullable=False)
    
    # Data values
    value = Column(Float, nullable=False)
    units = Column(String(100), nullable=False)
    frequency = Column(String(20), nullable=False)  # 'daily', 'weekly', 'monthly', 'quarterly', 'annual'
    
    # Time information
    observation_date = Column(DateTime(timezone=True), nullable=False, index=True)
    period = Column(String(20))  # Original period from source (e.g., "2024-Q3", "2024-10")
    
    # Metadata
    seasonal_adjustment = Column(String(50))
    revision_status = Column(String(20))  # 'preliminary', 'revised', 'final'
    
    # Change calculations
    period_change = Column(Float)  # Change from previous period
    year_over_year_change = Column(Float)  # Change from same period last year
    
    # Timestamps
    fetched_at = Column(DateTime(timezone=True), server_default=func.now())
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_economic_indicators_series_date', 'series_id', 'observation_date'),
        Index('idx_economic_indicators_source_date', 'source', 'observation_date'),
        Index('idx_economic_indicators_frequency', 'frequency'),
    )


class User(Base):
    """User accounts and authentication."""
    
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    
    # Profile information
    full_name = Column(String(200))
    organization = Column(String(200))
    role = Column(String(50), default='user')  # 'user', 'admin', 'analyst'
    
    # Account status
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    is_superuser = Column(Boolean, default=False, nullable=False)
    
    # Authentication
    last_login = Column(DateTime(timezone=True))
    failed_login_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime(timezone=True))
    
    # Preferences
    preferences = Column(JSON)  # User dashboard preferences
    api_key = Column(String(255), unique=True, index=True)  # For API access
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_users_active', 'is_active'),
        Index('idx_users_role', 'role'),
    )


class Alert(Base):
    """Risk alerts and notifications."""
    
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    alert_type = Column(String(50), nullable=False, index=True)  # 'risk_threshold', 'data_anomaly', 'system_error'
    severity = Column(String(20), nullable=False, index=True)  # 'low', 'medium', 'high', 'critical'
    title = Column(String(200), nullable=False)
    message = Column(Text, nullable=False)
    
    # Alert details
    triggered_by = Column(String(100))  # Risk factor or system component that triggered alert
    threshold_value = Column(Float)  # Threshold that was breached
    current_value = Column(Float)  # Current value that triggered alert
    
    # Status
    status = Column(String(20), default='active', nullable=False)  # 'active', 'acknowledged', 'resolved'
    acknowledged_by = Column(Integer)  # User ID who acknowledged
    acknowledged_at = Column(DateTime(timezone=True))
    resolved_at = Column(DateTime(timezone=True))
    
    # Additional data
    alert_metadata = Column(JSON)  # Additional alert-specific data
    
    # Timestamps
    triggered_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_alerts_type_severity', 'alert_type', 'severity'),
        Index('idx_alerts_status', 'status'),
        Index('idx_alerts_triggered', 'triggered_at'),
    )


class SystemMetric(Base):
    """System performance and health metrics."""
    
    __tablename__ = "system_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    metric_name = Column(String(100), nullable=False, index=True)
    metric_type = Column(String(50), nullable=False)  # 'counter', 'gauge', 'histogram', 'timing'
    
    # Metric values
    value = Column(Float, nullable=False)
    unit = Column(String(50))
    
    # Context
    component = Column(String(100), nullable=False, index=True)  # 'cache', 'database', 'api', 'worker'
    environment = Column(String(20), default='production')
    instance_id = Column(String(100))
    
    # Additional data
    tags = Column(JSON)  # Key-value tags for grouping/filtering
    metric_metadata = Column(JSON)  # Additional metric metadata
    
    # Timestamps
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_system_metrics_name_component', 'metric_name', 'component'),
        Index('idx_system_metrics_timestamp', 'timestamp'),
        Index('idx_system_metrics_type', 'metric_type'),
    )


class CacheEntry(Base):
    """Cache entries for L2 PostgreSQL caching."""
    
    __tablename__ = "cache_entries"
    
    cache_key = Column(String(255), primary_key=True, index=True)
    data = Column(Text, nullable=False)  # JSON serialized data
    ttl_seconds = Column(Integer)
    
    # Metadata
    data_source = Column(String(50))
    cache_tier = Column(String(10), default='L2')
    size_bytes = Column(Integer)
    
    # Timestamps
    cached_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    expires_at = Column(DateTime(timezone=True), index=True)
    accessed_at = Column(DateTime(timezone=True), server_default=func.now())
    access_count = Column(Integer, default=1)
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_cache_entries_expires', 'expires_at'),
        Index('idx_cache_entries_source', 'data_source'),
    )