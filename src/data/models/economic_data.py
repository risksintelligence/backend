"""
Economic data models for storing FRED and other economic indicators.
"""
from datetime import datetime
from typing import Optional

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Index, ForeignKey
from sqlalchemy.orm import relationship

from src.core.database import Base


class EconomicSeries(Base):
    """Economic time series metadata."""
    
    __tablename__ = "economic_series"
    
    id = Column(Integer, primary_key=True, index=True)
    series_id = Column(String(50), unique=True, index=True, nullable=False)
    title = Column(String(500), nullable=False)
    category = Column(String(100), nullable=True)  # interest_rates, inflation, etc.
    subcategory = Column(String(100), nullable=True)
    units = Column(String(100), nullable=True)
    frequency = Column(String(20), nullable=True)  # Daily, Monthly, Quarterly, Annual
    seasonal_adjustment = Column(String(50), nullable=True)
    source = Column(String(50), default="FRED")
    notes = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship to data points
    data_points = relationship("EconomicDataPoint", back_populates="series", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<EconomicSeries(series_id='{self.series_id}', title='{self.title}')>"


class EconomicDataPoint(Base):
    """Individual data points for economic time series."""
    
    __tablename__ = "economic_data_points"
    
    id = Column(Integer, primary_key=True, index=True)
    series_id = Column(String(50), ForeignKey("economic_series.series_id"), nullable=False)
    date = Column(DateTime, nullable=False)
    value = Column(Float, nullable=True)  # Some data points may be null
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship to series
    series = relationship("EconomicSeries", back_populates="data_points")
    
    __table_args__ = (
        Index("idx_series_date", "series_id", "date"),
        Index("idx_date_value", "date", "value"),
    )
    
    def __repr__(self):
        return f"<EconomicDataPoint(series_id='{self.series_id}', date='{self.date}', value={self.value})>"


class EconomicIndicator(Base):
    """Calculated economic indicators and derived metrics."""
    
    __tablename__ = "economic_indicators"
    
    id = Column(Integer, primary_key=True, index=True)
    indicator_name = Column(String(100), nullable=False)
    indicator_type = Column(String(50), nullable=False)  # composite, ratio, trend, etc.
    value = Column(Float, nullable=False)
    date = Column(DateTime, nullable=False)
    
    # Metadata about calculation
    calculation_method = Column(String(100), nullable=True)
    source_series = Column(Text, nullable=True)  # JSON list of source series IDs
    confidence_score = Column(Float, nullable=True)  # 0-1 confidence in calculation
    
    # Risk relevance
    risk_category = Column(String(50), nullable=True)  # financial, economic, supply_chain
    risk_level = Column(String(20), nullable=True)  # low, medium, high, critical
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index("idx_indicator_date", "indicator_name", "date"),
        Index("idx_risk_category", "risk_category", "risk_level"),
    )
    
    def __repr__(self):
        return f"<EconomicIndicator(name='{self.indicator_name}', value={self.value}, date='{self.date}')>"


class DataQualityCheck(Base):
    """Data quality monitoring for economic data."""
    
    __tablename__ = "data_quality_checks"
    
    id = Column(Integer, primary_key=True, index=True)
    series_id = Column(String(50), ForeignKey("economic_series.series_id"), nullable=False)
    check_type = Column(String(50), nullable=False)  # completeness, freshness, outlier, etc.
    check_date = Column(DateTime, default=datetime.utcnow)
    
    # Results
    passed = Column(String(10), nullable=False)  # "true", "false", "warning"
    score = Column(Float, nullable=True)  # 0-1 quality score
    details = Column(Text, nullable=True)  # JSON details about the check
    
    # Data range checked
    data_from = Column(DateTime, nullable=True)
    data_to = Column(DateTime, nullable=True)
    records_checked = Column(Integer, nullable=True)
    
    __table_args__ = (
        Index("idx_series_check", "series_id", "check_type", "check_date"),
    )
    
    def __repr__(self):
        return f"<DataQualityCheck(series_id='{self.series_id}', type='{self.check_type}', passed='{self.passed}')>"