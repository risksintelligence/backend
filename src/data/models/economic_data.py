from sqlalchemy import Column, Integer, String, DateTime, Float, Text, Boolean, JSON, Index
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()


class EconomicIndicator(Base):
    """Economic indicators from various government sources."""
    __tablename__ = "economic_indicators"

    id = Column(Integer, primary_key=True, index=True)
    series_id = Column(String(100), nullable=False, index=True)
    source = Column(String(50), nullable=False, index=True)  # FRED, BEA, BLS, Census
    indicator_name = Column(String(255), nullable=False)
    category = Column(String(100), nullable=False, index=True)  # gdp, employment, inflation, etc
    subcategory = Column(String(100), nullable=True)
    units = Column(String(100), nullable=False)
    frequency = Column(String(50), nullable=False)  # annual, quarterly, monthly, daily
    seasonal_adjustment = Column(String(50), nullable=True)
    last_updated = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    is_active = Column(Boolean, default=True)
    description = Column(Text, nullable=True)
    geographic_level = Column(String(100), nullable=True)  # national, regional, state
    
    # Create composite index for efficient queries
    __table_args__ = (
        Index('idx_source_category', 'source', 'category'),
        Index('idx_series_source', 'series_id', 'source'),
    )


class EconomicDataPoint(Base):
    """Individual data points for economic indicators."""
    __tablename__ = "economic_data_points"

    id = Column(Integer, primary_key=True, index=True)
    series_id = Column(String(100), nullable=False, index=True)
    date = Column(DateTime, nullable=False, index=True)
    value = Column(Float, nullable=False)
    preliminary = Column(Boolean, default=False)
    revised = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    source_timestamp = Column(DateTime, nullable=True)
    quality_flag = Column(String(10), nullable=True)  # A=good, B=caution, C=poor
    
    # Create composite index for efficient time series queries
    __table_args__ = (
        Index('idx_series_date', 'series_id', 'date'),
        Index('idx_date_series', 'date', 'series_id'),
    )


class MarketIndicator(Base):
    """Market-based indicators and financial metrics."""
    __tablename__ = "market_indicators"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(50), nullable=False, index=True)
    indicator_type = Column(String(100), nullable=False, index=True)  # index, rate, spread, volatility
    name = Column(String(255), nullable=False)
    current_value = Column(Float, nullable=False)
    previous_value = Column(Float, nullable=True)
    change_value = Column(Float, nullable=True)
    change_percent = Column(Float, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    market_open = Column(Boolean, default=True)
    data_source = Column(String(100), nullable=False)
    currency = Column(String(10), default="USD")


class FredSeries(Base):
    """FRED-specific series metadata and configuration."""
    __tablename__ = "fred_series"

    id = Column(Integer, primary_key=True, index=True)
    series_id = Column(String(100), unique=True, nullable=False, index=True)
    title = Column(String(500), nullable=False)
    observation_start = Column(DateTime, nullable=True)
    observation_end = Column(DateTime, nullable=True)
    frequency = Column(String(50), nullable=False)
    frequency_short = Column(String(10), nullable=False)
    units = Column(String(255), nullable=False)
    units_short = Column(String(50), nullable=False)
    seasonal_adjustment = Column(String(100), nullable=True)
    seasonal_adjustment_short = Column(String(10), nullable=True)
    last_updated = Column(DateTime, nullable=True, index=True)
    popularity = Column(Integer, default=0)
    group_popularity = Column(Integer, default=0)
    notes = Column(Text, nullable=True)
    is_monitored = Column(Boolean, default=False)
    update_frequency_minutes = Column(Integer, default=1440)  # daily by default


class BeaDataset(Base):
    """BEA dataset metadata and configuration."""
    __tablename__ = "bea_datasets"

    id = Column(Integer, primary_key=True, index=True)
    dataset_name = Column(String(100), nullable=False, index=True)
    table_name = Column(String(100), nullable=False, index=True)
    table_id = Column(String(50), nullable=False, index=True)
    line_code = Column(String(50), nullable=True)
    description = Column(Text, nullable=False)
    unit = Column(String(100), nullable=False)
    multiplier = Column(String(50), nullable=True)
    last_updated = Column(DateTime, nullable=True, index=True)
    is_monitored = Column(Boolean, default=False)
    data_type = Column(String(100), nullable=False)  # NIPA, Regional, Industry
    geographic_level = Column(String(50), default="National")


class BlsSeries(Base):
    """BLS series metadata and configuration."""
    __tablename__ = "bls_series"

    id = Column(Integer, primary_key=True, index=True)
    series_id = Column(String(100), unique=True, nullable=False, index=True)
    series_title = Column(String(500), nullable=False)
    survey_name = Column(String(200), nullable=False, index=True)
    measure_code = Column(String(50), nullable=True)
    seasonal = Column(String(10), nullable=True)  # S, U for seasonal/unadjusted
    periodicity_code = Column(String(10), nullable=False)
    area_code = Column(String(50), nullable=True)
    area_text = Column(String(200), nullable=True)
    industry_code = Column(String(50), nullable=True)
    industry_text = Column(String(200), nullable=True)
    occupation_code = Column(String(50), nullable=True)
    occupation_text = Column(String(200), nullable=True)
    last_updated = Column(DateTime, nullable=True, index=True)
    is_monitored = Column(Boolean, default=False)


class EconomicForecast(Base):
    """Economic forecasts and projections."""
    __tablename__ = "economic_forecasts"

    id = Column(Integer, primary_key=True, index=True)
    series_id = Column(String(100), nullable=False, index=True)
    forecast_date = Column(DateTime, nullable=False, index=True)  # Date forecast was made
    target_date = Column(DateTime, nullable=False, index=True)    # Date being forecasted
    forecasted_value = Column(Float, nullable=False)
    confidence_interval_lower = Column(Float, nullable=True)
    confidence_interval_upper = Column(Float, nullable=True)
    confidence_level = Column(Float, default=0.95)
    model_type = Column(String(100), nullable=False)  # ARIMA, ML, consensus, etc
    model_version = Column(String(50), nullable=True)
    forecast_horizon_days = Column(Integer, nullable=False)
    actual_value = Column(Float, nullable=True)  # Filled in once actual data available
    forecast_error = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Create composite index for efficient forecast queries
    __table_args__ = (
        Index('idx_series_target_date', 'series_id', 'target_date'),
        Index('idx_forecast_target_date', 'forecast_date', 'target_date'),
    )