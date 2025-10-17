"""
Economic data schemas for RiskX platform.
Pydantic models for economic indicators and macroeconomic data validation.
"""

from pydantic import BaseModel, Field, validator, root_validator
from typing import Optional, List, Dict, Any, Union
from datetime import datetime, date
from decimal import Decimal
from enum import Enum

from ...utils.constants import EconomicIndicators, BusinessRules


class EconomicIndicatorType(str, Enum):
    """Types of economic indicators."""
    GDP = "gdp"
    INFLATION = "inflation"
    UNEMPLOYMENT = "unemployment"
    INTEREST_RATE = "interest_rate"
    EXCHANGE_RATE = "exchange_rate"
    INDUSTRIAL_PRODUCTION = "industrial_production"
    RETAIL_SALES = "retail_sales"
    CONSUMER_CONFIDENCE = "consumer_confidence"
    BUSINESS_CONFIDENCE = "business_confidence"
    TRADE_BALANCE = "trade_balance"
    GOVERNMENT_DEBT = "government_debt"
    MONEY_SUPPLY = "money_supply"


class EconomicFrequency(str, Enum):
    """Frequency of economic data reporting."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUALLY = "annually"


class EconomicDataStatus(str, Enum):
    """Status of economic data."""
    PRELIMINARY = "preliminary"
    REVISED = "revised"
    FINAL = "final"
    ESTIMATED = "estimated"


class EconomicIndicator(BaseModel):
    """Base model for economic indicators."""
    
    indicator_id: str = Field(..., description="Unique identifier for the indicator")
    name: str = Field(..., description="Human-readable name of the indicator")
    description: Optional[str] = Field(None, description="Detailed description of the indicator")
    indicator_type: EconomicIndicatorType = Field(..., description="Type of economic indicator")
    unit: str = Field(..., description="Unit of measurement")
    frequency: EconomicFrequency = Field(..., description="Reporting frequency")
    source: str = Field(..., description="Data source (e.g., FRED, BEA, BLS)")
    source_id: Optional[str] = Field(None, description="Source-specific identifier")
    country_code: str = Field(default="US", description="ISO country code")
    region: Optional[str] = Field(None, description="Geographic region or subdivision")
    seasonal_adjustment: bool = Field(default=False, description="Whether data is seasonally adjusted")
    real_nominal: Optional[str] = Field(None, description="Real or nominal values")
    
    @validator('indicator_id')
    def validate_indicator_id(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Indicator ID cannot be empty')
        return v.strip().upper()
    
    @validator('country_code')
    def validate_country_code(cls, v):
        if len(v) != 2:
            raise ValueError('Country code must be 2 characters')
        return v.upper()


class EconomicDataPoint(BaseModel):
    """Single economic data point with metadata."""
    
    date: date = Field(..., description="Date of the observation")
    value: Decimal = Field(..., description="Observed value")
    status: EconomicDataStatus = Field(default=EconomicDataStatus.FINAL, description="Data status")
    revision_date: Optional[datetime] = Field(None, description="Date of last revision")
    confidence_interval: Optional[Dict[str, Decimal]] = Field(None, description="Confidence intervals if available")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('value')
    def validate_value(cls, v):
        if v is None:
            raise ValueError('Value cannot be None')
        return v
    
    class Config:
        json_encoders = {
            Decimal: lambda v: float(v),
            date: lambda v: v.isoformat(),
            datetime: lambda v: v.isoformat()
        }


class EconomicSeries(BaseModel):
    """Time series of economic data points."""
    
    indicator: EconomicIndicator = Field(..., description="Economic indicator metadata")
    data_points: List[EconomicDataPoint] = Field(..., description="List of data points")
    start_date: date = Field(..., description="Start date of the series")
    end_date: date = Field(..., description="End date of the series")
    total_observations: int = Field(..., description="Total number of observations")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    
    @root_validator
    def validate_series_consistency(cls, values):
        data_points = values.get('data_points', [])
        start_date = values.get('start_date')
        end_date = values.get('end_date')
        total_observations = values.get('total_observations', 0)
        
        if data_points:
            actual_count = len(data_points)
            if total_observations != actual_count:
                values['total_observations'] = actual_count
            
            dates = [dp.date for dp in data_points]
            if dates:
                actual_start = min(dates)
                actual_end = max(dates)
                
                if start_date and start_date != actual_start:
                    values['start_date'] = actual_start
                if end_date and end_date != actual_end:
                    values['end_date'] = actual_end
        
        return values


class GDPData(BaseModel):
    """Gross Domestic Product data."""
    
    nominal_gdp: Optional[Decimal] = Field(None, description="Nominal GDP value")
    real_gdp: Optional[Decimal] = Field(None, description="Real GDP value")
    gdp_growth_rate: Optional[Decimal] = Field(None, description="GDP growth rate")
    gdp_per_capita: Optional[Decimal] = Field(None, description="GDP per capita")
    gdp_deflator: Optional[Decimal] = Field(None, description="GDP deflator")
    currency: str = Field(default="USD", description="Currency denomination")
    base_year: Optional[int] = Field(None, description="Base year for real GDP calculations")
    
    @validator('gdp_growth_rate')
    def validate_gdp_growth_rate(cls, v):
        if v is not None and (v < -50 or v > 50):
            raise ValueError('GDP growth rate seems unrealistic')
        return v


class InflationData(BaseModel):
    """Consumer price index and inflation data."""
    
    cpi_value: Optional[Decimal] = Field(None, description="Consumer Price Index value")
    inflation_rate: Optional[Decimal] = Field(None, description="Inflation rate (year-over-year)")
    core_inflation: Optional[Decimal] = Field(None, description="Core inflation rate")
    ppi_value: Optional[Decimal] = Field(None, description="Producer Price Index value")
    pce_value: Optional[Decimal] = Field(None, description="Personal Consumption Expenditures value")
    base_period: str = Field(default="2020=100", description="Base period for index calculations")
    
    @validator('inflation_rate', 'core_inflation')
    def validate_inflation_rates(cls, v):
        if v is not None and (v < -25 or v > 100):
            raise ValueError('Inflation rate seems unrealistic')
        return v


class UnemploymentData(BaseModel):
    """Labor market and unemployment data."""
    
    unemployment_rate: Optional[Decimal] = Field(None, description="Unemployment rate")
    labor_force_participation: Optional[Decimal] = Field(None, description="Labor force participation rate")
    employment_population_ratio: Optional[Decimal] = Field(None, description="Employment to population ratio")
    total_unemployed: Optional[int] = Field(None, description="Total number of unemployed persons")
    total_employed: Optional[int] = Field(None, description="Total number of employed persons")
    labor_force_size: Optional[int] = Field(None, description="Total labor force size")
    nonfarm_payrolls: Optional[int] = Field(None, description="Nonfarm payroll employment")
    average_hourly_earnings: Optional[Decimal] = Field(None, description="Average hourly earnings")
    
    @validator('unemployment_rate', 'labor_force_participation', 'employment_population_ratio')
    def validate_rates(cls, v):
        if v is not None and (v < 0 or v > 100):
            raise ValueError('Rate must be between 0 and 100')
        return v


class InterestRateData(BaseModel):
    """Interest rate and monetary policy data."""
    
    federal_funds_rate: Optional[Decimal] = Field(None, description="Federal funds rate")
    prime_rate: Optional[Decimal] = Field(None, description="Prime lending rate")
    ten_year_treasury: Optional[Decimal] = Field(None, description="10-year Treasury yield")
    three_month_treasury: Optional[Decimal] = Field(None, description="3-month Treasury yield")
    thirty_year_mortgage: Optional[Decimal] = Field(None, description="30-year mortgage rate")
    yield_curve_spread: Optional[Decimal] = Field(None, description="Yield curve spread (10Y-3M)")
    real_interest_rate: Optional[Decimal] = Field(None, description="Real interest rate")
    
    @validator('federal_funds_rate', 'prime_rate', 'ten_year_treasury', 
              'three_month_treasury', 'thirty_year_mortgage')
    def validate_rates(cls, v):
        if v is not None and (v < -10 or v > 30):
            raise ValueError('Interest rate seems unrealistic')
        return v


class ExchangeRateData(BaseModel):
    """Foreign exchange rate data."""
    
    base_currency: str = Field(default="USD", description="Base currency code")
    quote_currency: str = Field(..., description="Quote currency code")
    exchange_rate: Decimal = Field(..., description="Exchange rate (base to quote)")
    bid_rate: Optional[Decimal] = Field(None, description="Bid exchange rate")
    ask_rate: Optional[Decimal] = Field(None, description="Ask exchange rate")
    volatility: Optional[Decimal] = Field(None, description="Exchange rate volatility")
    daily_change: Optional[Decimal] = Field(None, description="Daily change in exchange rate")
    daily_change_percent: Optional[Decimal] = Field(None, description="Daily change percentage")
    
    @validator('base_currency', 'quote_currency')
    def validate_currency_codes(cls, v):
        if len(v) != 3:
            raise ValueError('Currency code must be 3 characters')
        return v.upper()
    
    @validator('exchange_rate')
    def validate_exchange_rate(cls, v):
        if v <= 0:
            raise ValueError('Exchange rate must be positive')
        return v


class EconomicSummary(BaseModel):
    """Summary statistics for economic data."""
    
    indicator_type: EconomicIndicatorType = Field(..., description="Type of economic indicator")
    period_start: date = Field(..., description="Start date of summary period")
    period_end: date = Field(..., description="End date of summary period")
    observation_count: int = Field(..., description="Number of observations")
    mean_value: Optional[Decimal] = Field(None, description="Mean value")
    median_value: Optional[Decimal] = Field(None, description="Median value")
    min_value: Optional[Decimal] = Field(None, description="Minimum value")
    max_value: Optional[Decimal] = Field(None, description="Maximum value")
    standard_deviation: Optional[Decimal] = Field(None, description="Standard deviation")
    trend_direction: Optional[str] = Field(None, description="Overall trend direction")
    volatility_measure: Optional[Decimal] = Field(None, description="Volatility measure")
    last_value: Optional[Decimal] = Field(None, description="Most recent value")
    last_change: Optional[Decimal] = Field(None, description="Last period change")
    last_change_percent: Optional[Decimal] = Field(None, description="Last period change percentage")
    
    @validator('observation_count')
    def validate_observation_count(cls, v):
        if v < 0:
            raise ValueError('Observation count cannot be negative')
        return v
    
    @validator('trend_direction')
    def validate_trend_direction(cls, v):
        if v is not None and v.lower() not in ['up', 'down', 'stable', 'volatile']:
            raise ValueError('Trend direction must be one of: up, down, stable, volatile')
        return v.lower() if v else v