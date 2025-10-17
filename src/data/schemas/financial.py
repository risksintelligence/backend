"""
Financial data schemas for RiskX platform.
Pydantic models for financial instruments, market data, and risk metrics validation.
"""

from pydantic import BaseModel, Field, validator, root_validator
from typing import Optional, List, Dict, Any, Union
from datetime import datetime, date, time
from decimal import Decimal
from enum import Enum

from ...utils.constants import BusinessRules


class AssetClass(str, Enum):
    """Types of financial asset classes."""
    EQUITY = "equity"
    FIXED_INCOME = "fixed_income"
    COMMODITY = "commodity"
    CURRENCY = "currency"
    DERIVATIVE = "derivative"
    REAL_ESTATE = "real_estate"
    ALTERNATIVE = "alternative"


class MarketType(str, Enum):
    """Types of financial markets."""
    STOCK = "stock"
    BOND = "bond"
    FOREX = "forex"
    COMMODITY = "commodity"
    DERIVATIVE = "derivative"
    MONEY_MARKET = "money_market"
    CAPITAL_MARKET = "capital_market"


class RiskRating(str, Enum):
    """Risk rating classifications."""
    AAA = "AAA"
    AA = "AA"
    A = "A"
    BBB = "BBB"
    BB = "BB"
    B = "B"
    CCC = "CCC"
    CC = "CC"
    C = "C"
    D = "D"
    NR = "NR"  # Not Rated


class InstrumentType(str, Enum):
    """Types of financial instruments."""
    STOCK = "stock"
    BOND = "bond"
    OPTION = "option"
    FUTURE = "future"
    SWAP = "swap"
    ETF = "etf"
    MUTUAL_FUND = "mutual_fund"
    CD = "certificate_of_deposit"
    TREASURY = "treasury"
    CORPORATE_BOND = "corporate_bond"
    MUNICIPAL_BOND = "municipal_bond"


class FinancialInstrument(BaseModel):
    """Base model for financial instruments."""
    
    symbol: str = Field(..., description="Instrument symbol or ticker")
    name: str = Field(..., description="Full name of the instrument")
    instrument_type: InstrumentType = Field(..., description="Type of financial instrument")
    asset_class: AssetClass = Field(..., description="Asset class category")
    exchange: Optional[str] = Field(None, description="Exchange where traded")
    currency: str = Field(default="USD", description="Currency denomination")
    isin: Optional[str] = Field(None, description="International Securities Identification Number")
    cusip: Optional[str] = Field(None, description="Committee on Uniform Securities Identification Procedures")
    country: str = Field(default="US", description="Country of origin")
    sector: Optional[str] = Field(None, description="Economic sector")
    industry: Optional[str] = Field(None, description="Industry classification")
    market_cap_category: Optional[str] = Field(None, description="Market capitalization category")
    
    @validator('symbol')
    def validate_symbol(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Symbol cannot be empty')
        return v.strip().upper()
    
    @validator('currency', 'country')
    def validate_codes(cls, v):
        if len(v) < 2 or len(v) > 3:
            raise ValueError('Currency and country codes must be 2-3 characters')
        return v.upper()


class PriceData(BaseModel):
    """Price and quote data for financial instruments."""
    
    timestamp: datetime = Field(..., description="Timestamp of the price quote")
    open_price: Optional[Decimal] = Field(None, description="Opening price")
    high_price: Optional[Decimal] = Field(None, description="High price")
    low_price: Optional[Decimal] = Field(None, description="Low price")
    close_price: Optional[Decimal] = Field(None, description="Closing price")
    bid_price: Optional[Decimal] = Field(None, description="Bid price")
    ask_price: Optional[Decimal] = Field(None, description="Ask price")
    last_price: Optional[Decimal] = Field(None, description="Last traded price")
    previous_close: Optional[Decimal] = Field(None, description="Previous closing price")
    price_change: Optional[Decimal] = Field(None, description="Price change from previous close")
    price_change_percent: Optional[Decimal] = Field(None, description="Percentage price change")
    
    @validator('open_price', 'high_price', 'low_price', 'close_price', 
              'bid_price', 'ask_price', 'last_price', 'previous_close')
    def validate_prices(cls, v):
        if v is not None and v < 0:
            raise ValueError('Prices cannot be negative')
        return v
    
    @root_validator
    def validate_price_relationships(cls, values):
        high = values.get('high_price')
        low = values.get('low_price')
        open_price = values.get('open_price')
        close_price = values.get('close_price')
        
        if high is not None and low is not None and high < low:
            raise ValueError('High price cannot be less than low price')
        
        if high is not None and open_price is not None and open_price > high:
            raise ValueError('Open price cannot be higher than high price')
        
        if low is not None and close_price is not None and close_price < low:
            raise ValueError('Close price cannot be lower than low price')
        
        return values


class TradingVolume(BaseModel):
    """Trading volume and activity data."""
    
    volume: Optional[int] = Field(None, description="Number of shares/contracts traded")
    volume_weighted_avg_price: Optional[Decimal] = Field(None, description="Volume weighted average price")
    trade_count: Optional[int] = Field(None, description="Number of individual trades")
    turnover: Optional[Decimal] = Field(None, description="Total monetary value traded")
    average_trade_size: Optional[Decimal] = Field(None, description="Average size per trade")
    bid_size: Optional[int] = Field(None, description="Shares available at bid price")
    ask_size: Optional[int] = Field(None, description="Shares available at ask price")
    
    @validator('volume', 'trade_count', 'bid_size', 'ask_size')
    def validate_counts(cls, v):
        if v is not None and v < 0:
            raise ValueError('Counts cannot be negative')
        return v


class StockData(BaseModel):
    """Stock-specific data and metrics."""
    
    instrument: FinancialInstrument = Field(..., description="Stock instrument details")
    price: PriceData = Field(..., description="Current price information")
    volume: TradingVolume = Field(..., description="Trading volume data")
    shares_outstanding: Optional[int] = Field(None, description="Total shares outstanding")
    market_capitalization: Optional[Decimal] = Field(None, description="Market capitalization")
    book_value_per_share: Optional[Decimal] = Field(None, description="Book value per share")
    earnings_per_share: Optional[Decimal] = Field(None, description="Earnings per share")
    price_to_earnings: Optional[Decimal] = Field(None, description="Price to earnings ratio")
    price_to_book: Optional[Decimal] = Field(None, description="Price to book ratio")
    dividend_yield: Optional[Decimal] = Field(None, description="Dividend yield percentage")
    beta: Optional[Decimal] = Field(None, description="Beta coefficient")
    fifty_two_week_high: Optional[Decimal] = Field(None, description="52-week high price")
    fifty_two_week_low: Optional[Decimal] = Field(None, description="52-week low price")
    
    @validator('shares_outstanding')
    def validate_shares_outstanding(cls, v):
        if v is not None and v <= 0:
            raise ValueError('Shares outstanding must be positive')
        return v


class BondData(BaseModel):
    """Bond-specific data and metrics."""
    
    instrument: FinancialInstrument = Field(..., description="Bond instrument details")
    price: PriceData = Field(..., description="Current price information")
    face_value: Decimal = Field(..., description="Face value of the bond")
    coupon_rate: Decimal = Field(..., description="Annual coupon rate")
    yield_to_maturity: Optional[Decimal] = Field(None, description="Yield to maturity")
    current_yield: Optional[Decimal] = Field(None, description="Current yield")
    duration: Optional[Decimal] = Field(None, description="Duration measure")
    convexity: Optional[Decimal] = Field(None, description="Convexity measure")
    maturity_date: date = Field(..., description="Bond maturity date")
    issue_date: date = Field(..., description="Bond issue date")
    credit_rating: Optional[RiskRating] = Field(None, description="Credit rating")
    issuer: str = Field(..., description="Bond issuer")
    callable: bool = Field(default=False, description="Whether bond is callable")
    accrued_interest: Optional[Decimal] = Field(None, description="Accrued interest")
    
    @validator('face_value')
    def validate_face_value(cls, v):
        if v <= 0:
            raise ValueError('Face value must be positive')
        return v
    
    @validator('coupon_rate')
    def validate_coupon_rate(cls, v):
        if v < 0 or v > 100:
            raise ValueError('Coupon rate must be between 0 and 100')
        return v


class CurrencyData(BaseModel):
    """Foreign exchange currency data."""
    
    base_currency: str = Field(..., description="Base currency code")
    quote_currency: str = Field(..., description="Quote currency code")
    exchange_rate: Decimal = Field(..., description="Current exchange rate")
    bid_rate: Optional[Decimal] = Field(None, description="Bid exchange rate")
    ask_rate: Optional[Decimal] = Field(None, description="Ask exchange rate")
    spread: Optional[Decimal] = Field(None, description="Bid-ask spread")
    daily_high: Optional[Decimal] = Field(None, description="Daily high rate")
    daily_low: Optional[Decimal] = Field(None, description="Daily low rate")
    daily_change: Optional[Decimal] = Field(None, description="Daily change")
    daily_change_percent: Optional[Decimal] = Field(None, description="Daily change percentage")
    volatility: Optional[Decimal] = Field(None, description="Currency volatility")
    volume: Optional[Decimal] = Field(None, description="Trading volume")
    
    @validator('base_currency', 'quote_currency')
    def validate_currency_codes(cls, v):
        if len(v) != 3:
            raise ValueError('Currency codes must be 3 characters')
        return v.upper()


class DerivativeData(BaseModel):
    """Derivative instrument data."""
    
    instrument: FinancialInstrument = Field(..., description="Derivative instrument details")
    underlying_asset: str = Field(..., description="Underlying asset symbol")
    contract_size: Decimal = Field(..., description="Contract size")
    expiration_date: date = Field(..., description="Contract expiration date")
    strike_price: Optional[Decimal] = Field(None, description="Strike price for options")
    option_type: Optional[str] = Field(None, description="Call or Put for options")
    premium: Optional[Decimal] = Field(None, description="Option premium")
    implied_volatility: Optional[Decimal] = Field(None, description="Implied volatility")
    delta: Optional[Decimal] = Field(None, description="Option delta")
    gamma: Optional[Decimal] = Field(None, description="Option gamma")
    theta: Optional[Decimal] = Field(None, description="Option theta")
    vega: Optional[Decimal] = Field(None, description="Option vega")
    open_interest: Optional[int] = Field(None, description="Open interest")
    
    @validator('option_type')
    def validate_option_type(cls, v):
        if v is not None and v.upper() not in ['CALL', 'PUT']:
            raise ValueError('Option type must be CALL or PUT')
        return v.upper() if v else v


class MarketData(BaseModel):
    """Market-wide data and indices."""
    
    market_name: str = Field(..., description="Name of the market or index")
    symbol: str = Field(..., description="Market symbol or ticker")
    current_value: Decimal = Field(..., description="Current market value")
    previous_close: Decimal = Field(..., description="Previous closing value")
    daily_change: Decimal = Field(..., description="Daily change in value")
    daily_change_percent: Decimal = Field(..., description="Daily percentage change")
    volume: Optional[int] = Field(None, description="Market volume")
    market_cap: Optional[Decimal] = Field(None, description="Total market capitalization")
    pe_ratio: Optional[Decimal] = Field(None, description="Price to earnings ratio")
    dividend_yield: Optional[Decimal] = Field(None, description="Average dividend yield")
    volatility: Optional[Decimal] = Field(None, description="Market volatility")
    year_high: Optional[Decimal] = Field(None, description="52-week high")
    year_low: Optional[Decimal] = Field(None, description="52-week low")


class FinancialRisk(BaseModel):
    """Financial risk metrics and assessments."""
    
    instrument_symbol: str = Field(..., description="Financial instrument symbol")
    risk_type: str = Field(..., description="Type of financial risk")
    risk_score: Decimal = Field(..., description="Numerical risk score")
    risk_level: str = Field(..., description="Risk level classification")
    volatility: Optional[Decimal] = Field(None, description="Price volatility measure")
    var_95: Optional[Decimal] = Field(None, description="Value at Risk (95% confidence)")
    var_99: Optional[Decimal] = Field(None, description="Value at Risk (99% confidence)")
    expected_shortfall: Optional[Decimal] = Field(None, description="Expected shortfall")
    beta: Optional[Decimal] = Field(None, description="Market beta")
    correlation: Optional[Decimal] = Field(None, description="Market correlation")
    credit_spread: Optional[Decimal] = Field(None, description="Credit spread over benchmark")
    probability_of_default: Optional[Decimal] = Field(None, description="Probability of default")
    recovery_rate: Optional[Decimal] = Field(None, description="Expected recovery rate")
    
    @validator('risk_score')
    def validate_risk_score(cls, v):
        if v < 0 or v > 100:
            raise ValueError('Risk score must be between 0 and 100')
        return v
    
    @validator('risk_level')
    def validate_risk_level(cls, v):
        if v.upper() not in ['LOW', 'MODERATE', 'HIGH', 'CRITICAL']:
            raise ValueError('Risk level must be LOW, MODERATE, HIGH, or CRITICAL')
        return v.upper()


class PortfolioData(BaseModel):
    """Portfolio-level financial data."""
    
    portfolio_id: str = Field(..., description="Unique portfolio identifier")
    name: str = Field(..., description="Portfolio name")
    total_value: Decimal = Field(..., description="Total portfolio value")
    cash_position: Decimal = Field(..., description="Cash position")
    invested_amount: Decimal = Field(..., description="Total invested amount")
    unrealized_gain_loss: Decimal = Field(..., description="Unrealized gains/losses")
    realized_gain_loss: Decimal = Field(..., description="Realized gains/losses")
    daily_return: Optional[Decimal] = Field(None, description="Daily return percentage")
    annual_return: Optional[Decimal] = Field(None, description="Annualized return")
    sharpe_ratio: Optional[Decimal] = Field(None, description="Sharpe ratio")
    sortino_ratio: Optional[Decimal] = Field(None, description="Sortino ratio")
    max_drawdown: Optional[Decimal] = Field(None, description="Maximum drawdown")
    volatility: Optional[Decimal] = Field(None, description="Portfolio volatility")
    
    @validator('total_value', 'cash_position', 'invested_amount')
    def validate_amounts(cls, v):
        if v < 0:
            raise ValueError('Portfolio amounts cannot be negative')
        return v


class FinancialSeries(BaseModel):
    """Time series of financial data."""
    
    instrument: FinancialInstrument = Field(..., description="Financial instrument")
    data_type: str = Field(..., description="Type of financial data")
    start_date: date = Field(..., description="Start date of the series")
    end_date: date = Field(..., description="End date of the series")
    frequency: str = Field(..., description="Data frequency")
    total_observations: int = Field(..., description="Total number of observations")
    price_data: List[PriceData] = Field(default_factory=list, description="Historical price data")
    volume_data: List[TradingVolume] = Field(default_factory=list, description="Historical volume data")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    
    @validator('total_observations')
    def validate_observations(cls, v):
        if v < 0:
            raise ValueError('Total observations cannot be negative')
        return v


class FinancialSummary(BaseModel):
    """Summary statistics for financial data."""
    
    instrument_symbol: str = Field(..., description="Financial instrument symbol")
    asset_class: AssetClass = Field(..., description="Asset class")
    period_start: date = Field(..., description="Summary period start")
    period_end: date = Field(..., description="Summary period end")
    average_price: Optional[Decimal] = Field(None, description="Average price")
    median_price: Optional[Decimal] = Field(None, description="Median price")
    price_volatility: Optional[Decimal] = Field(None, description="Price volatility")
    total_return: Optional[Decimal] = Field(None, description="Total return")
    annualized_return: Optional[Decimal] = Field(None, description="Annualized return")
    max_price: Optional[Decimal] = Field(None, description="Maximum price")
    min_price: Optional[Decimal] = Field(None, description="Minimum price")
    total_volume: Optional[int] = Field(None, description="Total trading volume")
    average_daily_volume: Optional[int] = Field(None, description="Average daily volume")
    correlation_to_market: Optional[Decimal] = Field(None, description="Correlation to market index")
    beta: Optional[Decimal] = Field(None, description="Beta coefficient")
    alpha: Optional[Decimal] = Field(None, description="Alpha coefficient")
    
    @validator('price_volatility', 'total_return', 'annualized_return')
    def validate_percentages(cls, v):
        if v is not None and (v < -100 or v > 1000):
            raise ValueError('Percentage values seem unrealistic')
        return v