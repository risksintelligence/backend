"""
Data schemas package for RiskX platform.
Provides Pydantic schemas for data validation and serialization across all data domains.
"""

from .economic import (
    EconomicIndicator,
    GDPData,
    InflationData,
    UnemploymentData,
    InterestRateData,
    ExchangeRateData,
    EconomicDataPoint,
    EconomicSeries,
    EconomicSummary
)

from .financial import (
    FinancialInstrument,
    StockData,
    BondData,
    CurrencyData,
    DerivativeData,
    MarketData,
    FinancialRisk,
    PortfolioData,
    TradingVolume,
    PriceData,
    FinancialSeries,
    FinancialSummary
)

from .supply_chain import (
    SupplyChainNode,
    Supplier,
    Manufacturer,
    Distributor,
    Retailer,
    TransportationData,
    InventoryData,
    ShipmentData,
    SupplyChainRisk,
    LogisticsData,
    SupplyChainMetrics,
    SupplyChainSummary
)

from .disruption import (
    DisruptionEvent,
    NaturalDisaster,
    CyberIncident,
    GeopoliticalEvent,
    EconomicShock,
    SupplyChainDisruption,
    DisruptionImpact,
    DisruptionSeverity,
    DisruptionCategory,
    DisruptionAlert,
    DisruptionSummary
)

__all__ = [
    # Economic schemas
    'EconomicIndicator',
    'GDPData',
    'InflationData', 
    'UnemploymentData',
    'InterestRateData',
    'ExchangeRateData',
    'EconomicDataPoint',
    'EconomicSeries',
    'EconomicSummary',
    
    # Financial schemas
    'FinancialInstrument',
    'StockData',
    'BondData',
    'CurrencyData',
    'DerivativeData',
    'MarketData',
    'FinancialRisk',
    'PortfolioData',
    'TradingVolume',
    'PriceData',
    'FinancialSeries',
    'FinancialSummary',
    
    # Supply chain schemas
    'SupplyChainNode',
    'Supplier',
    'Manufacturer',
    'Distributor',
    'Retailer',
    'TransportationData',
    'InventoryData',
    'ShipmentData',
    'SupplyChainRisk',
    'LogisticsData',
    'SupplyChainMetrics',
    'SupplyChainSummary',
    
    # Disruption schemas
    'DisruptionEvent',
    'NaturalDisaster',
    'CyberIncident',
    'GeopoliticalEvent',
    'EconomicShock',
    'SupplyChainDisruption',
    'DisruptionImpact',
    'DisruptionSeverity',
    'DisruptionCategory',
    'DisruptionAlert',
    'DisruptionSummary'
]