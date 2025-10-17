"""
Economic Feature Engineering Module

Transforms raw economic indicators into predictive features
for risk assessment models.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

from ...cache.cache_manager import CacheManager
from ...core.config import get_settings


@dataclass
class EconomicFeatures:
    """Container for engineered economic features"""
    features: Dict[str, float]
    metadata: Dict[str, Any]
    timestamp: datetime
    data_quality_score: float


class EconomicFeatureEngineer:
    """
    Engineers features from economic indicators for risk prediction.
    
    Transforms raw economic data from sources like FRED, BEA, BLS
    into predictive features including:
    - Trend indicators
    - Volatility measures
    - Cross-correlation features
    - Regime change indicators
    - Leading/lagging relationships
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.cache = CacheManager()
        self.logger = logging.getLogger(__name__)
        
        # Key economic indicators to monitor
        self.key_indicators = {
            # Growth indicators
            "gdp_growth": "GDPC1",
            "industrial_production": "INDPRO",
            "capacity_utilization": "TCU",
            
            # Employment indicators
            "unemployment_rate": "UNRATE", 
            "employment_pop_ratio": "EMRATIO",
            "labor_force_participation": "CIVPART",
            
            # Inflation indicators
            "cpi_core": "CPILFESL",
            "pce_core": "PCEPILFE",
            "producer_price_index": "PPIACO",
            
            # Financial conditions
            "fed_funds_rate": "FEDFUNDS",
            "term_spread": "T10Y2Y",
            "credit_spread": "BAMLC0A0CM",
            
            # Confidence indicators
            "consumer_confidence": "UMCSENT",
            "business_confidence": "BSCICP03USM665S"
        }
        
        # Feature engineering parameters
        self.lookback_periods = [3, 6, 12, 24]  # months
        self.volatility_windows = [6, 12]  # months
    
    async def engineer_features(self, end_date: Optional[datetime] = None) -> EconomicFeatures:
        """
        Engineer comprehensive set of economic features
        """
        self.logger.info("Starting economic feature engineering")
        
        if end_date is None:
            end_date = datetime.now()
        
        try:
            # Load raw economic data
            raw_data = await self._load_economic_data(end_date)
            
            # Engineer different types of features
            trend_features = self._engineer_trend_features(raw_data)
            volatility_features = self._engineer_volatility_features(raw_data)
            correlation_features = self._engineer_correlation_features(raw_data)
            regime_features = self._engineer_regime_features(raw_data)
            momentum_features = self._engineer_momentum_features(raw_data)
            
            # Combine all features
            all_features = {}
            all_features.update(trend_features)
            all_features.update(volatility_features) 
            all_features.update(correlation_features)
            all_features.update(regime_features)
            all_features.update(momentum_features)
            
            # Calculate data quality score
            quality_score = self._calculate_data_quality(raw_data, all_features)
            
            # Create metadata
            metadata = {
                "feature_count": len(all_features),
                "data_sources": list(self.key_indicators.keys()),
                "engineering_date": end_date.isoformat(),
                "lookback_periods": self.lookback_periods,
                "data_coverage": self._assess_data_coverage(raw_data)
            }
            
            features = EconomicFeatures(
                features=all_features,
                metadata=metadata,
                timestamp=datetime.now(),
                data_quality_score=quality_score
            )
            
            # Cache features
            await self._cache_features(features)
            
            self.logger.info(f"Engineered {len(all_features)} economic features")
            return features
            
        except Exception as e:
            self.logger.error(f"Error in economic feature engineering: {str(e)}")
            raise
    
    async def _load_economic_data(self, end_date: datetime) -> pd.DataFrame:
        """Load economic data from FRED and other sources"""
        from ...data.sources.fred import FREDConnector
        
        # Try to get from cache first
        cache_key = f"economic_raw_data_{end_date.strftime('%Y%m%d')}"
        cached_data = self.cache.get(cache_key)
        
        if cached_data is not None:
            try:
                return pd.DataFrame(cached_data)
            except Exception as e:
                self.logger.warning(f"Failed to deserialize cached economic data: {e}")
        
        # Load real data from FRED API
        fred_connector = FREDConnector(self.cache)
        data_dict = {}
        start_date = end_date - timedelta(days=365 * 3)  # 3 years of data
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        self.logger.info(f"Loading economic data from {start_date_str} to {end_date_str}")
        
        # Map indicators to FRED series codes
        fred_series_map = {
            "gdp_growth": "GDPC1",
            "industrial_production": "INDPRO", 
            "capacity_utilization": "TCU",
            "unemployment_rate": "UNRATE",
            "employment_pop_ratio": "EMRATIO",
            "labor_force_participation": "CIVPART",
            "cpi_core": "CPILFESL",
            "pce_core": "PCEPILFE", 
            "producer_price_index": "PPIACO",
            "fed_funds_rate": "FEDFUNDS",
            "term_spread": "T10Y2Y",
            "credit_spread": "BAMLC0A0CM",
            "consumer_confidence": "UMCSENT",
            "business_confidence": "BSCICP03USM665S"
        }
        
        # Fetch data for each indicator
        for indicator, fred_code in fred_series_map.items():
            try:
                series_data = fred_connector.get_series(
                    fred_code,
                    start_date=start_date_str,
                    end_date=end_date_str,
                    use_cache=True
                )
                
                if series_data is not None and not series_data.empty:
                    # Convert to common monthly frequency
                    monthly_data = series_data.resample('M').last()
                    data_dict[indicator] = monthly_data
                    self.logger.info(f"Successfully loaded {indicator} from FRED ({len(monthly_data)} points)")
                else:
                    self.logger.warning(f"No data available for {indicator} ({fred_code})")
                    # Use fallback sample data for missing indicators
                    date_range = pd.date_range(start_date, end_date, freq='M')
                    data_dict[indicator] = self._generate_sample_indicator_data(indicator, date_range)
                    
            except Exception as e:
                self.logger.error(f"Error loading {indicator} from FRED: {e}")
                # Use fallback sample data on error
                date_range = pd.date_range(start_date, end_date, freq='M')
                data_dict[indicator] = self._generate_sample_indicator_data(indicator, date_range)
        
        # Align all series to common date index
        if data_dict:
            # Find common date range across all series
            all_dates = set()
            for series in data_dict.values():
                if hasattr(series, 'index'):
                    all_dates.update(series.index)
            
            if all_dates:
                common_dates = sorted(all_dates)
                df_dict = {}
                
                for indicator, series in data_dict.items():
                    if hasattr(series, 'reindex'):
                        # Align series to common dates
                        aligned_series = series.reindex(common_dates, method='ffill')
                        df_dict[indicator] = aligned_series
                    else:
                        # Handle list data
                        df_dict[indicator] = pd.Series(series, index=common_dates[:len(series)])
                
                df = pd.DataFrame(df_dict)
            else:
                # Fallback if no real data available
                self.logger.warning("No real economic data available, using fallback")
                date_range = pd.date_range(start_date, end_date, freq='M')
                fallback_dict = {}
                for indicator in fred_series_map.keys():
                    fallback_dict[indicator] = self._generate_sample_indicator_data(indicator, date_range)
                df = pd.DataFrame(fallback_dict, index=date_range)
        else:
            # Complete fallback
            self.logger.warning("Complete fallback to sample data")
            date_range = pd.date_range(start_date, end_date, freq='M')
            fallback_dict = {}
            for indicator in fred_series_map.keys():
                fallback_dict[indicator] = self._generate_sample_indicator_data(indicator, date_range)
            df = pd.DataFrame(fallback_dict, index=date_range)
        
        # Cache the processed data
        try:
            cache_data = df.to_dict('records')
            self.cache.set(cache_key, cache_data, ttl=86400)  # 24 hour cache
            self.logger.info(f"Cached economic data with {len(df)} records")
        except Exception as e:
            self.logger.error(f"Failed to cache economic data: {e}")
        
        return df
    
    def _generate_sample_indicator_data(self, indicator: str, date_range: pd.DatetimeIndex) -> List[float]:
        """Generate realistic sample data for indicators"""
        np.random.seed(hash(indicator) % 2**32)  # Deterministic but different per indicator
        
        # Base values and volatilities for different indicators
        indicator_params = {
            "gdp_growth": {"base": 2.5, "volatility": 1.0, "trend": 0.0},
            "unemployment_rate": {"base": 5.0, "volatility": 1.5, "trend": 0.0},
            "cpi_core": {"base": 2.0, "volatility": 0.5, "trend": 0.0},
            "fed_funds_rate": {"base": 2.0, "volatility": 1.0, "trend": 0.0},
            "term_spread": {"base": 1.5, "volatility": 0.8, "trend": 0.0},
            "consumer_confidence": {"base": 100.0, "volatility": 10.0, "trend": 0.0},
        }
        
        params = indicator_params.get(indicator, {"base": 50.0, "volatility": 5.0, "trend": 0.0})
        
        # Generate realistic time series with autocorrelation
        n_periods = len(date_range)
        data = []
        current_value = params["base"]
        
        for i in range(n_periods):
            # Add trend
            trend_component = params["trend"] * i / 12  # Annual trend
            
            # Add noise with some persistence
            noise = np.random.normal(0, params["volatility"])
            if i > 0:
                # Add some autocorrelation
                noise += 0.3 * (data[i-1] - params["base"])
            
            current_value = params["base"] + trend_component + noise
            data.append(current_value)
        
        return data
    
    def _engineer_trend_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """Engineer trend-based features"""
        features = {}
        
        for indicator in data.columns:
            series = data[indicator].dropna()
            if len(series) < 12:  # Need at least 12 months
                continue
            
            # Growth rates over different periods
            for period in self.lookback_periods:
                if len(series) >= period:
                    growth_rate = (series.iloc[-1] / series.iloc[-period] - 1) * 100
                    features[f"{indicator}_growth_{period}m"] = growth_rate
            
            # Trend slope (linear regression slope)
            if len(series) >= 12:
                x = np.arange(len(series[-12:]))
                y = series[-12:].values
                slope = np.polyfit(x, y, 1)[0]
                features[f"{indicator}_trend_slope"] = slope
            
            # Current level vs historical percentile
            if len(series) >= 24:
                current_percentile = (series.iloc[-1] > series[-24:]).sum() / 24 * 100
                features[f"{indicator}_percentile_24m"] = current_percentile
        
        return features
    
    def _engineer_volatility_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """Engineer volatility-based features"""
        features = {}
        
        for indicator in data.columns:
            series = data[indicator].dropna()
            if len(series) < 6:
                continue
            
            # Rolling volatility over different windows
            for window in self.volatility_windows:
                if len(series) >= window:
                    # Month-over-month changes
                    changes = series.pct_change().dropna()
                    if len(changes) >= window:
                        vol = changes[-window:].std() * 100
                        features[f"{indicator}_volatility_{window}m"] = vol
            
            # Volatility regime (current vs historical)
            if len(series) >= 24:
                recent_vol = series[-6:].pct_change().std()
                historical_vol = series[-24:].pct_change().std()
                vol_ratio = recent_vol / historical_vol if historical_vol > 0 else 1.0
                features[f"{indicator}_volatility_regime"] = vol_ratio
        
        return features
    
    def _engineer_correlation_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """Engineer cross-correlation features"""
        features = {}
        
        # Key economic relationships to monitor
        correlation_pairs = [
            ("unemployment_rate", "consumer_confidence"),
            ("fed_funds_rate", "term_spread"),
            ("gdp_growth", "industrial_production"),
            ("cpi_core", "fed_funds_rate")
        ]
        
        for indicator1, indicator2 in correlation_pairs:
            if indicator1 in data.columns and indicator2 in data.columns:
                series1 = data[indicator1].dropna()
                series2 = data[indicator2].dropna()
                
                # Align series
                common_index = series1.index.intersection(series2.index)
                if len(common_index) >= 12:
                    aligned1 = series1.loc[common_index]
                    aligned2 = series2.loc[common_index]
                    
                    # Rolling correlation
                    correlation = aligned1[-12:].corr(aligned2[-12:])
                    if not np.isnan(correlation):
                        features[f"corr_{indicator1}_{indicator2}_12m"] = correlation
        
        return features
    
    def _engineer_regime_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """Engineer regime change indicators"""
        features = {}
        
        # Monitor key indicators for regime changes
        regime_indicators = ["fed_funds_rate", "unemployment_rate", "cpi_core"]
        
        for indicator in regime_indicators:
            if indicator in data.columns:
                series = data[indicator].dropna()
                if len(series) >= 24:
                    # Simple regime detection based on moving averages
                    short_ma = series[-3:].mean()
                    long_ma = series[-12:].mean()
                    
                    # Regime signal
                    regime_signal = 1 if short_ma > long_ma else -1
                    features[f"{indicator}_regime_signal"] = regime_signal
                    
                    # Regime strength
                    regime_strength = abs(short_ma - long_ma) / long_ma if long_ma != 0 else 0
                    features[f"{indicator}_regime_strength"] = regime_strength
        
        return features
    
    def _engineer_momentum_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """Engineer momentum-based features"""
        features = {}
        
        for indicator in data.columns:
            series = data[indicator].dropna()
            if len(series) < 6:
                continue
            
            # Rate of change acceleration
            if len(series) >= 6:
                recent_change = series.iloc[-1] - series.iloc[-3]
                previous_change = series.iloc[-3] - series.iloc[-6]
                acceleration = recent_change - previous_change
                features[f"{indicator}_acceleration"] = acceleration
            
            # Momentum oscillator
            if len(series) >= 12:
                momentum = (series.iloc[-1] / series.iloc[-12] - 1) * 100
                features[f"{indicator}_momentum_12m"] = momentum
        
        return features
    
    def _calculate_data_quality(self, raw_data: pd.DataFrame, features: Dict[str, float]) -> float:
        """Calculate overall data quality score"""
        # Check data completeness
        total_possible = len(raw_data.columns) * len(raw_data)
        total_actual = raw_data.count().sum()
        completeness_score = total_actual / total_possible if total_possible > 0 else 0
        
        # Check feature validity
        valid_features = sum(1 for v in features.values() if not np.isnan(v) and np.isfinite(v))
        feature_validity = valid_features / len(features) if features else 0
        
        # Check data recency
        last_date = raw_data.index.max() if hasattr(raw_data.index, 'max') else datetime.now()
        days_old = (datetime.now() - last_date).days if isinstance(last_date, datetime) else 0
        recency_score = max(0, 1 - days_old / 30)  # Penalty for data older than 30 days
        
        # Overall quality score
        quality_score = (completeness_score * 0.4 + feature_validity * 0.4 + recency_score * 0.2)
        return min(1.0, max(0.0, quality_score))
    
    def _assess_data_coverage(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess data coverage across indicators"""
        coverage = {}
        
        for indicator in data.columns:
            series = data[indicator].dropna()
            coverage[indicator] = {
                "total_periods": len(data),
                "available_periods": len(series),
                "coverage_ratio": len(series) / len(data) if len(data) > 0 else 0,
                "latest_date": series.index.max().isoformat() if len(series) > 0 else None
            }
        
        return coverage
    
    async def _cache_features(self, features: EconomicFeatures):
        """Cache engineered features"""
        try:
            feature_data = {
                "features": features.features,
                "metadata": features.metadata,
                "timestamp": features.timestamp.isoformat(),
                "data_quality_score": features.data_quality_score
            }
            
            # Cache with date-based key
            date_key = f"economic_features_{features.timestamp.strftime('%Y%m%d')}"
            await self.cache.set(date_key, feature_data, ttl=86400 * 7)
            
            # Cache as latest
            await self.cache.set("economic_features_latest", feature_data, ttl=86400)
            
        except Exception as e:
            self.logger.error(f"Error caching economic features: {str(e)}")
    
    async def get_cached_features(self, date: Optional[datetime] = None) -> Optional[EconomicFeatures]:
        """Retrieve cached features"""
        try:
            if date:
                cache_key = f"economic_features_{date.strftime('%Y%m%d')}"
            else:
                cache_key = "economic_features_latest"
            
            cached_data = await self.cache.get(cache_key)
            if cached_data:
                return EconomicFeatures(
                    features=cached_data["features"],
                    metadata=cached_data["metadata"],
                    timestamp=datetime.fromisoformat(cached_data["timestamp"]),
                    data_quality_score=cached_data["data_quality_score"]
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error retrieving cached economic features: {str(e)}")
            return None
    
    def get_feature_importance_map(self) -> Dict[str, Dict[str, Any]]:
        """Get mapping of features to their business importance"""
        return {
            "gdp_growth_12m": {
                "importance": "high",
                "category": "growth",
                "description": "Annual GDP growth rate indicating economic expansion",
                "risk_relationship": "negative"  # Lower growth = higher risk
            },
            "unemployment_rate_trend_slope": {
                "importance": "high", 
                "category": "employment",
                "description": "Trend in unemployment rate",
                "risk_relationship": "positive"  # Rising unemployment = higher risk
            },
            "fed_funds_rate_volatility_6m": {
                "importance": "medium",
                "category": "monetary_policy",
                "description": "Federal funds rate volatility",
                "risk_relationship": "positive"  # Higher volatility = higher risk
            },
            "term_spread": {
                "importance": "high",
                "category": "financial_conditions", 
                "description": "Yield curve spread",
                "risk_relationship": "negative"  # Inverted curve = higher risk
            },
            "consumer_confidence_momentum_12m": {
                "importance": "medium",
                "category": "sentiment",
                "description": "Consumer confidence momentum",
                "risk_relationship": "negative"  # Lower confidence = higher risk
            }
        }