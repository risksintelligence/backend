"""
Economic Data Transformer

Transforms raw economic data from FRED, BEA, BLS into standardized format
and generates derived features for risk modeling.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import asyncio

from ...src.cache.cache_manager import CacheManager
from ...src.core.config import get_settings


@dataclass
class TransformedEconomicData:
    """Container for transformed economic data"""
    raw_indicators: pd.DataFrame
    derived_features: pd.DataFrame
    metadata: Dict[str, Any]
    transformation_timestamp: datetime
    data_quality_score: float


class EconomicTransformer:
    """
    Transforms raw economic data into standardized format and features.
    
    Handles data from:
    - FRED (Federal Reserve Economic Data)
    - BEA (Bureau of Economic Analysis)
    - BLS (Bureau of Labor Statistics)
    
    Transformations include:
    - Data standardization and normalization
    - Missing value imputation
    - Seasonal adjustment
    - Trend calculation
    - Volatility measures
    - Leading/lagging indicators
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.cache = CacheManager()
        self.logger = logging.getLogger(__name__)
        
        # Transformation configuration
        self.config = {
            'missing_value_threshold': 0.1,  # Max 10% missing values allowed
            'seasonal_adjustment_window': 12,  # 12-month seasonal adjustment
            'trend_calculation_window': 6,  # 6-month trend window
            'volatility_windows': [3, 6, 12],  # Multiple volatility windows
            'outlier_threshold': 3.0,  # Z-score threshold for outliers
        }
        
        # Economic indicator definitions
        self.indicator_definitions = {
            # Growth indicators
            'gdp_growth': {
                'source': 'bea',
                'fred_code': 'GDPC1',
                'frequency': 'quarterly',
                'seasonal_adjustment': True,
                'units': 'percent_change',
                'target_frequency': 'monthly'
            },
            'industrial_production': {
                'source': 'fred',
                'fred_code': 'INDPRO',
                'frequency': 'monthly',
                'seasonal_adjustment': True,
                'units': 'index',
                'target_frequency': 'monthly'
            },
            'capacity_utilization': {
                'source': 'fred',
                'fred_code': 'TCU',
                'frequency': 'monthly',
                'seasonal_adjustment': False,
                'units': 'percent',
                'target_frequency': 'monthly'
            },
            
            # Employment indicators
            'unemployment_rate': {
                'source': 'bls',
                'fred_code': 'UNRATE',
                'frequency': 'monthly',
                'seasonal_adjustment': True,
                'units': 'percent',
                'target_frequency': 'monthly'
            },
            'employment_pop_ratio': {
                'source': 'bls',
                'fred_code': 'EMRATIO',
                'frequency': 'monthly',
                'seasonal_adjustment': True,
                'units': 'percent',
                'target_frequency': 'monthly'
            },
            'labor_force_participation': {
                'source': 'bls',
                'fred_code': 'CIVPART',
                'frequency': 'monthly',
                'seasonal_adjustment': True,
                'units': 'percent',
                'target_frequency': 'monthly'
            },
            
            # Inflation indicators
            'cpi_core': {
                'source': 'bls',
                'fred_code': 'CPILFESL',
                'frequency': 'monthly',
                'seasonal_adjustment': False,
                'units': 'index',
                'target_frequency': 'monthly'
            },
            'pce_core': {
                'source': 'bea',
                'fred_code': 'PCEPILFE',
                'frequency': 'monthly',
                'seasonal_adjustment': False,
                'units': 'index',
                'target_frequency': 'monthly'
            },
            
            # Financial conditions
            'fed_funds_rate': {
                'source': 'fred',
                'fred_code': 'FEDFUNDS',
                'frequency': 'monthly',
                'seasonal_adjustment': False,
                'units': 'percent',
                'target_frequency': 'monthly'
            },
            'term_spread': {
                'source': 'fred',
                'fred_code': 'T10Y2Y',
                'frequency': 'daily',
                'seasonal_adjustment': False,
                'units': 'percentage_points',
                'target_frequency': 'monthly'
            }
        }
    
    async def transform_latest_data(self) -> Optional[TransformedEconomicData]:
        """
        Transform the latest economic data from all sources
        """
        self.logger.info("Starting economic data transformation")
        
        try:
            # Load raw data from cache or sources
            raw_data = await self._load_raw_economic_data()
            
            if raw_data is None or raw_data.empty:
                self.logger.warning("No raw economic data available for transformation")
                return None
            
            # Standardize the data
            standardized_data = await self._standardize_data(raw_data)
            
            # Handle missing values
            cleaned_data = self._handle_missing_values(standardized_data)
            
            # Apply transformations
            transformed_data = self._apply_transformations(cleaned_data)
            
            # Generate derived features
            derived_features = self._generate_derived_features(transformed_data)
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(transformed_data, derived_features)
            
            # Create metadata
            metadata = self._create_transformation_metadata(raw_data, transformed_data, derived_features)
            
            result = TransformedEconomicData(
                raw_indicators=transformed_data,
                derived_features=derived_features,
                metadata=metadata,
                transformation_timestamp=datetime.now(),
                data_quality_score=quality_score
            )
            
            # Cache transformed data
            await self._cache_transformed_data(result)
            
            self.logger.info(f"Economic data transformation completed. Quality score: {quality_score:.2f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in economic data transformation: {str(e)}")
            raise
    
    async def _load_raw_economic_data(self) -> Optional[pd.DataFrame]:
        """Load raw economic data from various sources"""
        
        try:
            all_data = {}
            
            # Load data for each indicator
            for indicator_name, config in self.indicator_definitions.items():
                try:
                    # Try to get from cache first
                    cache_key = f"{config['source']}_{config['fred_code']}_raw"
                    cached_data = await self.cache.get(cache_key)
                    
                    if cached_data:
                        # Convert to pandas Series
                        series_data = pd.Series(
                            cached_data['values'],
                            index=pd.to_datetime(cached_data['dates']),
                            name=indicator_name
                        )
                        all_data[indicator_name] = series_data
                    else:
                        # Generate sample data for demonstration
                        sample_data = self._generate_sample_economic_data(indicator_name, config)
                        all_data[indicator_name] = sample_data
                        
                except Exception as e:
                    self.logger.warning(f"Could not load data for {indicator_name}: {str(e)}")
            
            if all_data:
                # Combine all series into DataFrame
                combined_df = pd.DataFrame(all_data)
                return combined_df
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Error loading raw economic data: {str(e)}")
            return None
    
    def _generate_sample_economic_data(self, indicator_name: str, config: Dict) -> pd.Series:
        """Generate realistic sample data for testing purposes"""
        
        # Generate 3 years of data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * 3)
        
        # Create date range based on frequency
        if config['frequency'] == 'monthly':
            dates = pd.date_range(start_date, end_date, freq='M')
        elif config['frequency'] == 'quarterly':
            dates = pd.date_range(start_date, end_date, freq='Q')
        else:  # daily
            dates = pd.date_range(start_date, end_date, freq='D')
            dates = dates[dates.weekday < 5]  # Business days only
        
        # Generate realistic values based on indicator type
        np.random.seed(hash(indicator_name) % 2**32)  # Deterministic but different per indicator
        
        base_values = {
            'gdp_growth': {'mean': 2.5, 'std': 1.0, 'trend': 0.0},
            'unemployment_rate': {'mean': 5.0, 'std': 1.5, 'trend': 0.0},
            'cpi_core': {'mean': 120.0, 'std': 2.0, 'trend': 0.002},  # Slight upward trend
            'fed_funds_rate': {'mean': 2.0, 'std': 1.0, 'trend': 0.0},
            'term_spread': {'mean': 1.5, 'std': 0.8, 'trend': 0.0},
            'industrial_production': {'mean': 100.0, 'std': 3.0, 'trend': 0.001}
        }
        
        params = base_values.get(indicator_name, {'mean': 50.0, 'std': 5.0, 'trend': 0.0})
        
        # Generate time series with trend and seasonality
        n_periods = len(dates)
        values = []
        current_value = params['mean']
        
        for i in range(n_periods):
            # Add trend
            trend_component = params['trend'] * i
            
            # Add seasonal component (simplified)
            if config.get('seasonal_adjustment', False):
                seasonal_component = 2 * np.sin(2 * np.pi * i / 12) * params['std'] * 0.2
            else:
                seasonal_component = 0
            
            # Add noise with some persistence
            noise = np.random.normal(0, params['std'])
            if i > 0:
                # Add some autocorrelation
                noise += 0.3 * (values[i-1] - params['mean']) * 0.1
            
            current_value = params['mean'] + trend_component + seasonal_component + noise
            values.append(current_value)
        
        return pd.Series(values, index=dates, name=indicator_name)
    
    async def _standardize_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Standardize data formats and frequencies"""
        
        standardized_data = raw_data.copy()
        
        try:
            # Ensure datetime index
            if not isinstance(standardized_data.index, pd.DatetimeIndex):
                standardized_data.index = pd.to_datetime(standardized_data.index)
            
            # Sort by date
            standardized_data = standardized_data.sort_index()
            
            # Resample to monthly frequency (common target frequency)
            monthly_data = {}
            
            for column in standardized_data.columns:
                if column in self.indicator_definitions:
                    config = self.indicator_definitions[column]
                    series = standardized_data[column].dropna()
                    
                    if config['frequency'] == 'daily':
                        # Convert daily to monthly (last business day)
                        monthly_series = series.resample('M').last()
                    elif config['frequency'] == 'quarterly':
                        # Interpolate quarterly to monthly
                        monthly_series = series.resample('M').interpolate(method='linear')
                    else:
                        # Already monthly
                        monthly_series = series.resample('M').last()
                    
                    monthly_data[column] = monthly_series
                else:
                    # Default handling
                    monthly_data[column] = standardized_data[column].resample('M').last()
            
            result = pd.DataFrame(monthly_data)
            
            # Fill forward any remaining gaps (up to 2 months)
            result = result.fillna(method='ffill', limit=2)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in data standardization: {str(e)}")
            return raw_data
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values using appropriate imputation methods"""
        
        cleaned_data = data.copy()
        
        try:
            for column in cleaned_data.columns:
                series = cleaned_data[column]
                missing_ratio = series.isnull().sum() / len(series)
                
                if missing_ratio > self.config['missing_value_threshold']:
                    self.logger.warning(f"High missing value ratio for {column}: {missing_ratio:.1%}")
                
                if missing_ratio > 0:
                    if column in self.indicator_definitions:
                        config = self.indicator_definitions[column]
                        
                        # Use appropriate imputation method based on indicator type
                        if config['units'] in ['percent', 'percentage_points']:
                            # For rates, use forward fill then backward fill
                            cleaned_data[column] = series.fillna(method='ffill').fillna(method='bfill')
                        elif config['units'] == 'index':
                            # For indices, use interpolation
                            cleaned_data[column] = series.interpolate(method='linear')
                        else:
                            # Default: forward fill
                            cleaned_data[column] = series.fillna(method='ffill')
                    else:
                        # Default handling
                        cleaned_data[column] = series.fillna(method='ffill')
            
            return cleaned_data
            
        except Exception as e:
            self.logger.error(f"Error handling missing values: {str(e)}")
            return data
    
    def _apply_transformations(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply various transformations to the data"""
        
        transformed_data = data.copy()
        
        try:
            for column in transformed_data.columns:
                if column in self.indicator_definitions:
                    config = self.indicator_definitions[column]
                    series = transformed_data[column]
                    
                    # Apply seasonal adjustment if needed
                    if config.get('seasonal_adjustment', False):
                        adjusted_series = self._apply_seasonal_adjustment(series)
                        transformed_data[column] = adjusted_series
                    
                    # Convert to appropriate units
                    if config['units'] == 'percent_change':
                        # Convert to month-over-month percent change
                        pct_change = series.pct_change() * 100
                        transformed_data[column] = pct_change
            
            # Remove outliers
            transformed_data = self._remove_outliers(transformed_data)
            
            return transformed_data
            
        except Exception as e:
            self.logger.error(f"Error applying transformations: {str(e)}")
            return data
    
    def _apply_seasonal_adjustment(self, series: pd.Series) -> pd.Series:
        """Apply simple seasonal adjustment"""
        
        try:
            if len(series) < 24:  # Need at least 2 years of data
                return series
            
            # Calculate 12-month moving average for seasonal component
            seasonal_window = self.config['seasonal_adjustment_window']
            
            # Calculate seasonal factors
            seasonal_avg = series.rolling(window=seasonal_window, center=True).mean()
            seasonal_factors = series / seasonal_avg
            
            # Calculate average seasonal pattern
            seasonal_pattern = seasonal_factors.groupby(seasonal_factors.index.month).median()
            
            # Apply seasonal adjustment
            adjusted_series = series.copy()
            for month in range(1, 13):
                month_mask = adjusted_series.index.month == month
                if month in seasonal_pattern:
                    adjusted_series[month_mask] = adjusted_series[month_mask] / seasonal_pattern[month]
            
            return adjusted_series
            
        except Exception as e:
            self.logger.warning(f"Error in seasonal adjustment: {str(e)}")
            return series
    
    def _remove_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove statistical outliers from the data"""
        
        cleaned_data = data.copy()
        
        try:
            for column in cleaned_data.columns:
                series = cleaned_data[column].dropna()
                
                if len(series) > 10:  # Need sufficient data
                    # Calculate Z-scores
                    z_scores = np.abs((series - series.mean()) / series.std())
                    
                    # Identify outliers
                    outliers = z_scores > self.config['outlier_threshold']
                    
                    if outliers.sum() > 0:
                        self.logger.info(f"Removing {outliers.sum()} outliers from {column}")
                        
                        # Replace outliers with interpolated values
                        series_clean = series.copy()
                        series_clean[outliers] = np.nan
                        series_clean = series_clean.interpolate(method='linear')
                        
                        cleaned_data[column] = series_clean
            
            return cleaned_data
            
        except Exception as e:
            self.logger.warning(f"Error removing outliers: {str(e)}")
            return data
    
    def _generate_derived_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate derived features from transformed data"""
        
        features = pd.DataFrame(index=data.index)
        
        try:
            # Generate features for each indicator
            for column in data.columns:
                series = data[column].dropna()
                
                if len(series) < 12:  # Need sufficient data
                    continue
                
                # Growth rates
                features[f"{column}_growth_3m"] = series.pct_change(3) * 100
                features[f"{column}_growth_6m"] = series.pct_change(6) * 100
                features[f"{column}_growth_12m"] = series.pct_change(12) * 100
                
                # Volatility measures
                for window in self.config['volatility_windows']:
                    if len(series) >= window:
                        vol = series.pct_change().rolling(window=window).std() * 100
                        features[f"{column}_volatility_{window}m"] = vol
                
                # Trend indicators
                trend_window = self.config['trend_calculation_window']
                if len(series) >= trend_window:
                    # Linear trend slope
                    def calculate_slope(x):
                        if len(x) == trend_window:
                            slope = np.polyfit(range(trend_window), x, 1)[0]
                            return slope
                        return np.nan
                    
                    trend_slope = series.rolling(window=trend_window).apply(calculate_slope)
                    features[f"{column}_trend_{trend_window}m"] = trend_slope
                
                # Moving averages
                features[f"{column}_ma_3m"] = series.rolling(window=3).mean()
                features[f"{column}_ma_6m"] = series.rolling(window=6).mean()
                features[f"{column}_ma_12m"] = series.rolling(window=12).mean()
                
                # Percentile rankings (relative to historical)
                if len(series) >= 24:
                    rolling_percentile = series.rolling(window=24).apply(
                        lambda x: (x.iloc[-1] >= x).sum() / len(x) * 100
                    )
                    features[f"{column}_percentile_24m"] = rolling_percentile
            
            # Cross-indicator features
            features = self._generate_cross_indicator_features(data, features)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error generating derived features: {str(e)}")
            return pd.DataFrame(index=data.index)
    
    def _generate_cross_indicator_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Generate features based on relationships between indicators"""
        
        try:
            # Economic health score
            if all(col in data.columns for col in ['gdp_growth', 'unemployment_rate', 'cpi_core']):
                # Normalized economic health indicator
                gdp_norm = (data['gdp_growth'] - data['gdp_growth'].mean()) / data['gdp_growth'].std()
                unemp_norm = -(data['unemployment_rate'] - data['unemployment_rate'].mean()) / data['unemployment_rate'].std()
                inflation_norm = -(abs(data['cpi_core'].pct_change() * 100 - 2)) / 2  # Target 2% inflation
                
                features['economic_health_score'] = (gdp_norm + unemp_norm + inflation_norm) / 3
            
            # Financial stress indicator
            if all(col in data.columns for col in ['fed_funds_rate', 'term_spread']):
                # Simple financial stress measure
                rate_stress = (data['fed_funds_rate'] - data['fed_funds_rate'].rolling(24).mean()).abs()
                curve_stress = (data['term_spread'] < 0).astype(int) * 2  # Yield curve inversion
                
                features['financial_stress_indicator'] = rate_stress + curve_stress
            
            # Labor market strength
            if all(col in data.columns for col in ['unemployment_rate', 'employment_pop_ratio']):
                unemp_strength = -data['unemployment_rate']  # Lower unemployment = stronger
                emp_ratio_strength = data['employment_pop_ratio']
                
                features['labor_market_strength'] = (unemp_strength + emp_ratio_strength) / 2
            
            return features
            
        except Exception as e:
            self.logger.warning(f"Error generating cross-indicator features: {str(e)}")
            return features
    
    def _calculate_quality_score(self, raw_data: pd.DataFrame, features: pd.DataFrame) -> float:
        """Calculate overall data quality score"""
        
        try:
            scores = []
            
            # Data completeness score
            completeness = 1 - (raw_data.isnull().sum().sum() / (len(raw_data) * len(raw_data.columns)))
            scores.append(completeness)
            
            # Feature generation success rate
            feature_success = len(features.columns) / (len(raw_data.columns) * 10)  # Expect ~10 features per indicator
            feature_success = min(1.0, feature_success)
            scores.append(feature_success)
            
            # Data recency score
            if len(raw_data) > 0:
                latest_date = raw_data.index.max()
                days_old = (datetime.now() - latest_date).days
                recency_score = max(0, 1 - days_old / 30)  # Penalty for data older than 30 days
                scores.append(recency_score)
            
            # Overall quality score
            return sum(scores) / len(scores)
            
        except Exception as e:
            self.logger.warning(f"Error calculating quality score: {str(e)}")
            return 0.5
    
    def _create_transformation_metadata(self, raw_data: pd.DataFrame, transformed_data: pd.DataFrame, features: pd.DataFrame) -> Dict[str, Any]:
        """Create metadata about the transformation process"""
        
        return {
            'transformation_date': datetime.now().isoformat(),
            'raw_data_shape': raw_data.shape,
            'transformed_data_shape': transformed_data.shape,
            'features_shape': features.shape,
            'indicators_processed': list(raw_data.columns),
            'features_generated': list(features.columns),
            'data_date_range': {
                'start': raw_data.index.min().isoformat() if len(raw_data) > 0 else None,
                'end': raw_data.index.max().isoformat() if len(raw_data) > 0 else None
            },
            'transformation_config': self.config,
            'missing_value_summary': {
                col: int(transformed_data[col].isnull().sum()) 
                for col in transformed_data.columns
            }
        }
    
    async def _cache_transformed_data(self, result: TransformedEconomicData):
        """Cache transformed economic data"""
        
        try:
            # Prepare data for caching
            cache_data = {
                'raw_indicators': result.raw_indicators.to_dict('index'),
                'derived_features': result.derived_features.to_dict('index'),
                'metadata': result.metadata,
                'transformation_timestamp': result.transformation_timestamp.isoformat(),
                'data_quality_score': result.data_quality_score
            }
            
            # Cache latest transformation
            await self.cache.set('economic_transformed_latest', cache_data, ttl=86400)
            
            # Cache historical transformation
            date_key = f"economic_transformed_{result.transformation_timestamp.strftime('%Y%m%d')}"
            await self.cache.set(date_key, cache_data, ttl=86400 * 30)  # Keep for 30 days
            
        except Exception as e:
            self.logger.warning(f"Error caching transformed data: {str(e)}")


async def main():
    """Test economic data transformation"""
    transformer = EconomicTransformer()
    
    try:
        print("Testing economic data transformation...")
        result = await transformer.transform_latest_data()
        
        if result:
            print(f"Transformation successful!")
            print(f"Raw indicators shape: {result.raw_indicators.shape}")
            print(f"Derived features shape: {result.derived_features.shape}")
            print(f"Data quality score: {result.data_quality_score:.2f}")
            print(f"Features generated: {len(result.derived_features.columns)}")
        else:
            print("No data available for transformation")
            
    except Exception as e:
        print(f"Transformation failed: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())