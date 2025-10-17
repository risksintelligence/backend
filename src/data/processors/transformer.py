"""
Data transformation utilities for RiskX platform.
Provides comprehensive data transformation and normalization functions for all data sources.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from datetime import datetime, date, timedelta
import logging
from dataclasses import dataclass
from enum import Enum

from ...utils.helpers import normalize_value, calculate_percentage_change, parse_date_string
from ...utils.constants import BusinessRules, DataSources
from ...core.exceptions import ProcessingError, ValidationError

logger = logging.getLogger('riskx.data.processors.transformer')


class TransformationType(str, Enum):
    """Types of data transformations available."""
    NORMALIZE = "normalize"
    STANDARDIZE = "standardize"
    LOG_TRANSFORM = "log_transform"
    DIFFERENCE = "difference"
    PERCENTAGE_CHANGE = "percentage_change"
    MOVING_AVERAGE = "moving_average"
    SEASONAL_ADJUST = "seasonal_adjust"
    INTERPOLATE = "interpolate"
    AGGREGATE = "aggregate"


@dataclass
class TransformationConfig:
    """Configuration for data transformations."""
    transformation_type: TransformationType
    parameters: Dict[str, Any]
    apply_to_columns: Optional[List[str]] = None
    output_column_suffix: Optional[str] = None


@dataclass
class TransformationResult:
    """Result of data transformation operation."""
    transformed_data: pd.DataFrame
    transformation_log: List[str]
    statistics: Dict[str, Any]
    metadata: Dict[str, Any]


class DataTransformer:
    """Comprehensive data transformation utilities for economic and financial data."""
    
    def __init__(self):
        self.transformation_history = []
        self.statistics = {
            'transformations_applied': 0,
            'columns_transformed': 0,
            'data_points_processed': 0
        }
    
    def transform_dataframe(self, 
                          df: pd.DataFrame,
                          transformations: List[TransformationConfig]) -> TransformationResult:
        """
        Apply multiple transformations to a DataFrame in sequence.
        """
        try:
            transformed_df = df.copy()
            transformation_log = []
            statistics = {}
            
            for i, config in enumerate(transformations):
                logger.info(f"Applying transformation {i+1}/{len(transformations)}: {config.transformation_type}")
                
                result = self._apply_transformation(transformed_df, config)
                transformed_df = result['data']
                transformation_log.extend(result['log'])
                statistics[f'step_{i+1}'] = result['stats']
                
                self.statistics['transformations_applied'] += 1
            
            # Update global statistics
            self.statistics['data_points_processed'] += len(transformed_df) * len(transformed_df.columns)
            
            return TransformationResult(
                transformed_data=transformed_df,
                transformation_log=transformation_log,
                statistics=statistics,
                metadata={
                    'original_shape': df.shape,
                    'final_shape': transformed_df.shape,
                    'transformations_count': len(transformations),
                    'processing_timestamp': datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Error during data transformation: {e}")
            raise ProcessingError(f"Data transformation failed: {e}")
    
    def _apply_transformation(self, df: pd.DataFrame, config: TransformationConfig) -> Dict[str, Any]:
        """Apply a single transformation to the DataFrame."""
        columns_to_transform = config.apply_to_columns or df.select_dtypes(include=[np.number]).columns.tolist()
        transformed_df = df.copy()
        log_entries = []
        stats = {}
        
        try:
            if config.transformation_type == TransformationType.NORMALIZE:
                result = self._normalize_columns(transformed_df, columns_to_transform, config.parameters)
            
            elif config.transformation_type == TransformationType.STANDARDIZE:
                result = self._standardize_columns(transformed_df, columns_to_transform, config.parameters)
            
            elif config.transformation_type == TransformationType.LOG_TRANSFORM:
                result = self._log_transform_columns(transformed_df, columns_to_transform, config.parameters)
            
            elif config.transformation_type == TransformationType.DIFFERENCE:
                result = self._difference_columns(transformed_df, columns_to_transform, config.parameters)
            
            elif config.transformation_type == TransformationType.PERCENTAGE_CHANGE:
                result = self._percentage_change_columns(transformed_df, columns_to_transform, config.parameters)
            
            elif config.transformation_type == TransformationType.MOVING_AVERAGE:
                result = self._moving_average_columns(transformed_df, columns_to_transform, config.parameters)
            
            elif config.transformation_type == TransformationType.SEASONAL_ADJUST:
                result = self._seasonal_adjust_columns(transformed_df, columns_to_transform, config.parameters)
            
            elif config.transformation_type == TransformationType.INTERPOLATE:
                result = self._interpolate_columns(transformed_df, columns_to_transform, config.parameters)
            
            elif config.transformation_type == TransformationType.AGGREGATE:
                result = self._aggregate_columns(transformed_df, columns_to_transform, config.parameters)
            
            else:
                raise ValueError(f"Unknown transformation type: {config.transformation_type}")
            
            transformed_df = result['data']
            log_entries = result['log']
            stats = result['stats']
            
            self.statistics['columns_transformed'] += len(columns_to_transform)
            
            return {
                'data': transformed_df,
                'log': log_entries,
                'stats': stats
            }
            
        except Exception as e:
            error_msg = f"Failed to apply {config.transformation_type} transformation: {e}"
            logger.error(error_msg)
            raise ProcessingError(error_msg)
    
    def _normalize_columns(self, df: pd.DataFrame, columns: List[str], params: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize columns to specified range."""
        method = params.get('method', 'minmax')
        feature_range = params.get('feature_range', (0, 1))
        
        log_entries = []
        stats = {}
        
        for col in columns:
            if col not in df.columns:
                continue
                
            original_values = df[col].copy()
            
            if method == 'minmax':
                min_val = original_values.min()
                max_val = original_values.max()
                if max_val != min_val:
                    df[col] = (original_values - min_val) / (max_val - min_val)
                    df[col] = df[col] * (feature_range[1] - feature_range[0]) + feature_range[0]
                
                stats[col] = {'min': min_val, 'max': max_val, 'range': max_val - min_val}
                log_entries.append(f"Normalized {col} using min-max scaling to range {feature_range}")
            
            elif method == 'robust':
                median_val = original_values.median()
                q75 = original_values.quantile(0.75)
                q25 = original_values.quantile(0.25)
                iqr = q75 - q25
                
                if iqr != 0:
                    df[col] = (original_values - median_val) / iqr
                
                stats[col] = {'median': median_val, 'iqr': iqr, 'q25': q25, 'q75': q75}
                log_entries.append(f"Normalized {col} using robust scaling")
        
        return {'data': df, 'log': log_entries, 'stats': stats}
    
    def _standardize_columns(self, df: pd.DataFrame, columns: List[str], params: Dict[str, Any]) -> Dict[str, Any]:
        """Standardize columns to zero mean and unit variance."""
        log_entries = []
        stats = {}
        
        for col in columns:
            if col not in df.columns:
                continue
                
            original_values = df[col].copy()
            mean_val = original_values.mean()
            std_val = original_values.std()
            
            if std_val != 0:
                df[col] = (original_values - mean_val) / std_val
            
            stats[col] = {'mean': mean_val, 'std': std_val}
            log_entries.append(f"Standardized {col} (mean={mean_val:.4f}, std={std_val:.4f})")
        
        return {'data': df, 'log': log_entries, 'stats': stats}
    
    def _log_transform_columns(self, df: pd.DataFrame, columns: List[str], params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply logarithmic transformation to columns."""
        base = params.get('base', np.e)
        add_constant = params.get('add_constant', 1)  # Add 1 to handle zero values
        
        log_entries = []
        stats = {}
        
        for col in columns:
            if col not in df.columns:
                continue
                
            original_values = df[col].copy()
            
            # Add constant to handle negative or zero values
            adjusted_values = original_values + add_constant
            
            # Apply log transformation only to positive values
            mask = adjusted_values > 0
            if mask.any():
                if base == np.e:
                    df.loc[mask, col] = np.log(adjusted_values[mask])
                else:
                    df.loc[mask, col] = np.log(adjusted_values[mask]) / np.log(base)
            
            transformed_count = mask.sum()
            stats[col] = {
                'values_transformed': transformed_count,
                'values_skipped': len(df) - transformed_count,
                'constant_added': add_constant
            }
            log_entries.append(f"Log-transformed {col} (base={base}, constant={add_constant})")
        
        return {'data': df, 'log': log_entries, 'stats': stats}
    
    def _difference_columns(self, df: pd.DataFrame, columns: List[str], params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate differences between consecutive values."""
        periods = params.get('periods', 1)
        
        log_entries = []
        stats = {}
        
        for col in columns:
            if col not in df.columns:
                continue
                
            original_values = df[col].copy()
            df[col] = original_values.diff(periods=periods)
            
            stats[col] = {
                'periods': periods,
                'null_values_created': periods
            }
            log_entries.append(f"Calculated {periods}-period difference for {col}")
        
        return {'data': df, 'log': log_entries, 'stats': stats}
    
    def _percentage_change_columns(self, df: pd.DataFrame, columns: List[str], params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate percentage changes between consecutive values."""
        periods = params.get('periods', 1)
        
        log_entries = []
        stats = {}
        
        for col in columns:
            if col not in df.columns:
                continue
                
            original_values = df[col].copy()
            df[col] = original_values.pct_change(periods=periods)
            
            # Replace infinite values with NaN
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            
            stats[col] = {
                'periods': periods,
                'null_values_created': periods
            }
            log_entries.append(f"Calculated {periods}-period percentage change for {col}")
        
        return {'data': df, 'log': log_entries, 'stats': stats}
    
    def _moving_average_columns(self, df: pd.DataFrame, columns: List[str], params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate moving averages for columns."""
        window = params.get('window', 5)
        center = params.get('center', False)
        min_periods = params.get('min_periods', 1)
        
        log_entries = []
        stats = {}
        
        for col in columns:
            if col not in df.columns:
                continue
                
            original_values = df[col].copy()
            df[col] = original_values.rolling(
                window=window, 
                center=center, 
                min_periods=min_periods
            ).mean()
            
            stats[col] = {
                'window': window,
                'center': center,
                'min_periods': min_periods
            }
            log_entries.append(f"Calculated {window}-period moving average for {col}")
        
        return {'data': df, 'log': log_entries, 'stats': stats}
    
    def _seasonal_adjust_columns(self, df: pd.DataFrame, columns: List[str], params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply seasonal adjustment to time series columns."""
        period = params.get('period', 12)  # Default to monthly seasonality
        method = params.get('method', 'multiplicative')
        
        log_entries = []
        stats = {}
        
        for col in columns:
            if col not in df.columns:
                continue
                
            original_values = df[col].copy()
            
            try:
                # Simple seasonal adjustment using rolling statistics
                seasonal_component = original_values.rolling(window=period).mean()
                
                if method == 'additive':
                    df[col] = original_values - seasonal_component
                elif method == 'multiplicative':
                    # Avoid division by zero
                    mask = seasonal_component != 0
                    df.loc[mask, col] = original_values[mask] / seasonal_component[mask]
                
                stats[col] = {
                    'period': period,
                    'method': method,
                    'adjustment_applied': True
                }
                log_entries.append(f"Applied seasonal adjustment to {col} (period={period}, method={method})")
                
            except Exception as e:
                stats[col] = {
                    'period': period,
                    'method': method,
                    'adjustment_applied': False,
                    'error': str(e)
                }
                log_entries.append(f"Failed to apply seasonal adjustment to {col}: {e}")
        
        return {'data': df, 'log': log_entries, 'stats': stats}
    
    def _interpolate_columns(self, df: pd.DataFrame, columns: List[str], params: Dict[str, Any]) -> Dict[str, Any]:
        """Interpolate missing values in columns."""
        method = params.get('method', 'linear')
        limit = params.get('limit', None)
        
        log_entries = []
        stats = {}
        
        for col in columns:
            if col not in df.columns:
                continue
                
            missing_before = df[col].isnull().sum()
            df[col] = df[col].interpolate(method=method, limit=limit)
            missing_after = df[col].isnull().sum()
            
            values_interpolated = missing_before - missing_after
            
            stats[col] = {
                'method': method,
                'values_interpolated': values_interpolated,
                'missing_before': missing_before,
                'missing_after': missing_after
            }
            log_entries.append(f"Interpolated {values_interpolated} values in {col} using {method} method")
        
        return {'data': df, 'log': log_entries, 'stats': stats}
    
    def _aggregate_columns(self, df: pd.DataFrame, columns: List[str], params: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate columns using specified functions."""
        functions = params.get('functions', ['mean'])
        groupby_column = params.get('groupby', None)
        frequency = params.get('frequency', None)  # For time-based aggregation
        
        log_entries = []
        stats = {}
        
        try:
            if frequency and 'date' in df.columns:
                # Time-based aggregation
                df_grouped = df.set_index('date').groupby(pd.Grouper(freq=frequency))
                
                for col in columns:
                    if col not in df.columns:
                        continue
                    
                    for func in functions:
                        new_col_name = f"{col}_{func}_{frequency}"
                        df[new_col_name] = df_grouped[col].agg(func).values
                        
                        stats[new_col_name] = {
                            'original_column': col,
                            'function': func,
                            'frequency': frequency
                        }
                        log_entries.append(f"Aggregated {col} using {func} with {frequency} frequency")
            
            elif groupby_column and groupby_column in df.columns:
                # Group-based aggregation
                df_grouped = df.groupby(groupby_column)
                
                for col in columns:
                    if col not in df.columns:
                        continue
                    
                    for func in functions:
                        new_col_name = f"{col}_{func}_by_{groupby_column}"
                        aggregated = df_grouped[col].agg(func)
                        df[new_col_name] = df[groupby_column].map(aggregated)
                        
                        stats[new_col_name] = {
                            'original_column': col,
                            'function': func,
                            'groupby_column': groupby_column
                        }
                        log_entries.append(f"Aggregated {col} using {func} grouped by {groupby_column}")
            
            else:
                # Simple aggregation across all data
                for col in columns:
                    if col not in df.columns:
                        continue
                    
                    for func in functions:
                        if func == 'mean':
                            value = df[col].mean()
                        elif func == 'sum':
                            value = df[col].sum()
                        elif func == 'std':
                            value = df[col].std()
                        elif func == 'min':
                            value = df[col].min()
                        elif func == 'max':
                            value = df[col].max()
                        else:
                            continue
                        
                        new_col_name = f"{col}_{func}"
                        df[new_col_name] = value
                        
                        stats[new_col_name] = {
                            'original_column': col,
                            'function': func,
                            'value': value
                        }
                        log_entries.append(f"Calculated {func} for {col}: {value}")
        
        except Exception as e:
            error_msg = f"Aggregation failed: {e}"
            log_entries.append(error_msg)
            stats['error'] = error_msg
        
        return {'data': df, 'log': log_entries, 'stats': stats}
    
    def create_economic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create standardized economic indicators from raw data."""
        try:
            transformed_df = df.copy()
            
            # Economic indicator transformations
            transformations = [
                TransformationConfig(
                    transformation_type=TransformationType.PERCENTAGE_CHANGE,
                    parameters={'periods': 1},
                    apply_to_columns=['gdp', 'industrial_production', 'retail_sales']
                ),
                TransformationConfig(
                    transformation_type=TransformationType.MOVING_AVERAGE,
                    parameters={'window': 3},
                    apply_to_columns=['unemployment_rate', 'inflation_rate']
                ),
                TransformationConfig(
                    transformation_type=TransformationType.STANDARDIZE,
                    parameters={},
                    apply_to_columns=['interest_rate', 'exchange_rate']
                )
            ]
            
            result = self.transform_dataframe(transformed_df, transformations)
            logger.info(f"Created economic indicators from {len(df)} data points")
            
            return result.transformed_data
            
        except Exception as e:
            logger.error(f"Error creating economic indicators: {e}")
            raise ProcessingError(f"Economic indicator creation failed: {e}")
    
    def create_financial_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create standardized financial indicators from raw data."""
        try:
            transformed_df = df.copy()
            
            # Financial indicator transformations
            transformations = [
                TransformationConfig(
                    transformation_type=TransformationType.LOG_TRANSFORM,
                    parameters={'base': np.e, 'add_constant': 1},
                    apply_to_columns=['market_cap', 'volume', 'assets']
                ),
                TransformationConfig(
                    transformation_type=TransformationType.PERCENTAGE_CHANGE,
                    parameters={'periods': 1},
                    apply_to_columns=['price', 'yield', 'spread']
                ),
                TransformationConfig(
                    transformation_type=TransformationType.MOVING_AVERAGE,
                    parameters={'window': 5},
                    apply_to_columns=['volatility', 'returns']
                )
            ]
            
            result = self.transform_dataframe(transformed_df, transformations)
            logger.info(f"Created financial indicators from {len(df)} data points")
            
            return result.transformed_data
            
        except Exception as e:
            logger.error(f"Error creating financial indicators: {e}")
            raise ProcessingError(f"Financial indicator creation failed: {e}")
    
    def create_supply_chain_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create standardized supply chain indicators from raw data."""
        try:
            transformed_df = df.copy()
            
            # Supply chain indicator transformations
            transformations = [
                TransformationConfig(
                    transformation_type=TransformationType.NORMALIZE,
                    parameters={'method': 'minmax', 'feature_range': (0, 1)},
                    apply_to_columns=['delivery_time', 'inventory_level', 'capacity_utilization']
                ),
                TransformationConfig(
                    transformation_type=TransformationType.DIFFERENCE,
                    parameters={'periods': 1},
                    apply_to_columns=['shipment_volume', 'order_backlog']
                ),
                TransformationConfig(
                    transformation_type=TransformationType.SEASONAL_ADJUST,
                    parameters={'period': 12, 'method': 'multiplicative'},
                    apply_to_columns=['demand', 'production']
                )
            ]
            
            result = self.transform_dataframe(transformed_df, transformations)
            logger.info(f"Created supply chain indicators from {len(df)} data points")
            
            return result.transformed_data
            
        except Exception as e:
            logger.error(f"Error creating supply chain indicators: {e}")
            raise ProcessingError(f"Supply chain indicator creation failed: {e}")
    
    def get_transformation_summary(self) -> Dict[str, Any]:
        """Get summary of all transformations performed."""
        return {
            'statistics': self.statistics.copy(),
            'transformation_history': self.transformation_history.copy(),
            'available_transformations': [t.value for t in TransformationType]
        }
    
    def reset_statistics(self):
        """Reset transformation statistics."""
        self.statistics = {
            'transformations_applied': 0,
            'columns_transformed': 0,
            'data_points_processed': 0
        }
        self.transformation_history.clear()
        logger.info("Transformation statistics reset")