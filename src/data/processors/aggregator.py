"""
Data aggregation utilities for RiskX platform.
Provides comprehensive data aggregation and rollup functions for all data sources.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from datetime import datetime, date, timedelta
import logging
from dataclasses import dataclass
from enum import Enum

from ...utils.helpers import calculate_moving_average, calculate_percentile, safe_divide
from ...utils.constants import BusinessRules, DataSources
from ...core.exceptions import ProcessingError, ValidationError

logger = logging.getLogger('riskx.data.processors.aggregator')


class AggregationFunction(str, Enum):
    """Types of aggregation functions available."""
    MEAN = "mean"
    MEDIAN = "median"
    SUM = "sum"
    COUNT = "count"
    MIN = "min"
    MAX = "max"
    STD = "std"
    VAR = "var"
    QUANTILE = "quantile"
    FIRST = "first"
    LAST = "last"
    NUNIQUE = "nunique"
    WEIGHTED_MEAN = "weighted_mean"
    GEOMETRIC_MEAN = "geometric_mean"
    HARMONIC_MEAN = "harmonic_mean"


class AggregationLevel(str, Enum):
    """Levels of data aggregation."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    CUSTOM = "custom"


@dataclass
class AggregationConfig:
    """Configuration for data aggregation operations."""
    level: AggregationLevel
    functions: List[AggregationFunction]
    group_by_columns: Optional[List[str]] = None
    date_column: Optional[str] = 'date'
    weight_column: Optional[str] = None
    quantile_values: Optional[List[float]] = None
    custom_frequency: Optional[str] = None


@dataclass
class AggregationResult:
    """Result of data aggregation operation."""
    aggregated_data: pd.DataFrame
    summary_statistics: Dict[str, Any]
    metadata: Dict[str, Any]
    aggregation_log: List[str]


class DataAggregator:
    """Comprehensive data aggregation utilities for economic and financial data."""
    
    def __init__(self):
        self.aggregation_history = []
        self.statistics = {
            'aggregations_performed': 0,
            'rows_aggregated': 0,
            'columns_processed': 0
        }
    
    def aggregate_data(self, 
                      df: pd.DataFrame,
                      config: AggregationConfig) -> AggregationResult:
        """
        Aggregate data according to specified configuration.
        """
        try:
            logger.info(f"Starting data aggregation: level={config.level}, functions={config.functions}")
            
            # Validate input data
            if df.empty:
                raise ValueError("Input DataFrame is empty")
            
            # Prepare aggregation
            aggregated_df, log_entries = self._prepare_aggregation(df, config)
            
            # Perform aggregation
            result_df, agg_log = self._perform_aggregation(aggregated_df, config)
            log_entries.extend(agg_log)
            
            # Calculate summary statistics
            summary_stats = self._calculate_summary_statistics(df, result_df, config)
            
            # Update tracking statistics
            self.statistics['aggregations_performed'] += 1
            self.statistics['rows_aggregated'] += len(df)
            self.statistics['columns_processed'] += len(df.columns)
            
            # Create result
            result = AggregationResult(
                aggregated_data=result_df,
                summary_statistics=summary_stats,
                metadata={
                    'original_shape': df.shape,
                    'aggregated_shape': result_df.shape,
                    'aggregation_level': config.level,
                    'functions_applied': config.functions,
                    'processing_timestamp': datetime.utcnow().isoformat()
                },
                aggregation_log=log_entries
            )
            
            logger.info(f"Data aggregation completed: {df.shape} -> {result_df.shape}")
            return result
            
        except Exception as e:
            logger.error(f"Error during data aggregation: {e}")
            raise ProcessingError(f"Data aggregation failed: {e}")
    
    def _prepare_aggregation(self, df: pd.DataFrame, config: AggregationConfig) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare DataFrame for aggregation."""
        prepared_df = df.copy()
        log_entries = []
        
        # Ensure date column exists and is datetime
        if config.date_column and config.date_column in prepared_df.columns:
            if not pd.api.types.is_datetime64_any_dtype(prepared_df[config.date_column]):
                prepared_df[config.date_column] = pd.to_datetime(prepared_df[config.date_column])
                log_entries.append(f"Converted {config.date_column} to datetime")
        
        # Sort by date if available
        if config.date_column and config.date_column in prepared_df.columns:
            prepared_df = prepared_df.sort_values(config.date_column)
            log_entries.append(f"Sorted data by {config.date_column}")
        
        # Validate weight column if specified
        if config.weight_column and config.weight_column not in prepared_df.columns:
            logger.warning(f"Weight column {config.weight_column} not found, ignoring weights")
            config.weight_column = None
        
        return prepared_df, log_entries
    
    def _perform_aggregation(self, df: pd.DataFrame, config: AggregationConfig) -> Tuple[pd.DataFrame, List[str]]:
        """Perform the actual aggregation operation."""
        log_entries = []
        
        try:
            # Determine grouping strategy
            if config.level in [AggregationLevel.DAILY, AggregationLevel.WEEKLY, 
                               AggregationLevel.MONTHLY, AggregationLevel.QUARTERLY, 
                               AggregationLevel.YEARLY]:
                result_df, time_log = self._time_based_aggregation(df, config)
                log_entries.extend(time_log)
            
            elif config.group_by_columns:
                result_df, group_log = self._group_based_aggregation(df, config)
                log_entries.extend(group_log)
            
            else:
                result_df, summary_log = self._summary_aggregation(df, config)
                log_entries.extend(summary_log)
            
            return result_df, log_entries
            
        except Exception as e:
            error_msg = f"Aggregation operation failed: {e}"
            logger.error(error_msg)
            raise ProcessingError(error_msg)
    
    def _time_based_aggregation(self, df: pd.DataFrame, config: AggregationConfig) -> Tuple[pd.DataFrame, List[str]]:
        """Perform time-based aggregation."""
        log_entries = []
        
        if not config.date_column or config.date_column not in df.columns:
            raise ValueError("Date column required for time-based aggregation")
        
        # Define frequency mapping
        frequency_map = {
            AggregationLevel.DAILY: 'D',
            AggregationLevel.WEEKLY: 'W',
            AggregationLevel.MONTHLY: 'M',
            AggregationLevel.QUARTERLY: 'Q',
            AggregationLevel.YEARLY: 'Y'
        }
        
        frequency = frequency_map.get(config.level, config.custom_frequency)
        if not frequency:
            raise ValueError(f"No frequency defined for aggregation level: {config.level}")
        
        # Set date as index for grouping
        df_indexed = df.set_index(config.date_column)
        
        # Group by time frequency
        grouped = df_indexed.groupby(pd.Grouper(freq=frequency))
        
        # Apply aggregation functions
        result_frames = []
        numeric_columns = df_indexed.select_dtypes(include=[np.number]).columns
        
        for func in config.functions:
            func_result = self._apply_aggregation_function(grouped, numeric_columns, func, config)
            
            if func_result is not None:
                # Add function suffix to column names
                func_result.columns = [f"{col}_{func}" for col in func_result.columns]
                result_frames.append(func_result)
                log_entries.append(f"Applied {func} aggregation with {frequency} frequency")
        
        # Combine results
        if result_frames:
            result_df = pd.concat(result_frames, axis=1)
            result_df = result_df.reset_index()
        else:
            result_df = pd.DataFrame()
        
        return result_df, log_entries
    
    def _group_based_aggregation(self, df: pd.DataFrame, config: AggregationConfig) -> Tuple[pd.DataFrame, List[str]]:
        """Perform group-based aggregation."""
        log_entries = []
        
        if not config.group_by_columns:
            raise ValueError("Group by columns required for group-based aggregation")
        
        # Validate group by columns exist
        missing_columns = [col for col in config.group_by_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Group by columns not found: {missing_columns}")
        
        # Group by specified columns
        grouped = df.groupby(config.group_by_columns)
        
        # Apply aggregation functions
        result_frames = []
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for func in config.functions:
            func_result = self._apply_aggregation_function(grouped, numeric_columns, func, config)
            
            if func_result is not None:
                # Add function suffix to column names
                func_result.columns = [f"{col}_{func}" for col in func_result.columns]
                result_frames.append(func_result)
                log_entries.append(f"Applied {func} aggregation grouped by {config.group_by_columns}")
        
        # Combine results
        if result_frames:
            result_df = pd.concat(result_frames, axis=1)
            result_df = result_df.reset_index()
        else:
            result_df = pd.DataFrame()
        
        return result_df, log_entries
    
    def _summary_aggregation(self, df: pd.DataFrame, config: AggregationConfig) -> Tuple[pd.DataFrame, List[str]]:
        """Perform summary aggregation across entire dataset."""
        log_entries = []
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        result_data = {}
        
        for func in config.functions:
            if func == AggregationFunction.MEAN:
                values = df[numeric_columns].mean()
            elif func == AggregationFunction.MEDIAN:
                values = df[numeric_columns].median()
            elif func == AggregationFunction.SUM:
                values = df[numeric_columns].sum()
            elif func == AggregationFunction.COUNT:
                values = df[numeric_columns].count()
            elif func == AggregationFunction.MIN:
                values = df[numeric_columns].min()
            elif func == AggregationFunction.MAX:
                values = df[numeric_columns].max()
            elif func == AggregationFunction.STD:
                values = df[numeric_columns].std()
            elif func == AggregationFunction.VAR:
                values = df[numeric_columns].var()
            elif func == AggregationFunction.FIRST:
                values = df[numeric_columns].first()
            elif func == AggregationFunction.LAST:
                values = df[numeric_columns].last()
            elif func == AggregationFunction.NUNIQUE:
                values = df[numeric_columns].nunique()
            elif func == AggregationFunction.WEIGHTED_MEAN and config.weight_column:
                values = self._calculate_weighted_mean(df, numeric_columns, config.weight_column)
            elif func == AggregationFunction.GEOMETRIC_MEAN:
                values = self._calculate_geometric_mean(df, numeric_columns)
            elif func == AggregationFunction.HARMONIC_MEAN:
                values = self._calculate_harmonic_mean(df, numeric_columns)
            elif func == AggregationFunction.QUANTILE and config.quantile_values:
                for q in config.quantile_values:
                    quantile_values = df[numeric_columns].quantile(q)
                    for col in quantile_values.index:
                        result_data[f"{col}_quantile_{q}"] = quantile_values[col]
                continue
            else:
                continue
            
            # Add results to data dictionary
            for col in values.index:
                result_data[f"{col}_{func}"] = values[col]
            
            log_entries.append(f"Applied {func} summary aggregation")
        
        # Create result DataFrame
        result_df = pd.DataFrame([result_data]) if result_data else pd.DataFrame()
        
        return result_df, log_entries
    
    def _apply_aggregation_function(self, grouped, columns: List[str], 
                                   func: AggregationFunction, config: AggregationConfig) -> Optional[pd.DataFrame]:
        """Apply specific aggregation function to grouped data."""
        try:
            if func == AggregationFunction.MEAN:
                return grouped[columns].mean()
            elif func == AggregationFunction.MEDIAN:
                return grouped[columns].median()
            elif func == AggregationFunction.SUM:
                return grouped[columns].sum()
            elif func == AggregationFunction.COUNT:
                return grouped[columns].count()
            elif func == AggregationFunction.MIN:
                return grouped[columns].min()
            elif func == AggregationFunction.MAX:
                return grouped[columns].max()
            elif func == AggregationFunction.STD:
                return grouped[columns].std()
            elif func == AggregationFunction.VAR:
                return grouped[columns].var()
            elif func == AggregationFunction.FIRST:
                return grouped[columns].first()
            elif func == AggregationFunction.LAST:
                return grouped[columns].last()
            elif func == AggregationFunction.NUNIQUE:
                return grouped[columns].nunique()
            elif func == AggregationFunction.QUANTILE and config.quantile_values:
                return grouped[columns].quantile(config.quantile_values)
            elif func == AggregationFunction.WEIGHTED_MEAN and config.weight_column:
                return self._grouped_weighted_mean(grouped, columns, config.weight_column)
            elif func == AggregationFunction.GEOMETRIC_MEAN:
                return self._grouped_geometric_mean(grouped, columns)
            elif func == AggregationFunction.HARMONIC_MEAN:
                return self._grouped_harmonic_mean(grouped, columns)
            else:
                logger.warning(f"Unsupported aggregation function: {func}")
                return None
                
        except Exception as e:
            logger.error(f"Error applying aggregation function {func}: {e}")
            return None
    
    def _calculate_weighted_mean(self, df: pd.DataFrame, columns: List[str], weight_column: str) -> pd.Series:
        """Calculate weighted mean for specified columns."""
        result = {}
        weights = df[weight_column]
        
        for col in columns:
            if col in df.columns:
                weighted_sum = (df[col] * weights).sum()
                weight_sum = weights.sum()
                result[col] = safe_divide(weighted_sum, weight_sum)
        
        return pd.Series(result)
    
    def _calculate_geometric_mean(self, df: pd.DataFrame, columns: List[str]) -> pd.Series:
        """Calculate geometric mean for specified columns."""
        result = {}
        
        for col in columns:
            if col in df.columns:
                positive_values = df[df[col] > 0][col]
                if len(positive_values) > 0:
                    result[col] = np.exp(np.log(positive_values).mean())
                else:
                    result[col] = np.nan
        
        return pd.Series(result)
    
    def _calculate_harmonic_mean(self, df: pd.DataFrame, columns: List[str]) -> pd.Series:
        """Calculate harmonic mean for specified columns."""
        result = {}
        
        for col in columns:
            if col in df.columns:
                positive_values = df[df[col] > 0][col]
                if len(positive_values) > 0:
                    result[col] = len(positive_values) / (1 / positive_values).sum()
                else:
                    result[col] = np.nan
        
        return pd.Series(result)
    
    def _grouped_weighted_mean(self, grouped, columns: List[str], weight_column: str) -> pd.DataFrame:
        """Calculate weighted mean for grouped data."""
        def weighted_mean_func(group):
            weights = group[weight_column]
            result = {}
            for col in columns:
                if col in group.columns:
                    weighted_sum = (group[col] * weights).sum()
                    weight_sum = weights.sum()
                    result[col] = safe_divide(weighted_sum, weight_sum)
            return pd.Series(result)
        
        return grouped.apply(weighted_mean_func)
    
    def _grouped_geometric_mean(self, grouped, columns: List[str]) -> pd.DataFrame:
        """Calculate geometric mean for grouped data."""
        def geometric_mean_func(group):
            result = {}
            for col in columns:
                if col in group.columns:
                    positive_values = group[group[col] > 0][col]
                    if len(positive_values) > 0:
                        result[col] = np.exp(np.log(positive_values).mean())
                    else:
                        result[col] = np.nan
            return pd.Series(result)
        
        return grouped.apply(geometric_mean_func)
    
    def _grouped_harmonic_mean(self, grouped, columns: List[str]) -> pd.DataFrame:
        """Calculate harmonic mean for grouped data."""
        def harmonic_mean_func(group):
            result = {}
            for col in columns:
                if col in group.columns:
                    positive_values = group[group[col] > 0][col]
                    if len(positive_values) > 0:
                        result[col] = len(positive_values) / (1 / positive_values).sum()
                    else:
                        result[col] = np.nan
            return pd.Series(result)
        
        return grouped.apply(harmonic_mean_func)
    
    def _calculate_summary_statistics(self, original_df: pd.DataFrame, 
                                    aggregated_df: pd.DataFrame, 
                                    config: AggregationConfig) -> Dict[str, Any]:
        """Calculate summary statistics for the aggregation operation."""
        return {
            'original_rows': len(original_df),
            'aggregated_rows': len(aggregated_df),
            'reduction_ratio': safe_divide(len(aggregated_df), len(original_df)),
            'original_columns': len(original_df.columns),
            'aggregated_columns': len(aggregated_df.columns),
            'aggregation_level': config.level,
            'functions_applied': len(config.functions),
            'date_range': {
                'start': original_df[config.date_column].min().isoformat() if config.date_column and config.date_column in original_df.columns else None,
                'end': original_df[config.date_column].max().isoformat() if config.date_column and config.date_column in original_df.columns else None
            } if config.date_column else None
        }
    
    def aggregate_economic_data(self, df: pd.DataFrame, level: AggregationLevel = AggregationLevel.MONTHLY) -> pd.DataFrame:
        """Aggregate economic data with appropriate functions."""
        config = AggregationConfig(
            level=level,
            functions=[
                AggregationFunction.MEAN,
                AggregationFunction.LAST,
                AggregationFunction.STD
            ],
            date_column='date'
        )
        
        result = self.aggregate_data(df, config)
        logger.info(f"Aggregated economic data to {level} level")
        return result.aggregated_data
    
    def aggregate_financial_data(self, df: pd.DataFrame, level: AggregationLevel = AggregationLevel.DAILY) -> pd.DataFrame:
        """Aggregate financial data with appropriate functions."""
        config = AggregationConfig(
            level=level,
            functions=[
                AggregationFunction.MEAN,
                AggregationFunction.FIRST,
                AggregationFunction.LAST,
                AggregationFunction.MIN,
                AggregationFunction.MAX
            ],
            date_column='date'
        )
        
        result = self.aggregate_data(df, config)
        logger.info(f"Aggregated financial data to {level} level")
        return result.aggregated_data
    
    def aggregate_supply_chain_data(self, df: pd.DataFrame, level: AggregationLevel = AggregationLevel.WEEKLY) -> pd.DataFrame:
        """Aggregate supply chain data with appropriate functions."""
        config = AggregationConfig(
            level=level,
            functions=[
                AggregationFunction.MEAN,
                AggregationFunction.SUM,
                AggregationFunction.MAX
            ],
            date_column='date'
        )
        
        result = self.aggregate_data(df, config)
        logger.info(f"Aggregated supply chain data to {level} level")
        return result.aggregated_data
    
    def create_risk_aggregation(self, df: pd.DataFrame, risk_columns: List[str]) -> pd.DataFrame:
        """Create risk aggregation with weighted averages."""
        try:
            # Create equal weights if not specified
            weight_column = 'risk_weight'
            if weight_column not in df.columns:
                df[weight_column] = 1.0
            
            config = AggregationConfig(
                level=AggregationLevel.CUSTOM,
                functions=[
                    AggregationFunction.WEIGHTED_MEAN,
                    AggregationFunction.MAX,
                    AggregationFunction.STD
                ],
                weight_column=weight_column
            )
            
            # Filter to risk columns only
            risk_df = df[risk_columns + [weight_column]].copy()
            
            result = self.aggregate_data(risk_df, config)
            logger.info(f"Created risk aggregation for {len(risk_columns)} risk factors")
            
            return result.aggregated_data
            
        except Exception as e:
            logger.error(f"Error creating risk aggregation: {e}")
            raise ProcessingError(f"Risk aggregation failed: {e}")
    
    def get_aggregation_summary(self) -> Dict[str, Any]:
        """Get summary of all aggregations performed."""
        return {
            'statistics': self.statistics.copy(),
            'aggregation_history': self.aggregation_history.copy(),
            'available_functions': [f.value for f in AggregationFunction],
            'available_levels': [l.value for l in AggregationLevel]
        }
    
    def reset_statistics(self):
        """Reset aggregation statistics."""
        self.statistics = {
            'aggregations_performed': 0,
            'rows_aggregated': 0,
            'columns_processed': 0
        }
        self.aggregation_history.clear()
        logger.info("Aggregation statistics reset")