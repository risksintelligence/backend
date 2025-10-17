"""
Data cleaning utilities for RiskX platform.
Provides comprehensive data cleaning and normalization functions for all data sources.
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, date
import logging

from ...utils.helpers import clean_string, is_numeric, parse_date_string
from ...utils.constants import BusinessRules
from ...core.exceptions import ProcessingError, ValidationError

logger = logging.getLogger('riskx.data.processors.cleaner')


class DataCleaner:
    """Comprehensive data cleaning utilities for economic and financial data."""
    
    def __init__(self):
        self.cleaning_stats = {
            'rows_processed': 0,
            'rows_cleaned': 0,
            'values_imputed': 0,
            'outliers_removed': 0,
            'duplicates_removed': 0
        }
    
    def clean_dataframe(self, df: pd.DataFrame, 
                       config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Comprehensive DataFrame cleaning with configurable options.
        """
        if config is None:
            config = self._get_default_config()
        
        try:
            original_rows = len(df)
            self.cleaning_stats['rows_processed'] = original_rows
            
            # Create copy to avoid modifying original
            cleaned_df = df.copy()
            
            # Apply cleaning steps in order
            if config.get('remove_duplicates', True):
                cleaned_df = self._remove_duplicates(cleaned_df)
            
            if config.get('clean_column_names', True):
                cleaned_df = self._clean_column_names(cleaned_df)
            
            if config.get('standardize_types', True):
                cleaned_df = self._standardize_data_types(cleaned_df)
            
            if config.get('handle_missing', True):
                cleaned_df = self._handle_missing_values(
                    cleaned_df, 
                    config.get('missing_strategy', 'auto')
                )
            
            if config.get('remove_outliers', False):
                cleaned_df = self._remove_outliers(
                    cleaned_df,
                    config.get('outlier_method', 'iqr'),
                    config.get('outlier_threshold', 3.0)
                )
            
            if config.get('validate_ranges', True):
                cleaned_df = self._validate_value_ranges(cleaned_df)
            
            # Update stats
            self.cleaning_stats['rows_cleaned'] = len(cleaned_df)
            
            logger.info(f"Data cleaning completed: {original_rows} -> {len(cleaned_df)} rows")
            
            return cleaned_df
            
        except Exception as e:
            logger.error(f"Error during data cleaning: {e}")
            raise ProcessingError(f"Data cleaning failed: {e}")
    
    def clean_economic_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Specialized cleaning for economic indicator data."""
        config = {
            'remove_duplicates': True,
            'clean_column_names': True,
            'standardize_types': True,
            'handle_missing': True,
            'missing_strategy': 'interpolate',
            'remove_outliers': True,
            'outlier_method': 'zscore',
            'outlier_threshold': 3.0,
            'validate_ranges': True
        }
        
        return self.clean_dataframe(df, config)
    
    def clean_financial_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Specialized cleaning for financial market data."""
        config = {
            'remove_duplicates': True,
            'clean_column_names': True,
            'standardize_types': True,
            'handle_missing': True,
            'missing_strategy': 'forward_fill',
            'remove_outliers': False,  # Financial data can have legitimate extreme values
            'validate_ranges': True
        }
        
        return self.clean_dataframe(df, config)
    
    def clean_supply_chain_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Specialized cleaning for supply chain data."""
        config = {
            'remove_duplicates': True,
            'clean_column_names': True,
            'standardize_types': True,
            'handle_missing': True,
            'missing_strategy': 'auto',
            'remove_outliers': True,
            'outlier_method': 'iqr',
            'outlier_threshold': 2.5,
            'validate_ranges': True
        }
        
        return self.clean_dataframe(df, config)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default cleaning configuration."""
        return {
            'remove_duplicates': True,
            'clean_column_names': True,
            'standardize_types': True,
            'handle_missing': True,
            'missing_strategy': 'auto',
            'remove_outliers': False,
            'outlier_method': 'iqr',
            'outlier_threshold': 3.0,
            'validate_ranges': True
        }
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows from DataFrame."""
        initial_count = len(df)
        
        # Remove exact duplicates
        df_cleaned = df.drop_duplicates()
        
        # For time series data, also remove duplicates based on date column
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if date_cols:
            df_cleaned = df_cleaned.drop_duplicates(subset=date_cols, keep='last')
        
        duplicates_removed = initial_count - len(df_cleaned)
        self.cleaning_stats['duplicates_removed'] += duplicates_removed
        
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate rows")
        
        return df_cleaned
    
    def _clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names."""
        cleaned_columns = {}
        
        for col in df.columns:
            # Convert to lowercase
            clean_col = col.lower()
            
            # Replace spaces and special characters with underscores
            clean_col = re.sub(r'[^\w]', '_', clean_col)
            
            # Remove multiple consecutive underscores
            clean_col = re.sub(r'_+', '_', clean_col)
            
            # Remove leading/trailing underscores
            clean_col = clean_col.strip('_')
            
            # Ensure valid Python identifier
            if clean_col[0].isdigit():
                clean_col = 'col_' + clean_col
            
            cleaned_columns[col] = clean_col
        
        return df.rename(columns=cleaned_columns)
    
    def _standardize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize data types based on content analysis."""
        df_cleaned = df.copy()
        
        for col in df.columns:
            try:
                # Skip if column is already properly typed
                if df[col].dtype in ['datetime64[ns]', 'bool']:
                    continue
                
                # Try to convert to datetime if column name suggests it
                if any(keyword in col.lower() for keyword in ['date', 'time', 'timestamp']):
                    df_cleaned[col] = pd.to_datetime(df[col], errors='coerce')
                    continue
                
                # Try to convert to numeric
                if df[col].dtype == 'object':
                    # Remove common non-numeric characters
                    cleaned_values = df[col].astype(str).str.replace(r'[$,%]', '', regex=True)
                    
                    # Try converting to numeric
                    numeric_series = pd.to_numeric(cleaned_values, errors='coerce')
                    
                    # If most values converted successfully, use numeric type
                    if numeric_series.notna().sum() / len(df) > 0.8:
                        df_cleaned[col] = numeric_series
                        
                        # Choose int or float based on content
                        if numeric_series.notna().all() and (numeric_series == numeric_series.astype(int)).all():
                            df_cleaned[col] = numeric_series.astype('Int64')  # Nullable integer
                
            except Exception as e:
                logger.warning(f"Could not standardize data type for column {col}: {e}")
                continue
        
        return df_cleaned
    
    def _handle_missing_values(self, df: pd.DataFrame, strategy: str) -> pd.DataFrame:
        """Handle missing values with specified strategy."""
        df_cleaned = df.copy()
        initial_missing = df.isnull().sum().sum()
        
        for col in df.columns:
            if df[col].isnull().sum() == 0:
                continue
            
            missing_ratio = df[col].isnull().sum() / len(df)
            
            # If too many missing values, consider dropping the column
            if missing_ratio > 0.5:
                logger.warning(f"Column {col} has {missing_ratio:.1%} missing values")
                continue
            
            if strategy == 'auto':
                # Choose strategy based on data type and content
                if df[col].dtype in ['int64', 'float64', 'Int64']:
                    if 'price' in col.lower() or 'rate' in col.lower():
                        # Forward fill for prices and rates
                        df_cleaned[col] = df[col].fillna(method='ffill')
                    else:
                        # Interpolate for other numeric data
                        df_cleaned[col] = df[col].interpolate(method='linear')
                else:
                    # Forward fill for non-numeric data
                    df_cleaned[col] = df[col].fillna(method='ffill')
            
            elif strategy == 'interpolate':
                if df[col].dtype in ['int64', 'float64', 'Int64']:
                    df_cleaned[col] = df[col].interpolate(method='linear')
                else:
                    df_cleaned[col] = df[col].fillna(method='ffill')
            
            elif strategy == 'forward_fill':
                df_cleaned[col] = df[col].fillna(method='ffill')
            
            elif strategy == 'backward_fill':
                df_cleaned[col] = df[col].fillna(method='bfill')
            
            elif strategy == 'mean':
                if df[col].dtype in ['int64', 'float64', 'Int64']:
                    df_cleaned[col] = df[col].fillna(df[col].mean())
            
            elif strategy == 'median':
                if df[col].dtype in ['int64', 'float64', 'Int64']:
                    df_cleaned[col] = df[col].fillna(df[col].median())
            
            elif strategy == 'drop':
                df_cleaned = df_cleaned.dropna(subset=[col])
        
        final_missing = df_cleaned.isnull().sum().sum()
        values_imputed = initial_missing - final_missing
        self.cleaning_stats['values_imputed'] += values_imputed
        
        if values_imputed > 0:
            logger.info(f"Imputed {values_imputed} missing values using {strategy} strategy")
        
        return df_cleaned
    
    def _remove_outliers(self, df: pd.DataFrame, method: str, threshold: float) -> pd.DataFrame:
        """Remove outliers using specified method."""
        df_cleaned = df.copy()
        outliers_removed = 0
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outlier_mask = z_scores > threshold
            
            elif method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            
            elif method == 'percentile':
                lower_percentile = (100 - threshold) / 2
                upper_percentile = 100 - lower_percentile
                lower_bound = df[col].quantile(lower_percentile / 100)
                upper_bound = df[col].quantile(upper_percentile / 100)
                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            
            else:
                logger.warning(f"Unknown outlier detection method: {method}")
                continue
            
            # Count outliers before removal
            col_outliers = outlier_mask.sum()
            outliers_removed += col_outliers
            
            # Remove outliers
            df_cleaned = df_cleaned[~outlier_mask]
            
            if col_outliers > 0:
                logger.info(f"Removed {col_outliers} outliers from column {col}")
        
        self.cleaning_stats['outliers_removed'] += outliers_removed
        return df_cleaned
    
    def _validate_value_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate that values are within reasonable ranges for their type."""
        df_cleaned = df.copy()
        
        # Define reasonable ranges for common economic indicators
        range_validators = {
            'unemployment_rate': (0.0, 50.0),
            'inflation_rate': (-10.0, 50.0),
            'interest_rate': (-5.0, 30.0),
            'gdp_growth': (-20.0, 20.0),
            'percentage': (0.0, 100.0),
            'rate': (-50.0, 50.0)
        }
        
        for col in df.columns:
            col_lower = col.lower()
            
            # Check for specific indicators
            validator_key = None
            for key in range_validators:
                if key in col_lower:
                    validator_key = key
                    break
            
            if validator_key:
                min_val, max_val = range_validators[validator_key]
                
                # Remove values outside valid range
                valid_mask = (df[col] >= min_val) & (df[col] <= max_val)
                invalid_count = (~valid_mask).sum()
                
                if invalid_count > 0:
                    logger.warning(f"Removing {invalid_count} invalid values from {col}")
                    df_cleaned = df_cleaned[valid_mask]
        
        return df_cleaned
    
    def clean_text_data(self, text: str, 
                       remove_html: bool = True,
                       normalize_whitespace: bool = True,
                       remove_special_chars: bool = False) -> str:
        """Clean text data for news and event processing."""
        if not isinstance(text, str):
            return str(text) if text is not None else ""
        
        cleaned_text = text
        
        # Remove HTML tags
        if remove_html:
            cleaned_text = re.sub(r'<[^>]+>', '', cleaned_text)
        
        # Normalize whitespace
        if normalize_whitespace:
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        # Remove special characters (keep alphanumeric, spaces, and basic punctuation)
        if remove_special_chars:
            cleaned_text = re.sub(r'[^\w\s.,!?-]', '', cleaned_text)
        
        return cleaned_text
    
    def clean_currency_values(self, values: pd.Series) -> pd.Series:
        """Clean currency values by removing symbols and converting to numeric."""
        if values.dtype == 'object':
            # Remove currency symbols and formatting
            cleaned = values.astype(str).str.replace(r'[$€£¥,]', '', regex=True)
            cleaned = cleaned.str.replace(r'[^\d.-]', '', regex=True)
            
            # Convert to numeric
            return pd.to_numeric(cleaned, errors='coerce')
        
        return values
    
    def clean_percentage_values(self, values: pd.Series) -> pd.Series:
        """Clean percentage values and convert to decimal if needed."""
        if values.dtype == 'object':
            # Remove percentage symbol
            cleaned = values.astype(str).str.replace('%', '')
            cleaned = pd.to_numeric(cleaned, errors='coerce')
            
            # Convert to decimal if values appear to be percentages (>1)
            if cleaned.max() > 1:
                cleaned = cleaned / 100
            
            return cleaned
        
        return values
    
    def get_cleaning_report(self) -> Dict[str, Any]:
        """Get report of cleaning operations performed."""
        return {
            'cleaning_stats': self.cleaning_stats.copy(),
            'data_quality_score': self._calculate_data_quality_score(),
            'recommendations': self._get_cleaning_recommendations()
        }
    
    def _calculate_data_quality_score(self) -> float:
        """Calculate overall data quality score based on cleaning stats."""
        if self.cleaning_stats['rows_processed'] == 0:
            return 0.0
        
        # Base score
        score = 1.0
        
        # Penalize for high proportion of removed data
        removal_rate = (
            self.cleaning_stats['outliers_removed'] + 
            self.cleaning_stats['duplicates_removed']
        ) / self.cleaning_stats['rows_processed']
        
        score -= removal_rate * 0.5
        
        # Penalize for high imputation rate
        imputation_rate = self.cleaning_stats['values_imputed'] / (
            self.cleaning_stats['rows_processed'] * 10  # Assume average 10 columns
        )
        
        score -= imputation_rate * 0.3
        
        return max(0.0, min(1.0, score))
    
    def _get_cleaning_recommendations(self) -> List[str]:
        """Get recommendations based on cleaning results."""
        recommendations = []
        
        if self.cleaning_stats['duplicates_removed'] > 0:
            recommendations.append("Consider improving data collection to reduce duplicates")
        
        if self.cleaning_stats['values_imputed'] > 0:
            recommendations.append("Review missing value patterns and consider improving data sources")
        
        if self.cleaning_stats['outliers_removed'] > 0:
            recommendations.append("Investigate outlier patterns to identify data quality issues")
        
        data_quality = self._calculate_data_quality_score()
        if data_quality < 0.8:
            recommendations.append("Data quality is below recommended threshold - review cleaning process")
        
        return recommendations