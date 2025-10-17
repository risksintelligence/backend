"""
Data validation utilities for RiskX platform.
Provides comprehensive validation for data quality, schema compliance, and business rules.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from datetime import datetime, date, timedelta
from dataclasses import dataclass
import logging

from ...utils.helpers import is_numeric, validate_range, is_valid_email, is_valid_url
from ...utils.constants import BusinessRules, EconomicIndicators, RiskCategories
from ...core.exceptions import ValidationError, DataAccessError

logger = logging.getLogger('riskx.data.processors.validator')


@dataclass
class ValidationResult:
    """Result of data validation with detailed metrics."""
    
    is_valid: bool
    score: float  # 0.0 to 1.0
    total_checks: int
    passed_checks: int
    failed_checks: int
    errors: List[Dict[str, Any]]
    warnings: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'is_valid': self.is_valid,
            'score': self.score,
            'total_checks': self.total_checks,
            'passed_checks': self.passed_checks,
            'failed_checks': self.failed_checks,
            'errors': self.errors,
            'warnings': self.warnings,
            'metadata': self.metadata
        }


@dataclass
class ValidationRule:
    """Definition of a validation rule."""
    
    name: str
    description: str
    severity: str  # 'error', 'warning', 'info'
    validator_func: Callable
    column: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class DataValidator:
    """Comprehensive data validation for economic and financial data."""
    
    def __init__(self):
        self.validation_rules: List[ValidationRule] = []
        self.setup_default_rules()
    
    def setup_default_rules(self):
        """Setup default validation rules for common data types."""
        
        # Schema validation rules
        self.add_rule(
            name="required_columns",
            description="Check that required columns are present",
            severity="error",
            validator_func=self._validate_required_columns
        )
        
        self.add_rule(
            name="data_types",
            description="Validate data types match expected schema",
            severity="error",
            validator_func=self._validate_data_types
        )
        
        # Data quality rules
        self.add_rule(
            name="missing_values",
            description="Check missing value thresholds",
            severity="warning",
            validator_func=self._validate_missing_values,
            parameters={'threshold': 0.1}  # 10% threshold
        )
        
        self.add_rule(
            name="duplicate_records",
            description="Check for duplicate records",
            severity="warning",
            validator_func=self._validate_duplicates
        )
        
        # Value range validation
        self.add_rule(
            name="numeric_ranges",
            description="Validate numeric values are within reasonable ranges",
            severity="error",
            validator_func=self._validate_numeric_ranges
        )
        
        # Date validation
        self.add_rule(
            name="date_validity",
            description="Validate date values and ranges",
            severity="error",
            validator_func=self._validate_dates
        )
        
        # Business logic validation
        self.add_rule(
            name="business_rules",
            description="Validate business-specific rules",
            severity="error",
            validator_func=self._validate_business_rules
        )
    
    def add_rule(self, name: str, description: str, severity: str, 
                validator_func: Callable, column: Optional[str] = None,
                parameters: Optional[Dict[str, Any]] = None):
        """Add a custom validation rule."""
        rule = ValidationRule(
            name=name,
            description=description,
            severity=severity,
            validator_func=validator_func,
            column=column,
            parameters=parameters or {}
        )
        self.validation_rules.append(rule)
        logger.debug(f"Added validation rule: {name}")
    
    def validate_dataframe(self, df: pd.DataFrame, 
                          schema: Optional[Dict[str, Any]] = None,
                          custom_rules: Optional[List[ValidationRule]] = None) -> ValidationResult:
        """
        Comprehensive DataFrame validation.
        
        Args:
            df: DataFrame to validate
            schema: Expected schema definition
            custom_rules: Additional validation rules to apply
        """
        errors = []
        warnings = []
        total_checks = 0
        passed_checks = 0
        
        # Combine default rules with custom rules
        rules_to_apply = self.validation_rules.copy()
        if custom_rules:
            rules_to_apply.extend(custom_rules)
        
        # Run each validation rule
        for rule in rules_to_apply:
            total_checks += 1
            
            try:
                result = rule.validator_func(df, schema, rule.parameters)
                
                if result['passed']:
                    passed_checks += 1
                else:
                    error_info = {
                        'rule': rule.name,
                        'description': rule.description,
                        'severity': rule.severity,
                        'details': result.get('details', ''),
                        'column': rule.column,
                        'count': result.get('count', 0)
                    }
                    
                    if rule.severity == 'error':
                        errors.append(error_info)
                    else:
                        warnings.append(error_info)
                
            except Exception as e:
                logger.error(f"Error running validation rule {rule.name}: {e}")
                errors.append({
                    'rule': rule.name,
                    'description': f"Validation rule failed: {e}",
                    'severity': 'error',
                    'details': str(e)
                })
        
        # Calculate validation score
        failed_checks = total_checks - passed_checks
        score = passed_checks / total_checks if total_checks > 0 else 0.0
        
        # Determine overall validity
        is_valid = len(errors) == 0 and score >= BusinessRules.MIN_DATA_QUALITY_SCORE
        
        # Collect metadata
        metadata = {
            'validation_timestamp': datetime.utcnow().isoformat(),
            'dataframe_shape': df.shape,
            'columns': list(df.columns),
            'data_types': df.dtypes.to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        return ValidationResult(
            is_valid=is_valid,
            score=score,
            total_checks=total_checks,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            errors=errors,
            warnings=warnings,
            metadata=metadata
        )
    
    def validate_economic_data(self, df: pd.DataFrame) -> ValidationResult:
        """Specialized validation for economic indicator data."""
        schema = {
            'required_columns': ['date', 'value'],
            'column_types': {
                'date': 'datetime',
                'value': 'numeric'
            },
            'value_ranges': {
                'unemployment_rate': (0.0, 50.0),
                'inflation_rate': (-10.0, 50.0),
                'gdp_growth': (-20.0, 20.0),
                'interest_rate': (-5.0, 30.0)
            }
        }
        
        # Add economic-specific validation rules
        economic_rules = [
            ValidationRule(
                name="economic_indicator_ranges",
                description="Validate economic indicators are within historical ranges",
                severity="warning",
                validator_func=self._validate_economic_ranges
            ),
            ValidationRule(
                name="data_freshness",
                description="Check data freshness for economic indicators",
                severity="warning",
                validator_func=self._validate_data_freshness,
                parameters={'max_age_days': 7}
            )
        ]
        
        return self.validate_dataframe(df, schema, economic_rules)
    
    def validate_financial_data(self, df: pd.DataFrame) -> ValidationResult:
        """Specialized validation for financial market data."""
        schema = {
            'required_columns': ['date', 'value'],
            'column_types': {
                'date': 'datetime',
                'value': 'numeric'
            }
        }
        
        financial_rules = [
            ValidationRule(
                name="financial_data_continuity",
                description="Check for gaps in financial time series data",
                severity="warning",
                validator_func=self._validate_time_series_continuity
            ),
            ValidationRule(
                name="price_volatility",
                description="Check for unusual price volatility",
                severity="warning",
                validator_func=self._validate_price_volatility,
                parameters={'volatility_threshold': 0.1}  # 10% daily change
            )
        ]
        
        return self.validate_dataframe(df, schema, financial_rules)
    
    def validate_supply_chain_data(self, df: pd.DataFrame) -> ValidationResult:
        """Specialized validation for supply chain data."""
        schema = {
            'required_columns': ['date', 'metric', 'value'],
            'column_types': {
                'date': 'datetime',
                'metric': 'string',
                'value': 'numeric'
            }
        }
        
        supply_chain_rules = [
            ValidationRule(
                name="supply_chain_metrics",
                description="Validate supply chain metric values",
                severity="error",
                validator_func=self._validate_supply_chain_metrics
            )
        ]
        
        return self.validate_dataframe(df, schema, supply_chain_rules)
    
    # Validation rule implementations
    
    def _validate_required_columns(self, df: pd.DataFrame, schema: Optional[Dict], 
                                 params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that required columns are present."""
        if not schema or 'required_columns' not in schema:
            return {'passed': True}
        
        required_columns = schema['required_columns']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return {
                'passed': False,
                'details': f"Missing required columns: {missing_columns}",
                'count': len(missing_columns)
            }
        
        return {'passed': True}
    
    def _validate_data_types(self, df: pd.DataFrame, schema: Optional[Dict], 
                           params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data types match expected schema."""
        if not schema or 'column_types' not in schema:
            return {'passed': True}
        
        expected_types = schema['column_types']
        type_errors = []
        
        for column, expected_type in expected_types.items():
            if column not in df.columns:
                continue
            
            actual_dtype = str(df[column].dtype)
            
            # Map expected types to pandas dtypes
            type_mapping = {
                'numeric': ['int64', 'float64', 'Int64', 'Float64'],
                'string': ['object', 'string'],
                'datetime': ['datetime64[ns]', 'datetime64'],
                'boolean': ['bool', 'boolean']
            }
            
            if expected_type in type_mapping:
                valid_types = type_mapping[expected_type]
                if not any(valid_type in actual_dtype for valid_type in valid_types):
                    type_errors.append(f"{column}: expected {expected_type}, got {actual_dtype}")
        
        if type_errors:
            return {
                'passed': False,
                'details': f"Data type mismatches: {type_errors}",
                'count': len(type_errors)
            }
        
        return {'passed': True}
    
    def _validate_missing_values(self, df: pd.DataFrame, schema: Optional[Dict], 
                               params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate missing value thresholds."""
        threshold = params.get('threshold', 0.1)
        issues = []
        
        for column in df.columns:
            missing_ratio = df[column].isnull().sum() / len(df)
            if missing_ratio > threshold:
                issues.append(f"{column}: {missing_ratio:.1%} missing values")
        
        if issues:
            return {
                'passed': False,
                'details': f"High missing value ratios: {issues}",
                'count': len(issues)
            }
        
        return {'passed': True}
    
    def _validate_duplicates(self, df: pd.DataFrame, schema: Optional[Dict], 
                           params: Dict[str, Any]) -> Dict[str, Any]:
        """Check for duplicate records."""
        duplicate_count = df.duplicated().sum()
        
        if duplicate_count > 0:
            duplicate_ratio = duplicate_count / len(df)
            return {
                'passed': False,
                'details': f"Found {duplicate_count} duplicate records ({duplicate_ratio:.1%})",
                'count': duplicate_count
            }
        
        return {'passed': True}
    
    def _validate_numeric_ranges(self, df: pd.DataFrame, schema: Optional[Dict], 
                               params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate numeric values are within reasonable ranges."""
        if not schema or 'value_ranges' not in schema:
            return {'passed': True}
        
        value_ranges = schema['value_ranges']
        range_violations = []
        
        for column, (min_val, max_val) in value_ranges.items():
            if column not in df.columns:
                continue
            
            if df[column].dtype not in ['int64', 'float64', 'Int64', 'Float64']:
                continue
            
            out_of_range = ((df[column] < min_val) | (df[column] > max_val)).sum()
            if out_of_range > 0:
                range_violations.append(f"{column}: {out_of_range} values outside [{min_val}, {max_val}]")
        
        if range_violations:
            return {
                'passed': False,
                'details': f"Range violations: {range_violations}",
                'count': len(range_violations)
            }
        
        return {'passed': True}
    
    def _validate_dates(self, df: pd.DataFrame, schema: Optional[Dict], 
                       params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate date values and ranges."""
        date_columns = df.select_dtypes(include=['datetime64']).columns
        issues = []
        
        for column in date_columns:
            # Check for null dates
            null_dates = df[column].isnull().sum()
            if null_dates > 0:
                issues.append(f"{column}: {null_dates} null dates")
            
            # Check for future dates (if not expected)
            future_dates = (df[column] > datetime.now()).sum()
            if future_dates > 0:
                issues.append(f"{column}: {future_dates} future dates")
            
            # Check for very old dates (before 1900)
            very_old = (df[column] < datetime(1900, 1, 1)).sum()
            if very_old > 0:
                issues.append(f"{column}: {very_old} dates before 1900")
        
        if issues:
            return {
                'passed': False,
                'details': f"Date validation issues: {issues}",
                'count': len(issues)
            }
        
        return {'passed': True}
    
    def _validate_business_rules(self, df: pd.DataFrame, schema: Optional[Dict], 
                               params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate business-specific rules."""
        violations = []
        
        # Example business rules for economic data
        if 'unemployment_rate' in df.columns and 'employment_rate' in df.columns:
            # Unemployment + employment should approximately equal 100
            total_rate = df['unemployment_rate'] + df['employment_rate']
            unusual_totals = ((total_rate < 95) | (total_rate > 105)).sum()
            if unusual_totals > 0:
                violations.append(f"Unemployment + employment rate inconsistency: {unusual_totals} records")
        
        # Risk scores should be between 0 and 100
        risk_columns = [col for col in df.columns if 'risk' in col.lower() and 'score' in col.lower()]
        for column in risk_columns:
            if df[column].dtype in ['int64', 'float64']:
                invalid_scores = ((df[column] < 0) | (df[column] > 100)).sum()
                if invalid_scores > 0:
                    violations.append(f"{column}: {invalid_scores} values outside [0, 100]")
        
        if violations:
            return {
                'passed': False,
                'details': f"Business rule violations: {violations}",
                'count': len(violations)
            }
        
        return {'passed': True}
    
    def _validate_economic_ranges(self, df: pd.DataFrame, schema: Optional[Dict], 
                                params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate economic indicators are within historical ranges."""
        # Define historical ranges for major economic indicators
        historical_ranges = {
            'unemployment_rate': (0.0, 25.0),  # Great Depression max ~25%
            'inflation_rate': (-5.0, 20.0),    # Deflation to hyperinflation
            'gdp_growth': (-15.0, 15.0),       # Severe recession to boom
            'interest_rate': (0.0, 20.0),      # Zero bound to high inflation periods
            'debt_to_gdp': (0.0, 300.0)        # Some countries exceed 200%
        }
        
        violations = []
        
        for column in df.columns:
            col_lower = column.lower()
            
            # Find matching indicator
            for indicator, (min_val, max_val) in historical_ranges.items():
                if indicator.replace('_', '') in col_lower.replace('_', ''):
                    if df[column].dtype in ['int64', 'float64']:
                        extreme_values = ((df[column] < min_val) | (df[column] > max_val)).sum()
                        if extreme_values > 0:
                            violations.append(f"{column}: {extreme_values} historically extreme values")
                    break
        
        if violations:
            return {
                'passed': False,
                'details': f"Historical range violations: {violations}",
                'count': len(violations)
            }
        
        return {'passed': True}
    
    def _validate_data_freshness(self, df: pd.DataFrame, schema: Optional[Dict], 
                               params: Dict[str, Any]) -> Dict[str, Any]:
        """Check data freshness."""
        max_age_days = params.get('max_age_days', 7)
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        date_columns = df.select_dtypes(include=['datetime64']).columns
        
        if not date_columns.empty:
            latest_date = df[date_columns[0]].max()
            
            if pd.isna(latest_date) or latest_date < cutoff_date:
                return {
                    'passed': False,
                    'details': f"Data is not fresh. Latest date: {latest_date}, cutoff: {cutoff_date}",
                    'count': 1
                }
        
        return {'passed': True}
    
    def _validate_time_series_continuity(self, df: pd.DataFrame, schema: Optional[Dict], 
                                       params: Dict[str, Any]) -> Dict[str, Any]:
        """Check for gaps in time series data."""
        date_columns = df.select_dtypes(include=['datetime64']).columns
        
        if date_columns.empty:
            return {'passed': True}
        
        date_col = date_columns[0]
        df_sorted = df.sort_values(date_col)
        
        # Calculate gaps
        date_diffs = df_sorted[date_col].diff()
        
        # Identify large gaps (more than 7 days for daily data)
        large_gaps = (date_diffs > timedelta(days=7)).sum()
        
        if large_gaps > 0:
            return {
                'passed': False,
                'details': f"Found {large_gaps} large gaps in time series",
                'count': large_gaps
            }
        
        return {'passed': True}
    
    def _validate_price_volatility(self, df: pd.DataFrame, schema: Optional[Dict], 
                                 params: Dict[str, Any]) -> Dict[str, Any]:
        """Check for unusual price volatility."""
        volatility_threshold = params.get('volatility_threshold', 0.1)
        
        # Look for price or value columns
        value_columns = [col for col in df.columns if any(keyword in col.lower() 
                        for keyword in ['price', 'value', 'rate', 'yield'])]
        
        high_volatility_periods = 0
        
        for column in value_columns:
            if df[column].dtype in ['int64', 'float64']:
                # Calculate daily returns
                returns = df[column].pct_change().abs()
                high_vol = (returns > volatility_threshold).sum()
                high_volatility_periods += high_vol
        
        if high_volatility_periods > 0:
            return {
                'passed': False,
                'details': f"Found {high_volatility_periods} periods with high volatility",
                'count': high_volatility_periods
            }
        
        return {'passed': True}
    
    def _validate_supply_chain_metrics(self, df: pd.DataFrame, schema: Optional[Dict], 
                                     params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate supply chain specific metrics."""
        violations = []
        
        # Check for delivery time metrics
        if 'delivery_time' in df.columns:
            negative_times = (df['delivery_time'] < 0).sum()
            if negative_times > 0:
                violations.append(f"delivery_time: {negative_times} negative values")
        
        # Check for efficiency metrics (should be 0-1 or 0-100)
        efficiency_columns = [col for col in df.columns if 'efficiency' in col.lower()]
        for column in efficiency_columns:
            if df[column].dtype in ['int64', 'float64']:
                if df[column].max() <= 1:
                    # Assume 0-1 scale
                    invalid = ((df[column] < 0) | (df[column] > 1)).sum()
                else:
                    # Assume 0-100 scale
                    invalid = ((df[column] < 0) | (df[column] > 100)).sum()
                
                if invalid > 0:
                    violations.append(f"{column}: {invalid} values outside valid range")
        
        if violations:
            return {
                'passed': False,
                'details': f"Supply chain metric violations: {violations}",
                'count': len(violations)
            }
        
        return {'passed': True}
    
    def validate_api_response(self, data: Dict[str, Any], 
                            expected_schema: Dict[str, Any]) -> ValidationResult:
        """Validate API response data against expected schema."""
        errors = []
        warnings = []
        total_checks = 0
        passed_checks = 0
        
        # Check required fields
        if 'required_fields' in expected_schema:
            for field in expected_schema['required_fields']:
                total_checks += 1
                if field in data:
                    passed_checks += 1
                else:
                    errors.append({
                        'rule': 'required_field',
                        'description': f"Required field '{field}' is missing",
                        'severity': 'error',
                        'field': field
                    })
        
        # Check field types
        if 'field_types' in expected_schema:
            for field, expected_type in expected_schema['field_types'].items():
                if field in data:
                    total_checks += 1
                    value = data[field]
                    
                    type_valid = False
                    if expected_type == 'string' and isinstance(value, str):
                        type_valid = True
                    elif expected_type == 'number' and isinstance(value, (int, float)):
                        type_valid = True
                    elif expected_type == 'boolean' and isinstance(value, bool):
                        type_valid = True
                    elif expected_type == 'array' and isinstance(value, list):
                        type_valid = True
                    elif expected_type == 'object' and isinstance(value, dict):
                        type_valid = True
                    
                    if type_valid:
                        passed_checks += 1
                    else:
                        errors.append({
                            'rule': 'field_type',
                            'description': f"Field '{field}' has wrong type: expected {expected_type}, got {type(value).__name__}",
                            'severity': 'error',
                            'field': field
                        })
        
        # Calculate score and validity
        failed_checks = total_checks - passed_checks
        score = passed_checks / total_checks if total_checks > 0 else 0.0
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            score=score,
            total_checks=total_checks,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            errors=errors,
            warnings=warnings,
            metadata={'validation_type': 'api_response'}
        )