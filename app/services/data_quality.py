"""
RRIO Data Quality Validation Framework
Great Expectations-style validation for institutional-grade data integrity
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
import numpy as np
from great_expectations import DataContext
from great_expectations.dataset.pandas_dataset import PandasDataset
import structlog
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)
struct_logger = structlog.get_logger()

class ValidationSeverity(Enum):
    CRITICAL = "critical"
    HIGH = "high" 
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class ValidationCategory(Enum):
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    TIMELINESS = "timeliness"
    VALIDITY = "validity"
    INTEGRITY = "integrity"

@dataclass
class ValidationResult:
    check_name: str
    category: ValidationCategory
    severity: ValidationSeverity
    passed: bool
    message: str
    metadata: Dict[str, Any]
    timestamp: datetime
    affected_records: int = 0
    total_records: int = 0
    
    @property
    def success_rate(self) -> float:
        if self.total_records == 0:
            return 1.0
        return 1.0 - (self.affected_records / self.total_records)

@dataclass  
class DataQualityReport:
    validation_results: List[ValidationResult]
    overall_score: float
    critical_issues: int
    timestamp: datetime
    data_lineage: Dict[str, Any]
    
    @property
    def institutional_grade(self) -> bool:
        """Determine if data quality meets institutional standards"""
        return self.overall_score >= 0.95 and self.critical_issues == 0

class RRIODataQualityValidator:
    """
    RRIO-specific data quality validation framework
    Implements institutional-grade validation for economic data ingestion
    """
    
    def __init__(self):
        self.validation_results: List[ValidationResult] = []
        self.economic_indicators = [
            "VIX", "YIELD_CURVE", "CREDIT_SPREADS", "OIL_PRICES", 
            "USD_INDEX", "CPI_YOY", "UNEMPLOYMENT", "PMI_DIFFUSION",
            "FREIGHT_DIESEL", "BALTIC_DRY"
        ]
        
    def validate_observation_data(self, observations: List[Dict[str, Any]]) -> DataQualityReport:
        """
        Comprehensive validation of ingested observation data
        """
        struct_logger.info("Starting data quality validation", 
                          observations_count=len(observations))
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(observations)
        
        # Reset validation results
        self.validation_results = []
        
        # Core Data Quality Checks
        self._validate_completeness(df)
        self._validate_accuracy(df)
        self._validate_consistency(df)
        self._validate_timeliness(df)
        self._validate_validity(df)
        self._validate_integrity(df)
        
        # Calculate overall score
        overall_score = self._calculate_quality_score()
        critical_issues = sum(1 for r in self.validation_results 
                             if r.severity == ValidationSeverity.CRITICAL and not r.passed)
        
        # Create comprehensive report
        report = DataQualityReport(
            validation_results=self.validation_results,
            overall_score=overall_score,
            critical_issues=critical_issues,
            timestamp=datetime.utcnow(),
            data_lineage=self._generate_lineage_metadata(df)
        )
        
        struct_logger.info("Data quality validation completed",
                          overall_score=overall_score,
                          critical_issues=critical_issues,
                          institutional_grade=report.institutional_grade)
        
        return report
    
    def _validate_completeness(self, df: pd.DataFrame) -> None:
        """Validate data completeness for critical economic indicators"""
        
        # Check for required columns
        required_columns = ['series_id', 'value', 'observed_at']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        self.validation_results.append(ValidationResult(
            check_name="required_columns_present",
            category=ValidationCategory.COMPLETENESS,
            severity=ValidationSeverity.CRITICAL if missing_columns else ValidationSeverity.INFO,
            passed=len(missing_columns) == 0,
            message=f"Required columns check: {len(missing_columns)} missing" if missing_columns 
                   else "All required columns present",
            metadata={"missing_columns": missing_columns},
            timestamp=datetime.utcnow(),
            total_records=len(df)
        ))
        
        # Check data coverage for economic indicators
        present_indicators = set(df['series_id'].unique()) if 'series_id' in df.columns else set()
        missing_indicators = set(self.economic_indicators) - present_indicators
        coverage_rate = len(present_indicators) / len(self.economic_indicators)
        
        self.validation_results.append(ValidationResult(
            check_name="economic_indicator_coverage",
            category=ValidationCategory.COMPLETENESS,
            severity=ValidationSeverity.HIGH if coverage_rate < 0.8 else ValidationSeverity.INFO,
            passed=coverage_rate >= 0.8,
            message=f"Economic indicator coverage: {coverage_rate:.1%}",
            metadata={
                "coverage_rate": coverage_rate,
                "missing_indicators": list(missing_indicators),
                "present_indicators": list(present_indicators)
            },
            timestamp=datetime.utcnow(),
            total_records=len(self.economic_indicators)
        ))
        
        # Check for null values in critical fields
        if 'value' in df.columns:
            null_values = df['value'].isnull().sum()
            null_rate = null_values / len(df) if len(df) > 0 else 0
            
            self.validation_results.append(ValidationResult(
                check_name="null_value_check",
                category=ValidationCategory.COMPLETENESS,
                severity=ValidationSeverity.CRITICAL if null_rate > 0.05 else ValidationSeverity.LOW,
                passed=null_rate <= 0.05,
                message=f"Null value rate: {null_rate:.1%}",
                metadata={"null_count": int(null_values), "null_rate": null_rate},
                timestamp=datetime.utcnow(),
                affected_records=int(null_values),
                total_records=len(df)
            ))
    
    def _validate_accuracy(self, df: pd.DataFrame) -> None:
        """Validate data accuracy against expected ranges"""
        
        if 'value' not in df.columns or 'series_id' not in df.columns:
            return
            
        # Define expected ranges for economic indicators
        expected_ranges = {
            "VIX": (0, 100),
            "YIELD_CURVE": (-5, 5),
            "CREDIT_SPREADS": (0, 10),
            "OIL_PRICES": (0, 200),
            "USD_INDEX": (70, 130),
            "CPI_YOY": (-5, 20),
            "UNEMPLOYMENT": (0, 25),
            "PMI_DIFFUSION": (0, 100),
            "FREIGHT_DIESEL": (0, 10),
            "BALTIC_DRY": (0, 5000)
        }
        
        outliers_total = 0
        
        for series_id, (min_val, max_val) in expected_ranges.items():
            series_data = df[df['series_id'] == series_id]['value']
            if len(series_data) == 0:
                continue
                
            outliers = series_data[(series_data < min_val) | (series_data > max_val)]
            outlier_rate = len(outliers) / len(series_data)
            outliers_total += len(outliers)
            
            self.validation_results.append(ValidationResult(
                check_name=f"range_validation_{series_id}",
                category=ValidationCategory.ACCURACY,
                severity=ValidationSeverity.HIGH if outlier_rate > 0.1 else ValidationSeverity.MEDIUM,
                passed=outlier_rate <= 0.1,
                message=f"{series_id} range validation: {outlier_rate:.1%} outliers",
                metadata={
                    "expected_range": [min_val, max_val],
                    "outlier_count": len(outliers),
                    "outlier_rate": outlier_rate,
                    "series_id": series_id
                },
                timestamp=datetime.utcnow(),
                affected_records=len(outliers),
                total_records=len(series_data)
            ))
        
        # Statistical outlier detection using IQR
        numeric_data = pd.to_numeric(df['value'], errors='coerce').dropna()
        if len(numeric_data) > 0:
            Q1 = numeric_data.quantile(0.25)
            Q3 = numeric_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR  # More conservative than typical 1.5
            upper_bound = Q3 + 3 * IQR
            
            statistical_outliers = numeric_data[(numeric_data < lower_bound) | (numeric_data > upper_bound)]
            outlier_rate = len(statistical_outliers) / len(numeric_data)
            
            self.validation_results.append(ValidationResult(
                check_name="statistical_outlier_detection",
                category=ValidationCategory.ACCURACY,
                severity=ValidationSeverity.MEDIUM if outlier_rate > 0.05 else ValidationSeverity.LOW,
                passed=outlier_rate <= 0.05,
                message=f"Statistical outlier rate: {outlier_rate:.1%}",
                metadata={
                    "method": "IQR_3x",
                    "bounds": [float(lower_bound), float(upper_bound)],
                    "outlier_count": len(statistical_outliers)
                },
                timestamp=datetime.utcnow(),
                affected_records=len(statistical_outliers),
                total_records=len(numeric_data)
            ))
    
    def _validate_consistency(self, df: pd.DataFrame) -> None:
        """Validate data consistency across time and sources"""
        
        if 'observed_at' not in df.columns or 'value' not in df.columns:
            return
        
        # Check for duplicate observations (same series_id, observed_at)
        if 'series_id' in df.columns:
            duplicates = df.duplicated(subset=['series_id', 'observed_at'])
            duplicate_count = duplicates.sum()
            
            self.validation_results.append(ValidationResult(
                check_name="duplicate_observations",
                category=ValidationCategory.CONSISTENCY,
                severity=ValidationSeverity.HIGH if duplicate_count > 0 else ValidationSeverity.INFO,
                passed=duplicate_count == 0,
                message=f"Duplicate observations: {duplicate_count}",
                metadata={"duplicate_count": int(duplicate_count)},
                timestamp=datetime.utcnow(),
                affected_records=int(duplicate_count),
                total_records=len(df)
            ))
        
        # Check for temporal consistency (no future dates)
        df_copy = df.copy()
        df_copy['observed_at'] = pd.to_datetime(df_copy['observed_at'], errors='coerce')
        future_dates = df_copy['observed_at'] > datetime.utcnow()
        future_count = future_dates.sum() if not future_dates.empty else 0
        
        self.validation_results.append(ValidationResult(
            check_name="temporal_consistency",
            category=ValidationCategory.CONSISTENCY,
            severity=ValidationSeverity.CRITICAL if future_count > 0 else ValidationSeverity.INFO,
            passed=future_count == 0,
            message=f"Future dated observations: {future_count}",
            metadata={"future_count": int(future_count)},
            timestamp=datetime.utcnow(),
            affected_records=int(future_count),
            total_records=len(df)
        ))
    
    def _validate_timeliness(self, df: pd.DataFrame) -> None:
        """Validate data timeliness and freshness"""
        
        if 'fetched_at' not in df.columns:
            return
            
        # Check data freshness
        now = datetime.utcnow()
        df_copy = df.copy()
        df_copy['fetched_at'] = pd.to_datetime(df_copy['fetched_at'], errors='coerce')
        
        # Data older than 24 hours is considered stale for real-time indicators
        stale_threshold = now - timedelta(hours=24)
        stale_data = df_copy['fetched_at'] < stale_threshold
        stale_count = stale_data.sum() if not stale_data.empty else 0
        
        self.validation_results.append(ValidationResult(
            check_name="data_freshness",
            category=ValidationCategory.TIMELINESS,
            severity=ValidationSeverity.MEDIUM if stale_count > 0 else ValidationSeverity.INFO,
            passed=stale_count == 0,
            message=f"Stale data points: {stale_count}",
            metadata={
                "stale_threshold_hours": 24,
                "stale_count": int(stale_count)
            },
            timestamp=datetime.utcnow(),
            affected_records=int(stale_count),
            total_records=len(df)
        ))
    
    def _validate_validity(self, df: pd.DataFrame) -> None:
        """Validate data format and type validity"""
        
        # Check numeric value validity
        if 'value' in df.columns:
            numeric_conversion = pd.to_numeric(df['value'], errors='coerce')
            invalid_numeric = numeric_conversion.isnull().sum() - df['value'].isnull().sum()
            
            self.validation_results.append(ValidationResult(
                check_name="numeric_validity",
                category=ValidationCategory.VALIDITY,
                severity=ValidationSeverity.HIGH if invalid_numeric > 0 else ValidationSeverity.INFO,
                passed=invalid_numeric == 0,
                message=f"Invalid numeric values: {invalid_numeric}",
                metadata={"invalid_count": int(invalid_numeric)},
                timestamp=datetime.utcnow(),
                affected_records=int(invalid_numeric),
                total_records=len(df)
            ))
        
        # Check datetime validity
        if 'observed_at' in df.columns:
            datetime_conversion = pd.to_datetime(df['observed_at'], errors='coerce')
            invalid_dates = datetime_conversion.isnull().sum() - df['observed_at'].isnull().sum()
            
            self.validation_results.append(ValidationResult(
                check_name="datetime_validity",
                category=ValidationCategory.VALIDITY,
                severity=ValidationSeverity.HIGH if invalid_dates > 0 else ValidationSeverity.INFO,
                passed=invalid_dates == 0,
                message=f"Invalid datetime values: {invalid_dates}",
                metadata={"invalid_count": int(invalid_dates)},
                timestamp=datetime.utcnow(),
                affected_records=int(invalid_dates),
                total_records=len(df)
            ))
    
    def _validate_integrity(self, df: pd.DataFrame) -> None:
        """Validate referential integrity and business rules"""
        
        # Check series_id validity
        if 'series_id' in df.columns:
            valid_series = set(self.economic_indicators)
            invalid_series = set(df['series_id']) - valid_series
            invalid_count = df['series_id'].isin(invalid_series).sum()
            
            self.validation_results.append(ValidationResult(
                check_name="series_integrity", 
                category=ValidationCategory.INTEGRITY,
                severity=ValidationSeverity.HIGH if invalid_count > 0 else ValidationSeverity.INFO,
                passed=invalid_count == 0,
                message=f"Invalid series IDs: {len(invalid_series)}",
                metadata={
                    "invalid_series": list(invalid_series),
                    "invalid_count": int(invalid_count)
                },
                timestamp=datetime.utcnow(),
                affected_records=int(invalid_count),
                total_records=len(df)
            ))
    
    def _calculate_quality_score(self) -> float:
        """Calculate overall data quality score (0-1 scale)"""
        if not self.validation_results:
            return 0.0
        
        # Weight by severity
        severity_weights = {
            ValidationSeverity.CRITICAL: 1.0,
            ValidationSeverity.HIGH: 0.8,
            ValidationSeverity.MEDIUM: 0.6,
            ValidationSeverity.LOW: 0.4,
            ValidationSeverity.INFO: 0.1
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for result in self.validation_results:
            weight = severity_weights[result.severity]
            score = result.success_rate if result.passed else 0.0
            weighted_score += score * weight
            total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 1.0
    
    def _generate_lineage_metadata(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate data lineage metadata for transparency"""
        return {
            "validation_timestamp": datetime.utcnow().isoformat(),
            "validator_version": "1.0.0",
            "data_shape": df.shape,
            "data_sources": list(df.get('series_id', pd.Series()).unique()) if 'series_id' in df.columns else [],
            "validation_framework": "RRIO_DQF",
            "institutional_compliance": True,
            "quality_standards": ["completeness", "accuracy", "consistency", "timeliness", "validity", "integrity"]
        }

# Factory for creating validation reports
def validate_ingestion_output(observations: List[Dict[str, Any]]) -> DataQualityReport:
    """
    Main entry point for data quality validation
    """
    validator = RRIODataQualityValidator()
    return validator.validate_observation_data(observations)

# Export validation results for transparency endpoints
def export_validation_summary(report: DataQualityReport) -> Dict[str, Any]:
    """Export validation summary for API responses"""
    return {
        "overall_score": round(report.overall_score, 3),
        "institutional_grade": report.institutional_grade,
        "critical_issues": report.critical_issues,
        "total_checks": len(report.validation_results),
        "passed_checks": sum(1 for r in report.validation_results if r.passed),
        "timestamp": report.timestamp.isoformat(),
        "categories": {
            category.value: {
                "checks": sum(1 for r in report.validation_results if r.category == category),
                "passed": sum(1 for r in report.validation_results if r.category == category and r.passed)
            }
            for category in ValidationCategory
        },
        "data_lineage": report.data_lineage
    }