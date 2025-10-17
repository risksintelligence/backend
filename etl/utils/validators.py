"""
Data Validation Utilities

Provides comprehensive data validation capabilities for ETL pipelines,
including schema validation, data quality checks, and business rule validation.
"""

import logging
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime, timedelta
import json
import re
from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    """Result of a validation check"""
    is_valid: bool
    rule_name: str
    message: str
    severity: str = "error"  # error, warning, info
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ValidationReport:
    """Comprehensive validation report"""
    source_name: str
    total_checks: int
    passed_checks: int
    failed_checks: int
    warnings: int
    errors: List[ValidationResult] = field(default_factory=list)
    warnings_list: List[ValidationResult] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage"""
        if self.total_checks == 0:
            return 0.0
        return (self.passed_checks / self.total_checks) * 100
    
    @property
    def is_valid(self) -> bool:
        """Check if validation passed (no errors)"""
        return len(self.errors) == 0
    
    def add_result(self, result: ValidationResult):
        """Add a validation result to the report"""
        if result.severity == "error":
            self.errors.append(result)
            self.failed_checks += 1
        elif result.severity == "warning":
            self.warnings_list.append(result)
            self.warnings += 1
        else:
            self.passed_checks += 1
        
        self.total_checks += 1
    
    def finalize(self):
        """Finalize the report"""
        self.end_time = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary"""
        return {
            "source_name": self.source_name,
            "total_checks": self.total_checks,
            "passed_checks": self.passed_checks,
            "failed_checks": self.failed_checks,
            "warnings": self.warnings,
            "success_rate": self.success_rate,
            "is_valid": self.is_valid,
            "errors": [
                {
                    "rule_name": error.rule_name,
                    "message": error.message,
                    "details": error.details,
                    "timestamp": error.timestamp.isoformat()
                }
                for error in self.errors
            ],
            "warnings": [
                {
                    "rule_name": warning.rule_name,
                    "message": warning.message,
                    "details": warning.details,
                    "timestamp": warning.timestamp.isoformat()
                }
                for warning in self.warnings_list
            ],
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None
        }


class BaseValidator(ABC):
    """Abstract base class for all validators"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"validator.{name}")
    
    @abstractmethod
    async def validate(self, data: Any, config: Dict[str, Any] = None) -> ValidationReport:
        """Validate data according to rules"""
        pass


class SchemaValidator(BaseValidator):
    """Validates data against predefined schemas"""
    
    def __init__(self):
        super().__init__("schema")
        self.schemas = self._load_schemas()
    
    def _load_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Load validation schemas for different data sources"""
        return {
            "fred_data": {
                "required_fields": ["date", "value", "series_id"],
                "field_types": {
                    "date": "datetime",
                    "value": "float",
                    "series_id": "string"
                },
                "constraints": {
                    "date": {"min_date": "1990-01-01"},
                    "value": {"allow_null": True},
                    "series_id": {"pattern": r"^[A-Z0-9]+$"}
                }
            },
            "bea_data": {
                "required_fields": ["date", "value", "line_code", "table_name"],
                "field_types": {
                    "date": "datetime",
                    "value": "float",
                    "line_code": "int",
                    "table_name": "string"
                }
            },
            "trade_data": {
                "required_fields": ["date", "commodity_code", "country", "export_value", "import_value"],
                "field_types": {
                    "date": "datetime",
                    "commodity_code": "string",
                    "country": "string",
                    "export_value": "float",
                    "import_value": "float"
                }
            },
            "disruption_data": {
                "required_fields": ["date", "event_type", "severity", "location"],
                "field_types": {
                    "date": "datetime",
                    "event_type": "string",
                    "severity": "string",
                    "location": "string"
                }
            }
        }
    
    async def validate(self, data: Union[pd.DataFrame, List[Dict]], config: Dict[str, Any] = None) -> ValidationReport:
        """Validate data against schema"""
        report = ValidationReport(source_name=config.get("source_name", "unknown"))
        schema_name = config.get("schema_name", "generic")
        
        try:
            # Convert to DataFrame if needed
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, pd.DataFrame):
                df = data
            else:
                report.add_result(ValidationResult(
                    is_valid=False,
                    rule_name="data_format",
                    message=f"Invalid data format: {type(data)}",
                    severity="error"
                ))
                report.finalize()
                return report
            
            # Get schema
            schema = self.schemas.get(schema_name, {})
            
            if not schema:
                report.add_result(ValidationResult(
                    is_valid=False,
                    rule_name="schema_missing",
                    message=f"No schema found for: {schema_name}",
                    severity="error"
                ))
                report.finalize()
                return report
            
            # Validate required fields
            await self._validate_required_fields(df, schema, report)
            
            # Validate field types
            await self._validate_field_types(df, schema, report)
            
            # Validate constraints
            await self._validate_constraints(df, schema, report)
            
            # Check for empty dataset
            if len(df) == 0:
                report.add_result(ValidationResult(
                    is_valid=False,
                    rule_name="empty_dataset",
                    message="Dataset is empty",
                    severity="warning"
                ))
            else:
                report.add_result(ValidationResult(
                    is_valid=True,
                    rule_name="dataset_size",
                    message=f"Dataset contains {len(df)} records",
                    severity="info"
                ))
            
        except Exception as e:
            self.logger.error(f"Schema validation error: {str(e)}")
            report.add_result(ValidationResult(
                is_valid=False,
                rule_name="validation_error",
                message=f"Validation failed: {str(e)}",
                severity="error"
            ))
        
        report.finalize()
        return report
    
    async def _validate_required_fields(self, df: pd.DataFrame, schema: Dict, report: ValidationReport):
        """Validate required fields are present"""
        required_fields = schema.get("required_fields", [])
        missing_fields = [field for field in required_fields if field not in df.columns]
        
        if missing_fields:
            report.add_result(ValidationResult(
                is_valid=False,
                rule_name="missing_required_fields",
                message=f"Missing required fields: {missing_fields}",
                severity="error",
                details={"missing_fields": missing_fields}
            ))
        else:
            report.add_result(ValidationResult(
                is_valid=True,
                rule_name="required_fields_present",
                message="All required fields are present",
                severity="info"
            ))
    
    async def _validate_field_types(self, df: pd.DataFrame, schema: Dict, report: ValidationReport):
        """Validate field data types"""
        field_types = schema.get("field_types", {})
        
        for field, expected_type in field_types.items():
            if field not in df.columns:
                continue
            
            try:
                if expected_type == "datetime":
                    pd.to_datetime(df[field], errors='raise')
                elif expected_type == "float":
                    pd.to_numeric(df[field], errors='raise')
                elif expected_type == "int":
                    pd.to_numeric(df[field], errors='raise', downcast='integer')
                elif expected_type == "string":
                    df[field].astype(str)
                
                report.add_result(ValidationResult(
                    is_valid=True,
                    rule_name=f"field_type_{field}",
                    message=f"Field {field} has correct type: {expected_type}",
                    severity="info"
                ))
                
            except Exception as e:
                report.add_result(ValidationResult(
                    is_valid=False,
                    rule_name=f"field_type_{field}",
                    message=f"Field {field} type validation failed: {str(e)}",
                    severity="error",
                    details={"field": field, "expected_type": expected_type}
                ))
    
    async def _validate_constraints(self, df: pd.DataFrame, schema: Dict, report: ValidationReport):
        """Validate field constraints"""
        constraints = schema.get("constraints", {})
        
        for field, constraint_config in constraints.items():
            if field not in df.columns:
                continue
            
            # Null value constraints
            if not constraint_config.get("allow_null", True):
                null_count = df[field].isnull().sum()
                if null_count > 0:
                    report.add_result(ValidationResult(
                        is_valid=False,
                        rule_name=f"null_constraint_{field}",
                        message=f"Field {field} contains {null_count} null values",
                        severity="error",
                        details={"field": field, "null_count": int(null_count)}
                    ))
            
            # Pattern constraints
            if "pattern" in constraint_config:
                pattern = constraint_config["pattern"]
                non_matching = df[~df[field].astype(str).str.match(pattern, na=False)]
                if len(non_matching) > 0:
                    report.add_result(ValidationResult(
                        is_valid=False,
                        rule_name=f"pattern_constraint_{field}",
                        message=f"Field {field} has {len(non_matching)} values not matching pattern",
                        severity="error",
                        details={"field": field, "pattern": pattern, "violations": int(len(non_matching))}
                    ))
            
            # Date range constraints
            if "min_date" in constraint_config:
                try:
                    min_date = pd.to_datetime(constraint_config["min_date"])
                    date_series = pd.to_datetime(df[field])
                    violations = date_series < min_date
                    violation_count = violations.sum()
                    
                    if violation_count > 0:
                        report.add_result(ValidationResult(
                            is_valid=False,
                            rule_name=f"min_date_constraint_{field}",
                            message=f"Field {field} has {violation_count} dates before minimum",
                            severity="error",
                            details={"field": field, "min_date": str(min_date), "violations": int(violation_count)}
                        ))
                except Exception:
                    pass  # Skip if date conversion fails


class QualityValidator(BaseValidator):
    """Validates data quality metrics"""
    
    def __init__(self):
        super().__init__("quality")
    
    async def validate(self, data: pd.DataFrame, config: Dict[str, Any] = None) -> ValidationReport:
        """Validate data quality"""
        report = ValidationReport(source_name=config.get("source_name", "unknown"))
        
        try:
            # Data completeness checks
            await self._check_completeness(data, report, config)
            
            # Data freshness checks
            await self._check_freshness(data, report, config)
            
            # Data consistency checks
            await self._check_consistency(data, report, config)
            
            # Statistical quality checks
            await self._check_statistical_quality(data, report, config)
            
        except Exception as e:
            self.logger.error(f"Quality validation error: {str(e)}")
            report.add_result(ValidationResult(
                is_valid=False,
                rule_name="quality_validation_error",
                message=f"Quality validation failed: {str(e)}",
                severity="error"
            ))
        
        report.finalize()
        return report
    
    async def _check_completeness(self, df: pd.DataFrame, report: ValidationReport, config: Dict):
        """Check data completeness"""
        if len(df) == 0:
            report.add_result(ValidationResult(
                is_valid=False,
                rule_name="empty_dataset",
                message="Dataset is empty",
                severity="error"
            ))
            return
        
        # Calculate completeness for each column
        for column in df.columns:
            completeness = (1 - df[column].isnull().sum() / len(df)) * 100
            threshold = config.get("completeness_threshold", 80)
            
            if completeness < threshold:
                report.add_result(ValidationResult(
                    is_valid=False,
                    rule_name=f"completeness_{column}",
                    message=f"Column {column} completeness ({completeness:.1f}%) below threshold ({threshold}%)",
                    severity="warning" if completeness > threshold * 0.7 else "error",
                    details={"column": column, "completeness": completeness, "threshold": threshold}
                ))
            else:
                report.add_result(ValidationResult(
                    is_valid=True,
                    rule_name=f"completeness_{column}",
                    message=f"Column {column} completeness is acceptable ({completeness:.1f}%)",
                    severity="info"
                ))
    
    async def _check_freshness(self, df: pd.DataFrame, report: ValidationReport, config: Dict):
        """Check data freshness"""
        date_column = config.get("date_column", "date")
        max_age_days = config.get("max_age_days", 7)
        
        if date_column not in df.columns:
            report.add_result(ValidationResult(
                is_valid=False,
                rule_name="freshness_no_date_column",
                message=f"Date column '{date_column}' not found",
                severity="warning"
            ))
            return
        
        try:
            date_series = pd.to_datetime(df[date_column])
            latest_date = date_series.max()
            days_old = (datetime.now() - latest_date).days
            
            if days_old > max_age_days:
                report.add_result(ValidationResult(
                    is_valid=False,
                    rule_name="data_freshness",
                    message=f"Data is {days_old} days old, exceeds threshold of {max_age_days} days",
                    severity="warning" if days_old < max_age_days * 2 else "error",
                    details={"latest_date": str(latest_date), "days_old": days_old, "threshold": max_age_days}
                ))
            else:
                report.add_result(ValidationResult(
                    is_valid=True,
                    rule_name="data_freshness",
                    message=f"Data is fresh ({days_old} days old)",
                    severity="info"
                ))
                
        except Exception as e:
            report.add_result(ValidationResult(
                is_valid=False,
                rule_name="freshness_check_error",
                message=f"Error checking freshness: {str(e)}",
                severity="warning"
            ))
    
    async def _check_consistency(self, df: pd.DataFrame, report: ValidationReport, config: Dict):
        """Check data consistency"""
        # Check for duplicate records
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            report.add_result(ValidationResult(
                is_valid=False,
                rule_name="duplicate_records",
                message=f"Found {duplicate_count} duplicate records",
                severity="warning",
                details={"duplicate_count": int(duplicate_count)}
            ))
        else:
            report.add_result(ValidationResult(
                is_valid=True,
                rule_name="no_duplicates",
                message="No duplicate records found",
                severity="info"
            ))
    
    async def _check_statistical_quality(self, df: pd.DataFrame, report: ValidationReport, config: Dict):
        """Check statistical data quality"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            series = df[column].dropna()
            
            if len(series) == 0:
                continue
            
            # Check for outliers using IQR method
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = series[(series < lower_bound) | (series > upper_bound)]
            outlier_percentage = (len(outliers) / len(series)) * 100
            
            if outlier_percentage > config.get("outlier_threshold", 5):
                report.add_result(ValidationResult(
                    is_valid=False,
                    rule_name=f"outliers_{column}",
                    message=f"Column {column} has {outlier_percentage:.1f}% outliers",
                    severity="warning",
                    details={"column": column, "outlier_percentage": outlier_percentage}
                ))


class DataValidator:
    """Main data validator that orchestrates all validation types"""
    
    def __init__(self):
        self.schema_validator = SchemaValidator()
        self.quality_validator = QualityValidator()
        self.logger = logging.getLogger("data_validator")
    
    async def validate_data_source(self, data: Any, source_config: Dict[str, Any]) -> Dict[str, ValidationReport]:
        """Validate data from a specific source"""
        results = {}
        
        try:
            # Schema validation
            if source_config.get("validate_schema", True):
                schema_report = await self.schema_validator.validate(data, source_config)
                results["schema"] = schema_report
            
            # Quality validation
            if source_config.get("validate_quality", True) and isinstance(data, pd.DataFrame):
                quality_report = await self.quality_validator.validate(data, source_config)
                results["quality"] = quality_report
            
        except Exception as e:
            self.logger.error(f"Validation error for {source_config.get('source_name', 'unknown')}: {str(e)}")
            error_report = ValidationReport(source_name=source_config.get("source_name", "unknown"))
            error_report.add_result(ValidationResult(
                is_valid=False,
                rule_name="validation_exception",
                message=f"Validation failed with exception: {str(e)}",
                severity="error"
            ))
            error_report.finalize()
            results["error"] = error_report
        
        return results
    
    async def validate_multiple_sources(self, sources_data: Dict[str, Any], sources_config: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, ValidationReport]]:
        """Validate data from multiple sources"""
        all_results = {}
        
        for source_name, data in sources_data.items():
            source_config = sources_config.get(source_name, {})
            source_config["source_name"] = source_name
            
            source_results = await self.validate_data_source(data, source_config)
            all_results[source_name] = source_results
        
        return all_results
    
    def generate_summary_report(self, validation_results: Dict[str, Dict[str, ValidationReport]]) -> Dict[str, Any]:
        """Generate a summary report of all validations"""
        summary = {
            "total_sources": len(validation_results),
            "sources": {},
            "overall_status": "passed",
            "total_errors": 0,
            "total_warnings": 0,
            "timestamp": datetime.now().isoformat()
        }
        
        for source_name, source_results in validation_results.items():
            source_summary = {
                "validation_types": list(source_results.keys()),
                "status": "passed",
                "errors": 0,
                "warnings": 0
            }
            
            for validation_type, report in source_results.items():
                source_summary["errors"] += len(report.errors)
                source_summary["warnings"] += len(report.warnings_list)
                
                if not report.is_valid:
                    source_summary["status"] = "failed"
                    summary["overall_status"] = "failed"
            
            summary["total_errors"] += source_summary["errors"]
            summary["total_warnings"] += source_summary["warnings"]
            summary["sources"][source_name] = source_summary
        
        return summary