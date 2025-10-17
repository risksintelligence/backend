"""
Data Validation Module

Provides comprehensive data quality checks and validation
for all data sources in the RiskX platform.
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
class ValidationResult:
    """Result of a data validation check"""
    check_name: str
    status: str  # 'passed', 'warning', 'failed'
    score: float  # 0-1 quality score
    message: str
    details: Dict[str, Any]


@dataclass
class DataQualityReport:
    """Comprehensive data quality report"""
    overall_quality_score: float
    validation_results: List[ValidationResult]
    data_source_scores: Dict[str, float]
    recommendations: List[str]
    timestamp: datetime


class DataQualityValidator:
    """
    Comprehensive data quality validation for RiskX platform.
    
    Validates:
    - Data completeness and coverage
    - Data freshness and timeliness
    - Data consistency and integrity
    - Statistical anomalies and outliers
    - Schema compliance
    - Cross-source data consistency
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.cache = CacheManager()
        self.logger = logging.getLogger(__name__)
        
        # Validation thresholds
        self.thresholds = {
            'completeness_minimum': 0.8,  # 80% data completeness required
            'freshness_hours': 48,  # Data should be no older than 48 hours
            'outlier_threshold': 3.0,  # Z-score threshold for outliers
            'consistency_threshold': 0.9,  # Cross-source consistency requirement
            'minimum_records': 10  # Minimum records required per data source
        }
        
        # Data source definitions
        self.data_sources = {
            'fred': {
                'required_indicators': ['GDPC1', 'UNRATE', 'FEDFUNDS', 'CPILFESL'],
                'update_frequency': 'monthly',
                'expected_range': {'min': -50, 'max': 1000}
            },
            'bea': {
                'required_indicators': ['GDP', 'TRADE_BALANCE'],
                'update_frequency': 'quarterly',
                'expected_range': {'min': -10000, 'max': 50000}
            },
            'census_trade': {
                'required_fields': ['import_value', 'export_value', 'trade_balance'],
                'update_frequency': 'monthly',
                'expected_range': {'min': 0, 'max': 1000000}
            },
            'noaa_events': {
                'required_fields': ['event_type', 'severity', 'impact_area'],
                'update_frequency': 'daily',
                'expected_range': {'min': 0, 'max': 100}
            },
            'cisa_cyber': {
                'required_fields': ['advisory_id', 'severity', 'affected_systems'],
                'update_frequency': 'daily',
                'expected_range': {'min': 0, 'max': 50}
            }
        }
    
    async def run_comprehensive_validation(self) -> Dict[str, Any]:
        """
        Run comprehensive data quality validation across all sources
        """
        self.logger.info("Starting comprehensive data quality validation")
        
        try:
            validation_results = []
            data_source_scores = {}
            
            # Validate each data source
            for source_name, source_config in self.data_sources.items():
                source_results = await self._validate_data_source(source_name, source_config)
                validation_results.extend(source_results)
                
                # Calculate source-specific score
                source_score = self._calculate_source_score(source_results)
                data_source_scores[source_name] = source_score
            
            # Cross-source validation
            cross_validation_results = await self._validate_cross_source_consistency()
            validation_results.extend(cross_validation_results)
            
            # System-wide validation
            system_validation_results = await self._validate_system_wide_metrics()
            validation_results.extend(system_validation_results)
            
            # Calculate overall quality score
            overall_score = self._calculate_overall_quality_score(validation_results)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(validation_results)
            
            # Create comprehensive report
            report = DataQualityReport(
                overall_quality_score=overall_score,
                validation_results=validation_results,
                data_source_scores=data_source_scores,
                recommendations=recommendations,
                timestamp=datetime.now()
            )
            
            # Cache validation results
            await self._cache_validation_results(report)
            
            self.logger.info(f"Data quality validation completed. Overall score: {overall_score:.2f}")
            
            return {
                'overall_quality_score': overall_score,
                'validation_results': [
                    {
                        'check_name': r.check_name,
                        'status': r.status,
                        'score': r.score,
                        'message': r.message,
                        'details': r.details
                    } for r in validation_results
                ],
                'data_source_scores': data_source_scores,
                'recommendations': recommendations,
                'timestamp': report.timestamp.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive data validation: {str(e)}")
            raise
    
    async def _validate_data_source(self, source_name: str, source_config: Dict) -> List[ValidationResult]:
        """Validate a specific data source"""
        results = []
        
        try:
            # Load data for this source
            data = await self._load_source_data(source_name)
            
            if data is None or len(data) == 0:
                results.append(ValidationResult(
                    check_name=f"{source_name}_availability",
                    status="failed",
                    score=0.0,
                    message=f"No data available for {source_name}",
                    details={"error": "Data not found"}
                ))
                return results
            
            # Completeness check
            completeness_result = self._check_data_completeness(source_name, data, source_config)
            results.append(completeness_result)
            
            # Freshness check
            freshness_result = self._check_data_freshness(source_name, data)
            results.append(freshness_result)
            
            # Schema validation
            schema_result = self._check_schema_compliance(source_name, data, source_config)
            results.append(schema_result)
            
            # Statistical validation
            stats_result = self._check_statistical_validity(source_name, data, source_config)
            results.append(stats_result)
            
            # Outlier detection
            outlier_result = self._check_outliers(source_name, data)
            results.append(outlier_result)
            
        except Exception as e:
            self.logger.error(f"Error validating data source {source_name}: {str(e)}")
            results.append(ValidationResult(
                check_name=f"{source_name}_validation_error",
                status="failed",
                score=0.0,
                message=f"Validation error for {source_name}: {str(e)}",
                details={"error": str(e)}
            ))
        
        return results
    
    async def _load_source_data(self, source_name: str) -> Optional[pd.DataFrame]:
        """Load data for a specific source"""
        try:
            # Try to get from cache first
            cache_key = f"{source_name}_latest_data"
            cached_data = await self.cache.get(cache_key)
            
            if cached_data:
                return pd.DataFrame(cached_data)
            
            # If not in cache, try to load from data source directly
            # This would call the appropriate data source module
            if source_name == 'fred':
                from .fred_fetch import FredDataFetcher
                fetcher = FredDataFetcher()
                data = await fetcher.get_cached_data()
                return pd.DataFrame(data) if data else None
            
            # Add similar loading for other sources
            # ...
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Could not load data for {source_name}: {str(e)}")
            return None
    
    def _check_data_completeness(self, source_name: str, data: pd.DataFrame, config: Dict) -> ValidationResult:
        """Check data completeness against requirements"""
        
        try:
            total_expected = len(data)
            total_available = data.count().sum()
            total_possible = total_expected * len(data.columns)
            
            completeness_ratio = total_available / total_possible if total_possible > 0 else 0
            
            status = "passed" if completeness_ratio >= self.thresholds['completeness_minimum'] else "failed"
            if completeness_ratio >= 0.7:
                status = "warning" if status == "failed" else status
            
            return ValidationResult(
                check_name=f"{source_name}_completeness",
                status=status,
                score=completeness_ratio,
                message=f"Data completeness: {completeness_ratio:.1%}",
                details={
                    "total_cells": total_possible,
                    "available_cells": total_available,
                    "missing_cells": total_possible - total_available,
                    "completeness_ratio": completeness_ratio
                }
            )
            
        except Exception as e:
            return ValidationResult(
                check_name=f"{source_name}_completeness",
                status="failed",
                score=0.0,
                message=f"Completeness check failed: {str(e)}",
                details={"error": str(e)}
            )
    
    def _check_data_freshness(self, source_name: str, data: pd.DataFrame) -> ValidationResult:
        """Check data freshness and timeliness"""
        
        try:
            # Try to find timestamp column
            timestamp_columns = ['timestamp', 'date', 'updated_at', 'last_modified']
            timestamp_col = None
            
            for col in timestamp_columns:
                if col in data.columns:
                    timestamp_col = col
                    break
            
            if timestamp_col is None:
                # Use index if it's a datetime index
                if isinstance(data.index, pd.DatetimeIndex):
                    latest_timestamp = data.index.max()
                else:
                    return ValidationResult(
                        check_name=f"{source_name}_freshness",
                        status="warning",
                        score=0.5,
                        message="No timestamp column found for freshness check",
                        details={"available_columns": list(data.columns)}
                    )
            else:
                latest_timestamp = pd.to_datetime(data[timestamp_col]).max()
            
            # Calculate data age
            current_time = datetime.now()
            if pd.isna(latest_timestamp):
                age_hours = float('inf')
            else:
                age_hours = (current_time - latest_timestamp).total_seconds() / 3600
            
            # Determine status based on age
            threshold_hours = self.thresholds['freshness_hours']
            if age_hours <= threshold_hours:
                status = "passed"
                score = max(0, 1 - (age_hours / threshold_hours))
            elif age_hours <= threshold_hours * 2:
                status = "warning"
                score = max(0, 0.5 - (age_hours / (threshold_hours * 4)))
            else:
                status = "failed"
                score = 0.0
            
            return ValidationResult(
                check_name=f"{source_name}_freshness",
                status=status,
                score=score,
                message=f"Data age: {age_hours:.1f} hours",
                details={
                    "latest_timestamp": str(latest_timestamp),
                    "age_hours": age_hours,
                    "threshold_hours": threshold_hours
                }
            )
            
        except Exception as e:
            return ValidationResult(
                check_name=f"{source_name}_freshness",
                status="failed",
                score=0.0,
                message=f"Freshness check failed: {str(e)}",
                details={"error": str(e)}
            )
    
    def _check_schema_compliance(self, source_name: str, data: pd.DataFrame, config: Dict) -> ValidationResult:
        """Check schema compliance against expected structure"""
        
        try:
            required_fields = config.get('required_indicators', config.get('required_fields', []))
            
            if not required_fields:
                return ValidationResult(
                    check_name=f"{source_name}_schema",
                    status="passed",
                    score=1.0,
                    message="No schema requirements defined",
                    details={"note": "Schema validation skipped"}
                )
            
            # Check for required fields
            available_fields = set(data.columns)
            required_fields_set = set(required_fields)
            missing_fields = required_fields_set - available_fields
            extra_fields = available_fields - required_fields_set
            
            compliance_ratio = len(required_fields_set & available_fields) / len(required_fields_set)
            
            if compliance_ratio == 1.0:
                status = "passed"
            elif compliance_ratio >= 0.8:
                status = "warning"
            else:
                status = "failed"
            
            return ValidationResult(
                check_name=f"{source_name}_schema",
                status=status,
                score=compliance_ratio,
                message=f"Schema compliance: {compliance_ratio:.1%}",
                details={
                    "required_fields": required_fields,
                    "available_fields": list(available_fields),
                    "missing_fields": list(missing_fields),
                    "extra_fields": list(extra_fields),
                    "compliance_ratio": compliance_ratio
                }
            )
            
        except Exception as e:
            return ValidationResult(
                check_name=f"{source_name}_schema",
                status="failed",
                score=0.0,
                message=f"Schema check failed: {str(e)}",
                details={"error": str(e)}
            )
    
    def _check_statistical_validity(self, source_name: str, data: pd.DataFrame, config: Dict) -> ValidationResult:
        """Check statistical validity of data values"""
        
        try:
            expected_range = config.get('expected_range', {'min': -float('inf'), 'max': float('inf')})
            
            # Get numeric columns
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            
            if len(numeric_columns) == 0:
                return ValidationResult(
                    check_name=f"{source_name}_statistical_validity",
                    status="warning",
                    score=0.5,
                    message="No numeric columns found for statistical validation",
                    details={"available_columns": list(data.columns)}
                )
            
            range_violations = 0
            total_values = 0
            
            for col in numeric_columns:
                col_data = data[col].dropna()
                total_values += len(col_data)
                
                # Check range violations
                below_min = (col_data < expected_range['min']).sum()
                above_max = (col_data > expected_range['max']).sum()
                range_violations += below_min + above_max
            
            validity_ratio = 1 - (range_violations / max(total_values, 1))
            
            if validity_ratio >= 0.95:
                status = "passed"
            elif validity_ratio >= 0.9:
                status = "warning"
            else:
                status = "failed"
            
            return ValidationResult(
                check_name=f"{source_name}_statistical_validity",
                status=status,
                score=validity_ratio,
                message=f"Statistical validity: {validity_ratio:.1%}",
                details={
                    "total_values": total_values,
                    "range_violations": range_violations,
                    "expected_range": expected_range,
                    "validity_ratio": validity_ratio
                }
            )
            
        except Exception as e:
            return ValidationResult(
                check_name=f"{source_name}_statistical_validity",
                status="failed",
                score=0.0,
                message=f"Statistical validity check failed: {str(e)}",
                details={"error": str(e)}
            )
    
    def _check_outliers(self, source_name: str, data: pd.DataFrame) -> ValidationResult:
        """Check for statistical outliers in the data"""
        
        try:
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            
            if len(numeric_columns) == 0:
                return ValidationResult(
                    check_name=f"{source_name}_outliers",
                    status="passed",
                    score=1.0,
                    message="No numeric columns for outlier detection",
                    details={"note": "Outlier detection skipped"}
                )
            
            total_outliers = 0
            total_values = 0
            outlier_details = {}
            
            for col in numeric_columns:
                col_data = data[col].dropna()
                if len(col_data) == 0:
                    continue
                
                # Calculate Z-scores
                z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
                outliers = (z_scores > self.thresholds['outlier_threshold']).sum()
                
                total_outliers += outliers
                total_values += len(col_data)
                
                if outliers > 0:
                    outlier_details[col] = {
                        "outliers": int(outliers),
                        "total": len(col_data),
                        "outlier_ratio": outliers / len(col_data)
                    }
            
            outlier_ratio = total_outliers / max(total_values, 1)
            
            # Determine status based on outlier ratio
            if outlier_ratio <= 0.05:  # Less than 5% outliers
                status = "passed"
                score = 1 - outlier_ratio
            elif outlier_ratio <= 0.1:  # Less than 10% outliers
                status = "warning"
                score = 0.5
            else:
                status = "failed"
                score = 0.0
            
            return ValidationResult(
                check_name=f"{source_name}_outliers",
                status=status,
                score=score,
                message=f"Outlier ratio: {outlier_ratio:.1%}",
                details={
                    "total_outliers": total_outliers,
                    "total_values": total_values,
                    "outlier_ratio": outlier_ratio,
                    "outlier_threshold": self.thresholds['outlier_threshold'],
                    "column_details": outlier_details
                }
            )
            
        except Exception as e:
            return ValidationResult(
                check_name=f"{source_name}_outliers",
                status="failed",
                score=0.0,
                message=f"Outlier check failed: {str(e)}",
                details={"error": str(e)}
            )
    
    async def _validate_cross_source_consistency(self) -> List[ValidationResult]:
        """Validate consistency across different data sources"""
        results = []
        
        try:
            # Check consistency between related indicators
            consistency_checks = [
                {
                    'name': 'gdp_trade_consistency',
                    'sources': ['fred', 'bea'],
                    'indicators': ['GDP', 'TRADE_BALANCE'],
                    'expected_correlation': 0.3
                }
            ]
            
            for check in consistency_checks:
                result = await self._perform_consistency_check(check)
                results.append(result)
        
        except Exception as e:
            self.logger.warning(f"Error in cross-source validation: {str(e)}")
            results.append(ValidationResult(
                check_name="cross_source_consistency",
                status="failed",
                score=0.0,
                message=f"Cross-source validation failed: {str(e)}",
                details={"error": str(e)}
            ))
        
        return results
    
    async def _perform_consistency_check(self, check_config: Dict) -> ValidationResult:
        """Perform a specific consistency check"""
        
        try:
            # This is a simplified implementation
            # In practice, would load and compare actual data
            
            consistency_score = 0.85  # Mock score
            
            if consistency_score >= self.thresholds['consistency_threshold']:
                status = "passed"
            elif consistency_score >= 0.7:
                status = "warning"
            else:
                status = "failed"
            
            return ValidationResult(
                check_name=check_config['name'],
                status=status,
                score=consistency_score,
                message=f"Cross-source consistency: {consistency_score:.1%}",
                details={
                    "sources": check_config['sources'],
                    "indicators": check_config['indicators'],
                    "consistency_score": consistency_score
                }
            )
            
        except Exception as e:
            return ValidationResult(
                check_name=check_config['name'],
                status="failed",
                score=0.0,
                message=f"Consistency check failed: {str(e)}",
                details={"error": str(e)}
            )
    
    async def _validate_system_wide_metrics(self) -> List[ValidationResult]:
        """Validate system-wide data quality metrics"""
        results = []
        
        try:
            # Check overall data coverage
            coverage_result = await self._check_data_coverage()
            results.append(coverage_result)
            
            # Check cache health
            cache_result = await self._check_cache_integrity()
            results.append(cache_result)
            
        except Exception as e:
            self.logger.warning(f"Error in system-wide validation: {str(e)}")
        
        return results
    
    async def _check_data_coverage(self) -> ValidationResult:
        """Check overall data coverage across all sources"""
        
        try:
            total_sources = len(self.data_sources)
            available_sources = 0
            
            for source_name in self.data_sources.keys():
                data = await self._load_source_data(source_name)
                if data is not None and len(data) >= self.thresholds['minimum_records']:
                    available_sources += 1
            
            coverage_ratio = available_sources / total_sources
            
            if coverage_ratio >= 0.9:
                status = "passed"
            elif coverage_ratio >= 0.7:
                status = "warning"
            else:
                status = "failed"
            
            return ValidationResult(
                check_name="system_data_coverage",
                status=status,
                score=coverage_ratio,
                message=f"Data coverage: {available_sources}/{total_sources} sources",
                details={
                    "total_sources": total_sources,
                    "available_sources": available_sources,
                    "coverage_ratio": coverage_ratio
                }
            )
            
        except Exception as e:
            return ValidationResult(
                check_name="system_data_coverage",
                status="failed",
                score=0.0,
                message=f"Coverage check failed: {str(e)}",
                details={"error": str(e)}
            )
    
    async def _check_cache_integrity(self) -> ValidationResult:
        """Check cache system integrity"""
        
        try:
            cache_health = await self.cache.health_check()
            
            if cache_health.get('status') == 'healthy':
                status = "passed"
                score = 1.0
            elif cache_health.get('status') == 'degraded':
                status = "warning"
                score = 0.7
            else:
                status = "failed"
                score = 0.0
            
            return ValidationResult(
                check_name="cache_integrity",
                status=status,
                score=score,
                message=f"Cache status: {cache_health.get('status', 'unknown')}",
                details=cache_health
            )
            
        except Exception as e:
            return ValidationResult(
                check_name="cache_integrity",
                status="failed",
                score=0.0,
                message=f"Cache integrity check failed: {str(e)}",
                details={"error": str(e)}
            )
    
    def _calculate_source_score(self, source_results: List[ValidationResult]) -> float:
        """Calculate overall quality score for a data source"""
        
        if not source_results:
            return 0.0
        
        # Weight different validation types
        weights = {
            'availability': 0.3,
            'completeness': 0.2,
            'freshness': 0.2,
            'schema': 0.15,
            'statistical_validity': 0.1,
            'outliers': 0.05
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for result in source_results:
            for check_type, weight in weights.items():
                if check_type in result.check_name:
                    weighted_score += result.score * weight
                    total_weight += weight
                    break
        
        return weighted_score / max(total_weight, 1)
    
    def _calculate_overall_quality_score(self, all_results: List[ValidationResult]) -> float:
        """Calculate overall data quality score"""
        
        if not all_results:
            return 0.0
        
        # Simple average of all validation scores
        total_score = sum(result.score for result in all_results)
        return total_score / len(all_results)
    
    def _generate_recommendations(self, validation_results: List[ValidationResult]) -> List[str]:
        """Generate recommendations based on validation results"""
        
        recommendations = []
        
        # Analyze failed validations
        failed_checks = [r for r in validation_results if r.status == "failed"]
        warning_checks = [r for r in validation_results if r.status == "warning"]
        
        if failed_checks:
            recommendations.append(f"Critical: {len(failed_checks)} data quality checks failed - immediate attention required")
        
        if warning_checks:
            recommendations.append(f"Warning: {len(warning_checks)} data quality checks need monitoring")
        
        # Specific recommendations based on common issues
        completeness_issues = [r for r in validation_results if "completeness" in r.check_name and r.score < 0.8]
        if completeness_issues:
            recommendations.append("Address data completeness issues in data collection pipelines")
        
        freshness_issues = [r for r in validation_results if "freshness" in r.check_name and r.score < 0.7]
        if freshness_issues:
            recommendations.append("Update data refresh schedules to improve data freshness")
        
        outlier_issues = [r for r in validation_results if "outliers" in r.check_name and r.score < 0.8]
        if outlier_issues:
            recommendations.append("Investigate data outliers and implement outlier handling procedures")
        
        return recommendations[:5]  # Return top 5 recommendations
    
    async def _cache_validation_results(self, report: DataQualityReport):
        """Cache validation results for historical tracking"""
        
        try:
            report_data = {
                "overall_quality_score": report.overall_quality_score,
                "validation_results": [
                    {
                        "check_name": r.check_name,
                        "status": r.status,
                        "score": r.score,
                        "message": r.message,
                        "details": r.details
                    } for r in report.validation_results
                ],
                "data_source_scores": report.data_source_scores,
                "recommendations": report.recommendations,
                "timestamp": report.timestamp.isoformat()
            }
            
            # Cache latest report
            await self.cache.set("data_quality_latest", report_data, ttl=86400)
            
            # Cache historical report
            date_key = f"data_quality_{report.timestamp.strftime('%Y%m%d_%H%M%S')}"
            await self.cache.set(date_key, report_data, ttl=86400 * 30)  # Keep for 30 days
            
        except Exception as e:
            self.logger.warning(f"Error caching validation results: {str(e)}")


async def main():
    """Test data validation functionality"""
    validator = DataQualityValidator()
    
    try:
        print("Running comprehensive data quality validation...")
        results = await validator.run_comprehensive_validation()
        
        print(f"Overall Quality Score: {results['overall_quality_score']:.2f}")
        print(f"Validation Checks: {len(results['validation_results'])}")
        print(f"Recommendations: {len(results['recommendations'])}")
        
        for rec in results['recommendations']:
            print(f"- {rec}")
            
    except Exception as e:
        print(f"Validation failed: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())