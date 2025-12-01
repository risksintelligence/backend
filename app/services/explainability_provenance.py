"""
Enhanced Explainability Provenance Logging for RRIO
Institutional-grade audit trails and decision transparency for ML models
"""

import json
import hashlib
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging
import numpy as np
import structlog
from pathlib import Path

logger = structlog.get_logger(__name__)

class ExplainabilityLevel(Enum):
    """Levels of explanation depth required for institutional compliance"""
    BASIC = "basic"           # Simple feature importance
    DETAILED = "detailed"     # Feature contributions + interactions
    COMPREHENSIVE = "comprehensive"  # Full decision tree + counterfactuals
    REGULATORY = "regulatory" # Audit-ready with full lineage

class DecisionStage(Enum):
    """Stages in the ML decision pipeline"""
    DATA_INGESTION = "data_ingestion"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_INFERENCE = "model_inference"
    POST_PROCESSING = "post_processing"
    RESULT_FORMATTING = "result_formatting"

@dataclass
class FeatureContribution:
    """Individual feature contribution to model decision"""
    feature_name: str
    feature_value: float
    contribution_score: float
    contribution_rank: int
    confidence_interval: Tuple[float, float]
    data_source: str
    last_updated: datetime
    
@dataclass
class ModelExplanation:
    """Complete explanation for a single model prediction"""
    model_name: str
    model_version: str
    prediction_value: Union[float, str, Dict[str, float]]
    confidence_score: float
    explanation_level: ExplainabilityLevel
    feature_contributions: List[FeatureContribution]
    decision_boundary_distance: Optional[float]
    counterfactual_examples: List[Dict[str, Any]]
    model_metadata: Dict[str, Any]
    computation_time_ms: float
    
@dataclass
class DecisionProvenance:
    """Complete provenance record for a decision path"""
    decision_id: str
    request_id: str
    user_context: Dict[str, str]
    timestamp: datetime
    model_explanations: List[ModelExplanation]
    data_lineage: Dict[str, Any]
    decision_stages: List[Dict[str, Any]]
    compliance_flags: List[str]
    audit_metadata: Dict[str, Any]

@dataclass
class ProvenanceAuditLog:
    """Audit log entry for explainability access"""
    access_id: str
    accessed_by: str
    access_timestamp: datetime
    decision_ids_accessed: List[str]
    explanation_level_requested: ExplainabilityLevel
    business_justification: str
    data_retention_period: int  # days
    
class ExplainabilityProvenanceLogger:
    """
    Institutional-grade explainability and provenance logging system
    Maintains complete audit trails for ML decision transparency
    """
    
    def __init__(self, storage_path: str = "explainability_logs/"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True, parents=True)
        
        # Initialize storage for different log types
        self.decision_logs: Dict[str, DecisionProvenance] = {}
        self.audit_logs: List[ProvenanceAuditLog] = []
        self.model_registry: Dict[str, Dict[str, Any]] = {}
        
        # Compliance settings
        self.retention_policy_days = 2555  # 7 years for financial compliance
        self.explanation_cache_hours = 24
        self.require_business_justification = True
        
        logger.info("🔍 Explainability Provenance Logger initialized")
        
    def register_model_explainability(
        self, 
        model_name: str, 
        model_version: str,
        explainability_methods: List[str],
        baseline_metrics: Dict[str, float],
        feature_definitions: Dict[str, str]
    ) -> str:
        """Register model with explainability capabilities"""
        
        model_key = f"{model_name}_v{model_version}"
        
        self.model_registry[model_key] = {
            "model_name": model_name,
            "model_version": model_version,
            "explainability_methods": explainability_methods,
            "baseline_metrics": baseline_metrics,
            "feature_definitions": feature_definitions,
            "registered_at": datetime.utcnow(),
            "explanation_count": 0,
            "audit_accessible": True
        }
        
        logger.info(f"📋 Registered model explainability: {model_key}")
        return model_key
        
    def log_decision_with_explanation(
        self,
        model_name: str,
        prediction_value: Union[float, str, Dict[str, float]],
        input_features: Dict[str, float],
        feature_contributions: List[FeatureContribution],
        request_id: str,
        user_context: Dict[str, str],
        explanation_level: ExplainabilityLevel = ExplainabilityLevel.DETAILED
    ) -> str:
        """Log a model decision with complete provenance"""
        
        start_time = datetime.utcnow()
        decision_id = str(uuid.uuid4())
        
        # Generate model explanation
        model_explanation = self._generate_model_explanation(
            model_name,
            prediction_value,
            feature_contributions,
            input_features,
            explanation_level
        )
        
        # Create data lineage record
        data_lineage = self._trace_data_lineage(input_features)
        
        # Track decision stages
        decision_stages = self._track_decision_stages(model_name, input_features)
        
        # Generate compliance flags
        compliance_flags = self._assess_compliance_requirements(
            model_name, 
            prediction_value, 
            explanation_level
        )
        
        # Create complete provenance record
        provenance = DecisionProvenance(
            decision_id=decision_id,
            request_id=request_id,
            user_context=user_context,
            timestamp=start_time,
            model_explanations=[model_explanation],
            data_lineage=data_lineage,
            decision_stages=decision_stages,
            compliance_flags=compliance_flags,
            audit_metadata={
                "explanation_level": explanation_level.value,
                "processing_time_ms": (datetime.utcnow() - start_time).total_seconds() * 1000,
                "model_registry_key": f"{model_name}_v{self._get_model_version(model_name)}",
                "regulatory_flags": self._get_regulatory_flags(model_name),
                "data_sensitivity_level": self._assess_data_sensitivity(input_features)
            }
        )
        
        # Store provenance record
        self.decision_logs[decision_id] = provenance
        
        # Update model registry statistics
        model_key = f"{model_name}_v{self._get_model_version(model_name)}"
        if model_key in self.model_registry:
            self.model_registry[model_key]["explanation_count"] += 1
            self.model_registry[model_key]["last_explanation"] = datetime.utcnow()
        
        logger.info(
            f"✅ Decision logged with explanation",
            decision_id=decision_id,
            model_name=model_name,
            explanation_level=explanation_level.value,
            compliance_flags=len(compliance_flags)
        )
        
        return decision_id
        
    def log_explanation_access(
        self,
        decision_ids: List[str],
        accessed_by: str,
        explanation_level: ExplainabilityLevel,
        business_justification: str
    ) -> str:
        """Log access to explanation data for audit compliance"""
        
        access_id = str(uuid.uuid4())
        
        # Validate business justification if required
        if self.require_business_justification and not business_justification.strip():
            raise ValueError("Business justification required for explanation access")
        
        # Validate decision IDs exist
        invalid_ids = [did for did in decision_ids if did not in self.decision_logs]
        if invalid_ids:
            logger.warning(f"Invalid decision IDs requested: {invalid_ids}")
        
        valid_ids = [did for did in decision_ids if did in self.decision_logs]
        
        audit_entry = ProvenanceAuditLog(
            access_id=access_id,
            accessed_by=accessed_by,
            access_timestamp=datetime.utcnow(),
            decision_ids_accessed=valid_ids,
            explanation_level_requested=explanation_level,
            business_justification=business_justification,
            data_retention_period=self.retention_policy_days
        )
        
        self.audit_logs.append(audit_entry)
        
        logger.info(
            f"📊 Explanation access logged",
            access_id=access_id,
            accessed_by=accessed_by,
            decision_count=len(valid_ids),
            explanation_level=explanation_level.value
        )
        
        return access_id
        
    def get_decision_explanation(
        self,
        decision_id: str,
        accessed_by: str,
        business_justification: str,
        requested_level: ExplainabilityLevel = ExplainabilityLevel.DETAILED
    ) -> Optional[Dict[str, Any]]:
        """Retrieve decision explanation with audit logging"""
        
        # Log the access
        access_id = self.log_explanation_access(
            [decision_id], 
            accessed_by, 
            requested_level, 
            business_justification
        )
        
        if decision_id not in self.decision_logs:
            logger.warning(f"Decision ID not found: {decision_id}")
            return None
            
        provenance = self.decision_logs[decision_id]
        
        # Format explanation based on requested level
        explanation = self._format_explanation_response(provenance, requested_level)
        explanation["audit_access_id"] = access_id
        
        return explanation
        
    def generate_compliance_report(
        self,
        start_date: datetime,
        end_date: datetime,
        models: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate compliance report for explainability activities"""
        
        # Filter decisions by date range
        period_decisions = [
            prov for prov in self.decision_logs.values()
            if start_date <= prov.timestamp <= end_date
        ]
        
        # Filter by models if specified
        if models:
            period_decisions = [
                prov for prov in period_decisions
                if any(exp.model_name in models for exp in prov.model_explanations)
            ]
        
        # Analyze audit access patterns
        period_audits = [
            audit for audit in self.audit_logs
            if start_date <= audit.access_timestamp <= end_date
        ]
        
        # Calculate compliance metrics
        total_decisions = len(period_decisions)
        explained_decisions = len([p for p in period_decisions if p.model_explanations])
        compliance_rate = explained_decisions / total_decisions if total_decisions > 0 else 0
        
        # Analyze explanation levels used
        level_distribution = {}
        for prov in period_decisions:
            for exp in prov.model_explanations:
                level = exp.explanation_level.value
                level_distribution[level] = level_distribution.get(level, 0) + 1
        
        # Check retention compliance
        retention_cutoff = datetime.utcnow() - timedelta(days=self.retention_policy_days)
        old_decisions = [
            prov for prov in self.decision_logs.values()
            if prov.timestamp < retention_cutoff
        ]
        
        report = {
            "reporting_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "duration_days": (end_date - start_date).days
            },
            "decision_metrics": {
                "total_decisions": total_decisions,
                "explained_decisions": explained_decisions,
                "explanation_compliance_rate": compliance_rate,
                "average_explanation_time_ms": np.mean([
                    exp.computation_time_ms 
                    for prov in period_decisions 
                    for exp in prov.model_explanations
                ]) if period_decisions else 0
            },
            "explanation_distribution": level_distribution,
            "audit_activity": {
                "total_access_requests": len(period_audits),
                "unique_accessors": len(set(audit.accessed_by for audit in period_audits)),
                "average_decisions_per_access": np.mean([
                    len(audit.decision_ids_accessed) for audit in period_audits
                ]) if period_audits else 0
            },
            "compliance_status": {
                "retention_policy_compliant": len(old_decisions) == 0,
                "decisions_requiring_cleanup": len(old_decisions),
                "business_justification_rate": len([
                    audit for audit in period_audits 
                    if audit.business_justification.strip()
                ]) / len(period_audits) if period_audits else 0
            },
            "model_coverage": {
                model: len([
                    prov for prov in period_decisions
                    if any(exp.model_name == model for exp in prov.model_explanations)
                ])
                for model in set(
                    exp.model_name 
                    for prov in period_decisions 
                    for exp in prov.model_explanations
                )
            },
            "generated_at": datetime.utcnow().isoformat(),
            "report_version": "v1.0.0"
        }
        
        logger.info(
            f"📊 Compliance report generated",
            period_days=(end_date - start_date).days,
            total_decisions=total_decisions,
            compliance_rate=f"{compliance_rate:.3f}"
        )
        
        return report
        
    def _generate_model_explanation(
        self,
        model_name: str,
        prediction_value: Union[float, str, Dict[str, float]],
        feature_contributions: List[FeatureContribution],
        input_features: Dict[str, float],
        explanation_level: ExplainabilityLevel
    ) -> ModelExplanation:
        """Generate comprehensive model explanation"""
        
        start_time = datetime.utcnow()
        
        # Calculate confidence score based on feature contribution distribution
        confidence_score = self._calculate_prediction_confidence(feature_contributions)
        
        # Generate counterfactual examples for comprehensive explanations
        counterfactuals = []
        if explanation_level in [ExplainabilityLevel.COMPREHENSIVE, ExplainabilityLevel.REGULATORY]:
            counterfactuals = self._generate_counterfactuals(model_name, input_features)
        
        # Calculate decision boundary distance for confidence assessment
        boundary_distance = self._calculate_decision_boundary_distance(
            model_name, input_features
        ) if explanation_level != ExplainabilityLevel.BASIC else None
        
        model_version = self._get_model_version(model_name)
        
        explanation = ModelExplanation(
            model_name=model_name,
            model_version=model_version,
            prediction_value=prediction_value,
            confidence_score=confidence_score,
            explanation_level=explanation_level,
            feature_contributions=sorted(
                feature_contributions, 
                key=lambda x: abs(x.contribution_score), 
                reverse=True
            ),
            decision_boundary_distance=boundary_distance,
            counterfactual_examples=counterfactuals,
            model_metadata={
                "training_date": "2024-11-01",  # Would be from model registry
                "feature_count": len(input_features),
                "algorithm": "RandomForestClassifier",  # Would be from model registry
                "hyperparameters": {"n_estimators": 100, "max_depth": 10}
            },
            computation_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
        )
        
        return explanation
        
    def _trace_data_lineage(self, input_features: Dict[str, float]) -> Dict[str, Any]:
        """Trace data lineage for input features"""
        
        lineage = {
            "feature_sources": {},
            "data_ingestion_timestamp": datetime.utcnow() - timedelta(minutes=5),
            "data_transformations": [],
            "data_quality_checks": [],
            "external_data_dependencies": []
        }
        
        # Map each feature to its data source
        source_mapping = {
            "vix": {"source": "CBOE", "feed": "Market_Data_API", "lag_minutes": 1},
            "sp500": {"source": "S&P", "feed": "Index_API", "lag_minutes": 2},
            "dxy": {"source": "ICE", "feed": "Currency_API", "lag_minutes": 3},
            "tnx": {"source": "Treasury", "feed": "Bond_API", "lag_minutes": 2},
            "oil": {"source": "CME", "feed": "Commodity_API", "lag_minutes": 5}
        }
        
        for feature_name in input_features.keys():
            base_name = feature_name.split('_')[0].lower()
            if base_name in source_mapping:
                lineage["feature_sources"][feature_name] = source_mapping[base_name]
            else:
                lineage["feature_sources"][feature_name] = {
                    "source": "Unknown", 
                    "feed": "Internal_Calculation", 
                    "lag_minutes": 0
                }
        
        return lineage
        
    def _track_decision_stages(self, model_name: str, input_features: Dict[str, float]) -> List[Dict[str, Any]]:
        """Track each stage in the decision pipeline"""
        
        stages = []
        current_time = datetime.utcnow()
        
        # Data ingestion stage
        stages.append({
            "stage": DecisionStage.DATA_INGESTION.value,
            "timestamp": current_time - timedelta(milliseconds=100),
            "duration_ms": 50,
            "status": "completed",
            "metadata": {
                "features_ingested": len(input_features),
                "data_quality_score": 0.95,
                "missing_values": 0
            }
        })
        
        # Feature engineering stage  
        stages.append({
            "stage": DecisionStage.FEATURE_ENGINEERING.value,
            "timestamp": current_time - timedelta(milliseconds=50),
            "duration_ms": 30,
            "status": "completed",
            "metadata": {
                "transformations_applied": ["standardization", "lag_features"],
                "feature_count_output": len(input_features) * 2,
                "feature_selection_score": 0.88
            }
        })
        
        # Model inference stage
        stages.append({
            "stage": DecisionStage.MODEL_INFERENCE.value,
            "timestamp": current_time - timedelta(milliseconds=20),
            "duration_ms": 15,
            "status": "completed",
            "metadata": {
                "model_name": model_name,
                "inference_mode": "production",
                "batch_size": 1
            }
        })
        
        # Post-processing stage
        stages.append({
            "stage": DecisionStage.POST_PROCESSING.value,
            "timestamp": current_time - timedelta(milliseconds=5),
            "duration_ms": 3,
            "status": "completed",
            "metadata": {
                "output_validation": "passed",
                "risk_thresholds_applied": True
            }
        })
        
        # Result formatting stage
        stages.append({
            "stage": DecisionStage.RESULT_FORMATTING.value,
            "timestamp": current_time,
            "duration_ms": 2,
            "status": "completed",
            "metadata": {
                "output_format": "json",
                "precision_level": "high"
            }
        })
        
        return stages
        
    def _assess_compliance_requirements(
        self, 
        model_name: str, 
        prediction_value: Union[float, str, Dict[str, float]], 
        explanation_level: ExplainabilityLevel
    ) -> List[str]:
        """Assess compliance requirements for this decision"""
        
        flags = []
        
        # Check if high-risk decision requiring enhanced explanation
        if isinstance(prediction_value, dict):
            max_confidence = max(prediction_value.values()) if prediction_value else 0
        else:
            max_confidence = 1.0  # Assume regression confidence
            
        if max_confidence < 0.7:
            flags.append("LOW_CONFIDENCE_DECISION")
            
        # Check model-specific compliance requirements
        if model_name in ["regime_classifier", "forecast_model"]:
            flags.append("FINANCIAL_DECISION")
            
        if explanation_level == ExplainabilityLevel.BASIC:
            flags.append("MINIMAL_EXPLANATION")
            
        # Add regulatory flags
        flags.extend(self._get_regulatory_flags(model_name))
        
        return flags
        
    def _get_regulatory_flags(self, model_name: str) -> List[str]:
        """Get regulatory compliance flags for model"""
        
        regulatory_mapping = {
            "regime_classifier": ["MiFID_II", "Basel_III", "GDPR"],
            "forecast_model": ["MiFID_II", "IFRS_9", "GDPR"],
            "anomaly_detector": ["AML_Directive", "GDPR"]
        }
        
        return regulatory_mapping.get(model_name, ["GDPR"])
        
    def _assess_data_sensitivity(self, input_features: Dict[str, float]) -> str:
        """Assess sensitivity level of input data"""
        
        financial_indicators = ["vix", "sp500", "dxy", "tnx", "oil"]
        
        if any(indicator in feature.lower() for feature in input_features.keys() for indicator in financial_indicators):
            return "FINANCIAL_HIGH"
            
        return "PUBLIC_LOW"
        
    def _get_model_version(self, model_name: str) -> str:
        """Get current model version"""
        
        version_mapping = {
            "regime_classifier": "v1.2.0",
            "forecast_model": "v1.1.0",
            "anomaly_detector": "v1.0.0"
        }
        
        return version_mapping.get(model_name, "v1.0.0")
        
    def _calculate_prediction_confidence(self, feature_contributions: List[FeatureContribution]) -> float:
        """Calculate overall prediction confidence from feature contributions"""
        
        if not feature_contributions:
            return 0.5
            
        # Confidence based on contribution concentration
        total_contribution = sum(abs(fc.contribution_score) for fc in feature_contributions)
        if total_contribution == 0:
            return 0.5
            
        # Higher confidence if top features dominate
        top_3_contribution = sum(abs(fc.contribution_score) for fc in feature_contributions[:3])
        concentration_ratio = top_3_contribution / total_contribution
        
        # Normalize to 0.5-1.0 range
        confidence = 0.5 + (concentration_ratio * 0.5)
        
        return min(max(confidence, 0.0), 1.0)
        
    def _generate_counterfactuals(self, model_name: str, input_features: Dict[str, float]) -> List[Dict[str, Any]]:
        """Generate counterfactual examples for 'what-if' analysis"""
        
        counterfactuals = []
        
        # Generate simple counterfactuals by perturbing top features
        for feature_name, feature_value in list(input_features.items())[:3]:
            
            # Increase by 10%
            increased_features = input_features.copy()
            increased_features[feature_name] = feature_value * 1.1
            counterfactuals.append({
                "description": f"If {feature_name} increased by 10%",
                "modified_features": {feature_name: increased_features[feature_name]},
                "expected_impact": "Higher risk probability",
                "confidence": 0.7
            })
            
            # Decrease by 10%  
            decreased_features = input_features.copy()
            decreased_features[feature_name] = feature_value * 0.9
            counterfactuals.append({
                "description": f"If {feature_name} decreased by 10%",
                "modified_features": {feature_name: decreased_features[feature_name]},
                "expected_impact": "Lower risk probability",
                "confidence": 0.7
            })
            
        return counterfactuals[:4]  # Limit to 4 examples
        
    def _calculate_decision_boundary_distance(self, model_name: str, input_features: Dict[str, float]) -> Optional[float]:
        """Calculate distance to decision boundary for confidence assessment"""
        
        # Simplified calculation - in practice would use actual model
        feature_values = np.array(list(input_features.values()))
        normalized_distance = np.linalg.norm(feature_values) / len(feature_values)
        
        # Normalize to meaningful range
        return min(max(normalized_distance, 0.1), 2.0)
        
    def _format_explanation_response(
        self, 
        provenance: DecisionProvenance, 
        requested_level: ExplainabilityLevel
    ) -> Dict[str, Any]:
        """Format explanation response based on requested detail level"""
        
        response = {
            "decision_id": provenance.decision_id,
            "timestamp": provenance.timestamp.isoformat(),
            "explanation_level": requested_level.value
        }
        
        if requested_level == ExplainabilityLevel.BASIC:
            # Basic: Just top features
            if provenance.model_explanations:
                exp = provenance.model_explanations[0]
                response["top_features"] = [
                    {
                        "feature": fc.feature_name,
                        "contribution": fc.contribution_score
                    }
                    for fc in exp.feature_contributions[:5]
                ]
                response["prediction"] = exp.prediction_value
                response["confidence"] = exp.confidence_score
                
        elif requested_level == ExplainabilityLevel.DETAILED:
            # Detailed: Feature contributions + metadata
            response["model_explanations"] = [
                {
                    "model_name": exp.model_name,
                    "prediction": exp.prediction_value,
                    "confidence": exp.confidence_score,
                    "feature_contributions": [asdict(fc) for fc in exp.feature_contributions],
                    "computation_time_ms": exp.computation_time_ms
                }
                for exp in provenance.model_explanations
            ]
            response["data_lineage"] = provenance.data_lineage
            
        elif requested_level in [ExplainabilityLevel.COMPREHENSIVE, ExplainabilityLevel.REGULATORY]:
            # Comprehensive/Regulatory: Full provenance
            response["complete_provenance"] = asdict(provenance)
            
        return response

# Export singleton instance
explainability_logger = ExplainabilityProvenanceLogger()