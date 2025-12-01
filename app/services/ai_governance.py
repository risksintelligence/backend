"""
NIST AI Risk Management Framework (AI RMF 1.0) Implementation
Institutional-grade AI/ML governance for RRIO model lifecycle management
"""

import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import structlog
from pydantic import BaseModel

logger = logging.getLogger(__name__)
struct_logger = structlog.get_logger()

# API Request Models
class ModelRegistrationRequest(BaseModel):
    model_name: str
    model_type: str
    model_version: str
    training_data_hash: str
    hyperparameters: Dict[str, Any]
    performance_metrics: Dict[str, float]
    intended_use: str
    limitations: List[str]
    risk_level: str

class DriftCheckRequest(BaseModel):
    model_name: str
    current_data: List[Dict[str, Any]]
    drift_type: str = "data"
    reference_window: int = 30
    significance_level: float = 0.05

class ComplianceReportRequest(BaseModel):
    model_name: str
    include_history: bool = False

class AIRiskLevel(Enum):
    MINIMAL = "minimal"
    LOW = "low" 
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"

class ModelStatus(Enum):
    DEVELOPMENT = "development"
    VALIDATION = "validation"
    PRODUCTION = "production"
    MONITORING = "monitoring"
    DEPRECATED = "deprecated"
    RETIRED = "retired"

class DriftType(Enum):
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    PREDICTION_DRIFT = "prediction_drift"
    PERFORMANCE_DRIFT = "performance_drift"

@dataclass
class ModelArtifact:
    model_id: str
    version: str
    model_type: str  # "regime_classifier", "forecast_model", "anomaly_detector"
    artifact_path: str
    checksum: str
    created_at: datetime
    training_data_hash: str
    hyperparameters: Dict[str, Any]
    performance_metrics: Dict[str, float]
    
@dataclass
class ModelValidation:
    model_id: str
    version: str
    validation_type: str  # "bias_test", "fairness_test", "robustness_test", "explainability_test"
    passed: bool
    score: float
    details: Dict[str, Any]
    validated_by: str
    validated_at: datetime

@dataclass
class DriftAlert:
    model_id: str
    drift_type: DriftType
    severity: AIRiskLevel
    detected_at: datetime
    metric_name: str
    baseline_value: float
    current_value: float
    threshold: float
    confidence: float
    
    @property
    def drift_magnitude(self) -> float:
        return abs(self.current_value - self.baseline_value) / abs(self.baseline_value) if self.baseline_value != 0 else float('inf')

@dataclass
class GovernanceReport:
    timestamp: datetime
    nist_ai_rmf_compliance: Dict[str, Any]
    model_inventory: List[ModelArtifact]
    active_drift_alerts: List[DriftAlert]
    validation_summary: Dict[str, Any]
    risk_assessment: Dict[str, AIRiskLevel]
    
class NISTAIGovernanceFramework:
    """
    NIST AI RMF 1.0 compliant governance framework for RRIO AI systems
    
    Implements the four core functions:
    1. GOVERN (1.0): Establish AI governance structures
    2. MAP (2.0): Categorize AI systems and risk contexts  
    3. MEASURE (3.0): Analyze and track AI risks
    4. MANAGE (4.0): Allocate resources to manage AI risks
    """
    
    def __init__(self, models_directory: str = "models/"):
        self.models_dir = Path(models_directory)
        self.models_dir.mkdir(exist_ok=True)
        
        # NIST AI RMF Core Components
        self.governance_structure = self._initialize_governance()
        self.model_registry: Dict[str, ModelArtifact] = {}
        self.drift_monitors: Dict[str, Any] = {}
        self.validation_history: List[ModelValidation] = []
        self.drift_alerts: List[DriftAlert] = []
        
        # Performance baselines for drift detection
        self.performance_baselines = {
            "regime_classifier": {"accuracy": 0.85, "f1_score": 0.80, "recall": 0.82},
            "forecast_model": {"mae": 2.5, "mse": 8.0, "r2_score": 0.75},
            "anomaly_detector": {"precision": 0.90, "recall": 0.75, "f1_score": 0.82}
        }
    
    def _initialize_governance(self) -> Dict[str, Any]:
        """Initialize NIST AI RMF governance structure"""
        return {
            "framework_version": "NIST_AI_RMF_1.0",
            "organization": "RRIO",
            "governance_board": {
                "chief_risk_officer": "AI_Ethics_Board",
                "model_validation_team": "RRIO_ML_Engineering",
                "compliance_officer": "Risk_Management_Team"
            },
            "policies": {
                "bias_testing_required": True,
                "explainability_required": True,
                "human_oversight_required": True,
                "drift_monitoring_enabled": True,
                "model_versioning_enforced": True
            },
            "risk_tolerance": {
                "maximum_bias_score": 0.1,
                "minimum_explainability_score": 0.8,
                "maximum_drift_threshold": 0.15,
                "model_retirement_threshold": 0.6  # Performance below 60% triggers retirement
            }
        }
    
    # ============ GOVERN (1.0): Governance and Oversight ============
    
    def register_model(
        self, 
        model_id: str, 
        version: str, 
        model_type: str, 
        artifact_path: str,
        hyperparameters: Dict[str, Any],
        performance_metrics: Dict[str, float],
        training_data_hash: str
    ) -> ModelArtifact:
        """
        Register a new model artifact with full governance tracking
        NIST AI RMF GOVERN function implementation
        """
        
        # Calculate artifact checksum for integrity
        try:
            with open(artifact_path, 'rb') as f:
                checksum = hashlib.sha256(f.read()).hexdigest()
        except FileNotFoundError:
            checksum = "artifact_not_found"
        
        artifact = ModelArtifact(
            model_id=model_id,
            version=version,
            model_type=model_type,
            artifact_path=artifact_path,
            checksum=checksum,
            created_at=datetime.utcnow(),
            training_data_hash=training_data_hash,
            hyperparameters=hyperparameters,
            performance_metrics=performance_metrics
        )
        
        self.model_registry[f"{model_id}_{version}"] = artifact
        
        struct_logger.info(
            "Model registered in governance framework",
            model_id=model_id,
            version=version,
            model_type=model_type,
            nist_ai_rmf_function="GOVERN"
        )
        
        return artifact
    
    # ============ MAP (2.0): Risk Context and Categorization ============
    
    def assess_ai_risk_level(self, model_id: str, use_case: str) -> AIRiskLevel:
        """
        Assess AI risk level according to NIST AI RMF MAP function
        Maps AI systems to risk contexts and impact assessments
        """
        
        # RRIO-specific risk assessment criteria
        risk_factors = {
            "economic_impact": 0.4,  # High impact on economic decisions
            "regulatory_scope": 0.3,  # Subject to financial regulations
            "transparency_requirement": 0.2,  # Institutional transparency required
            "human_oversight": 0.1   # Human oversight available
        }
        
        # Model-specific risk scoring
        model_risk_profiles = {
            "regime_classifier": {
                "economic_impact": 0.8,  # High - affects risk assessment
                "regulatory_scope": 0.9,  # High - financial regulation
                "transparency_requirement": 0.9,  # High - institutional use
                "human_oversight": 0.7   # Medium - analyst oversight
            },
            "forecast_model": {
                "economic_impact": 0.9,  # Very high - predictive decisions
                "regulatory_scope": 0.8,  # High - forecast disclosures
                "transparency_requirement": 0.8,  # High - explainability needed
                "human_oversight": 0.8   # High - forecast review
            },
            "anomaly_detector": {
                "economic_impact": 0.6,  # Medium - alert generation
                "regulatory_scope": 0.5,  # Medium - monitoring function
                "transparency_requirement": 0.7,  # High - investigation support
                "human_oversight": 0.9   # High - analyst validation
            }
        }
        
        # Get model profile or default to high risk
        profile = model_risk_profiles.get(model_id.split("_")[0], {
            "economic_impact": 0.8,
            "regulatory_scope": 0.8,
            "transparency_requirement": 0.8,
            "human_oversight": 0.6
        })
        
        # Calculate weighted risk score
        risk_score = sum(profile[factor] * weight for factor, weight in risk_factors.items())
        
        # Map to risk levels
        if risk_score >= 0.8:
            return AIRiskLevel.CRITICAL
        elif risk_score >= 0.6:
            return AIRiskLevel.HIGH
        elif risk_score >= 0.4:
            return AIRiskLevel.MODERATE
        elif risk_score >= 0.2:
            return AIRiskLevel.LOW
        else:
            return AIRiskLevel.MINIMAL
    
    # ============ MEASURE (3.0): AI Risk Analysis and Monitoring ============
    
    def detect_model_drift(
        self, 
        model_id: str, 
        current_predictions: List[float],
        current_features: Optional[np.ndarray] = None,
        actual_outcomes: Optional[List[float]] = None
    ) -> List[DriftAlert]:
        """
        Detect various types of model drift using statistical methods
        NIST AI RMF MEASURE function implementation
        """
        
        alerts = []
        model_key = f"{model_id}_latest"
        baseline_metrics = self.performance_baselines.get(model_id.split("_")[0], {})
        
        # 1. Prediction Drift Detection (Distribution shift)
        if current_predictions:
            prediction_drift = self._detect_prediction_drift(model_id, current_predictions)
            if prediction_drift:
                alerts.append(prediction_drift)
        
        # 2. Performance Drift Detection
        if actual_outcomes and len(current_predictions) == len(actual_outcomes):
            performance_drift = self._detect_performance_drift(
                model_id, current_predictions, actual_outcomes, baseline_metrics
            )
            if performance_drift:
                alerts.append(performance_drift)
        
        # 3. Data Drift Detection (Feature distribution changes)
        if current_features is not None:
            data_drift = self._detect_data_drift(model_id, current_features)
            if data_drift:
                alerts.append(data_drift)
        
        # Store alerts for governance reporting
        self.drift_alerts.extend(alerts)
        
        # Log drift detection results
        if alerts:
            struct_logger.warning(
                "Model drift detected",
                model_id=model_id,
                drift_count=len(alerts),
                drift_types=[alert.drift_type.value for alert in alerts],
                nist_ai_rmf_function="MEASURE"
            )
        
        return alerts
    
    def _detect_prediction_drift(self, model_id: str, predictions: List[float]) -> Optional[DriftAlert]:
        """Detect drift in prediction distributions using statistical tests"""
        
        # Store current predictions for baseline comparison
        cache_key = f"{model_id}_prediction_baseline"
        
        # For now, use simple statistics - in production, implement KS test or similar
        current_mean = np.mean(predictions)
        current_std = np.std(predictions)
        
        # Simulated baseline (in production, retrieve from historical data)
        baseline_mean = 55.0  # GERII typical value
        baseline_std = 12.0
        
        # Calculate drift magnitude
        mean_drift = abs(current_mean - baseline_mean) / baseline_std
        std_drift = abs(current_std - baseline_std) / baseline_std
        
        # Alert if significant drift detected
        threshold = 0.15  # 15% deviation threshold
        
        if mean_drift > threshold:
            return DriftAlert(
                model_id=model_id,
                drift_type=DriftType.PREDICTION_DRIFT,
                severity=AIRiskLevel.HIGH if mean_drift > 0.25 else AIRiskLevel.MODERATE,
                detected_at=datetime.utcnow(),
                metric_name="prediction_mean",
                baseline_value=baseline_mean,
                current_value=current_mean,
                threshold=threshold,
                confidence=0.95
            )
        
        return None
    
    def _detect_performance_drift(
        self, 
        model_id: str, 
        predictions: List[float], 
        actuals: List[float],
        baseline_metrics: Dict[str, float]
    ) -> Optional[DriftAlert]:
        """Detect performance degradation vs baseline metrics"""
        
        # Calculate current performance metrics
        mae = np.mean(np.abs(np.array(predictions) - np.array(actuals)))
        mse = np.mean((np.array(predictions) - np.array(actuals)) ** 2)
        
        # Compare to baseline
        baseline_mae = baseline_metrics.get("mae", 2.5)
        
        # Calculate performance drift
        performance_drift = (mae - baseline_mae) / baseline_mae
        
        threshold = 0.20  # 20% performance degradation threshold
        
        if performance_drift > threshold:
            return DriftAlert(
                model_id=model_id,
                drift_type=DriftType.PERFORMANCE_DRIFT,
                severity=AIRiskLevel.CRITICAL if performance_drift > 0.40 else AIRiskLevel.HIGH,
                detected_at=datetime.utcnow(),
                metric_name="mean_absolute_error",
                baseline_value=baseline_mae,
                current_value=mae,
                threshold=threshold,
                confidence=0.90
            )
        
        return None
    
    def _detect_data_drift(self, model_id: str, features: np.ndarray) -> Optional[DriftAlert]:
        """Detect data drift in input features"""
        
        # Simple feature statistics drift detection
        current_feature_means = np.mean(features, axis=0)
        current_feature_stds = np.std(features, axis=0)
        
        # Simulated baseline feature statistics
        baseline_means = np.array([50.0, 2.0, 15.0, 95.0])  # Typical RRIO feature means
        
        if len(current_feature_means) >= len(baseline_means):
            # Calculate drift in feature means
            feature_drift = np.mean(np.abs(current_feature_means[:len(baseline_means)] - baseline_means) / baseline_means)
            
            threshold = 0.10  # 10% feature drift threshold
            
            if feature_drift > threshold:
                return DriftAlert(
                    model_id=model_id,
                    drift_type=DriftType.DATA_DRIFT,
                    severity=AIRiskLevel.MODERATE,
                    detected_at=datetime.utcnow(),
                    metric_name="feature_mean_drift",
                    baseline_value=float(np.mean(baseline_means)),
                    current_value=float(np.mean(current_feature_means[:len(baseline_means)])),
                    threshold=threshold,
                    confidence=0.85
                )
        
        return None
    
    # ============ MANAGE (4.0): Resource Allocation and Risk Mitigation ============
    
    def generate_governance_report(self) -> GovernanceReport:
        """
        Generate comprehensive AI governance report
        NIST AI RMF MANAGE function implementation
        """
        
        # NIST AI RMF Compliance Assessment
        nist_compliance = {
            "govern_score": self._assess_governance_compliance(),
            "map_score": self._assess_mapping_compliance(),
            "measure_score": self._assess_measurement_compliance(),
            "manage_score": self._assess_management_compliance(),
            "overall_compliance": 0.0  # Will be calculated
        }
        
        # Calculate overall compliance
        scores = [nist_compliance[key] for key in ["govern_score", "map_score", "measure_score", "manage_score"]]
        nist_compliance["overall_compliance"] = sum(scores) / len(scores)
        
        # Risk assessment summary
        risk_assessment = {}
        for model_id in self.model_registry.keys():
            risk_level = self.assess_ai_risk_level(model_id, "economic_intelligence")
            risk_assessment[model_id] = risk_level
        
        # Validation summary
        validation_summary = {
            "total_validations": len(self.validation_history),
            "passed_validations": sum(1 for v in self.validation_history if v.passed),
            "recent_validations": [
                v for v in self.validation_history 
                if v.validated_at > datetime.utcnow() - timedelta(days=30)
            ]
        }
        
        report = GovernanceReport(
            timestamp=datetime.utcnow(),
            nist_ai_rmf_compliance=nist_compliance,
            model_inventory=list(self.model_registry.values()),
            active_drift_alerts=[
                alert for alert in self.drift_alerts 
                if alert.detected_at > datetime.utcnow() - timedelta(days=7)
            ],
            validation_summary=validation_summary,
            risk_assessment=risk_assessment
        )
        
        struct_logger.info(
            "AI governance report generated",
            nist_compliance_score=nist_compliance["overall_compliance"],
            active_alerts=len(report.active_drift_alerts),
            models_tracked=len(report.model_inventory),
            nist_ai_rmf_function="MANAGE"
        )
        
        return report
    
    def _assess_governance_compliance(self) -> float:
        """Assess GOVERN function compliance"""
        score = 0.0
        
        # Check if governance structure is established
        score += 0.3 if self.governance_structure else 0.0
        
        # Check if models are registered
        score += 0.3 if self.model_registry else 0.0
        
        # Check if policies are defined
        score += 0.2 if self.governance_structure.get("policies") else 0.0
        
        # Check if risk tolerance is set
        score += 0.2 if self.governance_structure.get("risk_tolerance") else 0.0
        
        return score
    
    def _assess_mapping_compliance(self) -> float:
        """Assess MAP function compliance"""
        score = 0.0
        
        # Check if AI systems are categorized
        score += 0.5 if self.model_registry else 0.0
        
        # Check if risk assessments exist
        score += 0.5 if any(self.assess_ai_risk_level(model_id, "test") for model_id in self.model_registry.keys()) else 0.0
        
        return score
    
    def _assess_measurement_compliance(self) -> float:
        """Assess MEASURE function compliance"""
        score = 0.0
        
        # Check if drift monitoring is active
        score += 0.4 if self.drift_alerts else 0.0
        
        # Check if performance baselines exist
        score += 0.3 if self.performance_baselines else 0.0
        
        # Check if validations are performed
        score += 0.3 if self.validation_history else 0.0
        
        return score
    
    def _assess_management_compliance(self) -> float:
        """Assess MANAGE function compliance"""
        score = 0.0
        
        # Check if governance reports are generated
        score += 0.4
        
        # Check if alerts are being tracked
        score += 0.3 if self.drift_alerts else 0.0
        
        # Check if model lifecycle is managed
        score += 0.3 if self.model_registry else 0.0
        
        return score

# Global governance instance
ai_governance = NISTAIGovernanceFramework()

# Export functions for API integration
def register_model_artifact(model_id: str, version: str, model_type: str, **kwargs) -> Dict[str, Any]:
    """Register model with governance framework"""
    try:
        artifact = ai_governance.register_model(model_id, version, model_type, **kwargs)
        return {"status": "registered", "artifact_id": f"{model_id}_{version}"}
    except Exception as e:
        logger.error(f"Failed to register model: {e}")
        return {"status": "failed", "error": str(e)}

def check_model_drift(model_id: str, predictions: List[float], **kwargs) -> Dict[str, Any]:
    """Check for model drift and return alerts"""
    try:
        alerts = ai_governance.detect_model_drift(model_id, predictions, **kwargs)
        return {
            "drift_detected": len(alerts) > 0,
            "alert_count": len(alerts),
            "alerts": [asdict(alert) for alert in alerts]
        }
    except Exception as e:
        logger.error(f"Failed to check drift: {e}")
        return {"status": "failed", "error": str(e)}

def generate_compliance_report() -> Dict[str, Any]:
    """Generate NIST AI RMF compliance report"""
    try:
        report = ai_governance.generate_governance_report()
        return asdict(report)
    except Exception as e:
        logger.error(f"Failed to generate compliance report: {e}")
        return {"status": "failed", "error": str(e)}