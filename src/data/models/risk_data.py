"""
Risk assessment and scoring models.
"""
from datetime import datetime
from typing import Optional

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Index, Boolean, JSON

from src.core.database import Base


class RiskScore(Base):
    """Risk scores calculated by the system."""
    
    __tablename__ = "risk_scores"
    
    id = Column(Integer, primary_key=True, index=True)
    score_type = Column(String(50), nullable=False)  # overall, financial, supply_chain, etc.
    entity_type = Column(String(50), nullable=False)  # national, sector, institution, etc.
    entity_id = Column(String(100), nullable=True)  # specific entity identifier
    
    # Score values
    score = Column(Float, nullable=False)  # 0-100 risk score
    confidence = Column(Float, nullable=True)  # 0-1 confidence in score
    risk_level = Column(String(20), nullable=False)  # low, medium, high, critical
    
    # Temporal information
    score_date = Column(DateTime, nullable=False)
    valid_from = Column(DateTime, nullable=False)
    valid_to = Column(DateTime, nullable=True)
    
    # Calculation metadata
    model_version = Column(String(20), nullable=True)
    input_data_hash = Column(String(64), nullable=True)  # Hash of input data for reproducibility
    calculation_time = Column(Float, nullable=True)  # Seconds to calculate
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index("idx_score_entity", "score_type", "entity_type", "entity_id"),
        Index("idx_score_date", "score_date", "score_type"),
        Index("idx_risk_level", "risk_level", "score_date"),
    )
    
    def __repr__(self):
        return f"<RiskScore(type='{self.score_type}', score={self.score}, level='{self.risk_level}')>"


class RiskFactor(Base):
    """Individual risk factors contributing to overall risk scores."""
    
    __tablename__ = "risk_factors"
    
    id = Column(Integer, primary_key=True, index=True)
    factor_name = Column(String(100), nullable=False)
    factor_category = Column(String(50), nullable=False)  # economic, financial, geopolitical, etc.
    
    # Factor values
    weight = Column(Float, nullable=False)  # Weight in overall calculation (0-1)
    value = Column(Float, nullable=False)  # Current factor value
    normalized_value = Column(Float, nullable=False)  # Normalized to 0-1 scale
    contribution = Column(Float, nullable=False)  # Contribution to total score
    
    # Temporal information
    factor_date = Column(DateTime, nullable=False)
    
    # Data sources
    source_series = Column(Text, nullable=True)  # JSON list of source data series
    calculation_method = Column(String(100), nullable=True)
    
    # Associated risk score
    risk_score_id = Column(Integer, nullable=True)  # Can be null for standalone factors
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index("idx_factor_category", "factor_category", "factor_date"),
        Index("idx_factor_contribution", "contribution", "factor_date"),
    )
    
    def __repr__(self):
        return f"<RiskFactor(name='{self.factor_name}', value={self.value}, weight={self.weight})>"


class RiskEvent(Base):
    """Identified risk events and disruptions."""
    
    __tablename__ = "risk_events"
    
    id = Column(Integer, primary_key=True, index=True)
    event_type = Column(String(50), nullable=False)  # cyber, natural, economic, political, etc.
    event_name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    
    # Event characteristics
    severity = Column(String(20), nullable=False)  # low, medium, high, critical
    probability = Column(Float, nullable=True)  # 0-1 probability of occurrence
    impact_score = Column(Float, nullable=True)  # 0-100 potential impact
    
    # Geographic and sectoral scope
    geographic_scope = Column(String(100), nullable=True)  # local, regional, national, global
    affected_sectors = Column(Text, nullable=True)  # JSON list of affected sectors
    
    # Temporal information
    event_date = Column(DateTime, nullable=True)  # When event occurred/will occur
    detected_date = Column(DateTime, default=datetime.utcnow)  # When we detected it
    start_date = Column(DateTime, nullable=True)
    end_date = Column(DateTime, nullable=True)
    
    # Event status
    status = Column(String(20), default="active")  # active, resolved, monitoring, etc.
    is_active = Column(Boolean, default=True)
    
    # Data sources
    source = Column(String(50), nullable=True)  # CISA, NOAA, news, model, etc.
    source_url = Column(String(500), nullable=True)
    confidence = Column(Float, nullable=True)  # 0-1 confidence in event data
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index("idx_event_type_date", "event_type", "event_date"),
        Index("idx_severity_status", "severity", "status"),
        Index("idx_active_events", "is_active", "event_date"),
    )
    
    def __repr__(self):
        return f"<RiskEvent(type='{self.event_type}', name='{self.event_name}', severity='{self.severity}')>"


class RiskPrediction(Base):
    """Risk predictions generated by ML models."""
    
    __tablename__ = "risk_predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    prediction_type = Column(String(50), nullable=False)  # recession, disruption, default, etc.
    target_date = Column(DateTime, nullable=False)  # Date being predicted
    horizon_days = Column(Integer, nullable=False)  # Prediction horizon in days
    
    # Prediction values
    probability = Column(Float, nullable=False)  # 0-1 probability
    confidence_interval_lower = Column(Float, nullable=True)
    confidence_interval_upper = Column(Float, nullable=True)
    confidence_level = Column(Float, nullable=True)  # e.g., 0.95 for 95% CI
    
    # Model information
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(20), nullable=False)
    features_used = Column(Text, nullable=True)  # JSON list of features
    feature_importance = Column(Text, nullable=True)  # JSON dict of feature importances
    
    # Explainability
    shap_values = Column(Text, nullable=True)  # JSON SHAP values for explanation
    top_drivers = Column(Text, nullable=True)  # JSON list of top prediction drivers
    
    # Validation
    prediction_date = Column(DateTime, default=datetime.utcnow)
    actual_outcome = Column(String(20), nullable=True)  # To be filled when target_date passes
    accuracy_score = Column(Float, nullable=True)  # Calculated post-prediction
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index("idx_prediction_type_date", "prediction_type", "target_date"),
        Index("idx_model_prediction", "model_name", "prediction_date"),
        Index("idx_horizon", "horizon_days", "probability"),
    )
    
    def __repr__(self):
        return f"<RiskPrediction(type='{self.prediction_type}', probability={self.probability}, target='{self.target_date}')>"


class ModelPerformance(Base):
    """Model performance tracking and validation."""
    
    __tablename__ = "model_performance"
    
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(20), nullable=False)
    evaluation_date = Column(DateTime, default=datetime.utcnow)
    
    # Performance metrics
    accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    auc_roc = Column(Float, nullable=True)
    
    # Bias and fairness metrics
    bias_score = Column(Float, nullable=True)  # 0-1, lower is better
    fairness_metrics = Column(Text, nullable=True)  # JSON dict of fairness metrics
    
    # Data characteristics
    training_data_size = Column(Integer, nullable=True)
    validation_data_size = Column(Integer, nullable=True)
    test_data_size = Column(Integer, nullable=True)
    
    # Time-based performance
    prediction_horizon = Column(Integer, nullable=True)  # Days
    temporal_stability = Column(Float, nullable=True)  # Performance over time
    
    # Additional details
    evaluation_details = Column(Text, nullable=True)  # JSON with detailed metrics
    notes = Column(Text, nullable=True)
    
    __table_args__ = (
        Index("idx_model_version", "model_name", "model_version"),
        Index("idx_performance_date", "evaluation_date", "accuracy"),
    )
    
    def __repr__(self):
        return f"<ModelPerformance(model='{self.model_name}', version='{self.model_version}', accuracy={self.accuracy})>"