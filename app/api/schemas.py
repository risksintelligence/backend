from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class Alert(BaseModel):
    id: str
    severity: Literal["critical", "high", "medium", "low"]
    message: str
    driver: str
    timestamp: str  # ISO string; leave as str to avoid coercion issues


class AnomalySummary(BaseModel):
    total_anomalies: int
    max_severity: Literal["critical", "high", "medium", "low"]
    updated_at: str


class AnomalyResponse(BaseModel):
    anomalies: List[Alert]
    summary: AnomalySummary

    class Config:
        extra = "allow"


class NetworkNode(BaseModel):
    id: str
    name: str
    sector: str
    risk: float


class Vulnerability(BaseModel):
    node: str
    risk: float
    description: str


class PartnerDependency(BaseModel):
    partner: str
    dependency: str
    status: Literal["stable", "watch", "critical"]


class ProviderHealthEntry(BaseModel):
    reliability_score: float
    failure_count: int
    last_failure: Optional[str]
    rate_limit_per_minute: Optional[int]
    should_skip: bool
    supported_series: List[str]

    class Config:
        extra = "allow"


class ProviderHealthSummary(BaseModel):
    total_providers: int
    healthy_providers: int
    unhealthy_providers: int
    average_reliability: float
    overall_health: str


class ProviderHealthResponse(BaseModel):
    nodes: List[NetworkNode]
    criticalPaths: List[str]
    summary: str
    updatedAt: str
    vulnerabilities: List[Vulnerability]
    partnerDependencies: List[PartnerDependency]
    providerHealth: Dict[str, ProviderHealthEntry]
    summaryStats: ProviderHealthSummary

    class Config:
        extra = "allow"


class TransparencyFreshnessResponse(BaseModel):
    timestamp: str
    overall_status: Any
    cache_layers: Dict[str, Any]
    series_freshness: Dict[str, Any]
    background_refresh: Any
    provider_health: Any
    compliance: Dict[str, Any]

    class Config:
        extra = "allow"


# Supply Chain Cascade Schemas
class CascadeNode(BaseModel):
    id: str
    name: str
    type: str  # port | plant | region | dc | hub
    lat: float
    lng: float
    risk_operational: float
    risk_financial: float
    risk_policy: float
    industry_impacts: Dict[str, float] = Field(default_factory=dict)


class CascadeEdge(BaseModel):
    from_id: str = Field(..., alias="from")
    to_id: str = Field(..., alias="to")
    mode: str  # sea | air | rail | road | pipeline
    flow: float
    congestion: float
    eta_delay_hours: float
    criticality: float

    class Config:
        allow_population_by_field_name = True


class CascadeDisruption(BaseModel):
    id: str
    type: str  # geo_event | congestion | policy | supplier
    severity: Literal["critical", "high", "medium", "low"]
    location: List[float]  # [lat, lng]
    description: str
    source: Optional[str] = None


class CascadeSnapshotResponse(BaseModel):
    as_of: str
    nodes: List[CascadeNode]
    edges: List[CascadeEdge]
    critical_paths: List[List[str]]
    disruptions: List[CascadeDisruption]


class CascadeTimePoint(BaseModel):
    t: str
    v: float


class CascadeSeries(BaseModel):
    metric: str
    points: List[CascadeTimePoint]


class CascadeHistoryResponse(BaseModel):
    series: List[CascadeSeries]


class CascadeImpactsResponse(BaseModel):
    financial: Dict[str, Any]
    policy: Dict[str, Any]
    industry: Dict[str, Any]

    class Config:
        extra = "allow"


# Predictive Analysis Schemas
class DisruptionPrediction(BaseModel):
    disruption_type: str
    probability: float
    risk_level: str
    estimated_impact_usd: float
    confidence_score: float
    time_horizon_days: int
    affected_regions: List[str]
    affected_commodities: List[str]
    risk_triggers: List[str]
    mitigation_strategies: List[str]


class CascadeImpactModel(BaseModel):
    node_id: str
    node_name: str
    direct_impact_probability: float
    indirect_impact_probability: float
    cascade_delay_hours: float
    economic_impact_usd: float
    recovery_time_days: int


class PredictionSummary(BaseModel):
    total_predictions: int
    high_risk_predictions: int
    total_economic_risk_usd: float
    average_confidence: float
    cascade_nodes_analyzed: int


class DisruptionPredictionsResponse(BaseModel):
    as_of: str
    forecast_horizon_days: int
    predictions: List[DisruptionPrediction]
    cascade_impacts: List[CascadeImpactModel]
    summary: PredictionSummary

    class Config:
        extra = "allow"


class CascadeImpactResponse(BaseModel):
    as_of: str
    disruption_correlations: Dict[str, float]
    amplification_factors: Dict[str, Any]
    high_risk_combinations: List[str]
    analysis_notes: Dict[str, Any]

    class Config:
        extra = "allow"


class ActiveAlert(BaseModel):
    disruption_type: str
    risk_level: str
    probability: float
    estimated_impact_usd: float
    affected_regions: List[str]
    immediate_actions: List[str]


class AlertThreshold(BaseModel):
    probability_threshold: float
    impact_threshold_usd: float
    confidence_threshold: Optional[float] = None


class SystemConfiguration(BaseModel):
    update_frequency: str
    escalation_protocols: List[str]


class EarlyWarningRecommendations(BaseModel):
    immediate_actions: List[str]
    strategic_actions: List[str]


class EarlyWarningResponse(BaseModel):
    as_of: str
    alert_level: str
    active_alerts: List[ActiveAlert]
    alert_thresholds: Dict[str, AlertThreshold]
    monitoring_indicators: Dict[str, List[str]]
    system_configuration: SystemConfiguration
    recommendations: EarlyWarningRecommendations

    class Config:
        extra = "allow"
