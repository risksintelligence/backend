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
    free_intelligence_status: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    external_api_count: int = Field(default=0)

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
    geopolitical_events: List["GeopoliticalEvent"] = Field(default_factory=list)
    data_sources: List[str] = Field(default=["GDELT", "Free Maritime Intelligence"])


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
    geopolitical_risks: Dict[str, Any] = Field(default_factory=dict)
    supply_chain_disruptions: List["SupplyChainDisruption"] = Field(default_factory=list)

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
    geopolitical_factors: List["GeopoliticalEvent"] = Field(default_factory=list)
    maritime_factors: List["ShippingDelay"] = Field(default_factory=list)
    data_sources: List[str] = Field(default=["GDELT", "Free Maritime Intelligence"])

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


# Geopolitical Intelligence Schemas
class GeopoliticalEvent(BaseModel):
    event_id: str
    event_type: str
    sub_event_type: str
    event_date: str  # ISO string
    country: str
    region: str
    location: List[float]  # [lat, lng]
    impact_score: float
    confidence: float
    source: str
    description: str
    affected_trade_routes: List[str]
    estimated_disruption_days: int
    source_url: Optional[str] = None


class SupplyChainDisruption(BaseModel):
    disruption_id: str
    severity: Literal["low", "medium", "high", "critical"]
    event_type: str
    location: List[float]  # [lat, lng]
    description: str
    source: str
    start_date: str  # ISO string
    estimated_duration_days: int
    affected_commodities: List[str]
    affected_trade_routes: List[str]
    economic_impact_usd: float
    confidence_score: float
    mitigation_strategies: List[str]


class GeopoliticalDisruptionsResponse(BaseModel):
    as_of: str
    disruptions: List[SupplyChainDisruption]
    summary: Dict[str, Any]

    class Config:
        extra = "allow"


# Maritime Intelligence Schemas
class VesselInfo(BaseModel):
    mmsi: int
    vessel_name: str
    vessel_type: str
    lat: float
    lng: float
    speed: Optional[float] = None
    course: Optional[float] = None
    timestamp: str
    source: str
    port_destination: Optional[str] = None
    eta: Optional[str] = None
    cargo_type: Optional[str] = None


class PortCongestion(BaseModel):
    port_code: str
    port_name: str
    vessels_at_anchor: int
    vessels_at_berth: int
    average_wait_time_hours: Optional[float] = None
    congestion_level: Literal["low", "medium", "high", "severe"]
    last_updated: str
    source_breakdown: Dict[str, int]


class ShippingDelay(BaseModel):
    route_name: str
    origin_port: str
    destination_port: str
    typical_transit_days: int
    current_delay_days: int
    delay_reasons: List[str]
    severity: Literal["minor", "moderate", "major", "critical"]
    affected_vessels: int


class MaritimeProvider(BaseModel):
    id: str
    name: str
    coverage: str
    data_types: List[str]
    rate_limit: int
    requires_auth: bool


class ProviderHealth(BaseModel):
    overall_health: Literal["healthy", "degraded", "critical"]
    health_score: float
    providers: Dict[str, bool]
    healthy_providers: int
    total_providers: int


class PortCongestionResponse(BaseModel):
    ports: List[PortCongestion]
    summary: Dict[str, Any]

    class Config:
        extra = "allow"


class ShippingDelaysResponse(BaseModel):
    delays: List[ShippingDelay]
    summary: Dict[str, Any]

    class Config:
        extra = "allow"


class MaritimeRiskAssessment(BaseModel):
    overall_risk_score: float
    risk_level: Literal["low", "medium", "high"]
    high_risk_ports: List[str]
    critical_delays: int
    port_congestion_summary: Dict[str, Dict[str, Any]]
    shipping_delays_summary: List[Dict[str, Any]]
    data_sources: List[str]
    provider_health: Dict[str, bool]
    last_updated: str

    class Config:
        extra = "allow"


class VesselsNearPortResponse(BaseModel):
    search_params: Dict[str, float]
    vessels: List[VesselInfo]
    summary: Dict[str, Any]

    class Config:
        extra = "allow"


class MaritimeProvidersResponse(BaseModel):
    providers: List[MaritimeProvider]
    total_providers: int
    coverage_areas: List[str]
    advantages: List[str]

    class Config:
        extra = "allow"


class MaritimeIntelligenceHealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"]
    last_update: str
    data_sources: List[str]
    port_coverage: int
    vessel_tracking_active: bool
    api_response_time_ms: Optional[float] = None
    error_message: Optional[str] = None

    class Config:
        extra = "allow"


# Resolve forward references after all models are defined
CascadeSnapshotResponse.model_rebuild()
CascadeImpactsResponse.model_rebuild()
DisruptionPredictionsResponse.model_rebuild()
GeopoliticalDisruptionsResponse.model_rebuild()
