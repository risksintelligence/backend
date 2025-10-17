"""
Disruption event schemas for RiskX platform.
Pydantic models for disruption events, natural disasters, cyber incidents, and impact assessment.
"""

from pydantic import BaseModel, Field, validator, root_validator
from typing import Optional, List, Dict, Any, Union
from datetime import datetime, date
from decimal import Decimal
from enum import Enum

from ...utils.constants import BusinessRules


class DisruptionType(str, Enum):
    """Types of disruption events."""
    NATURAL_DISASTER = "natural_disaster"
    CYBER_INCIDENT = "cyber_incident"
    GEOPOLITICAL = "geopolitical"
    ECONOMIC_SHOCK = "economic_shock"
    SUPPLY_CHAIN = "supply_chain"
    PANDEMIC = "pandemic"
    TERRORISM = "terrorism"
    LABOR_DISPUTE = "labor_dispute"
    REGULATORY = "regulatory"
    TECHNOLOGICAL = "technological"


class SeverityLevel(str, Enum):
    """Severity levels for disruption events."""
    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    SEVERE = "severe"
    CATASTROPHIC = "catastrophic"


class DisruptionStatus(str, Enum):
    """Status of disruption events."""
    MONITORING = "monitoring"
    DEVELOPING = "developing"
    ACTIVE = "active"
    RESOLVING = "resolving"
    RESOLVED = "resolved"
    CLOSED = "closed"


class ImpactType(str, Enum):
    """Types of impact from disruptions."""
    ECONOMIC = "economic"
    OPERATIONAL = "operational"
    FINANCIAL = "financial"
    REPUTATIONAL = "reputational"
    REGULATORY = "regulatory"
    HUMAN = "human"
    ENVIRONMENTAL = "environmental"


class GeographicScope(str, Enum):
    """Geographic scope of disruption events."""
    LOCAL = "local"
    REGIONAL = "regional"
    NATIONAL = "national"
    INTERNATIONAL = "international"
    GLOBAL = "global"


class AlertLevel(str, Enum):
    """Alert levels for disruption events."""
    WATCH = "watch"
    ADVISORY = "advisory"
    WARNING = "warning"
    EMERGENCY = "emergency"
    CRITICAL = "critical"


class DisruptionEvent(BaseModel):
    """Base model for disruption events."""
    
    event_id: str = Field(..., description="Unique event identifier")
    title: str = Field(..., description="Event title or name")
    description: str = Field(..., description="Detailed event description")
    disruption_type: DisruptionType = Field(..., description="Type of disruption")
    severity: SeverityLevel = Field(..., description="Severity level")
    status: DisruptionStatus = Field(..., description="Current event status")
    geographic_scope: GeographicScope = Field(..., description="Geographic scope of impact")
    affected_regions: List[str] = Field(..., description="List of affected regions/countries")
    start_time: datetime = Field(..., description="Event start time")
    end_time: Optional[datetime] = Field(None, description="Event end time")
    detection_time: datetime = Field(..., description="When event was first detected")
    source: str = Field(..., description="Information source")
    confidence_level: Decimal = Field(..., description="Confidence level of information")
    tags: Optional[List[str]] = Field(default_factory=list, description="Event tags")
    related_events: Optional[List[str]] = Field(default_factory=list, description="Related event IDs")
    
    @validator('event_id')
    def validate_event_id(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Event ID cannot be empty')
        return v.strip().upper()
    
    @validator('confidence_level')
    def validate_confidence_level(cls, v):
        if v < 0 or v > 100:
            raise ValueError('Confidence level must be between 0 and 100')
        return v
    
    @root_validator
    def validate_time_sequence(cls, values):
        start_time = values.get('start_time')
        end_time = values.get('end_time')
        detection_time = values.get('detection_time')
        
        if end_time and start_time and end_time < start_time:
            raise ValueError('End time cannot be before start time')
        
        if detection_time and start_time and detection_time > start_time:
            # Detection can be after start for historical events
            pass
        
        return values


class NaturalDisaster(BaseModel):
    """Natural disaster specific data."""
    
    event: DisruptionEvent = Field(..., description="Base disruption event")
    disaster_type: str = Field(..., description="Specific type of natural disaster")
    magnitude: Optional[Decimal] = Field(None, description="Magnitude or intensity measure")
    scale: Optional[str] = Field(None, description="Scale used for measurement")
    epicenter_latitude: Optional[Decimal] = Field(None, description="Epicenter latitude")
    epicenter_longitude: Optional[Decimal] = Field(None, description="Epicenter longitude")
    affected_area: Optional[Decimal] = Field(None, description="Affected area in square kilometers")
    casualties: Optional[int] = Field(None, description="Number of casualties")
    displaced_persons: Optional[int] = Field(None, description="Number of displaced persons")
    economic_damage: Optional[Decimal] = Field(None, description="Estimated economic damage")
    infrastructure_damage: Optional[List[str]] = Field(default_factory=list, description="Types of infrastructure damaged")
    weather_conditions: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Weather conditions")
    warning_issued: bool = Field(default=False, description="Whether early warning was issued")
    warning_time: Optional[datetime] = Field(None, description="Time warning was issued")
    
    @validator('magnitude')
    def validate_magnitude(cls, v):
        if v is not None and v < 0:
            raise ValueError('Magnitude cannot be negative')
        return v
    
    @validator('casualties', 'displaced_persons')
    def validate_people_counts(cls, v):
        if v is not None and v < 0:
            raise ValueError('People counts cannot be negative')
        return v


class CyberIncident(BaseModel):
    """Cyber security incident specific data."""
    
    event: DisruptionEvent = Field(..., description="Base disruption event")
    incident_type: str = Field(..., description="Type of cyber incident")
    attack_vector: Optional[str] = Field(None, description="Attack vector or method")
    targeted_systems: List[str] = Field(..., description="List of targeted systems")
    affected_organizations: List[str] = Field(..., description="List of affected organizations")
    data_compromised: bool = Field(default=False, description="Whether data was compromised")
    data_types: Optional[List[str]] = Field(default_factory=list, description="Types of data compromised")
    records_affected: Optional[int] = Field(None, description="Number of records affected")
    attribution: Optional[str] = Field(None, description="Attribution information")
    malware_family: Optional[str] = Field(None, description="Malware family if applicable")
    vulnerabilities_exploited: Optional[List[str]] = Field(default_factory=list, description="Vulnerabilities exploited")
    mitigation_actions: Optional[List[str]] = Field(default_factory=list, description="Mitigation actions taken")
    recovery_time: Optional[Decimal] = Field(None, description="Recovery time in hours")
    estimated_cost: Optional[Decimal] = Field(None, description="Estimated incident cost")
    
    @validator('records_affected')
    def validate_records_affected(cls, v):
        if v is not None and v < 0:
            raise ValueError('Records affected cannot be negative')
        return v


class GeopoliticalEvent(BaseModel):
    """Geopolitical event specific data."""
    
    event: DisruptionEvent = Field(..., description="Base disruption event")
    event_type: str = Field(..., description="Type of geopolitical event")
    countries_involved: List[str] = Field(..., description="Countries involved in the event")
    international_response: Optional[str] = Field(None, description="International community response")
    sanctions_imposed: bool = Field(default=False, description="Whether sanctions were imposed")
    trade_restrictions: Optional[List[str]] = Field(default_factory=list, description="Trade restrictions imposed")
    market_impact: Optional[str] = Field(None, description="Impact on financial markets")
    commodity_impact: Optional[Dict[str, str]] = Field(default_factory=dict, description="Impact on specific commodities")
    supply_chain_disruption: bool = Field(default=False, description="Supply chain disruption occurred")
    refugee_count: Optional[int] = Field(None, description="Number of refugees generated")
    military_involvement: bool = Field(default=False, description="Military involvement")
    diplomatic_status: Optional[str] = Field(None, description="Current diplomatic status")
    resolution_prospects: Optional[str] = Field(None, description="Prospects for resolution")
    
    @validator('refugee_count')
    def validate_refugee_count(cls, v):
        if v is not None and v < 0:
            raise ValueError('Refugee count cannot be negative')
        return v


class EconomicShock(BaseModel):
    """Economic shock specific data."""
    
    event: DisruptionEvent = Field(..., description="Base disruption event")
    shock_type: str = Field(..., description="Type of economic shock")
    affected_sectors: List[str] = Field(..., description="Economic sectors affected")
    gdp_impact: Optional[Decimal] = Field(None, description="Estimated GDP impact percentage")
    unemployment_impact: Optional[Decimal] = Field(None, description="Unemployment rate impact")
    inflation_impact: Optional[Decimal] = Field(None, description="Inflation rate impact")
    currency_impact: Optional[Decimal] = Field(None, description="Currency devaluation percentage")
    market_indices_impact: Optional[Dict[str, Decimal]] = Field(default_factory=dict, description="Stock market indices impact")
    commodity_prices_impact: Optional[Dict[str, Decimal]] = Field(default_factory=dict, description="Commodity price impacts")
    trade_volume_impact: Optional[Decimal] = Field(None, description="Trade volume impact percentage")
    fiscal_response: Optional[str] = Field(None, description="Government fiscal response")
    monetary_response: Optional[str] = Field(None, description="Central bank monetary response")
    recovery_timeframe: Optional[str] = Field(None, description="Expected recovery timeframe")
    systemic_risk: bool = Field(default=False, description="Poses systemic risk")
    
    @validator('gdp_impact', 'unemployment_impact', 'inflation_impact', 'currency_impact', 'trade_volume_impact')
    def validate_economic_impacts(cls, v):
        if v is not None and (v < -100 or v > 1000):
            raise ValueError('Economic impact percentages seem unrealistic')
        return v


class SupplyChainDisruption(BaseModel):
    """Supply chain disruption specific data."""
    
    event: DisruptionEvent = Field(..., description="Base disruption event")
    disruption_category: str = Field(..., description="Category of supply chain disruption")
    affected_nodes: List[str] = Field(..., description="Supply chain nodes affected")
    affected_products: List[str] = Field(..., description="Products affected")
    affected_routes: Optional[List[str]] = Field(default_factory=list, description="Transportation routes affected")
    capacity_reduction: Optional[Decimal] = Field(None, description="Capacity reduction percentage")
    delivery_delays: Optional[Decimal] = Field(None, description="Average delivery delays in hours")
    cost_increase: Optional[Decimal] = Field(None, description="Cost increase percentage")
    alternative_sources: Optional[List[str]] = Field(default_factory=list, description="Alternative supply sources")
    inventory_impact: Optional[str] = Field(None, description="Impact on inventory levels")
    customer_impact: Optional[str] = Field(None, description="Impact on customers")
    recovery_actions: Optional[List[str]] = Field(default_factory=list, description="Recovery actions taken")
    lessons_learned: Optional[List[str]] = Field(default_factory=list, description="Lessons learned")
    
    @validator('capacity_reduction', 'cost_increase')
    def validate_percentages(cls, v):
        if v is not None and (v < 0 or v > 100):
            raise ValueError('Percentages must be between 0 and 100')
        return v
    
    @validator('delivery_delays')
    def validate_delays(cls, v):
        if v is not None and v < 0:
            raise ValueError('Delivery delays cannot be negative')
        return v


class DisruptionImpact(BaseModel):
    """Impact assessment for disruption events."""
    
    event_id: str = Field(..., description="Related disruption event ID")
    impact_type: ImpactType = Field(..., description="Type of impact")
    sector: str = Field(..., description="Affected sector")
    impact_description: str = Field(..., description="Description of the impact")
    severity_score: Decimal = Field(..., description="Impact severity score")
    financial_impact: Optional[Decimal] = Field(None, description="Financial impact amount")
    operational_impact: Optional[str] = Field(None, description="Operational impact description")
    duration: Optional[Decimal] = Field(None, description="Impact duration in hours")
    affected_entities: Optional[List[str]] = Field(default_factory=list, description="Specific entities affected")
    mitigation_effectiveness: Optional[Decimal] = Field(None, description="Effectiveness of mitigation measures")
    recovery_status: Optional[str] = Field(None, description="Current recovery status")
    long_term_effects: Optional[str] = Field(None, description="Expected long-term effects")
    
    @validator('severity_score')
    def validate_severity_score(cls, v):
        if v < 0 or v > 100:
            raise ValueError('Severity score must be between 0 and 100')
        return v
    
    @validator('mitigation_effectiveness')
    def validate_mitigation_effectiveness(cls, v):
        if v is not None and (v < 0 or v > 100):
            raise ValueError('Mitigation effectiveness must be between 0 and 100')
        return v


class DisruptionSeverity(BaseModel):
    """Severity assessment framework for disruptions."""
    
    event_id: str = Field(..., description="Related disruption event ID")
    overall_severity: SeverityLevel = Field(..., description="Overall severity level")
    human_impact_score: Decimal = Field(..., description="Human impact severity score")
    economic_impact_score: Decimal = Field(..., description="Economic impact severity score")
    environmental_impact_score: Decimal = Field(..., description="Environmental impact severity score")
    infrastructure_impact_score: Decimal = Field(..., description="Infrastructure impact severity score")
    systemic_risk_score: Decimal = Field(..., description="Systemic risk score")
    cascading_effect_potential: Decimal = Field(..., description="Potential for cascading effects")
    recovery_complexity: Decimal = Field(..., description="Recovery complexity score")
    international_significance: bool = Field(default=False, description="International significance")
    media_attention_level: Optional[str] = Field(None, description="Level of media attention")
    
    @validator('human_impact_score', 'economic_impact_score', 'environmental_impact_score',
              'infrastructure_impact_score', 'systemic_risk_score', 'cascading_effect_potential',
              'recovery_complexity')
    def validate_impact_scores(cls, v):
        if v < 0 or v > 100:
            raise ValueError('Impact scores must be between 0 and 100')
        return v


class DisruptionCategory(BaseModel):
    """Categorization system for disruption events."""
    
    category_id: str = Field(..., description="Category identifier")
    name: str = Field(..., description="Category name")
    description: str = Field(..., description="Category description")
    parent_category: Optional[str] = Field(None, description="Parent category ID")
    subcategories: Optional[List[str]] = Field(default_factory=list, description="Subcategory IDs")
    typical_duration: Optional[str] = Field(None, description="Typical event duration")
    common_indicators: Optional[List[str]] = Field(default_factory=list, description="Common early indicators")
    impact_patterns: Optional[Dict[str, str]] = Field(default_factory=dict, description="Typical impact patterns")
    mitigation_strategies: Optional[List[str]] = Field(default_factory=list, description="Common mitigation strategies")
    historical_frequency: Optional[str] = Field(None, description="Historical frequency of occurrence")
    
    @validator('category_id')
    def validate_category_id(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Category ID cannot be empty')
        return v.strip().upper()


class DisruptionAlert(BaseModel):
    """Alert system for disruption events."""
    
    alert_id: str = Field(..., description="Unique alert identifier")
    event_id: Optional[str] = Field(None, description="Related event ID if applicable")
    alert_level: AlertLevel = Field(..., description="Alert level")
    title: str = Field(..., description="Alert title")
    message: str = Field(..., description="Alert message")
    affected_regions: List[str] = Field(..., description="Regions covered by alert")
    valid_from: datetime = Field(..., description="Alert valid from time")
    valid_until: Optional[datetime] = Field(None, description="Alert valid until time")
    issuing_authority: str = Field(..., description="Authority issuing the alert")
    alert_type: str = Field(..., description="Type of alert")
    recommended_actions: Optional[List[str]] = Field(default_factory=list, description="Recommended actions")
    contact_information: Optional[Dict[str, str]] = Field(default_factory=dict, description="Emergency contact information")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last update time")
    
    @validator('alert_id')
    def validate_alert_id(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Alert ID cannot be empty')
        return v.strip().upper()
    
    @root_validator
    def validate_alert_timeframe(cls, values):
        valid_from = values.get('valid_from')
        valid_until = values.get('valid_until')
        
        if valid_until and valid_from and valid_until < valid_from:
            raise ValueError('Alert valid until time cannot be before valid from time')
        
        return values


class DisruptionSummary(BaseModel):
    """Summary statistics for disruption events."""
    
    summary_period_start: date = Field(..., description="Summary period start date")
    summary_period_end: date = Field(..., description="Summary period end date")
    total_events: int = Field(..., description="Total number of events")
    events_by_type: Dict[str, int] = Field(..., description="Event count by type")
    events_by_severity: Dict[str, int] = Field(..., description="Event count by severity")
    average_duration: Optional[Decimal] = Field(None, description="Average event duration in hours")
    total_economic_impact: Optional[Decimal] = Field(None, description="Total economic impact")
    most_affected_regions: List[str] = Field(default_factory=list, description="Most affected regions")
    emerging_threats: Optional[List[str]] = Field(default_factory=list, description="Emerging threat patterns")
    trend_analysis: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Trend analysis results")
    alert_statistics: Optional[Dict[str, int]] = Field(default_factory=dict, description="Alert issuance statistics")
    recovery_statistics: Optional[Dict[str, Decimal]] = Field(default_factory=dict, description="Recovery time statistics")
    
    @validator('total_events')
    def validate_total_events(cls, v):
        if v < 0:
            raise ValueError('Total events cannot be negative')
        return v