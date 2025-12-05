from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from app.db import Base
import uuid
from datetime import datetime

class ObservationModel(Base):
    __tablename__ = 'observations'
    id = Column(Integer, primary_key=True, index=True)
    series_id = Column(String, index=True)
    observed_at = Column(DateTime)
    value = Column(Float)
    
    # Data lineage fields per architecture requirements
    source = Column(String, index=True)  # Provider name (fred, eia, etc.)
    source_url = Column(String)  # Original API endpoint
    fetched_at = Column(DateTime, index=True)  # When data was retrieved
    checksum = Column(String)  # Data integrity verification
    derivation_flag = Column(String)  # raw, derived, or blended
    
    # TTL tracking for cache management
    soft_ttl = Column(Integer)  # Seconds until background refresh
    hard_ttl = Column(Integer)  # Seconds until marked stale

class SubmissionModel(Base):
    __tablename__ = 'submissions'
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    title = Column(String, nullable=False)
    summary = Column(Text)
    author = Column(String, nullable=False)
    author_email = Column(String, nullable=False)
    content_url = Column(String)
    submission_type = Column(String, nullable=False)  # analysis, lab, policy
    status = Column(String, default='pending')  # pending, approved, rejected
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime)
    meta_data = Column(JSON)

class JudgingLogModel(Base):
    __tablename__ = 'judging_logs'
    id = Column(Integer, primary_key=True, index=True)
    submission_id = Column(String, nullable=False)
    judge_id = Column(String, nullable=False)
    action = Column(String, nullable=False)  # approve, reject, comment
    notes = Column(Text)
    timestamp = Column(DateTime, nullable=False)
    meta_data = Column(JSON)

class TransparencyLogModel(Base):
    __tablename__ = 'transparency_logs'
    id = Column(Integer, primary_key=True, index=True)
    event_type = Column(String, nullable=False)  # data_update, model_retrain, system_change
    description = Column(Text)
    timestamp = Column(DateTime, nullable=False)
    meta_data = Column(JSON)

class ModelMetadataModel(Base):
    __tablename__ = 'model_metadata'
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, nullable=False, unique=True)
    version = Column(String, nullable=False)
    trained_at = Column(DateTime, nullable=False)
    training_window_start = Column(DateTime, nullable=False)
    training_window_end = Column(DateTime, nullable=False)
    performance_metrics = Column(JSON)
    is_active = Column(Boolean, default=False)
    file_path = Column(String)  # Path to the .pkl file
    created_at = Column(DateTime, nullable=False)

# Analytics Models
class UserMetrics(Base):
    __tablename__ = "user_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    page_path = Column(String(500), index=True)
    user_agent = Column(Text)
    referrer = Column(String(500))
    viewport = Column(String(100))
    country = Column(String(100))
    device_type = Column(String(50))
    session_duration = Column(Float)

class PageView(Base):
    __tablename__ = "page_views"
    
    id = Column(Integer, primary_key=True, index=True)
    path = Column(String(500), index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    user_agent = Column(Text)
    referrer = Column(String(500))
    viewport = Column(String(100))

class UserEvent(Base):
    __tablename__ = "user_events"
    
    id = Column(Integer, primary_key=True, index=True)
    event_name = Column(String(255), index=True)
    event_data = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    path = Column(String(500))
    user_session = Column(String(255))

class UserFeedback(Base):
    __tablename__ = "user_feedback"
    
    id = Column(Integer, primary_key=True, index=True)
    page = Column(String(500), index=True)
    rating = Column(Integer)
    comment = Column(Text)
    category = Column(String(100))
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    user_session = Column(String(255))

# Community Models
class CommunityUser(Base):
    __tablename__ = "community_users"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    username = Column(String(100), nullable=False, unique=True, index=True)
    email = Column(String(255), nullable=False, unique=True, index=True)
    full_name = Column(String(255))
    title = Column(String(255))
    company = Column(String(255))
    bio = Column(Text)
    verified = Column(Boolean, default=False)
    professional_category = Column(String(100))  # portfolio-manager, supply-chain-director, risk-analyst, etc
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow)
    
    # Professional verification
    verification_document_url = Column(String(500))
    verification_status = Column(String(50), default="pending")  # pending, verified, rejected
    verified_at = Column(DateTime)
    
    # Relationships
    insights = relationship("CommunityInsight", back_populates="user")
    comments = relationship("InsightComment", back_populates="user") 
    likes = relationship("InsightLike", back_populates="user")

class CommunityInsight(Base):
    __tablename__ = "community_insights"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    title = Column(String(500), nullable=False, index=True)
    content = Column(Text, nullable=False)
    category = Column(String(100), nullable=False, index=True)  # market-analysis, supply-chain, etc
    risk_score = Column(Float, index=True)
    impact_level = Column(String(50), index=True)  # low, medium, high, critical
    tags = Column(JSON)  # Array of tag strings
    
    # Author information
    author_id = Column(String, ForeignKey("community_users.id"), nullable=False)
    
    # Engagement metrics
    likes_count = Column(Integer, default=0, index=True)
    comments_count = Column(Integer, default=0, index=True)
    shares_count = Column(Integer, default=0, index=True)
    views_count = Column(Integer, default=0, index=True)
    
    # Content moderation
    status = Column(String(50), default="published")  # draft, published, under_review, hidden
    moderated_at = Column(DateTime)
    moderated_by = Column(String)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow)
    published_at = Column(DateTime)
    
    # Relationships
    user = relationship("CommunityUser", back_populates="insights")
    comments = relationship("InsightComment", back_populates="insight", cascade="all, delete-orphan")
    likes = relationship("InsightLike", back_populates="insight", cascade="all, delete-orphan")

class InsightComment(Base):
    __tablename__ = "insight_comments"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    insight_id = Column(String, ForeignKey("community_insights.id"), nullable=False)
    user_id = Column(String, ForeignKey("community_users.id"), nullable=False)
    content = Column(Text, nullable=False)
    parent_comment_id = Column(String, ForeignKey("insight_comments.id"))  # For threaded comments
    
    # Engagement
    likes_count = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    insight = relationship("CommunityInsight", back_populates="comments")
    user = relationship("CommunityUser", back_populates="comments")
    parent = relationship("InsightComment", remote_side=[id])

class InsightLike(Base):
    __tablename__ = "insight_likes"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    insight_id = Column(String, ForeignKey("community_insights.id"), nullable=False)
    user_id = Column(String, ForeignKey("community_users.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationships
    insight = relationship("CommunityInsight", back_populates="likes")
    user = relationship("CommunityUser", back_populates="likes")

# Weekly Intelligence Models
class WeeklyBrief(Base):
    __tablename__ = "weekly_briefs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    week_start_date = Column(DateTime, nullable=False, index=True)
    week_end_date = Column(DateTime, nullable=False)
    title = Column(String(500), nullable=False)
    executive_summary = Column(Text, nullable=False)
    sections = Column(JSON, nullable=False)  # Array of section objects
    
    # Data snapshot at time of generation
    geri_score = Column(Float, index=True)
    current_regime = Column(String(100))
    forecast_delta = Column(Float)
    
    # Generation metadata
    format_type = Column(String(50))  # executive, detailed, technical
    generated_at = Column(DateTime, default=datetime.utcnow, index=True)
    generated_by = Column(String)  # system or user_id
    version = Column(String(50))
    
    # Distribution
    subscribers_count = Column(Integer, default=0)
    sent_at = Column(DateTime)
    pdf_url = Column(String(500))
    
    # Status
    status = Column(String(50), default="draft")  # draft, published, sent

class WeeklyBriefSubscription(Base):
    __tablename__ = "weekly_brief_subscriptions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    email = Column(String(255), nullable=False, unique=True, index=True)
    user_id = Column(String, ForeignKey("community_users.id"))  # Optional - can subscribe without account
    format_preference = Column(String(50), default="executive")  # executive, detailed, technical
    
    # Professional information
    name = Column(String(255))
    company = Column(String(255))
    title = Column(String(255))
    
    # Subscription settings
    active = Column(Boolean, default=True)
    subscribed_at = Column(DateTime, default=datetime.utcnow, index=True)
    unsubscribed_at = Column(DateTime)
    last_sent_at = Column(DateTime)
    
    # Email preferences
    send_time_preference = Column(String(50), default="monday_8am")
    timezone = Column(String(100), default="UTC")


# ================================
# Supply Chain Risk Models
# ================================

class SupplyChainNode(Base):
    """Represents entities (companies, ports, regions) in the supply chain network"""
    __tablename__ = "supply_chain_nodes"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    node_type = Column(String(50), nullable=False, index=True)  # company, port, region, sector
    name = Column(String(500), nullable=False, index=True)
    identifier = Column(String(255), unique=True, index=True)  # External ID (LEI, port code, etc.)
    
    # Geographic information
    country = Column(String(100), index=True)
    region = Column(String(100), index=True)
    latitude = Column(Float)
    longitude = Column(Float)
    
    # Classification
    industry_sector = Column(String(100), index=True)
    sub_sector = Column(String(100))
    tier_level = Column(Integer, index=True)  # Supply chain tier (1=direct supplier, 2=sub-supplier)
    
    # Risk metrics
    overall_risk_score = Column(Float, index=True)
    financial_health_score = Column(Float)
    operational_risk_score = Column(Float)
    geopolitical_risk_score = Column(Float)
    
    # Metadata
    data_sources = Column(JSON)  # List of data sources
    last_updated = Column(DateTime, default=datetime.utcnow, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True, index=True)
    
    # Relationships
    upstream_relationships = relationship("SupplyChainRelationship", foreign_keys="SupplyChainRelationship.downstream_node_id", back_populates="downstream_node")
    downstream_relationships = relationship("SupplyChainRelationship", foreign_keys="SupplyChainRelationship.upstream_node_id", back_populates="upstream_node")


class SupplyChainRelationship(Base):
    """Represents connections between nodes in the supply chain"""
    __tablename__ = "supply_chain_relationships"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    upstream_node_id = Column(String, ForeignKey("supply_chain_nodes.id"), nullable=False)
    downstream_node_id = Column(String, ForeignKey("supply_chain_nodes.id"), nullable=False)
    
    # Relationship characteristics
    relationship_type = Column(String(50), nullable=False)  # supplier, customer, partner, dependency
    relationship_strength = Column(Float)  # 0-1, strength of dependency
    trade_volume_usd = Column(Float)
    percentage_of_downstream_supply = Column(Float)  # What % of downstream's supply comes from upstream
    
    # Risk factors
    criticality_score = Column(Float, index=True)
    vulnerability_score = Column(Float)
    alternative_suppliers_count = Column(Integer)
    substitution_difficulty = Column(String(50))  # low, medium, high, impossible
    
    # Temporal data
    established_date = Column(DateTime)
    last_transaction_date = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True, index=True)
    
    # Relationships
    upstream_node = relationship("SupplyChainNode", foreign_keys=[upstream_node_id], back_populates="downstream_relationships")
    downstream_node = relationship("SupplyChainNode", foreign_keys=[downstream_node_id], back_populates="upstream_relationships")


class CascadeEvent(Base):
    """Records supply chain disruption events and cascading impacts"""
    __tablename__ = "cascade_events"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    cascade_id = Column(String, nullable=False, index=True)  # Groups related events
    event_type = Column(String(100), nullable=False, index=True)  # disruption, secondary_impact, recovery
    severity = Column(String(50), nullable=False, index=True)  # low, medium, high, critical
    
    # Event details
    title = Column(String(500), nullable=False)
    description = Column(Text)
    trigger_event = Column(String(500))  # What caused this event
    
    # Geographic and sector impact
    affected_countries = Column(JSON)  # List of country codes
    affected_regions = Column(JSON)  # List of region names
    affected_sectors = Column(JSON)  # List of industry sectors
    affected_nodes = Column(JSON)  # List of supply chain node IDs
    
    # Impact metrics
    estimated_cost_usd = Column(Float)
    affected_companies_count = Column(Integer)
    supply_disruption_percentage = Column(Float)
    recovery_time_days = Column(Integer)
    
    # Temporal data
    event_start = Column(DateTime, nullable=False, index=True)
    event_end = Column(DateTime, index=True)
    detected_at = Column(DateTime, default=datetime.utcnow)
    recovery_start = Column(DateTime)
    full_recovery_date = Column(DateTime)
    
    # Cascade propagation
    propagation_speed = Column(String(50))  # immediate, hours, days, weeks
    cascade_depth = Column(Integer)  # How many degrees of separation affected
    parent_event_id = Column(String, ForeignKey("cascade_events.id"))  # For cascading events
    
    # Data provenance
    data_sources = Column(JSON)
    confidence_level = Column(Float)  # 0-100
    detection_method = Column(String(100))  # manual, automated, ml_prediction
    
    # Status
    status = Column(String(50), default="active", index=True)  # active, resolved, ongoing
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    parent_event = relationship("CascadeEvent", remote_side=[id])
    child_events = relationship("CascadeEvent", overlaps="parent_event")


class SectorVulnerabilityAssessment(Base):
    """Stores comprehensive vulnerability assessments for industry sectors"""
    __tablename__ = "sector_vulnerability_assessments"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    assessment_id = Column(String, unique=True, nullable=False, index=True)
    sector = Column(String(100), nullable=False, index=True)
    sector_name = Column(String(200), nullable=False)
    
    # Assessment metadata
    assessment_date = Column(DateTime, nullable=False, index=True)
    assessment_version = Column(String(50), default="1.0")
    assessment_methodology = Column(String(100))
    
    # Overall metrics
    overall_risk_score = Column(Float, nullable=False, index=True)
    vulnerability_count = Column(Integer, default=0)
    critical_vulnerabilities = Column(Integer, default=0)
    high_vulnerabilities = Column(Integer, default=0)
    medium_vulnerabilities = Column(Integer, default=0)
    low_vulnerabilities = Column(Integer, default=0)
    
    # Sector profile scores
    complexity_score = Column(Float)
    globalization_index = Column(Float)
    regulatory_burden = Column(Float)
    technology_dependency = Column(Float)
    environmental_sensitivity = Column(Float)
    geopolitical_exposure = Column(Float)
    
    # Sector characteristics
    key_suppliers = Column(JSON)  # List of critical supplier countries/companies
    critical_regions = Column(JSON)  # Geopolitically sensitive areas
    primary_risks = Column(JSON)  # List of main risk categories
    seasonal_factors = Column(JSON)  # Time-based vulnerabilities
    compliance_requirements = Column(JSON)  # Regulatory frameworks
    
    # Recommendations
    recommendations = Column(JSON)  # List of mitigation recommendations
    next_assessment_date = Column(DateTime, index=True)
    
    # Data provenance
    data_sources = Column(JSON)
    analyst_notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True, index=True)
    
    # Relationships
    vulnerabilities = relationship("SectorVulnerability", back_populates="assessment")


class SectorVulnerability(Base):
    """Individual vulnerabilities identified in sector assessments"""
    __tablename__ = "sector_vulnerabilities"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    vulnerability_id = Column(String, unique=True, nullable=False, index=True)
    assessment_id = Column(String, ForeignKey("sector_vulnerability_assessments.id"), nullable=False)
    
    # Vulnerability details
    category = Column(String(100), nullable=False, index=True)  # supply_chain, cyber, regulatory, etc.
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=False)
    severity = Column(String(50), nullable=False, index=True)
    
    # Risk metrics
    likelihood = Column(Float, nullable=False)  # 0-100
    impact_score = Column(Float, nullable=False)  # 0-100
    risk_score = Column(Float, nullable=False, index=True)  # Calculated from likelihood * impact
    
    # Impact details
    affected_regions = Column(JSON)
    critical_components = Column(JSON)  # Key systems/processes at risk
    dependencies = Column(JSON)  # External dependencies that create vulnerability
    
    # Mitigation
    mitigation_strategies = Column(JSON)  # List of recommended actions
    mitigation_complexity = Column(String(50))  # low, medium, high
    estimated_mitigation_cost_usd = Column(Float)
    
    # Temporal factors
    time_horizon_days = Column(Integer)  # How soon this could manifest
    seasonal_variation = Column(Boolean, default=False)
    confidence_level = Column(Float)  # 0-100
    
    # Data provenance
    last_assessed = Column(DateTime, default=datetime.utcnow)
    data_sources = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True, index=True)
    
    # Relationships
    assessment = relationship("SectorVulnerabilityAssessment", back_populates="vulnerabilities")


class TimelineCascadeEvent(Base):
    """Timeline view of cascade events for visualization"""
    __tablename__ = "timeline_cascade_events"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    event_id = Column(String, unique=True, nullable=False, index=True)
    cascade_id = Column(String, nullable=False, index=True)
    
    # Event classification
    event_type = Column(String(100), nullable=False, index=True)  # initial_disruption, secondary_impact, recovery
    severity = Column(String(50), nullable=False, index=True)
    phase = Column(String(100), nullable=False, index=True)  # onset, propagation, peak_impact, recovery
    
    # Event details
    title = Column(String(500), nullable=False)
    description = Column(Text)
    timestamp = Column(DateTime, nullable=False, index=True)
    
    # Impact scope
    affected_entities = Column(JSON)  # Companies, regions, sectors affected
    affected_sectors = Column(JSON)
    affected_regions = Column(JSON)
    
    # Impact metrics
    impact_metrics = Column(JSON)  # Cost, disruption duration, etc.
    location_data = Column(JSON)  # Geographic coordinates if applicable
    
    # Event relationships
    related_events = Column(JSON)  # IDs of related events
    cascade_triggers = Column(JSON)  # What this event triggered
    
    # Data quality
    source = Column(String(200), nullable=False)
    confidence_level = Column(Float)  # 0-100
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)


class SPGlobalIntelligence(Base):
    """S&P Global market intelligence data"""
    __tablename__ = "sp_global_intelligence"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    data_type = Column(String(100), nullable=False, index=True)  # supplier_risk, market_intel, financial_health
    entity_id = Column(String, nullable=False, index=True)  # Company/supplier identifier
    
    # Entity details
    entity_name = Column(String(500), nullable=False)
    entity_type = Column(String(100))  # company, subsidiary, division
    country = Column(String(100), index=True)
    industry_sector = Column(String(100), index=True)
    
    # Risk assessment data
    overall_risk_level = Column(String(50), index=True)  # low, medium, high, critical
    financial_health_score = Column(Float)
    operational_risk_score = Column(Float)
    compliance_risk_score = Column(Float)
    cyber_risk_score = Column(Float)
    
    # Financial metrics
    revenue_usd = Column(Float)
    market_cap_usd = Column(Float)
    debt_to_equity_ratio = Column(Float)
    credit_rating = Column(String(50))
    
    # Risk factors
    risk_factors = Column(JSON)  # List of identified risk factors
    esg_score = Column(Float)
    governance_score = Column(Float)
    
    # Market intelligence
    market_trends = Column(JSON)
    competitive_position = Column(String(100))
    growth_outlook = Column(String(100))
    
    # Data metadata
    data_as_of_date = Column(DateTime, nullable=False, index=True)
    sp_data_source = Column(String(200))  # Which S&P dataset
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    is_current = Column(Boolean, default=True, index=True)


class ResilienceMetric(Base):
    """Supply chain resilience metrics and scores"""
    __tablename__ = "resilience_metrics"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    metric_id = Column(String, unique=True, nullable=False, index=True)
    entity_id = Column(String, nullable=False, index=True)  # What this metric applies to
    entity_type = Column(String(100), nullable=False, index=True)  # company, sector, region, network
    
    # Metric details
    metric_name = Column(String(200), nullable=False)
    metric_category = Column(String(100), nullable=False, index=True)  # diversity, redundancy, adaptability
    
    # Resilience scores
    overall_resilience_score = Column(Float, nullable=False, index=True)  # 0-100
    supplier_diversity_score = Column(Float)
    geographic_distribution_score = Column(Float)
    financial_stability_score = Column(Float)
    operational_flexibility_score = Column(Float)
    risk_management_maturity = Column(Float)
    
    # Component metrics
    supplier_concentration = Column(Float)  # Herfindahl index or similar
    geographic_concentration = Column(Float)
    critical_dependency_count = Column(Integer)
    alternative_sources_count = Column(Integer)
    recovery_time_estimate_days = Column(Integer)
    
    # Assessment details
    assessment_methodology = Column(String(200))
    assessment_date = Column(DateTime, nullable=False, index=True)
    assessment_period_start = Column(DateTime)
    assessment_period_end = Column(DateTime)
    
    # Recommendations
    improvement_recommendations = Column(JSON)
    priority_actions = Column(JSON)
    estimated_improvement_cost_usd = Column(Float)
    
    # Data quality
    data_completeness_percentage = Column(Float)
    confidence_level = Column(Float)
    data_sources = Column(JSON)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True, index=True)


class RealTimeDataFeed(Base):
    """Tracks real-time data integration and refresh status"""
    __tablename__ = "realtime_data_feeds"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    feed_name = Column(String(200), nullable=False, unique=True, index=True)
    feed_type = Column(String(100), nullable=False, index=True)  # api, websocket, file_upload
    data_source = Column(String(200), nullable=False, index=True)  # free_geopolitical_intelligence, comtrade, sp_global, etc.
    
    # Configuration
    endpoint_url = Column(String(1000))
    refresh_frequency_minutes = Column(Integer, default=60)
    priority = Column(String(50), default="medium", index=True)  # low, medium, high, critical
    
    # Status tracking
    status = Column(String(50), default="active", index=True)  # active, paused, error, maintenance
    last_successful_refresh = Column(DateTime, index=True)
    last_refresh_attempt = Column(DateTime, index=True)
    next_scheduled_refresh = Column(DateTime, index=True)
    
    # Performance metrics
    success_rate_percentage = Column(Float, default=100.0)
    average_refresh_time_seconds = Column(Float)
    total_refreshes = Column(Integer, default=0)
    successful_refreshes = Column(Integer, default=0)
    failed_refreshes = Column(Integer, default=0)
    
    # Data statistics
    last_record_count = Column(Integer)
    total_records_processed = Column(Integer, default=0)
    data_quality_score = Column(Float)
    
    # Error tracking
    last_error_message = Column(Text)
    last_error_timestamp = Column(DateTime)
    consecutive_failures = Column(Integer, default=0)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(String(200))
    is_active = Column(Boolean, default=True, index=True)


class DataRefreshLog(Base):
    """Log of all data refresh operations"""
    __tablename__ = "data_refresh_logs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    feed_id = Column(String, ForeignKey("realtime_data_feeds.id"), nullable=False)
    refresh_id = Column(String, unique=True, nullable=False, index=True)
    
    # Refresh details
    refresh_type = Column(String(50), nullable=False)  # scheduled, manual, retry
    triggered_by = Column(String(200))  # system, user_id, error_recovery
    
    # Execution details
    started_at = Column(DateTime, nullable=False, index=True)
    completed_at = Column(DateTime, index=True)
    duration_seconds = Column(Float)
    status = Column(String(50), nullable=False, index=True)  # running, completed, failed, cancelled
    
    # Results
    records_retrieved = Column(Integer, default=0)
    records_processed = Column(Integer, default=0)
    records_updated = Column(Integer, default=0)
    records_inserted = Column(Integer, default=0)
    records_failed = Column(Integer, default=0)
    
    # Change detection
    data_hash_before = Column(String(255))
    data_hash_after = Column(String(255))
    changes_detected = Column(Boolean, default=False, index=True)
    
    # Error details
    error_message = Column(Text)
    error_code = Column(String(50))
    retry_count = Column(Integer, default=0)
    
    # Performance metrics
    api_response_time_ms = Column(Float)
    processing_time_ms = Column(Float)
    memory_usage_mb = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    feed = relationship("RealTimeDataFeed")


# ================================
# External Data Integration Models
# ================================

class GeopoliticalEvent(Base):
    """Free geopolitical intelligence event data"""
    __tablename__ = "geopolitical_events"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    event_id = Column(String, unique=True, nullable=False, index=True)
    
    # Event details
    event_type = Column(String(200), nullable=False, index=True)
    sub_event_type = Column(String(200), index=True)
    event_date = Column(DateTime, nullable=False, index=True)
    
    # Location
    country = Column(String(100), nullable=False, index=True)
    region = Column(String(200), index=True)
    location = Column(String(500))
    latitude = Column(Float)
    longitude = Column(Float)
    
    # Impact assessment
    fatalities = Column(Integer, default=0)
    severity_score = Column(Float)  # Our calculated severity
    supply_chain_relevance = Column(Boolean, default=False, index=True)
    economic_impact_estimate = Column(Float)
    
    # Classification
    actors_involved = Column(JSON)
    sectors_affected = Column(JSON)  # Industries potentially impacted
    transportation_impact = Column(Boolean, default=False)
    port_impact = Column(Boolean, default=False)
    
    # Notes and sources
    notes = Column(Text)
    source_scale = Column(String(200))
    data_sources = Column(JSON)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True, index=True)


class ComtradeTradeFlow(Base):
    """UN Comtrade international trade flow data"""
    __tablename__ = "comtrade_trade_flows"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    trade_flow_id = Column(String, unique=True, nullable=False, index=True)
    
    # Trade relationship
    reporter_country = Column(String(100), nullable=False, index=True)
    partner_country = Column(String(100), nullable=False, index=True)
    trade_flow_direction = Column(String(50), nullable=False, index=True)  # import, export
    
    # Product classification
    commodity_code = Column(String(50), nullable=False, index=True)
    commodity_description = Column(String(1000))
    
    # Trade values
    trade_value_usd = Column(Float, nullable=False)
    quantity = Column(Float)
    quantity_unit = Column(String(50))
    
    # Temporal data
    reference_year = Column(Integer, nullable=False, index=True)
    reference_month = Column(Integer, index=True)
    reference_period = Column(String(50))
    
    # Supply chain context
    supply_chain_criticality = Column(String(50))  # low, medium, high, critical
    sector_relevance = Column(JSON)  # Industries dependent on this trade flow
    
    # Data provenance
    data_source = Column(String(100), default="UN_Comtrade")
    data_quality_flag = Column(String(50))
    last_updated = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)


class WTOTradeStatistic(Base):
    """WTO trade statistics and market access data"""
    __tablename__ = "wto_trade_statistics"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    statistic_id = Column(String, unique=True, nullable=False, index=True)
    
    # Geographic scope
    country = Column(String(100), nullable=False, index=True)
    partner_country = Column(String(100), index=True)  # Null for total trade stats
    region = Column(String(200), index=True)
    
    # Trade metrics
    metric_type = Column(String(100), nullable=False, index=True)  # total_trade, market_access, tariff_rate
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String(50))
    
    # Product/sector classification
    sector = Column(String(100), index=True)
    product_category = Column(String(200))
    
    # Temporal data
    reference_year = Column(Integer, nullable=False, index=True)
    reference_quarter = Column(Integer, index=True)
    
    # Supply chain impact
    supply_chain_relevance_score = Column(Float)  # How relevant to supply chain risk
    critical_supply_indicator = Column(Boolean, default=False, index=True)
    
    # Data provenance
    data_source = Column(String(100), default="WTO")
    last_updated = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
