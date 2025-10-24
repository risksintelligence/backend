"""
Network analysis database models for risk propagation and vulnerability assessment.
"""
from sqlalchemy import Column, Integer, Float, String, DateTime, Text, Boolean, JSON, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from src.core.database import Base


class NetworkNode(Base):
    """Network nodes representing entities in the risk network."""
    
    __tablename__ = "network_nodes"
    
    id = Column(Integer, primary_key=True, index=True)
    node_id = Column(String(100), unique=True, index=True, nullable=False)
    name = Column(String(200), nullable=False)
    node_type = Column(String(50), nullable=False, index=True)  # 'company', 'sector', 'country', 'asset'
    
    # Network position
    x_position = Column(Float)
    y_position = Column(Float)
    z_position = Column(Float)  # For 3D networks
    
    # Node properties
    size = Column(Float, default=1.0)  # Visual size indicator
    weight = Column(Float, default=1.0)  # Importance weight
    risk_level = Column(Float, default=0.0)  # Current risk level (0-100)
    
    # Centrality measures
    betweenness_centrality = Column(Float, default=0.0)
    closeness_centrality = Column(Float, default=0.0)
    eigenvector_centrality = Column(Float, default=0.0)
    pagerank_centrality = Column(Float, default=0.0)
    
    # Risk metrics
    vulnerability_score = Column(Float, default=0.0)
    influence_score = Column(Float, default=0.0)
    systemic_importance = Column(Float, default=0.0)
    
    # Status
    is_active = Column(Boolean, default=True)
    is_critical = Column(Boolean, default=False)
    
    # Metadata
    attributes = Column(JSON)  # Additional node attributes
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class NetworkEdge(Base):
    """Network edges representing connections between nodes."""
    
    __tablename__ = "network_edges"
    
    id = Column(Integer, primary_key=True, index=True)
    edge_id = Column(String(100), unique=True, index=True, nullable=False)
    source_node_id = Column(String(100), ForeignKey('network_nodes.node_id'), nullable=False, index=True)
    target_node_id = Column(String(100), ForeignKey('network_nodes.node_id'), nullable=False, index=True)
    
    # Edge properties
    edge_type = Column(String(50), nullable=False, index=True)  # 'supplier', 'customer', 'dependency', 'correlation'
    weight = Column(Float, default=1.0)  # Connection strength
    direction = Column(String(20), default='undirected')  # 'directed', 'undirected', 'bidirectional'
    
    # Risk propagation
    propagation_probability = Column(Float, default=0.0)  # Probability of risk transmission
    propagation_delay = Column(Integer, default=0)  # Delay in risk transmission (hours)
    amplification_factor = Column(Float, default=1.0)  # Risk amplification through this edge
    
    # Status
    is_active = Column(Boolean, default=True)
    is_critical = Column(Boolean, default=False)
    
    # Metadata
    attributes = Column(JSON)  # Additional edge attributes
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    source_node = relationship("NetworkNode", foreign_keys=[source_node_id])
    target_node = relationship("NetworkNode", foreign_keys=[target_node_id])


class NetworkSnapshot(Base):
    """Snapshots of network state for temporal analysis."""
    
    __tablename__ = "network_snapshots"
    
    id = Column(Integer, primary_key=True, index=True)
    snapshot_id = Column(String(100), unique=True, index=True, nullable=False)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    
    # Network metrics
    node_count = Column(Integer, nullable=False)
    edge_count = Column(Integer, nullable=False)
    density = Column(Float, nullable=False)
    clustering_coefficient = Column(Float)
    average_path_length = Column(Float)
    diameter = Column(Integer)
    
    # Risk metrics
    overall_risk_level = Column(Float, default=0.0)
    systemic_risk_score = Column(Float, default=0.0)
    vulnerability_index = Column(Float, default=0.0)
    resilience_score = Column(Float, default=0.0)
    
    # Critical components
    critical_nodes = Column(JSON)  # List of critical node IDs
    critical_edges = Column(JSON)  # List of critical edge IDs
    single_points_of_failure = Column(JSON)  # SPOF analysis
    
    # Snapshot metadata
    network_type = Column(String(50), default='risk_network')
    snapshot_timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    is_baseline = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class RiskPropagationSimulation(Base):
    """Risk propagation simulation results."""
    
    __tablename__ = "risk_propagation_simulations"
    
    id = Column(Integer, primary_key=True, index=True)
    simulation_id = Column(String(100), unique=True, index=True, nullable=False)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    
    # Simulation parameters
    initial_shock_nodes = Column(JSON, nullable=False)  # Nodes where shock originated
    shock_magnitude = Column(Float, nullable=False)  # Initial shock strength
    simulation_steps = Column(Integer, default=100)
    time_horizon = Column(Integer, default=24)  # Hours
    
    # Results
    affected_nodes = Column(JSON)  # Nodes affected by propagation
    propagation_path = Column(JSON)  # Path of risk propagation
    final_network_state = Column(JSON)  # Final state of all nodes
    
    # Metrics
    total_affected_nodes = Column(Integer)
    max_propagation_distance = Column(Integer)
    cascade_threshold = Column(Float)
    recovery_time_estimate = Column(Integer)  # Hours to recovery
    
    # Analysis
    bottleneck_nodes = Column(JSON)  # Nodes that acted as bottlenecks
    amplification_paths = Column(JSON)  # Paths with high amplification
    containment_strategies = Column(JSON)  # Suggested containment measures
    
    # Simulation metadata
    algorithm_used = Column(String(50), default='monte_carlo')
    confidence_interval = Column(Float, default=0.95)
    
    # Timestamps
    simulation_timestamp = Column(DateTime(timezone=True), server_default=func.now())
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class NetworkVulnerabilityAssessment(Base):
    """Network vulnerability assessment results."""
    
    __tablename__ = "network_vulnerability_assessments"
    
    id = Column(Integer, primary_key=True, index=True)
    assessment_id = Column(String(100), unique=True, index=True, nullable=False)
    network_snapshot_id = Column(String(100), ForeignKey('network_snapshots.snapshot_id'), nullable=False)
    
    # Overall vulnerability
    overall_vulnerability_score = Column(Float, nullable=False)
    vulnerability_level = Column(String(20), nullable=False)  # 'low', 'medium', 'high', 'critical'
    
    # Specific vulnerabilities
    single_points_of_failure = Column(JSON)  # Critical nodes/edges
    cascade_vulnerabilities = Column(JSON)  # Cascade failure risks
    structural_vulnerabilities = Column(JSON)  # Network structure weaknesses
    
    # Recommendations
    mitigation_strategies = Column(JSON)  # Suggested mitigation measures
    redundancy_recommendations = Column(JSON)  # Redundancy improvements
    monitoring_priorities = Column(JSON)  # Priority monitoring targets
    
    # Assessment metadata
    assessment_method = Column(String(50), default='comprehensive')
    confidence_score = Column(Float, default=0.8)
    
    # Timestamps
    assessment_timestamp = Column(DateTime(timezone=True), server_default=func.now())
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    network_snapshot = relationship("NetworkSnapshot", foreign_keys=[network_snapshot_id])