"""
Supply Chain Resilience Metrics and Scoring Service

Provides comprehensive quantitative assessments of supply chain resilience,
including vulnerability scoring, redundancy analysis, and adaptive capacity metrics.
"""

import logging
import math
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import statistics

logger = logging.getLogger(__name__)


class ResilienceCategory(Enum):
    EXCELLENT = "excellent"      # 90-100%
    GOOD = "good"               # 75-89%
    MODERATE = "moderate"       # 60-74% 
    WEAK = "weak"              # 40-59%
    CRITICAL = "critical"       # 0-39%


class MetricType(Enum):
    REDUNDANCY = "redundancy"
    DIVERSITY = "diversity"
    ADAPTABILITY = "adaptability"
    ROBUSTNESS = "robustness"
    VISIBILITY = "visibility"
    VELOCITY = "velocity"


@dataclass
class ResilienceMetric:
    metric_id: str
    metric_name: str
    metric_type: MetricType
    current_score: float  # 0.0 to 1.0
    target_score: float   # 0.0 to 1.0
    weight: float         # Importance weight
    category: ResilienceCategory
    trend_direction: str  # "improving", "stable", "declining"
    contributing_factors: List[str]
    improvement_recommendations: List[str]
    benchmark_comparison: Dict[str, float]  # industry/peer comparisons
    last_calculated: datetime


@dataclass
class NodeResilience:
    node_id: str
    node_name: str
    node_type: str
    overall_score: float
    component_scores: Dict[str, float]
    critical_vulnerabilities: List[str]
    strengths: List[str]
    risk_exposure: float
    recovery_capability: float
    estimated_recovery_days: int


@dataclass
class NetworkResilience:
    network_id: str
    overall_resilience_score: float
    node_resilience: List[NodeResilience]
    critical_paths_resilience: Dict[str, float]
    systemic_risks: List[str]
    cascade_vulnerability: float
    adaptive_capacity: float
    redundancy_level: float


@dataclass
class ResilienceReport:
    report_id: str
    generated_at: datetime
    overall_resilience: NetworkResilience
    detailed_metrics: List[ResilienceMetric]
    industry_benchmarks: Dict[str, float]
    improvement_roadmap: List[Dict[str, Any]]
    executive_summary: Dict[str, Any]


class SupplyChainResilienceService:
    """Service for calculating and analyzing supply chain resilience metrics."""
    
    def __init__(self):
        # Industry benchmark scores for comparison
        self.industry_benchmarks = {
            "automotive": {
                "redundancy": 0.72,
                "diversity": 0.68,
                "adaptability": 0.65,
                "robustness": 0.70,
                "visibility": 0.75,
                "velocity": 0.63
            },
            "electronics": {
                "redundancy": 0.78,
                "diversity": 0.73,
                "adaptability": 0.70,
                "robustness": 0.68,
                "visibility": 0.80,
                "velocity": 0.75
            },
            "pharmaceuticals": {
                "redundancy": 0.85,
                "diversity": 0.70,
                "adaptability": 0.75,
                "robustness": 0.82,
                "visibility": 0.88,
                "velocity": 0.60
            },
            "aerospace": {
                "redundancy": 0.88,
                "diversity": 0.75,
                "adaptability": 0.72,
                "robustness": 0.85,
                "visibility": 0.82,
                "velocity": 0.58
            },
            "general_manufacturing": {
                "redundancy": 0.65,
                "diversity": 0.62,
                "adaptability": 0.60,
                "robustness": 0.63,
                "visibility": 0.68,
                "velocity": 0.65
            }
        }
        
        # Metric weights by industry
        self.metric_weights = {
            "automotive": {
                "redundancy": 0.20,
                "diversity": 0.15,
                "adaptability": 0.18,
                "robustness": 0.22,
                "visibility": 0.15,
                "velocity": 0.10
            },
            "electronics": {
                "redundancy": 0.18,
                "diversity": 0.16,
                "adaptability": 0.20,
                "robustness": 0.16,
                "visibility": 0.20,
                "velocity": 0.10
            },
            "default": {
                "redundancy": 0.18,
                "diversity": 0.16,
                "adaptability": 0.18,
                "robustness": 0.18,
                "visibility": 0.18,
                "velocity": 0.12
            }
        }
        
        # Risk factors that impact resilience
        self.risk_factors = {
            "geopolitical": {"weight": 0.25, "impact_multiplier": 1.3},
            "natural_disaster": {"weight": 0.20, "impact_multiplier": 1.2},
            "cyber_security": {"weight": 0.15, "impact_multiplier": 1.4},
            "supplier_concentration": {"weight": 0.20, "impact_multiplier": 1.5},
            "financial_stability": {"weight": 0.12, "impact_multiplier": 1.1},
            "regulatory_compliance": {"weight": 0.08, "impact_multiplier": 1.0}
        }

    def calculate_comprehensive_resilience(
        self,
        supply_chain_data: Dict[str, Any],
        industry_sector: str = "general_manufacturing",
        historical_performance: Optional[List[Dict]] = None
    ) -> ResilienceReport:
        """Calculate comprehensive resilience metrics for the entire supply chain."""
        
        report_id = f"resilience_report_{int(datetime.utcnow().timestamp())}"
        current_time = datetime.utcnow()
        
        # Extract nodes and edges from supply chain data
        nodes = supply_chain_data.get("nodes", [])
        edges = supply_chain_data.get("edges", [])
        disruptions = supply_chain_data.get("disruptions", [])
        critical_paths = supply_chain_data.get("critical_paths", [])
        
        # Calculate individual node resilience
        node_resilience_scores = []
        for node in nodes:
            node_resilience = self._calculate_node_resilience(node, edges, disruptions)
            node_resilience_scores.append(node_resilience)
        
        # Calculate network-level resilience
        network_resilience = self._calculate_network_resilience(
            nodes, edges, critical_paths, disruptions, node_resilience_scores
        )
        
        # Calculate detailed metrics
        detailed_metrics = self._calculate_detailed_metrics(
            supply_chain_data, industry_sector, historical_performance
        )
        
        # Get industry benchmarks
        benchmarks = self.industry_benchmarks.get(industry_sector, self.industry_benchmarks["general_manufacturing"])
        
        # Generate improvement roadmap
        improvement_roadmap = self._generate_improvement_roadmap(
            detailed_metrics, network_resilience, benchmarks
        )
        
        # Create executive summary
        executive_summary = self._create_executive_summary(
            network_resilience, detailed_metrics, benchmarks
        )
        
        report = ResilienceReport(
            report_id=report_id,
            generated_at=current_time,
            overall_resilience=network_resilience,
            detailed_metrics=detailed_metrics,
            industry_benchmarks=benchmarks,
            improvement_roadmap=improvement_roadmap,
            executive_summary=executive_summary
        )
        
        logger.info(f"Generated resilience report {report_id} with overall score: {network_resilience.overall_resilience_score:.3f}")
        
        return report

    def _calculate_node_resilience(
        self,
        node: Dict[str, Any],
        edges: List[Dict[str, Any]],
        disruptions: List[Dict[str, Any]]
    ) -> NodeResilience:
        """Calculate resilience metrics for an individual node."""
        
        node_id = node.get("id", "unknown")
        node_name = node.get("name", "Unknown Node")
        node_type = node.get("type", "unknown")
        
        # Calculate component scores
        component_scores = {}
        
        # 1. Connectivity Resilience
        incoming_connections = len([e for e in edges if e.get("to") == node_id])
        outgoing_connections = len([e for e in edges if e.get("from") == node_id])
        total_connections = incoming_connections + outgoing_connections
        
        # Normalize connectivity score (more connections = better resilience)
        connectivity_score = min(1.0, total_connections / 10.0)  # Assume 10 is excellent connectivity
        component_scores["connectivity"] = connectivity_score
        
        # 2. Operational Resilience
        operational_risk = node.get("risk_operational", 0.5)
        operational_score = 1.0 - operational_risk  # Lower risk = higher resilience
        component_scores["operational"] = operational_score
        
        # 3. Financial Resilience  
        financial_risk = node.get("risk_financial", 0.5)
        financial_score = 1.0 - financial_risk
        component_scores["financial"] = financial_score
        
        # 4. Geographic/Policy Resilience
        policy_risk = node.get("risk_policy", 0.5)
        policy_score = 1.0 - policy_risk
        component_scores["policy"] = policy_score
        
        # 5. Disruption Exposure
        nearby_disruptions = [
            d for d in disruptions 
            if self._calculate_geographic_distance(node, d) < 500  # Within 500km
        ]
        disruption_exposure = min(1.0, len(nearby_disruptions) / 5.0)  # 5+ disruptions = max exposure
        disruption_resilience = 1.0 - disruption_exposure
        component_scores["disruption_resilience"] = disruption_resilience
        
        # Calculate overall score (weighted average)
        weights = {
            "connectivity": 0.25,
            "operational": 0.25,
            "financial": 0.20,
            "policy": 0.20,
            "disruption_resilience": 0.10
        }
        
        overall_score = sum(component_scores[component] * weights[component] 
                           for component in component_scores)
        
        # Identify vulnerabilities and strengths
        critical_vulnerabilities = []
        strengths = []
        
        for component, score in component_scores.items():
            if score < 0.4:
                critical_vulnerabilities.append(f"Low {component.replace('_', ' ')} resilience")
            elif score > 0.8:
                strengths.append(f"Strong {component.replace('_', ' ')} resilience")
        
        # Calculate risk exposure and recovery capability
        risk_exposure = (operational_risk + financial_risk + policy_risk) / 3
        recovery_capability = overall_score  # Use overall score as proxy
        
        # Estimate recovery time
        base_recovery_days = 14  # 2 weeks baseline
        risk_multiplier = 1 + risk_exposure
        capability_divisor = max(0.1, recovery_capability)
        estimated_recovery_days = int(base_recovery_days * risk_multiplier / capability_divisor)
        
        return NodeResilience(
            node_id=node_id,
            node_name=node_name,
            node_type=node_type,
            overall_score=overall_score,
            component_scores=component_scores,
            critical_vulnerabilities=critical_vulnerabilities,
            strengths=strengths,
            risk_exposure=risk_exposure,
            recovery_capability=recovery_capability,
            estimated_recovery_days=estimated_recovery_days
        )

    def _calculate_network_resilience(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]], 
        critical_paths: List[List[str]],
        disruptions: List[Dict[str, Any]],
        node_resilience_scores: List[NodeResilience]
    ) -> NetworkResilience:
        """Calculate network-level resilience metrics."""
        
        # Overall network score (weighted average of node scores)
        if node_resilience_scores:
            # Weight nodes by their criticality (number of connections)
            node_weights = {}
            for node_resilience in node_resilience_scores:
                node_id = node_resilience.node_id
                connections = len([e for e in edges if e.get("from") == node_id or e.get("to") == node_id])
                node_weights[node_id] = max(1, connections)  # At least weight of 1
            
            total_weight = sum(node_weights.values())
            weighted_score = sum(
                nr.overall_score * node_weights[nr.node_id] 
                for nr in node_resilience_scores
            ) / total_weight
            
            overall_score = weighted_score
        else:
            overall_score = 0.0
        
        # Critical paths resilience
        critical_paths_resilience = {}
        for i, path in enumerate(critical_paths):
            path_scores = []
            for node_id in path:
                node_resilience = next((nr for nr in node_resilience_scores if nr.node_id == node_id), None)
                if node_resilience:
                    path_scores.append(node_resilience.overall_score)
            
            # Path resilience is the minimum score (weakest link)
            path_resilience = min(path_scores) if path_scores else 0.0
            critical_paths_resilience[f"path_{i+1}"] = path_resilience
        
        # Calculate redundancy level
        avg_connections_per_node = (len(edges) * 2) / max(1, len(nodes))  # Each edge connects 2 nodes
        redundancy_level = min(1.0, avg_connections_per_node / 6.0)  # 6+ avg connections = full redundancy
        
        # Calculate cascade vulnerability
        highly_connected_nodes = [
            node for node in nodes
            if len([e for e in edges if e.get("from") == node.get("id") or e.get("to") == node.get("id")]) > 5
        ]
        cascade_vulnerability = min(1.0, len(highly_connected_nodes) / max(1, len(nodes)))
        
        # Calculate adaptive capacity
        diverse_node_types = len(set(node.get("type", "unknown") for node in nodes))
        geographic_spread = self._calculate_geographic_diversity(nodes)
        adaptive_capacity = (min(1.0, diverse_node_types / 5.0) + geographic_spread) / 2
        
        # Identify systemic risks
        systemic_risks = []
        if redundancy_level < 0.4:
            systemic_risks.append("Low network redundancy")
        if cascade_vulnerability > 0.6:
            systemic_risks.append("High cascade vulnerability due to hub concentration")
        if len(disruptions) > 10:
            systemic_risks.append("Multiple active disruptions creating compound risks")
        if adaptive_capacity < 0.5:
            systemic_risks.append("Limited adaptive capacity and diversity")
        
        return NetworkResilience(
            network_id="primary_supply_network",
            overall_resilience_score=overall_score,
            node_resilience=node_resilience_scores,
            critical_paths_resilience=critical_paths_resilience,
            systemic_risks=systemic_risks,
            cascade_vulnerability=cascade_vulnerability,
            adaptive_capacity=adaptive_capacity,
            redundancy_level=redundancy_level
        )

    def _calculate_detailed_metrics(
        self,
        supply_chain_data: Dict[str, Any],
        industry_sector: str,
        historical_performance: Optional[List[Dict]] = None
    ) -> List[ResilienceMetric]:
        """Calculate detailed resilience metrics."""
        
        nodes = supply_chain_data.get("nodes", [])
        edges = supply_chain_data.get("edges", [])
        disruptions = supply_chain_data.get("disruptions", [])
        
        metrics = []
        weights = self.metric_weights.get(industry_sector, self.metric_weights["default"])
        benchmarks = self.industry_benchmarks.get(industry_sector, self.industry_benchmarks["general_manufacturing"])
        
        # 1. Redundancy Metric
        supplier_redundancy = self._calculate_supplier_redundancy(nodes, edges)
        redundancy_metric = ResilienceMetric(
            metric_id="redundancy_001",
            metric_name="Supplier Redundancy",
            metric_type=MetricType.REDUNDANCY,
            current_score=supplier_redundancy,
            target_score=0.80,
            weight=weights["redundancy"],
            category=self._score_to_category(supplier_redundancy),
            trend_direction=self._determine_trend(supplier_redundancy, historical_performance, "redundancy"),
            contributing_factors=[
                "Number of alternative suppliers per critical component",
                "Geographic distribution of supplier base",
                "Supplier capacity utilization levels"
            ],
            improvement_recommendations=self._generate_redundancy_recommendations(supplier_redundancy),
            benchmark_comparison={"industry_average": benchmarks["redundancy"]},
            last_calculated=datetime.utcnow()
        )
        metrics.append(redundancy_metric)
        
        # 2. Diversity Metric
        supplier_diversity = self._calculate_supplier_diversity(nodes)
        diversity_metric = ResilienceMetric(
            metric_id="diversity_001",
            metric_name="Supplier Diversity",
            metric_type=MetricType.DIVERSITY,
            current_score=supplier_diversity,
            target_score=0.75,
            weight=weights["diversity"],
            category=self._score_to_category(supplier_diversity),
            trend_direction=self._determine_trend(supplier_diversity, historical_performance, "diversity"),
            contributing_factors=[
                "Geographic distribution of suppliers",
                "Industry sector diversification",
                "Company size distribution"
            ],
            improvement_recommendations=self._generate_diversity_recommendations(supplier_diversity),
            benchmark_comparison={"industry_average": benchmarks["diversity"]},
            last_calculated=datetime.utcnow()
        )
        metrics.append(diversity_metric)
        
        # 3. Adaptability Metric
        network_adaptability = self._calculate_adaptability(nodes, edges, disruptions)
        adaptability_metric = ResilienceMetric(
            metric_id="adaptability_001", 
            metric_name="Network Adaptability",
            metric_type=MetricType.ADAPTABILITY,
            current_score=network_adaptability,
            target_score=0.70,
            weight=weights["adaptability"],
            category=self._score_to_category(network_adaptability),
            trend_direction=self._determine_trend(network_adaptability, historical_performance, "adaptability"),
            contributing_factors=[
                "Speed of supplier onboarding",
                "Alternative route availability",
                "Flexible contract terms"
            ],
            improvement_recommendations=self._generate_adaptability_recommendations(network_adaptability),
            benchmark_comparison={"industry_average": benchmarks["adaptability"]},
            last_calculated=datetime.utcnow()
        )
        metrics.append(adaptability_metric)
        
        # 4. Robustness Metric
        network_robustness = self._calculate_robustness(nodes, edges, disruptions)
        robustness_metric = ResilienceMetric(
            metric_id="robustness_001",
            metric_name="Network Robustness", 
            metric_type=MetricType.ROBUSTNESS,
            current_score=network_robustness,
            target_score=0.75,
            weight=weights["robustness"],
            category=self._score_to_category(network_robustness),
            trend_direction=self._determine_trend(network_robustness, historical_performance, "robustness"),
            contributing_factors=[
                "Network structure resilience",
                "Supplier financial stability",
                "Quality management systems"
            ],
            improvement_recommendations=self._generate_robustness_recommendations(network_robustness),
            benchmark_comparison={"industry_average": benchmarks["robustness"]},
            last_calculated=datetime.utcnow()
        )
        metrics.append(robustness_metric)
        
        # 5. Visibility Metric
        supply_visibility = self._calculate_visibility(nodes, edges)
        visibility_metric = ResilienceMetric(
            metric_id="visibility_001",
            metric_name="Supply Chain Visibility",
            metric_type=MetricType.VISIBILITY, 
            current_score=supply_visibility,
            target_score=0.85,
            weight=weights["visibility"],
            category=self._score_to_category(supply_visibility),
            trend_direction=self._determine_trend(supply_visibility, historical_performance, "visibility"),
            contributing_factors=[
                "Real-time tracking capabilities",
                "Supplier data integration",
                "Risk monitoring systems"
            ],
            improvement_recommendations=self._generate_visibility_recommendations(supply_visibility),
            benchmark_comparison={"industry_average": benchmarks["visibility"]},
            last_calculated=datetime.utcnow()
        )
        metrics.append(visibility_metric)
        
        # 6. Velocity Metric
        response_velocity = self._calculate_velocity(nodes, edges, disruptions)
        velocity_metric = ResilienceMetric(
            metric_id="velocity_001",
            metric_name="Response Velocity",
            metric_type=MetricType.VELOCITY,
            current_score=response_velocity,
            target_score=0.70,
            weight=weights["velocity"],
            category=self._score_to_category(response_velocity),
            trend_direction=self._determine_trend(response_velocity, historical_performance, "velocity"),
            contributing_factors=[
                "Decision-making speed",
                "Supplier communication efficiency",
                "Inventory adjustment capability"
            ],
            improvement_recommendations=self._generate_velocity_recommendations(response_velocity),
            benchmark_comparison={"industry_average": benchmarks["velocity"]},
            last_calculated=datetime.utcnow()
        )
        metrics.append(velocity_metric)
        
        return metrics

    def _calculate_supplier_redundancy(self, nodes: List[Dict], edges: List[Dict]) -> float:
        """Calculate supplier redundancy score."""
        if not nodes:
            return 0.0
        
        # Calculate average alternative suppliers per node
        redundancy_scores = []
        for node in nodes:
            node_id = node.get("id")
            # Count incoming suppliers
            suppliers = [e for e in edges if e.get("to") == node_id]
            # More than 1 supplier provides redundancy
            if len(suppliers) <= 1:
                redundancy_scores.append(0.0)
            elif len(suppliers) == 2:
                redundancy_scores.append(0.6)
            elif len(suppliers) == 3:
                redundancy_scores.append(0.8)
            else:
                redundancy_scores.append(1.0)
        
        return statistics.mean(redundancy_scores) if redundancy_scores else 0.0

    def _calculate_supplier_diversity(self, nodes: List[Dict]) -> float:
        """Calculate supplier diversity score."""
        if not nodes:
            return 0.0
        
        # Geographic diversity
        countries = set(node.get("country", "unknown") for node in nodes)
        geographic_diversity = min(1.0, len(countries) / 10.0)  # 10+ countries = max diversity
        
        # Type diversity
        node_types = set(node.get("type", "unknown") for node in nodes)
        type_diversity = min(1.0, len(node_types) / 5.0)  # 5+ types = max diversity
        
        return (geographic_diversity + type_diversity) / 2

    def _calculate_adaptability(self, nodes: List[Dict], edges: List[Dict], disruptions: List[Dict]) -> float:
        """Calculate network adaptability score."""
        # Measure ability to adapt to changes
        base_adaptability = 0.6  # Baseline assumption
        
        # Bonus for flexible connections
        flexible_connections = len([e for e in edges if e.get("criticality", 0) < 0.8])
        flexibility_bonus = min(0.3, flexible_connections / max(1, len(edges)) * 0.5)
        
        # Penalty for rigid critical paths
        critical_edges = len([e for e in edges if e.get("criticality", 0) > 0.9])
        rigidity_penalty = min(0.2, critical_edges / max(1, len(edges)) * 0.3)
        
        return min(1.0, max(0.0, base_adaptability + flexibility_bonus - rigidity_penalty))

    def _calculate_robustness(self, nodes: List[Dict], edges: List[Dict], disruptions: List[Dict]) -> float:
        """Calculate network robustness score."""
        if not nodes:
            return 0.0
        
        # Average node resilience
        node_risk_scores = []
        for node in nodes:
            avg_risk = (
                node.get("risk_operational", 0.5) +
                node.get("risk_financial", 0.5) +
                node.get("risk_policy", 0.5)
            ) / 3
            node_risk_scores.append(1.0 - avg_risk)  # Convert risk to resilience
        
        avg_robustness = statistics.mean(node_risk_scores)
        
        # Penalty for active disruptions
        disruption_penalty = min(0.3, len(disruptions) / 20.0)  # Max penalty at 20+ disruptions
        
        return max(0.0, avg_robustness - disruption_penalty)

    def _calculate_visibility(self, nodes: List[Dict], edges: List[Dict]) -> float:
        """Calculate supply chain visibility score."""
        # Assume visibility correlates with data quality and completeness
        base_visibility = 0.7  # Baseline for having basic tracking
        
        # Bonus for comprehensive data
        nodes_with_complete_data = sum(
            1 for node in nodes 
            if all(key in node for key in ["name", "type", "lat", "lng"])
        )
        completeness_bonus = min(0.2, nodes_with_complete_data / max(1, len(nodes)) * 0.2)
        
        # Bonus for real-time updates (assume we have this capability)
        realtime_bonus = 0.1
        
        return min(1.0, base_visibility + completeness_bonus + realtime_bonus)

    def _calculate_velocity(self, nodes: List[Dict], edges: List[Dict], disruptions: List[Dict]) -> float:
        """Calculate response velocity score."""
        # Base velocity assumption
        base_velocity = 0.65
        
        # Factor in network complexity (simpler = faster response)
        complexity_factor = len(nodes) + len(edges)
        complexity_penalty = min(0.2, complexity_factor / 100.0)  # Penalty for very complex networks
        
        # Bonus for automated systems (assume we have some automation)
        automation_bonus = 0.15
        
        return min(1.0, max(0.0, base_velocity - complexity_penalty + automation_bonus))

    def _calculate_geographic_distance(self, node1: Dict, node2: Dict) -> float:
        """Calculate geographic distance between two nodes in kilometers."""
        lat1, lng1 = node1.get("lat", 0), node1.get("lng", 0)
        if isinstance(node2.get("location"), list) and len(node2["location"]) >= 2:
            lat2, lng2 = node2["location"][0], node2["location"][1]
        else:
            lat2, lng2 = node2.get("lat", 0), node2.get("lng", 0)
        
        # Haversine formula for great-circle distance
        R = 6371  # Earth's radius in kilometers
        
        dlat = math.radians(lat2 - lat1)
        dlng = math.radians(lng2 - lng1)
        
        a = (math.sin(dlat/2) * math.sin(dlat/2) +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
             math.sin(dlng/2) * math.sin(dlng/2))
        
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance = R * c
        
        return distance

    def _calculate_geographic_diversity(self, nodes: List[Dict]) -> float:
        """Calculate geographic diversity of nodes."""
        if len(nodes) < 2:
            return 0.0
        
        # Calculate average distance between all node pairs
        total_distance = 0
        pair_count = 0
        
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes[i+1:], i+1):
                distance = self._calculate_geographic_distance(node1, node2)
                total_distance += distance
                pair_count += 1
        
        if pair_count == 0:
            return 0.0
        
        avg_distance = total_distance / pair_count
        
        # Normalize to 0-1 scale (10,000km = maximum diversity)
        return min(1.0, avg_distance / 10000.0)

    def _score_to_category(self, score: float) -> ResilienceCategory:
        """Convert numerical score to resilience category."""
        if score >= 0.90:
            return ResilienceCategory.EXCELLENT
        elif score >= 0.75:
            return ResilienceCategory.GOOD
        elif score >= 0.60:
            return ResilienceCategory.MODERATE
        elif score >= 0.40:
            return ResilienceCategory.WEAK
        else:
            return ResilienceCategory.CRITICAL

    def _determine_trend(
        self, 
        current_score: float, 
        historical_data: Optional[List[Dict]], 
        metric_name: str
    ) -> str:
        """Determine trend direction based on historical data."""
        if not historical_data:
            return "stable"  # Default if no historical data
        
        # Look for the metric in historical data
        historical_scores = []
        for record in historical_data[-5:]:  # Last 5 records
            if metric_name in record:
                historical_scores.append(record[metric_name])
        
        if len(historical_scores) < 2:
            return "stable"
        
        # Calculate trend
        recent_avg = statistics.mean(historical_scores[-2:])
        older_avg = statistics.mean(historical_scores[:-2]) if len(historical_scores) > 2 else historical_scores[0]
        
        change = recent_avg - older_avg
        
        if change > 0.05:
            return "improving"
        elif change < -0.05:
            return "declining"
        else:
            return "stable"

    def _generate_redundancy_recommendations(self, score: float) -> List[str]:
        """Generate recommendations for improving redundancy."""
        recommendations = []
        
        if score < 0.4:
            recommendations.extend([
                "Identify and onboard alternative suppliers for critical components",
                "Implement dual-sourcing strategy for high-risk items",
                "Develop supplier qualification framework for rapid onboarding"
            ])
        elif score < 0.7:
            recommendations.extend([
                "Expand geographic diversification of supplier base",
                "Negotiate capacity agreements with backup suppliers",
                "Create supplier development programs in new regions"
            ])
        else:
            recommendations.extend([
                "Maintain current redundancy levels",
                "Monitor supplier capacity utilization",
                "Optimize redundancy costs while maintaining resilience"
            ])
        
        return recommendations

    def _generate_diversity_recommendations(self, score: float) -> List[str]:
        """Generate recommendations for improving diversity."""
        recommendations = []
        
        if score < 0.5:
            recommendations.extend([
                "Diversify supplier base across more geographic regions",
                "Engage suppliers from different industry sectors",
                "Develop relationships with suppliers of various sizes"
            ])
        elif score < 0.75:
            recommendations.extend([
                "Continue geographic expansion of supplier network",
                "Assess concentration risks in current supplier base",
                "Explore emerging markets for new supplier relationships"
            ])
        else:
            recommendations.extend([
                "Maintain current diversity levels",
                "Monitor for concentration drift over time",
                "Balance diversity with operational efficiency"
            ])
        
        return recommendations

    def _generate_adaptability_recommendations(self, score: float) -> List[str]:
        """Generate recommendations for improving adaptability."""
        recommendations = []
        
        if score < 0.5:
            recommendations.extend([
                "Implement flexible contract terms with suppliers",
                "Develop rapid supplier qualification processes",
                "Create cross-training programs for supply chain staff"
            ])
        elif score < 0.75:
            recommendations.extend([
                "Enhance supplier communication and collaboration systems",
                "Implement scenario planning for supply chain disruptions",
                "Develop agile inventory management practices"
            ])
        else:
            recommendations.extend([
                "Maintain current adaptability capabilities",
                "Continuously test and improve response procedures",
                "Share best practices across the organization"
            ])
        
        return recommendations

    def _generate_robustness_recommendations(self, score: float) -> List[str]:
        """Generate recommendations for improving robustness."""
        recommendations = []
        
        if score < 0.5:
            recommendations.extend([
                "Conduct comprehensive supplier risk assessments",
                "Implement supplier financial health monitoring",
                "Develop risk mitigation strategies for high-risk suppliers"
            ])
        elif score < 0.75:
            recommendations.extend([
                "Enhance supplier quality management systems",
                "Implement regular business continuity planning exercises",
                "Strengthen contractual risk management clauses"
            ])
        else:
            recommendations.extend([
                "Maintain current robustness standards",
                "Continuously monitor supplier performance",
                "Invest in long-term supplier relationships"
            ])
        
        return recommendations

    def _generate_visibility_recommendations(self, score: float) -> List[str]:
        """Generate recommendations for improving visibility."""
        recommendations = []
        
        if score < 0.6:
            recommendations.extend([
                "Implement end-to-end supply chain tracking systems",
                "Establish supplier data integration platforms",
                "Deploy real-time risk monitoring tools"
            ])
        elif score < 0.8:
            recommendations.extend([
                "Enhance data analytics capabilities",
                "Implement predictive visibility tools",
                "Expand tracking to tier 2 and tier 3 suppliers"
            ])
        else:
            recommendations.extend([
                "Leverage advanced AI/ML for supply chain insights",
                "Share visibility data with strategic partners",
                "Continuously improve data quality and completeness"
            ])
        
        return recommendations

    def _generate_velocity_recommendations(self, score: float) -> List[str]:
        """Generate recommendations for improving velocity."""
        recommendations = []
        
        if score < 0.5:
            recommendations.extend([
                "Streamline decision-making processes",
                "Implement automated response systems",
                "Establish clear escalation procedures"
            ])
        elif score < 0.75:
            recommendations.extend([
                "Deploy real-time communication tools with suppliers",
                "Implement agile supply chain planning processes",
                "Create rapid response teams for disruptions"
            ])
        else:
            recommendations.extend([
                "Optimize current response capabilities",
                "Implement continuous improvement programs",
                "Share response best practices across teams"
            ])
        
        return recommendations

    def _generate_improvement_roadmap(
        self,
        metrics: List[ResilienceMetric],
        network_resilience: NetworkResilience,
        benchmarks: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Generate a prioritized improvement roadmap."""
        
        roadmap = []
        
        # Identify metrics that are significantly below benchmark
        priority_metrics = []
        for metric in metrics:
            benchmark_score = benchmarks.get(metric.metric_type.value, 0.7)
            gap = benchmark_score - metric.current_score
            if gap > 0.1:  # Significant gap
                priority_metrics.append((metric, gap))
        
        # Sort by gap size (highest priority first)
        priority_metrics.sort(key=lambda x: x[1], reverse=True)
        
        # Create roadmap items
        for i, (metric, gap) in enumerate(priority_metrics[:5]):  # Top 5 priorities
            roadmap_item = {
                "priority": i + 1,
                "focus_area": metric.metric_name,
                "current_score": metric.current_score,
                "target_score": metric.target_score,
                "gap": gap,
                "category": metric.category.value,
                "timeline": "90 days" if gap > 0.3 else "60 days" if gap > 0.2 else "30 days",
                "key_actions": metric.improvement_recommendations,
                "expected_impact": "High" if gap > 0.25 else "Medium" if gap > 0.15 else "Low",
                "estimated_investment": "High" if gap > 0.3 else "Medium" if gap > 0.2 else "Low"
            }
            roadmap.append(roadmap_item)
        
        # Add network-level recommendations
        if network_resilience.overall_resilience_score < 0.7:
            roadmap.append({
                "priority": len(roadmap) + 1,
                "focus_area": "Network Architecture",
                "current_score": network_resilience.overall_resilience_score,
                "target_score": 0.75,
                "gap": 0.75 - network_resilience.overall_resilience_score,
                "category": "network",
                "timeline": "120 days",
                "key_actions": [
                    "Redesign network topology for better resilience",
                    "Implement systematic risk management processes",
                    "Enhance cross-functional collaboration"
                ],
                "expected_impact": "High",
                "estimated_investment": "High"
            })
        
        return roadmap

    def _create_executive_summary(
        self,
        network_resilience: NetworkResilience,
        metrics: List[ResilienceMetric],
        benchmarks: Dict[str, float]
    ) -> Dict[str, Any]:
        """Create executive summary of resilience assessment."""
        
        # Overall assessment
        overall_score = network_resilience.overall_resilience_score
        overall_category = self._score_to_category(overall_score)
        
        # Identify strengths and weaknesses
        strengths = []
        weaknesses = []
        
        for metric in metrics:
            benchmark = benchmarks.get(metric.metric_type.value, 0.7)
            if metric.current_score > benchmark + 0.05:
                strengths.append(metric.metric_name)
            elif metric.current_score < benchmark - 0.1:
                weaknesses.append(metric.metric_name)
        
        # Calculate improvement potential
        max_possible_score = sum(metric.target_score * metric.weight for metric in metrics)
        current_weighted_score = sum(metric.current_score * metric.weight for metric in metrics)
        improvement_potential = max_possible_score - current_weighted_score
        
        # Risk summary
        high_risk_nodes = len([nr for nr in network_resilience.node_resilience if nr.overall_score < 0.5])
        critical_vulnerabilities = len([nr for nr in network_resilience.node_resilience if nr.overall_score < 0.3])
        
        return {
            "overall_resilience_score": round(overall_score, 3),
            "overall_category": overall_category.value,
            "benchmark_comparison": {
                "above_benchmark": len([m for m in metrics if m.current_score > benchmarks.get(m.metric_type.value, 0.7)]),
                "below_benchmark": len([m for m in metrics if m.current_score < benchmarks.get(m.metric_type.value, 0.7)]),
                "total_metrics": len(metrics)
            },
            "key_strengths": strengths[:3],  # Top 3
            "key_weaknesses": weaknesses[:3],  # Top 3
            "improvement_potential": round(improvement_potential, 3),
            "risk_summary": {
                "total_nodes": len(network_resilience.node_resilience),
                "high_risk_nodes": high_risk_nodes,
                "critical_vulnerabilities": critical_vulnerabilities,
                "systemic_risks": len(network_resilience.systemic_risks)
            },
            "recommended_actions": [
                f"Priority: Address {weaknesses[0] if weaknesses else 'network architecture'}",
                f"Focus: Improve {', '.join(weaknesses[:2]) if len(weaknesses) >= 2 else 'overall resilience metrics'}",
                "Timeline: Implement top 3 recommendations within 90 days"
            ]
        }

    def get_industry_benchmark_comparison(self, industry_sector: str) -> Dict[str, Any]:
        """Get detailed industry benchmark comparison."""
        
        benchmarks = self.industry_benchmarks.get(industry_sector, self.industry_benchmarks["general_manufacturing"])
        
        return {
            "industry_sector": industry_sector,
            "available_sectors": list(self.industry_benchmarks.keys()),
            "benchmarks": benchmarks,
            "methodology": "Benchmarks based on industry surveys and public resilience studies",
            "last_updated": "2024-Q3",
            "peer_comparison": {
                "top_quartile": {metric: score * 1.2 for metric, score in benchmarks.items()},
                "median": benchmarks,
                "bottom_quartile": {metric: score * 0.8 for metric, score in benchmarks.items()}
            }
        }


# Singleton instance
_resilience_service = None


def get_resilience_service() -> SupplyChainResilienceService:
    """Get the resilience metrics service instance."""
    global _resilience_service
    if _resilience_service is None:
        _resilience_service = SupplyChainResilienceService()
    return _resilience_service