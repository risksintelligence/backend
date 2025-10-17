"""
Network analysis for systemic risk propagation modeling.
"""
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import networkx as nx
import numpy as np
from dataclasses import dataclass

@dataclass
class NetworkNode:
    """Represents a node in the risk network."""
    id: str
    name: str
    category: str  # 'financial', 'supply_chain', 'economic', 'government'
    risk_level: float  # 0-100
    systemic_importance: float  # 0-1
    description: str

@dataclass
class NetworkEdge:
    """Represents an edge in the risk network."""
    source: str
    target: str
    strength: float  # 0-1
    risk_propagation: float  # 0-1
    connection_type: str  # 'trade', 'financial', 'supply', 'regulatory'
    description: str

@dataclass
class NetworkAnalysis:
    """Network analysis results."""
    nodes: List[NetworkNode]
    edges: List[NetworkEdge]
    centrality_metrics: Dict[str, float]
    vulnerability_score: float
    systemic_risk_score: float
    critical_paths: List[List[str]]
    timestamp: datetime


class RiskNetworkAnalyzer:
    """Analyzes systemic risk propagation through network structures."""
    
    def __init__(self):
        self.network = nx.DiGraph()
        self.node_data = {}
        self.edge_data = {}
        
    def _create_sample_network(self) -> Tuple[List[NetworkNode], List[NetworkEdge]]:
        """Create a sample risk network for demonstration."""
        
        # Sample nodes representing major economic entities
        nodes = [
            NetworkNode(
                id="fed",
                name="Federal Reserve",
                category="government",
                risk_level=25.0,
                systemic_importance=0.95,
                description="Central banking system of the United States"
            ),
            NetworkNode(
                id="treasury",
                name="US Treasury",
                category="government", 
                risk_level=20.0,
                systemic_importance=0.90,
                description="Department of the Treasury"
            ),
            NetworkNode(
                id="jpmorgan",
                name="JPMorgan Chase",
                category="financial",
                risk_level=42.0,
                systemic_importance=0.85,
                description="Largest bank in the United States"
            ),
            NetworkNode(
                id="goldman",
                name="Goldman Sachs",
                category="financial",
                risk_level=48.0,
                systemic_importance=0.75,
                description="Investment banking and financial services"
            ),
            NetworkNode(
                id="bofa",
                name="Bank of America",
                category="financial",
                risk_level=39.0,
                systemic_importance=0.80,
                description="Major commercial bank"
            ),
            NetworkNode(
                id="apple",
                name="Apple Inc",
                category="supply_chain",
                risk_level=35.0,
                systemic_importance=0.70,
                description="Technology company with global supply chain"
            ),
            NetworkNode(
                id="amazon",
                name="Amazon",
                category="supply_chain",
                risk_level=45.0,
                systemic_importance=0.75,
                description="E-commerce and cloud services"
            ),
            NetworkNode(
                id="labor_market",
                name="Labor Market",
                category="economic",
                risk_level=52.0,
                systemic_importance=0.85,
                description="US employment and wage conditions"
            ),
            NetworkNode(
                id="housing",
                name="Housing Market",
                category="economic",
                risk_level=38.0,
                systemic_importance=0.70,
                description="Residential real estate market"
            ),
            NetworkNode(
                id="china_trade",
                name="China Trade Relations",
                category="supply_chain",
                risk_level=68.0,
                systemic_importance=0.80,
                description="Trade relationship with China"
            ),
            NetworkNode(
                id="energy",
                name="Energy Sector",
                category="supply_chain",
                risk_level=55.0,
                systemic_importance=0.85,
                description="Oil, gas, and renewable energy"
            ),
            NetworkNode(
                id="tech_sector",
                name="Technology Sector",
                category="economic",
                risk_level=42.0,
                systemic_importance=0.75,
                description="Technology industry aggregate"
            )
        ]
        
        # Sample edges representing relationships
        edges = [
            # Government regulatory relationships
            NetworkEdge("fed", "jpmorgan", 0.85, 0.75, "regulatory", "Monetary policy transmission"),
            NetworkEdge("fed", "goldman", 0.80, 0.70, "regulatory", "Financial regulation"),
            NetworkEdge("fed", "bofa", 0.85, 0.75, "regulatory", "Banking oversight"),
            NetworkEdge("fed", "treasury", 0.95, 0.85, "regulatory", "Monetary-fiscal coordination"),
            NetworkEdge("treasury", "labor_market", 0.60, 0.50, "regulatory", "Fiscal policy impact"),
            
            # Financial sector interconnections
            NetworkEdge("jpmorgan", "goldman", 0.70, 0.60, "financial", "Interbank lending"),
            NetworkEdge("jpmorgan", "bofa", 0.75, 0.65, "financial", "Financial markets"),
            NetworkEdge("goldman", "bofa", 0.65, 0.55, "financial", "Investment banking"),
            
            # Financial-corporate relationships
            NetworkEdge("jpmorgan", "apple", 0.60, 0.50, "financial", "Corporate banking"),
            NetworkEdge("goldman", "amazon", 0.65, 0.55, "financial", "Investment services"),
            NetworkEdge("bofa", "tech_sector", 0.55, 0.45, "financial", "Commercial lending"),
            
            # Supply chain relationships
            NetworkEdge("apple", "china_trade", 0.90, 0.85, "supply", "Manufacturing dependency"),
            NetworkEdge("amazon", "china_trade", 0.75, 0.70, "supply", "Product sourcing"),
            NetworkEdge("tech_sector", "china_trade", 0.80, 0.75, "supply", "Components and assembly"),
            NetworkEdge("energy", "china_trade", 0.70, 0.65, "trade", "Energy trade"),
            
            # Economic interdependencies
            NetworkEdge("labor_market", "housing", 0.75, 0.70, "trade", "Employment-housing link"),
            NetworkEdge("labor_market", "tech_sector", 0.65, 0.60, "trade", "Tech employment"),
            NetworkEdge("housing", "jpmorgan", 0.80, 0.75, "financial", "Mortgage lending"),
            NetworkEdge("housing", "bofa", 0.75, 0.70, "financial", "Real estate financing"),
            
            # Energy dependencies
            NetworkEdge("energy", "labor_market", 0.55, 0.50, "trade", "Energy sector employment"),
            NetworkEdge("energy", "tech_sector", 0.60, 0.55, "supply", "Energy for data centers"),
            NetworkEdge("energy", "amazon", 0.70, 0.65, "supply", "Logistics fuel costs")
        ]
        
        return nodes, edges
    
    def analyze_network(self, custom_data: Optional[Dict] = None) -> NetworkAnalysis:
        """Perform comprehensive network analysis."""
        
        # Use sample data for now
        nodes, edges = self._create_sample_network()
        
        # Build NetworkX graph
        self.network.clear()
        
        # Add nodes
        for node in nodes:
            self.network.add_node(
                node.id,
                name=node.name,
                category=node.category,
                risk_level=node.risk_level,
                systemic_importance=node.systemic_importance
            )
            self.node_data[node.id] = node
        
        # Add edges
        for edge in edges:
            self.network.add_edge(
                edge.source,
                edge.target,
                strength=edge.strength,
                risk_propagation=edge.risk_propagation,
                connection_type=edge.connection_type
            )
            self.edge_data[(edge.source, edge.target)] = edge
        
        # Calculate centrality metrics
        centrality_metrics = self._calculate_centrality_metrics()
        
        # Calculate systemic risk scores
        vulnerability_score = self._calculate_vulnerability_score()
        systemic_risk_score = self._calculate_systemic_risk_score()
        
        # Find critical paths
        critical_paths = self._find_critical_paths()
        
        return NetworkAnalysis(
            nodes=nodes,
            edges=edges,
            centrality_metrics=centrality_metrics,
            vulnerability_score=vulnerability_score,
            systemic_risk_score=systemic_risk_score,
            critical_paths=critical_paths,
            timestamp=datetime.utcnow()
        )
    
    def _calculate_centrality_metrics(self) -> Dict[str, float]:
        """Calculate various centrality measures."""
        try:
            # Degree centrality
            degree_centrality = nx.degree_centrality(self.network)
            
            # Betweenness centrality
            betweenness_centrality = nx.betweenness_centrality(self.network)
            
            # Eigenvector centrality
            try:
                eigenvector_centrality = nx.eigenvector_centrality(self.network, max_iter=1000)
            except:
                # Fallback if eigenvector doesn't converge
                eigenvector_centrality = {node: 0.1 for node in self.network.nodes()}
            
            # PageRank (good for directed graphs)
            pagerank = nx.pagerank(self.network)
            
            # Calculate aggregate scores
            centrality_scores = {}
            for node in self.network.nodes():
                centrality_scores[node] = (
                    degree_centrality.get(node, 0) * 0.3 +
                    betweenness_centrality.get(node, 0) * 0.3 +
                    eigenvector_centrality.get(node, 0) * 0.2 +
                    pagerank.get(node, 0) * 0.2
                )
            
            return centrality_scores
            
        except Exception as e:
            # Return default values if calculation fails
            return {node: 0.1 for node in self.network.nodes()}
    
    def _calculate_vulnerability_score(self) -> float:
        """Calculate overall network vulnerability."""
        try:
            # Average risk level weighted by systemic importance
            total_weighted_risk = 0
            total_weight = 0
            
            for node_id in self.network.nodes():
                node = self.node_data[node_id]
                weight = node.systemic_importance
                total_weighted_risk += node.risk_level * weight
                total_weight += weight
            
            if total_weight == 0:
                return 50.0
            
            return total_weighted_risk / total_weight
            
        except Exception:
            return 50.0
    
    def _calculate_systemic_risk_score(self) -> float:
        """Calculate systemic risk based on network structure."""
        try:
            # Factors affecting systemic risk:
            # 1. Network density
            density = nx.density(self.network)
            
            # 2. Average clustering coefficient
            try:
                clustering = nx.average_clustering(self.network.to_undirected())
            except:
                clustering = 0.3
            
            # 3. Average risk propagation strength
            edge_weights = []
            for edge in self.network.edges(data=True):
                edge_weights.append(edge[2].get('risk_propagation', 0.5))
            
            avg_propagation = np.mean(edge_weights) if edge_weights else 0.5
            
            # 4. Presence of high-risk highly-connected nodes
            high_risk_central_score = 0
            centrality = self._calculate_centrality_metrics()
            
            for node_id in self.network.nodes():
                node = self.node_data[node_id]
                if node.risk_level > 60 and centrality.get(node_id, 0) > 0.3:
                    high_risk_central_score += 0.2
            
            # Combine factors
            systemic_risk = (
                density * 30 +  # Network connectivity
                clustering * 25 +  # Clustering effect
                avg_propagation * 35 +  # Risk propagation
                min(high_risk_central_score, 1.0) * 10  # High-risk central nodes
            )
            
            return min(max(systemic_risk, 0), 100)
            
        except Exception:
            return 50.0
    
    def _find_critical_paths(self) -> List[List[str]]:
        """Find critical risk propagation paths."""
        try:
            critical_paths = []
            
            # Find paths from high-risk nodes to systemically important nodes
            high_risk_nodes = [
                node_id for node_id in self.network.nodes()
                if self.node_data[node_id].risk_level > 60
            ]
            
            important_nodes = [
                node_id for node_id in self.network.nodes()
                if self.node_data[node_id].systemic_importance > 0.8
            ]
            
            for source in high_risk_nodes:
                for target in important_nodes:
                    if source != target:
                        try:
                            if nx.has_path(self.network, source, target):
                                path = nx.shortest_path(self.network, source, target)
                                if len(path) <= 4:  # Only include short paths
                                    critical_paths.append(path)
                        except:
                            continue
            
            # Sort by path risk score and return top 5
            def path_risk_score(path):
                score = 0
                for i in range(len(path) - 1):
                    edge_data = self.network.get_edge_data(path[i], path[i+1])
                    if edge_data:
                        score += edge_data.get('risk_propagation', 0.5)
                return score
            
            critical_paths.sort(key=path_risk_score, reverse=True)
            return critical_paths[:5]
            
        except Exception:
            return []
    
    def simulate_shock(self, shocked_node: str, shock_magnitude: float) -> Dict[str, float]:
        """Simulate a risk shock propagating through the network."""
        try:
            # Initialize risk levels
            current_risks = {}
            for node_id in self.network.nodes():
                current_risks[node_id] = self.node_data[node_id].risk_level
            
            # Apply initial shock
            if shocked_node in current_risks:
                current_risks[shocked_node] = min(100, current_risks[shocked_node] + shock_magnitude)
            
            # Propagate shock through network (simplified model)
            for iteration in range(3):  # 3 rounds of propagation
                new_risks = current_risks.copy()
                
                for node_id in self.network.nodes():
                    # Calculate risk from neighbors
                    neighbor_risk = 0
                    neighbor_count = 0
                    
                    for predecessor in self.network.predecessors(node_id):
                        edge_data = self.network.get_edge_data(predecessor, node_id)
                        if edge_data:
                            propagation_factor = edge_data.get('risk_propagation', 0.5)
                            neighbor_risk += current_risks[predecessor] * propagation_factor * 0.3
                            neighbor_count += 1
                    
                    if neighbor_count > 0:
                        additional_risk = neighbor_risk / neighbor_count
                        new_risks[node_id] = min(100, current_risks[node_id] + additional_risk)
                
                current_risks = new_risks
            
            return current_risks
            
        except Exception:
            return {node_id: 50.0 for node_id in self.network.nodes()}
    
    async def update_network_data(self) -> bool:
        """
        Update network data with latest information.
        
        Returns:
            True if update was successful
        """
        try:
            # In a real implementation, this would fetch fresh data from various sources
            # For now, we'll just verify the network can be analyzed
            analysis = self.analyze_network()
            
            if analysis and analysis.nodes:
                return True
            else:
                return False
                
        except Exception as e:
            return False