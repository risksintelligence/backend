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
        
    def _create_real_network(self, cache_manager) -> Tuple[List[NetworkNode], List[NetworkEdge]]:
        """Create network based on real cached economic data."""
        from src.cache.cache_manager import CacheManager
        
        if not cache_manager:
            cache_manager = CacheManager()
        
        nodes = []
        
        # Federal Reserve - use real federal funds rate data
        fed_data = cache_manager.get("fred:FEDFUNDS:latest")
        fed_risk = self._calculate_fed_risk(fed_data)
        print(f"DEBUG: fed_data = {fed_data}")  # Debug line
        nodes.append(NetworkNode(
            id="fed",
            name="Federal Reserve",
            category="government",
            risk_level=fed_risk,
            systemic_importance=0.95,
            description=f"Federal funds rate: {fed_data.get('value', 'N/A')}%" if fed_data else "Central banking system - no cache data"
        ))
        
        # Labor Market - use real unemployment data
        unemployment_data = cache_manager.get("fred:UNRATE:latest")
        labor_risk = self._calculate_labor_risk(unemployment_data)
        nodes.append(NetworkNode(
            id="labor_market",
            name="Labor Market",
            category="economic",
            risk_level=labor_risk,
            systemic_importance=0.85,
            description=f"Unemployment rate: {unemployment_data.get('value', 'N/A')}%" if unemployment_data else "Employment conditions"
        ))
        
        # Inflation/Economic Stability - use real CPI data
        cpi_data = cache_manager.get("fred:CPIAUCSL:latest")
        inflation_risk = self._calculate_inflation_risk(cpi_data)
        nodes.append(NetworkNode(
            id="inflation_sector",
            name="Inflation Sector",
            category="economic",
            risk_level=inflation_risk,
            systemic_importance=0.80,
            description=f"CPI inflation indicator" if cpi_data else "Price stability sector"
        ))
        
        # Financial System - use financial stress index
        stress_data = cache_manager.get("fred:STLFSI4:latest")
        financial_risk = self._calculate_financial_risk(stress_data)
        nodes.append(NetworkNode(
            id="financial_system",
            name="Financial System",
            category="financial",
            risk_level=financial_risk,
            systemic_importance=0.90,
            description=f"Financial stress index: {stress_data.get('value', 'N/A')}" if stress_data else "Banking and financial institutions"
        ))
        
        # GDP/Economic Growth - use real GDP data
        gdp_data = cache_manager.get("fred:GDP:latest")
        growth_risk = self._calculate_growth_risk(gdp_data)
        nodes.append(NetworkNode(
            id="economic_growth",
            name="Economic Growth",
            category="economic",
            risk_level=growth_risk,
            systemic_importance=0.85,
            description=f"GDP indicator" if gdp_data else "Economic output and growth"
        ))
        
        # Treasury/Government Finances - use yield curve data
        treasury_data = cache_manager.get("fred:DGS10:latest")
        treasury_risk = self._calculate_treasury_risk(treasury_data)
        nodes.append(NetworkNode(
            id="treasury",
            name="US Treasury",
            category="government",
            risk_level=treasury_risk,
            systemic_importance=0.90,
            description=f"10-year yield: {treasury_data.get('value', 'N/A')}%" if treasury_data else "Government debt and financing"
        ))
        
        # Add major financial institutions based on systemic importance
        nodes.extend([
            NetworkNode(
                id="jpmorgan",
                name="Major Banks",
                category="financial",
                risk_level=min(financial_risk + 10, 100),  # Bank risk related to financial stress
                systemic_importance=0.85,
                description="Systemically important financial institutions"
            ),
            NetworkNode(
                id="supply_chain",
                name="Supply Chain",
                category="supply_chain",
                risk_level=45.0,  # Moderate baseline
                systemic_importance=0.75,
                description="Global supply chain networks"
            ),
            NetworkNode(
                id="housing",
                name="Housing Market", 
                category="economic",
                risk_level=max(20, labor_risk - 10),  # Housing tied to employment
                systemic_importance=0.70,
                description="Residential real estate sector"
            ),
            NetworkNode(
                id="energy",
                name="Energy Sector",
                category="supply_chain", 
                risk_level=50.0,  # Moderate baseline
                systemic_importance=0.80,
                description="Energy production and distribution"
            )
        ])
        
        # Create edges based on economic relationships
        edges = self._create_real_edges(nodes)
        
        return nodes, edges
    
    def _calculate_fed_risk(self, fed_data) -> float:
        """Calculate Federal Reserve risk based on federal funds rate."""
        if not fed_data:
            return 30.0
        
        rate = fed_data.get('value', 5.0)
        # Risk increases with very high or very low rates
        if rate < 1.0:  # Too low - emergency conditions
            return 60.0
        elif rate > 6.0:  # Too high - restrictive policy
            return min(70.0, 40.0 + (rate - 6.0) * 5)
        else:  # Normal range
            return 20.0 + (rate * 2)  # Moderate scaling
    
    def _calculate_labor_risk(self, unemployment_data) -> float:
        """Calculate labor market risk based on unemployment rate."""
        if not unemployment_data:
            return 40.0
        
        rate = unemployment_data.get('value', 5.0)
        if rate < 3.0:  # Very low unemployment - potential overheating
            return 30.0
        elif rate > 7.0:  # High unemployment - recession risk
            return min(80.0, 50.0 + (rate - 7.0) * 10)
        else:  # Normal range
            return 20.0 + (rate * 4)
    
    def _calculate_inflation_risk(self, cpi_data) -> float:
        """Calculate inflation risk based on CPI data.""" 
        if not cpi_data:
            return 35.0
        
        # Assuming CPI value is year-over-year percentage
        rate = abs(cpi_data.get('value', 2.5))
        target = 2.0  # Fed's inflation target
        
        deviation = abs(rate - target)
        if deviation < 0.5:  # Close to target
            return 20.0
        elif deviation > 3.0:  # Far from target
            return min(75.0, 40.0 + deviation * 10)
        else:  # Moderate deviation
            return 25.0 + deviation * 8
    
    def _calculate_financial_risk(self, stress_data) -> float:
        """Calculate financial system risk based on stress index."""
        if not stress_data:
            return 45.0
        
        stress_value = stress_data.get('value', 0.0)
        # Financial stress index typically ranges from -2 to +4
        # Negative values indicate low stress, positive indicate stress
        if stress_value < -0.5:  # Very low stress
            return 25.0
        elif stress_value > 1.0:  # High stress
            return min(85.0, 50.0 + stress_value * 20)
        else:  # Normal range
            return 35.0 + max(0, stress_value * 15)
    
    def _calculate_growth_risk(self, gdp_data) -> float:
        """Calculate economic growth risk based on GDP data."""
        if not gdp_data:
            return 40.0
        
        # GDP typically in trillions, we want growth rate
        # This is simplified - in practice we'd calculate growth rate
        return 35.0  # Moderate baseline
    
    def _calculate_treasury_risk(self, treasury_data) -> float:
        """Calculate treasury/government finance risk based on yield data."""
        if not treasury_data:
            return 25.0
        
        yield_value = treasury_data.get('value', 4.0)
        # 10-year treasury yield risk assessment
        if yield_value < 2.0:  # Very low yields
            return 35.0
        elif yield_value > 6.0:  # Very high yields
            return min(65.0, 30.0 + (yield_value - 6.0) * 8)
        else:  # Normal range
            return 20.0 + yield_value * 2
    
    def _create_real_edges(self, nodes) -> List[NetworkEdge]:
        """Create edges based on economic relationships between real data nodes."""
        edges = []
        
        # Government policy transmission channels
        edges.extend([
            NetworkEdge("fed", "financial_system", 0.90, 0.80, "regulatory", "Monetary policy transmission"),
            NetworkEdge("fed", "treasury", 0.85, 0.70, "regulatory", "Monetary-fiscal coordination"),
            NetworkEdge("treasury", "economic_growth", 0.70, 0.60, "regulatory", "Fiscal policy impact"),
            NetworkEdge("treasury", "labor_market", 0.65, 0.55, "regulatory", "Government spending effects"),
        ])
        
        # Financial system connections
        edges.extend([
            NetworkEdge("financial_system", "jpmorgan", 0.95, 0.85, "financial", "Systemic banking connection"),
            NetworkEdge("financial_system", "housing", 0.85, 0.75, "financial", "Mortgage and real estate financing"),
            NetworkEdge("financial_system", "economic_growth", 0.80, 0.70, "financial", "Credit availability"),
        ])
        
        # Economic interdependencies
        edges.extend([
            NetworkEdge("labor_market", "housing", 0.80, 0.70, "economic", "Employment-housing demand link"),
            NetworkEdge("labor_market", "inflation_sector", 0.75, 0.65, "economic", "Wage-price spiral relationship"),
            NetworkEdge("economic_growth", "labor_market", 0.85, 0.75, "economic", "Growth-employment correlation"),
            NetworkEdge("inflation_sector", "fed", 0.70, 0.60, "economic", "Inflation targeting feedback"),
        ])
        
        # Supply chain and energy relationships
        edges.extend([
            NetworkEdge("energy", "inflation_sector", 0.75, 0.65, "supply", "Energy price inflation transmission"),
            NetworkEdge("energy", "supply_chain", 0.80, 0.70, "supply", "Energy costs for logistics"),
            NetworkEdge("supply_chain", "inflation_sector", 0.70, 0.60, "supply", "Supply chain cost pressures"),
            NetworkEdge("supply_chain", "economic_growth", 0.65, 0.55, "supply", "Supply availability for growth"),
        ])
        
        # Banking and institutional connections
        edges.extend([
            NetworkEdge("jpmorgan", "housing", 0.85, 0.75, "financial", "Mortgage origination"),
            NetworkEdge("jpmorgan", "energy", 0.60, 0.50, "financial", "Energy sector financing"),
            NetworkEdge("jpmorgan", "supply_chain", 0.70, 0.60, "financial", "Trade finance and working capital"),
        ])
        
        return edges
    
    def analyze_network(self, custom_data: Optional[Dict] = None) -> NetworkAnalysis:
        """Perform comprehensive network analysis using real cached data."""
        from src.cache.cache_manager import CacheManager
        
        cache_manager = CacheManager()
        nodes, edges = self._create_real_network(cache_manager)
        
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