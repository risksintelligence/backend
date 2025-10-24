"""
Network analysis engine for risk propagation and vulnerability assessment.
"""
import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import json
import random
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class NetworkMetrics:
    """Network-wide metrics and statistics."""
    node_count: int
    edge_count: int
    density: float
    clustering_coefficient: float
    average_path_length: float
    diameter: int
    connected_components: int
    largest_component_size: int


@dataclass
class NodeAnalysis:
    """Individual node analysis results."""
    node_id: str
    centrality_measures: Dict[str, float]
    vulnerability_score: float
    influence_score: float
    local_clustering: float
    neighborhood_size: int
    risk_level: float


@dataclass
class PropagationResult:
    """Risk propagation simulation result."""
    simulation_id: str
    initial_nodes: List[str]
    affected_nodes: List[str]
    propagation_steps: List[Dict]
    final_impact: Dict[str, float]
    recovery_estimate: int
    containment_strategies: List[str]


class NetworkAnalyzer:
    """
    Advanced network analysis for risk assessment and vulnerability detection.
    """
    
    def __init__(self):
        self.graph = nx.Graph()
        self.directed_graph = nx.DiGraph()
        self.node_attributes = {}
        self.edge_attributes = {}
        
    def build_network_from_data(self, nodes_data: List[Dict], edges_data: List[Dict]) -> None:
        """Build network from nodes and edges data."""
        logger.info(f"Building network with {len(nodes_data)} nodes and {len(edges_data)} edges")
        
        # Clear existing graphs
        self.graph.clear()
        self.directed_graph.clear()
        
        # Add nodes
        for node in nodes_data:
            node_id = node['node_id']
            attributes = {
                'name': node.get('name', ''),
                'type': node.get('node_type', ''),
                'risk_level': node.get('risk_level', 0.0),
                'weight': node.get('weight', 1.0),
                'position': (node.get('x_position', 0), node.get('y_position', 0))
            }
            
            self.graph.add_node(node_id, **attributes)
            self.directed_graph.add_node(node_id, **attributes)
            self.node_attributes[node_id] = attributes
        
        # Add edges
        for edge in edges_data:
            source = edge['source_node_id']
            target = edge['target_node_id']
            weight = edge.get('weight', 1.0)
            edge_type = edge.get('edge_type', '')
            propagation_prob = edge.get('propagation_probability', 0.1)
            
            edge_attrs = {
                'weight': weight,
                'type': edge_type,
                'propagation_probability': propagation_prob,
                'amplification_factor': edge.get('amplification_factor', 1.0)
            }
            
            # Add to undirected graph
            self.graph.add_edge(source, target, **edge_attrs)
            
            # Add to directed graph based on direction
            direction = edge.get('direction', 'undirected')
            if direction == 'directed':
                self.directed_graph.add_edge(source, target, **edge_attrs)
            elif direction == 'bidirectional' or direction == 'undirected':
                self.directed_graph.add_edge(source, target, **edge_attrs)
                self.directed_graph.add_edge(target, source, **edge_attrs)
            
            self.edge_attributes[f"{source}-{target}"] = edge_attrs
        
        logger.info("Network construction completed")
    
    def calculate_network_metrics(self) -> NetworkMetrics:
        """Calculate comprehensive network metrics."""
        logger.info("Calculating network metrics")
        
        try:
            # Basic metrics
            node_count = self.graph.number_of_nodes()
            edge_count = self.graph.number_of_edges()
            
            if node_count == 0:
                return NetworkMetrics(0, 0, 0.0, 0.0, 0.0, 0, 0, 0)
            
            # Density
            density = nx.density(self.graph)
            
            # Clustering coefficient
            clustering = nx.average_clustering(self.graph)
            
            # Path-related metrics (only for connected components)
            largest_cc = max(nx.connected_components(self.graph), key=len) if nx.is_connected(self.graph) else set()
            
            if len(largest_cc) > 1:
                largest_subgraph = self.graph.subgraph(largest_cc)
                avg_path_length = nx.average_shortest_path_length(largest_subgraph)
                diameter = nx.diameter(largest_subgraph)
            else:
                avg_path_length = 0.0
                diameter = 0
            
            # Connected components
            connected_components = nx.number_connected_components(self.graph)
            largest_component_size = len(largest_cc)
            
            return NetworkMetrics(
                node_count=node_count,
                edge_count=edge_count,
                density=density,
                clustering_coefficient=clustering,
                average_path_length=avg_path_length,
                diameter=diameter,
                connected_components=connected_components,
                largest_component_size=largest_component_size
            )
            
        except Exception as e:
            logger.error(f"Error calculating network metrics: {e}")
            return NetworkMetrics(0, 0, 0.0, 0.0, 0.0, 0, 0, 0)
    
    def calculate_centrality_measures(self) -> Dict[str, Dict[str, float]]:
        """Calculate various centrality measures for all nodes."""
        logger.info("Calculating centrality measures")
        
        try:
            # Betweenness centrality
            betweenness = nx.betweenness_centrality(self.graph, weight='weight')
            
            # Closeness centrality
            closeness = nx.closeness_centrality(self.graph, distance='weight')
            
            # Eigenvector centrality
            try:
                eigenvector = nx.eigenvector_centrality(self.graph, weight='weight', max_iter=1000)
            except (nx.NetworkXError, np.linalg.LinAlgError):
                eigenvector = {node: 0.0 for node in self.graph.nodes()}
            
            # PageRank centrality
            pagerank = nx.pagerank(self.directed_graph, weight='weight')
            
            # Degree centrality
            degree = nx.degree_centrality(self.graph)
            
            # Combine all centrality measures
            centrality_measures = {}
            for node in self.graph.nodes():
                centrality_measures[node] = {
                    'betweenness': betweenness.get(node, 0.0),
                    'closeness': closeness.get(node, 0.0),
                    'eigenvector': eigenvector.get(node, 0.0),
                    'pagerank': pagerank.get(node, 0.0),
                    'degree': degree.get(node, 0.0)
                }
            
            return centrality_measures
            
        except Exception as e:
            logger.error(f"Error calculating centrality measures: {e}")
            return {}
    
    def identify_critical_nodes(self, threshold: float = 0.8) -> List[str]:
        """Identify critical nodes based on centrality measures."""
        centrality_measures = self.calculate_centrality_measures()
        
        critical_nodes = []
        for node, measures in centrality_measures.items():
            # Calculate composite criticality score
            criticality_score = (
                measures['betweenness'] * 0.3 +
                measures['eigenvector'] * 0.25 +
                measures['pagerank'] * 0.25 +
                measures['closeness'] * 0.2
            )
            
            if criticality_score >= threshold:
                critical_nodes.append(node)
        
        return critical_nodes
    
    def find_vulnerabilities(self) -> Dict[str, Any]:
        """Identify network vulnerabilities and single points of failure."""
        logger.info("Analyzing network vulnerabilities")
        
        vulnerabilities = {
            'single_points_of_failure': [],
            'bridge_edges': [],
            'articulation_points': [],
            'vulnerable_components': [],
            'bottleneck_analysis': {}
        }
        
        try:
            # Articulation points (nodes whose removal disconnects the graph)
            articulation_points = list(nx.articulation_points(self.graph))
            vulnerabilities['articulation_points'] = articulation_points
            
            # Bridge edges (edges whose removal disconnects the graph)
            bridges = list(nx.bridges(self.graph))
            vulnerabilities['bridge_edges'] = [f"{u}-{v}" for u, v in bridges]
            
            # Single points of failure (critical nodes)
            spof = self.identify_critical_nodes(threshold=0.9)
            vulnerabilities['single_points_of_failure'] = spof
            
            # Bottleneck analysis using betweenness centrality
            centrality = nx.betweenness_centrality(self.graph, weight='weight')
            bottlenecks = {node: score for node, score in centrality.items() if score > 0.1}
            vulnerabilities['bottleneck_analysis'] = bottlenecks
            
            # Vulnerable components (small connected components)
            components = list(nx.connected_components(self.graph))
            vulnerable_components = [list(comp) for comp in components if len(comp) < 5]
            vulnerabilities['vulnerable_components'] = vulnerable_components
            
            return vulnerabilities
            
        except Exception as e:
            logger.error(f"Error analyzing vulnerabilities: {e}")
            return vulnerabilities
    
    def simulate_risk_propagation(
        self, 
        initial_nodes: List[str], 
        shock_magnitude: float = 1.0,
        steps: int = 50,
        containment_threshold: float = 0.1
    ) -> PropagationResult:
        """Simulate risk propagation through the network."""
        logger.info(f"Simulating risk propagation from {len(initial_nodes)} initial nodes")
        
        simulation_id = f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize node states
        node_states = {node: 0.0 for node in self.graph.nodes()}
        for node in initial_nodes:
            if node in node_states:
                node_states[node] = shock_magnitude
        
        propagation_steps = []
        affected_nodes = set(initial_nodes)
        
        for step in range(steps):
            new_states = node_states.copy()
            step_changes = {}
            
            # Propagate risk through edges
            for edge in self.graph.edges(data=True):
                source, target, attrs = edge
                
                if node_states[source] > containment_threshold:
                    # Calculate propagated risk
                    propagation_prob = attrs.get('propagation_probability', 0.1)
                    amplification = attrs.get('amplification_factor', 1.0)
                    
                    # Risk transmission with randomness
                    if random.random() < propagation_prob:
                        transmitted_risk = node_states[source] * amplification * 0.8  # 80% transmission
                        new_states[target] = max(new_states[target], transmitted_risk)
                        
                        if transmitted_risk > containment_threshold:
                            affected_nodes.add(target)
                            step_changes[target] = transmitted_risk
            
            # Apply decay to existing risk levels
            for node in new_states:
                if node not in initial_nodes:  # Don't decay initial shock nodes immediately
                    new_states[node] *= 0.95  # 5% decay per step
            
            # Record step
            propagation_steps.append({
                'step': step,
                'active_nodes': [n for n, risk in new_states.items() if risk > containment_threshold],
                'new_infections': list(step_changes.keys()),
                'total_risk': sum(new_states.values())
            })
            
            node_states = new_states
            
            # Check convergence
            if sum(new_states.values()) < containment_threshold * len(initial_nodes):
                break
        
        # Calculate recovery estimate
        max_risk = max(node_states.values())
        recovery_estimate = int(max_risk * 24)  # Rough estimate in hours
        
        # Generate containment strategies
        containment_strategies = self._generate_containment_strategies(affected_nodes, node_states)
        
        return PropagationResult(
            simulation_id=simulation_id,
            initial_nodes=initial_nodes,
            affected_nodes=list(affected_nodes),
            propagation_steps=propagation_steps,
            final_impact={node: risk for node, risk in node_states.items() if risk > 0.01},
            recovery_estimate=recovery_estimate,
            containment_strategies=containment_strategies
        )
    
    def _generate_containment_strategies(self, affected_nodes: set, node_states: Dict[str, float]) -> List[str]:
        """Generate containment strategies based on propagation results."""
        strategies = []
        
        # Identify high-risk nodes
        high_risk_nodes = [node for node, risk in node_states.items() if risk > 0.5]
        if high_risk_nodes:
            strategies.append(f"Priority monitoring of {len(high_risk_nodes)} high-risk nodes")
            strategies.append(f"Implement emergency protocols for: {', '.join(high_risk_nodes[:3])}")
        
        # Check for critical infrastructure
        critical_nodes = self.identify_critical_nodes(threshold=0.7)
        affected_critical = set(affected_nodes) & set(critical_nodes)
        if affected_critical:
            strategies.append(f"Activate redundancy systems for {len(affected_critical)} critical components")
        
        # Network isolation strategies
        if len(affected_nodes) > len(self.graph.nodes()) * 0.3:
            strategies.append("Consider network segmentation to contain spread")
        
        # Recovery strategies
        strategies.append("Establish alternative communication channels")
        strategies.append("Activate backup systems and redundant pathways")
        
        return strategies
    
    def calculate_shortest_paths(self, source: str, targets: List[str] = None) -> Dict[str, Any]:
        """Calculate shortest paths and critical path analysis."""
        if source not in self.graph.nodes():
            return {}
        
        if targets is None:
            targets = list(self.graph.nodes())
        
        try:
            # Single source shortest paths
            paths = nx.single_source_shortest_path_length(self.graph, source, weight='weight')
            
            # Shortest path tree
            path_tree = nx.shortest_path(self.graph, source, weight='weight')
            
            # Critical paths (paths through high-centrality nodes)
            centrality = nx.betweenness_centrality(self.graph, weight='weight')
            critical_paths = {}
            
            for target in targets:
                if target in path_tree and target != source:
                    path = path_tree[target]
                    path_criticality = sum(centrality.get(node, 0) for node in path) / len(path)
                    critical_paths[target] = {
                        'path': path,
                        'length': paths.get(target, float('inf')),
                        'criticality': path_criticality
                    }
            
            return {
                'source': source,
                'distances': paths,
                'paths': critical_paths,
                'reachable_nodes': len(paths),
                'unreachable_nodes': len(self.graph.nodes()) - len(paths)
            }
            
        except Exception as e:
            logger.error(f"Error calculating shortest paths: {e}")
            return {}
    
    def analyze_node(self, node_id: str) -> Optional[NodeAnalysis]:
        """Comprehensive analysis of individual node."""
        if node_id not in self.graph.nodes():
            return None
        
        try:
            # Centrality measures
            centrality_measures = self.calculate_centrality_measures()
            node_centrality = centrality_measures.get(node_id, {})
            
            # Local clustering
            local_clustering = nx.clustering(self.graph, node_id)
            
            # Neighborhood analysis
            neighbors = list(self.graph.neighbors(node_id))
            neighborhood_size = len(neighbors)
            
            # Calculate vulnerability score
            vulnerability_score = self._calculate_node_vulnerability(node_id)
            
            # Calculate influence score
            influence_score = (
                node_centrality.get('betweenness', 0) * 0.4 +
                node_centrality.get('eigenvector', 0) * 0.3 +
                node_centrality.get('pagerank', 0) * 0.3
            )
            
            # Get current risk level
            risk_level = self.node_attributes.get(node_id, {}).get('risk_level', 0.0)
            
            return NodeAnalysis(
                node_id=node_id,
                centrality_measures=node_centrality,
                vulnerability_score=vulnerability_score,
                influence_score=influence_score,
                local_clustering=local_clustering,
                neighborhood_size=neighborhood_size,
                risk_level=risk_level
            )
            
        except Exception as e:
            logger.error(f"Error analyzing node {node_id}: {e}")
            return None
    
    def _calculate_node_vulnerability(self, node_id: str) -> float:
        """Calculate node vulnerability score."""
        try:
            # Check if node is articulation point
            articulation_points = list(nx.articulation_points(self.graph))
            is_articulation = node_id in articulation_points
            
            # Check degree (low degree = more vulnerable)
            degree = self.graph.degree(node_id)
            max_degree = max(dict(self.graph.degree()).values()) if self.graph.nodes() else 1
            degree_vulnerability = 1.0 - (degree / max_degree)
            
            # Check clustering (low clustering = more vulnerable)
            clustering = nx.clustering(self.graph, node_id)
            clustering_vulnerability = 1.0 - clustering
            
            # Combine factors
            vulnerability = (
                (1.0 if is_articulation else 0.0) * 0.4 +
                degree_vulnerability * 0.3 +
                clustering_vulnerability * 0.3
            )
            
            return min(vulnerability, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating vulnerability for {node_id}: {e}")
            return 0.5
    
    def generate_network_visualization_data(self) -> Dict[str, Any]:
        """Generate data for network visualization."""
        centrality_measures = self.calculate_centrality_measures()
        
        # Prepare nodes data
        nodes = []
        for node_id in self.graph.nodes():
            attrs = self.node_attributes.get(node_id, {})
            centrality = centrality_measures.get(node_id, {})
            
            nodes.append({
                'id': node_id,
                'name': attrs.get('name', node_id),
                'type': attrs.get('type', 'unknown'),
                'risk_level': attrs.get('risk_level', 0.0),
                'size': centrality.get('pagerank', 0.1) * 100,
                'color': self._get_node_color(attrs.get('risk_level', 0.0)),
                'position': attrs.get('position', (0, 0)),
                'centrality': centrality
            })
        
        # Prepare edges data
        edges = []
        for source, target, attrs in self.graph.edges(data=True):
            edges.append({
                'source': source,
                'target': target,
                'weight': attrs.get('weight', 1.0),
                'type': attrs.get('type', 'unknown'),
                'width': attrs.get('weight', 1.0) * 3,
                'color': self._get_edge_color(attrs.get('propagation_probability', 0.1))
            })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'metadata': {
                'node_count': len(nodes),
                'edge_count': len(edges),
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def _get_node_color(self, risk_level: float) -> str:
        """Get color for node based on risk level."""
        if risk_level >= 80:
            return '#d32f2f'  # Red - High risk
        elif risk_level >= 60:
            return '#f57c00'  # Orange - Medium-high risk
        elif risk_level >= 40:
            return '#fbc02d'  # Yellow - Medium risk
        elif risk_level >= 20:
            return '#689f38'  # Light green - Low-medium risk
        else:
            return '#388e3c'  # Green - Low risk
    
    def _get_edge_color(self, propagation_probability: float) -> str:
        """Get color for edge based on propagation probability."""
        if propagation_probability >= 0.8:
            return '#d32f2f'  # Red - High propagation
        elif propagation_probability >= 0.6:
            return '#f57c00'  # Orange - Medium-high propagation
        elif propagation_probability >= 0.4:
            return '#fbc02d'  # Yellow - Medium propagation
        else:
            return '#757575'  # Gray - Low propagation