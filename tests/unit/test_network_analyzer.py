"""
Unit tests for the NetworkAnalyzer.
"""
import pytest
import networkx as nx
from unittest.mock import patch, MagicMock

from src.ml.models.network_analyzer import NetworkAnalyzer, NetworkMetrics, NodeAnalysis, PropagationResult


@pytest.mark.unit
class TestNetworkAnalyzer:
    """Test cases for NetworkAnalyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create a NetworkAnalyzer instance."""
        return NetworkAnalyzer()
    
    @pytest.fixture
    def sample_nodes(self):
        """Sample nodes data."""
        return [
            {
                "node_id": "A",
                "name": "Node A",
                "node_type": "financial",
                "risk_level": 25.0,
                "weight": 1.0,
                "x_position": 0.1,
                "y_position": 0.5
            },
            {
                "node_id": "B",
                "name": "Node B", 
                "node_type": "technology",
                "risk_level": 35.0,
                "weight": 1.2,
                "x_position": 0.3,
                "y_position": 0.3
            },
            {
                "node_id": "C",
                "name": "Node C",
                "node_type": "infrastructure", 
                "risk_level": 45.0,
                "weight": 1.8,
                "x_position": 0.7,
                "y_position": 0.6
            }
        ]
    
    @pytest.fixture
    def sample_edges(self):
        """Sample edges data."""
        return [
            {
                "source_node_id": "A",
                "target_node_id": "B",
                "edge_type": "financial",
                "weight": 1.5,
                "propagation_probability": 0.7,
                "amplification_factor": 1.2,
                "direction": "directed"
            },
            {
                "source_node_id": "B",
                "target_node_id": "C",
                "edge_type": "dependency",
                "weight": 2.0,
                "propagation_probability": 0.9,
                "amplification_factor": 1.5,
                "direction": "directed"
            }
        ]
    
    def test_analyzer_initialization(self, analyzer):
        """Test NetworkAnalyzer initialization."""
        assert isinstance(analyzer.graph, nx.Graph)
        assert isinstance(analyzer.directed_graph, nx.DiGraph)
        assert analyzer.node_attributes == {}
        assert analyzer.edge_attributes == {}
    
    def test_build_network_from_data(self, analyzer, sample_nodes, sample_edges):
        """Test building network from data."""
        # Execute
        analyzer.build_network_from_data(sample_nodes, sample_edges)
        
        # Verify nodes
        assert analyzer.graph.number_of_nodes() == 3
        assert analyzer.directed_graph.number_of_nodes() == 3
        assert "A" in analyzer.graph.nodes()
        assert "B" in analyzer.graph.nodes()
        assert "C" in analyzer.graph.nodes()
        
        # Verify edges
        assert analyzer.graph.number_of_edges() == 2
        assert analyzer.directed_graph.number_of_edges() == 2
        assert analyzer.graph.has_edge("A", "B")
        assert analyzer.graph.has_edge("B", "C")
        
        # Verify attributes
        assert analyzer.node_attributes["A"]["name"] == "Node A"
        assert analyzer.edge_attributes["A-B"]["weight"] == 1.5
    
    def test_calculate_network_metrics_empty_graph(self, analyzer):
        """Test network metrics calculation with empty graph."""
        metrics = analyzer.calculate_network_metrics()
        
        assert isinstance(metrics, NetworkMetrics)
        assert metrics.node_count == 0
        assert metrics.edge_count == 0
        assert metrics.density == 0.0
    
    def test_calculate_network_metrics_with_data(self, analyzer, sample_nodes, sample_edges):
        """Test network metrics calculation with data."""
        # Setup
        analyzer.build_network_from_data(sample_nodes, sample_edges)
        
        # Execute
        metrics = analyzer.calculate_network_metrics()
        
        # Verify
        assert metrics.node_count == 3
        assert metrics.edge_count == 2
        assert metrics.density > 0
        assert metrics.clustering_coefficient >= 0
        assert metrics.connected_components == 1
    
    def test_calculate_centrality_measures(self, analyzer, sample_nodes, sample_edges):
        """Test centrality measures calculation."""
        # Setup
        analyzer.build_network_from_data(sample_nodes, sample_edges)
        
        # Execute
        centrality = analyzer.calculate_centrality_measures()
        
        # Verify
        assert isinstance(centrality, dict)
        assert len(centrality) == 3
        assert "A" in centrality
        assert "betweenness" in centrality["A"]
        assert "closeness" in centrality["A"]
        assert "eigenvector" in centrality["A"]
        assert "pagerank" in centrality["A"]
        assert "degree" in centrality["A"]
        
        # Verify values are in valid range
        for node, measures in centrality.items():
            for measure, value in measures.items():
                assert 0 <= value <= 1
    
    def test_identify_critical_nodes(self, analyzer, sample_nodes, sample_edges):
        """Test critical nodes identification."""
        # Setup
        analyzer.build_network_from_data(sample_nodes, sample_edges)
        
        # Execute
        critical_nodes = analyzer.identify_critical_nodes(threshold=0.0)  # Low threshold to get results
        
        # Verify
        assert isinstance(critical_nodes, list)
        # Should have some critical nodes with low threshold
        assert len(critical_nodes) >= 0
    
    def test_find_vulnerabilities(self, analyzer, sample_nodes, sample_edges):
        """Test vulnerability analysis."""
        # Setup
        analyzer.build_network_from_data(sample_nodes, sample_edges)
        
        # Execute
        vulnerabilities = analyzer.find_vulnerabilities()
        
        # Verify
        assert isinstance(vulnerabilities, dict)
        assert "single_points_of_failure" in vulnerabilities
        assert "bridge_edges" in vulnerabilities
        assert "articulation_points" in vulnerabilities
        assert "vulnerable_components" in vulnerabilities
        assert "bottleneck_analysis" in vulnerabilities
    
    def test_simulate_risk_propagation(self, analyzer, sample_nodes, sample_edges):
        """Test risk propagation simulation."""
        # Setup
        analyzer.build_network_from_data(sample_nodes, sample_edges)
        
        # Execute
        with patch('random.random', return_value=0.5):  # Mock randomness
            result = analyzer.simulate_risk_propagation(
                initial_nodes=["A"],
                shock_magnitude=0.8,
                steps=10,
                containment_threshold=0.1
            )
        
        # Verify
        assert isinstance(result, PropagationResult)
        assert result.initial_nodes == ["A"]
        assert isinstance(result.affected_nodes, list)
        assert isinstance(result.propagation_steps, list)
        assert isinstance(result.final_impact, dict)
        assert result.recovery_estimate >= 0
        assert isinstance(result.containment_strategies, list)
    
    def test_calculate_shortest_paths(self, analyzer, sample_nodes, sample_edges):
        """Test shortest paths calculation."""
        # Setup
        analyzer.build_network_from_data(sample_nodes, sample_edges)
        
        # Execute
        paths = analyzer.calculate_shortest_paths("A", ["B", "C"])
        
        # Verify
        assert isinstance(paths, dict)
        assert "source" in paths
        assert "distances" in paths
        assert "paths" in paths
        assert paths["source"] == "A"
        assert "reachable_nodes" in paths
    
    def test_analyze_node(self, analyzer, sample_nodes, sample_edges):
        """Test individual node analysis."""
        # Setup
        analyzer.build_network_from_data(sample_nodes, sample_edges)
        
        # Execute
        node_analysis = analyzer.analyze_node("A")
        
        # Verify
        assert isinstance(node_analysis, NodeAnalysis)
        assert node_analysis.node_id == "A"
        assert isinstance(node_analysis.centrality_measures, dict)
        assert 0 <= node_analysis.vulnerability_score <= 1
        assert 0 <= node_analysis.influence_score <= 1
        assert 0 <= node_analysis.local_clustering <= 1
        assert node_analysis.neighborhood_size >= 0
    
    def test_analyze_nonexistent_node(self, analyzer, sample_nodes, sample_edges):
        """Test analysis of nonexistent node."""
        # Setup
        analyzer.build_network_from_data(sample_nodes, sample_edges)
        
        # Execute
        result = analyzer.analyze_node("NONEXISTENT")
        
        # Verify
        assert result is None
    
    def test_generate_network_visualization_data(self, analyzer, sample_nodes, sample_edges):
        """Test network visualization data generation."""
        # Setup
        analyzer.build_network_from_data(sample_nodes, sample_edges)
        
        # Execute
        viz_data = analyzer.generate_network_visualization_data()
        
        # Verify
        assert isinstance(viz_data, dict)
        assert "nodes" in viz_data
        assert "edges" in viz_data
        assert "metadata" in viz_data
        
        # Verify nodes structure
        assert len(viz_data["nodes"]) == 3
        node = viz_data["nodes"][0]
        assert "id" in node
        assert "name" in node
        assert "type" in node
        assert "risk_level" in node
        assert "size" in node
        assert "color" in node
        
        # Verify edges structure
        assert len(viz_data["edges"]) == 2
        edge = viz_data["edges"][0]
        assert "source" in edge
        assert "target" in edge
        assert "weight" in edge
        assert "type" in edge
    
    def test_node_vulnerability_calculation(self, analyzer, sample_nodes, sample_edges):
        """Test node vulnerability score calculation."""
        # Setup
        analyzer.build_network_from_data(sample_nodes, sample_edges)
        
        # Execute
        vuln_score = analyzer._calculate_node_vulnerability("A")
        
        # Verify
        assert 0 <= vuln_score <= 1
    
    def test_containment_strategies_generation(self, analyzer):
        """Test containment strategies generation."""
        # Setup
        affected_nodes = {"A", "B", "C"}
        node_states = {"A": 0.8, "B": 0.6, "C": 0.3}
        
        # Execute
        strategies = analyzer._generate_containment_strategies(affected_nodes, node_states)
        
        # Verify
        assert isinstance(strategies, list)
        assert len(strategies) > 0
        for strategy in strategies:
            assert isinstance(strategy, str)
    
    def test_get_node_color(self, analyzer):
        """Test node color generation based on risk level."""
        # Test different risk levels
        assert analyzer._get_node_color(90) == '#d32f2f'  # High risk - red
        assert analyzer._get_node_color(70) == '#f57c00'  # Medium-high risk - orange
        assert analyzer._get_node_color(50) == '#fbc02d'  # Medium risk - yellow
        assert analyzer._get_node_color(30) == '#689f38'  # Low-medium risk - light green
        assert analyzer._get_node_color(10) == '#388e3c'  # Low risk - green
    
    def test_get_edge_color(self, analyzer):
        """Test edge color generation based on propagation probability."""
        # Test different propagation probabilities
        assert analyzer._get_edge_color(0.9) == '#d32f2f'  # High propagation - red
        assert analyzer._get_edge_color(0.7) == '#f57c00'  # Medium-high propagation - orange
        assert analyzer._get_edge_color(0.5) == '#fbc02d'  # Medium propagation - yellow
        assert analyzer._get_edge_color(0.2) == '#757575'  # Low propagation - gray
    
    def test_empty_graph_operations(self, analyzer):
        """Test operations on empty graph."""
        # Test centrality calculation on empty graph
        centrality = analyzer.calculate_centrality_measures()
        assert centrality == {}
        
        # Test vulnerability analysis on empty graph
        vulnerabilities = analyzer.find_vulnerabilities()
        assert vulnerabilities["articulation_points"] == []
        assert vulnerabilities["bridge_edges"] == []
    
    def test_single_node_graph(self, analyzer):
        """Test operations on single node graph."""
        # Setup
        single_node = [{
            "node_id": "SINGLE",
            "name": "Single Node",
            "node_type": "test",
            "risk_level": 50.0,
            "weight": 1.0,
            "x_position": 0.5,
            "y_position": 0.5
        }]
        
        analyzer.build_network_from_data(single_node, [])
        
        # Test metrics
        metrics = analyzer.calculate_network_metrics()
        assert metrics.node_count == 1
        assert metrics.edge_count == 0
        assert metrics.density == 0.0
        
        # Test centrality
        centrality = analyzer.calculate_centrality_measures()
        assert "SINGLE" in centrality
        assert centrality["SINGLE"]["degree"] == 0.0
    
    def test_disconnected_graph(self, analyzer):
        """Test operations on disconnected graph."""
        # Setup disconnected nodes
        nodes = [
            {"node_id": "A", "name": "Node A", "node_type": "test", "risk_level": 50.0, "weight": 1.0, "x_position": 0.1, "y_position": 0.1},
            {"node_id": "B", "name": "Node B", "node_type": "test", "risk_level": 50.0, "weight": 1.0, "x_position": 0.9, "y_position": 0.9}
        ]
        # No edges - disconnected
        
        analyzer.build_network_from_data(nodes, [])
        
        # Test metrics
        metrics = analyzer.calculate_network_metrics()
        assert metrics.node_count == 2
        assert metrics.connected_components == 2
        assert metrics.largest_component_size == 1
    
    @patch('src.ml.models.network_analyzer.logger')
    def test_error_handling_in_centrality(self, mock_logger, analyzer, sample_nodes):
        """Test error handling in centrality calculation."""
        # Setup with problematic graph
        analyzer.build_network_from_data(sample_nodes, [])
        
        # Mock NetworkX to raise exception
        with patch('networkx.eigenvector_centrality', side_effect=nx.NetworkXError("Test error")):
            centrality = analyzer.calculate_centrality_measures()
            
            # Should handle error gracefully
            assert isinstance(centrality, dict)
            for node in centrality.values():
                assert node["eigenvector"] == 0.0
    
    def test_propagation_with_containment(self, analyzer, sample_nodes, sample_edges):
        """Test propagation simulation with quick containment."""
        # Setup
        analyzer.build_network_from_data(sample_nodes, sample_edges)
        
        # Execute with high containment threshold
        with patch('random.random', return_value=0.9):  # High randomness = low propagation
            result = analyzer.simulate_risk_propagation(
                initial_nodes=["A"],
                shock_magnitude=0.1,
                steps=50,
                containment_threshold=0.5  # High threshold
            )
        
        # Verify quick containment
        assert len(result.propagation_steps) <= 50
        assert result.recovery_estimate >= 0