"""
End-to-end tests for risk assessment workflows.
"""
import pytest
import asyncio
from unittest.mock import patch


@pytest.mark.e2e
class TestRiskAssessmentWorkflow:
    """Test complete risk assessment workflow."""
    
    @pytest.mark.asyncio
    async def test_complete_risk_assessment_flow(self, async_client):
        """Test complete risk assessment from data to visualization."""
        
        # Step 1: Check system health
        health_response = await async_client.get("/api/v1/health")
        assert health_response.status_code == 200
        assert health_response.json()["status"] == "healthy"
        
        # Step 2: Get economic indicators
        economic_response = await async_client.get("/api/v1/economic/indicators")
        assert economic_response.status_code == 200
        economic_data = economic_response.json()
        assert economic_data["status"] == "success"
        
        # Step 3: Get market data
        market_response = await async_client.get("/api/v1/economic/market")
        assert market_response.status_code == 200
        market_data = market_response.json()
        assert market_data["status"] == "success"
        
        # Step 4: Calculate risk scores
        risk_response = await async_client.get("/api/v1/risk/overview")
        assert risk_response.status_code == 200
        risk_data = risk_response.json()
        assert risk_data["status"] == "success"
        assert "data" in risk_data
        
        # Step 5: Get risk factors
        factors_response = await async_client.get("/api/v1/risk/factors")
        assert factors_response.status_code == 200
        factors_data = factors_response.json()
        assert factors_data["status"] == "success"
        
        # Step 6: Perform network analysis
        network_response = await async_client.get("/api/v1/network/overview")
        assert network_response.status_code == 200
        network_data = network_response.json()
        assert network_data["status"] == "success"
        assert "visualization_data" in network_data["data"]
        
        # Step 7: Assess vulnerabilities
        vuln_response = await async_client.get("/api/v1/network/vulnerabilities")
        assert vuln_response.status_code == 200
        vuln_data = vuln_response.json()
        assert vuln_data["status"] == "success"
        
        # Step 8: Run propagation simulation
        prop_response = await async_client.get("/api/v1/network/propagation?scenario=default")
        assert prop_response.status_code == 200
        prop_data = prop_response.json()
        assert prop_data["status"] == "success"
        
        # Verify data consistency across endpoints
        assert "timestamp" in risk_data
        assert "timestamp" in network_data
    
    @pytest.mark.asyncio
    async def test_custom_simulation_workflow(self, async_client):
        """Test custom simulation workflow."""
        
        # Step 1: Get network overview to identify nodes
        network_response = await async_client.get("/api/v1/network/overview")
        assert network_response.status_code == 200
        network_data = network_response.json()["data"]
        
        # Step 2: Run custom simulation
        simulation_request = {
            "initial_nodes": ["BANK_A", "ENERGY_GRID"],
            "shock_magnitude": 0.8,
            "simulation_steps": 30,
            "containment_threshold": 0.15
        }
        
        sim_response = await async_client.post("/api/v1/network/simulation", json=simulation_request)
        assert sim_response.status_code == 200
        sim_data = sim_response.json()
        assert sim_data["status"] == "success"
        
        # Step 3: Verify simulation results
        sim_results = sim_data["data"]
        assert "simulation_metadata" in sim_results
        assert "simulation_results" in sim_results
        assert "cascade_analysis" in sim_results
        assert "recovery_analysis" in sim_results
        
        # Step 4: Check containment strategies
        assert "containment_strategies" in sim_results
        assert isinstance(sim_results["containment_strategies"], list)
        
        # Step 5: Verify recommendations
        assert "recommendations" in sim_results
        assert isinstance(sim_results["recommendations"], list)


@pytest.mark.e2e
class TestDataPipelineWorkflow:
    """Test data pipeline from external sources to cache."""
    
    @pytest.mark.asyncio
    async def test_external_data_integration_flow(self, async_client):
        """Test external data integration pipeline."""
        
        # Step 1: Check external API health
        with patch('src.data.sources.fred.health_check', return_value=True):
            with patch('src.data.sources.bea.health_check', return_value=True):
                health_response = await async_client.get("/api/v1/external/health")
                assert health_response.status_code == 200
                health_data = health_response.json()
                assert health_data["status"] == "success"
        
        # Step 2: Fetch FRED data
        with patch('src.data.sources.fred.get_key_indicators', return_value={"indicators": {"gdp": {"value": 27000000}}}):
            fred_response = await async_client.get("/api/v1/external/fred/indicators")
            assert fred_response.status_code == 200
            fred_data = fred_response.json()
            assert fred_data["status"] == "success"
        
        # Step 3: Check cache metrics
        cache_response = await async_client.get("/api/v1/cache/metrics")
        assert cache_response.status_code == 200
        cache_data = cache_response.json()
        assert cache_data["status"] == "success"
        
        # Step 4: Verify cached data
        cache_status_response = await async_client.get("/api/v1/cache/status")
        assert cache_status_response.status_code == 200
        
        # Step 5: Test cache warming
        warm_response = await async_client.post("/api/v1/cache/warm")
        assert warm_response.status_code == 200
        warm_data = warm_response.json()
        assert warm_data["status"] == "success"


@pytest.mark.e2e
class TestNetworkAnalysisWorkflow:
    """Test complete network analysis workflow."""
    
    @pytest.mark.asyncio
    async def test_network_analysis_complete_flow(self, async_client):
        """Test complete network analysis from topology to simulation."""
        
        # Step 1: Get network topology
        overview_response = await async_client.get("/api/v1/network/overview")
        assert overview_response.status_code == 200
        overview_data = overview_response.json()["data"]
        
        # Extract sample node for further analysis
        viz_data = overview_data["visualization_data"]
        if viz_data["nodes"]:
            sample_node = viz_data["nodes"][0]["id"]
        else:
            sample_node = "BANK_A"  # Default from sample data
        
        # Step 2: Analyze individual nodes
        nodes_response = await async_client.get(f"/api/v1/network/nodes?node_id={sample_node}")
        if nodes_response.status_code == 200:
            nodes_data = nodes_response.json()
            assert nodes_data["status"] == "success"
        
        # Step 3: Get centrality measures
        centrality_response = await async_client.get("/api/v1/network/centrality?metric=all&top_n=10")
        assert centrality_response.status_code == 200
        centrality_data = centrality_response.json()
        assert centrality_data["status"] == "success"
        
        # Step 4: Assess vulnerabilities
        vuln_response = await async_client.get("/api/v1/network/vulnerabilities?assessment_type=comprehensive")
        assert vuln_response.status_code == 200
        vuln_data = vuln_response.json()
        assert vuln_data["status"] == "success"
        
        # Step 5: Analyze risk propagation
        prop_response = await async_client.get("/api/v1/network/propagation?scenario=high_impact")
        assert prop_response.status_code == 200
        prop_data = prop_response.json()
        assert prop_data["status"] == "success"
        
        # Step 6: Get critical paths
        paths_response = await async_client.get(f"/api/v1/network/critical-paths?source_node={sample_node}&max_paths=5")
        # This might return 404 if node doesn't exist, which is acceptable
        assert paths_response.status_code in [200, 404]
        
        # Step 7: Run shock simulation
        simulation_request = {
            "initial_nodes": [sample_node],
            "shock_magnitude": 0.6,
            "simulation_steps": 25,
            "containment_threshold": 0.1
        }
        
        sim_response = await async_client.post("/api/v1/network/simulation", json=simulation_request)
        assert sim_response.status_code == 200
        sim_data = sim_response.json()
        assert sim_data["status"] == "success"
        
        # Verify complete workflow data consistency
        assert "network_metrics" in overview_data
        assert "critical_components" in overview_data
        assert "overall_assessment" in vuln_data["data"]
        assert "simulation_summary" in prop_data["data"]


@pytest.mark.e2e
class TestCacheEfficiencyWorkflow:
    """Test cache efficiency across multiple requests."""
    
    @pytest.mark.asyncio
    async def test_cache_performance_workflow(self, async_client):
        """Test cache performance and efficiency."""
        
        # Step 1: Get initial cache metrics
        initial_metrics_response = await async_client.get("/api/v1/cache/metrics")
        assert initial_metrics_response.status_code == 200
        initial_metrics = initial_metrics_response.json()["data"]
        
        # Step 2: Make multiple requests to same endpoint
        endpoints_to_test = [
            "/api/v1/risk/overview",
            "/api/v1/network/overview",
            "/api/v1/economic/indicators"
        ]
        
        for endpoint in endpoints_to_test:
            # First request (likely cache miss)
            response1 = await async_client.get(endpoint)
            assert response1.status_code == 200
            
            # Second request (likely cache hit)
            response2 = await async_client.get(endpoint)
            assert response2.status_code == 200
            
            # Third request (should be cache hit)
            response3 = await async_client.get(endpoint)
            assert response3.status_code == 200
        
        # Step 3: Check final cache metrics
        final_metrics_response = await async_client.get("/api/v1/cache/metrics")
        assert final_metrics_response.status_code == 200
        final_metrics = final_metrics_response.json()["data"]
        
        # Step 4: Verify cache efficiency improved
        if "cache_operations" in final_metrics:
            # Should have more operations than initial
            assert final_metrics["total_operations"] >= initial_metrics.get("total_operations", 0)
        
        # Step 5: Test cache clearing
        clear_response = await async_client.delete("/api/v1/cache/clear")
        assert clear_response.status_code == 200


@pytest.mark.e2e
class TestSystemResilienceWorkflow:
    """Test system resilience and error recovery."""
    
    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self, async_client):
        """Test system recovery from various error conditions."""
        
        # Step 1: Test with invalid parameters
        invalid_response = await async_client.get("/api/v1/network/nodes?limit=invalid")
        assert invalid_response.status_code == 422
        
        # Step 2: Verify system still operational after error
        health_response = await async_client.get("/api/v1/health")
        assert health_response.status_code == 200
        
        # Step 3: Test with edge case values
        edge_case_response = await async_client.get("/api/v1/network/centrality?top_n=1000")
        assert edge_case_response.status_code in [200, 422]  # Either works or validates
        
        # Step 4: Test concurrent error scenarios
        invalid_tasks = []
        for _ in range(5):
            invalid_tasks.append(async_client.get("/api/v1/nonexistent/endpoint"))
        
        responses = await asyncio.gather(*invalid_tasks, return_exceptions=True)
        
        # All should return 404 or similar error
        for response in responses:
            if hasattr(response, 'status_code'):
                assert response.status_code == 404
        
        # Step 5: Verify system still healthy after errors
        final_health_response = await async_client.get("/api/v1/health")
        assert final_health_response.status_code == 200
        assert final_health_response.json()["status"] == "healthy"


@pytest.mark.e2e
@pytest.mark.slow
class TestLongRunningWorkflow:
    """Test long-running operations and workflows."""
    
    @pytest.mark.asyncio
    async def test_extended_simulation_workflow(self, async_client):
        """Test extended simulation with many steps."""
        
        # Step 1: Run long simulation
        long_simulation_request = {
            "initial_nodes": ["BANK_A", "TECH_CORP", "ENERGY_GRID"],
            "shock_magnitude": 0.9,
            "simulation_steps": 100,  # Long simulation
            "containment_threshold": 0.05
        }
        
        sim_response = await async_client.post("/api/v1/network/simulation", json=long_simulation_request)
        assert sim_response.status_code == 200
        sim_data = sim_response.json()
        assert sim_data["status"] == "success"
        
        # Step 2: Verify simulation completed
        results = sim_data["data"]
        assert "simulation_results" in results
        assert "cascade_analysis" in results
        
        # Step 3: Check system health after long operation
        health_response = await async_client.get("/api/v1/health")
        assert health_response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_batch_operations_workflow(self, async_client):
        """Test batch operations workflow."""
        
        # Step 1: Batch network analysis requests
        network_endpoints = [
            "/api/v1/network/overview",
            "/api/v1/network/centrality?metric=betweenness",
            "/api/v1/network/centrality?metric=pagerank",
            "/api/v1/network/vulnerabilities?assessment_type=spof",
            "/api/v1/network/vulnerabilities?assessment_type=bridges"
        ]
        
        # Make concurrent requests
        tasks = []
        for endpoint in network_endpoints:
            tasks.append(async_client.get(endpoint))
        
        responses = await asyncio.gather(*tasks)
        
        # Step 2: Verify all succeeded
        for response in responses:
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
        
        # Step 3: Check cache efficiency
        cache_response = await async_client.get("/api/v1/cache/metrics")
        assert cache_response.status_code == 200


@pytest.mark.e2e
class TestComprehensiveIntegration:
    """Test comprehensive integration across all systems."""
    
    @pytest.mark.asyncio
    async def test_full_platform_integration(self, async_client):
        """Test full platform integration workflow."""
        
        # Phase 1: System Initialization
        health_response = await async_client.get("/api/v1/health")
        assert health_response.status_code == 200
        
        platform_response = await async_client.get("/api/v1/platform/info")
        assert platform_response.status_code == 200
        
        # Phase 2: Data Collection
        economic_response = await async_client.get("/api/v1/economic/indicators")
        assert economic_response.status_code == 200
        
        # Phase 3: Risk Analysis
        risk_response = await async_client.get("/api/v1/risk/overview")
        assert risk_response.status_code == 200
        
        factors_response = await async_client.get("/api/v1/risk/factors")
        assert factors_response.status_code == 200
        
        # Phase 4: Network Analysis
        network_response = await async_client.get("/api/v1/network/overview")
        assert network_response.status_code == 200
        
        vuln_response = await async_client.get("/api/v1/network/vulnerabilities")
        assert vuln_response.status_code == 200
        
        # Phase 5: Simulation and Modeling
        prop_response = await async_client.get("/api/v1/network/propagation")
        assert prop_response.status_code == 200
        
        simulation_request = {
            "initial_nodes": ["BANK_A"],
            "shock_magnitude": 0.7,
            "simulation_steps": 50,
            "containment_threshold": 0.1
        }
        
        sim_response = await async_client.post("/api/v1/network/simulation", json=simulation_request)
        assert sim_response.status_code == 200
        
        # Phase 6: Cache and Performance
        cache_response = await async_client.get("/api/v1/cache/metrics")
        assert cache_response.status_code == 200
        
        # Phase 7: WebSocket Infrastructure
        ws_status_response = await async_client.get("/api/v1/ws/status")
        assert ws_status_response.status_code == 200
        
        # Phase 8: Database Operations
        db_schema_response = await async_client.get("/api/v1/database/schema")
        assert db_schema_response.status_code == 200
        
        # Verify all responses indicate success
        all_responses = [
            health_response, platform_response, economic_response,
            risk_response, factors_response, network_response,
            vuln_response, prop_response, sim_response,
            cache_response, ws_status_response, db_schema_response
        ]
        
        for response in all_responses:
            assert response.status_code == 200
            if "status" in response.json():
                assert response.json()["status"] in ["success", "healthy", "active"]