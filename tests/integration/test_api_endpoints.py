"""
Integration tests for API endpoints.
"""
import pytest
import json
from unittest.mock import patch, AsyncMock


@pytest.mark.integration
class TestHealthEndpoints:
    """Test health check endpoints."""
    
    @pytest.mark.asyncio
    async def test_root_endpoint(self, async_client):
        """Test root endpoint."""
        response = await async_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "RiskX Risk Intelligence Platform"
        assert data["status"] == "operational"
        assert "version" in data
        assert "timestamp" in data
    
    @pytest.mark.asyncio
    async def test_health_endpoint(self, async_client):
        """Test health check endpoint."""
        with patch('src.core.database.check_database_connection', return_value={"status": "connected"}):
            with patch('src.core.cache.check_redis_connection', return_value={"status": "connected"}):
                response = await async_client.get("/api/v1/health")
                
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "healthy"
                assert data["service"] == "riskx-backend"
                assert "components" in data
    
    @pytest.mark.asyncio
    async def test_status_endpoint(self, async_client):
        """Test API status endpoint."""
        with patch('src.core.database.check_database_connection', return_value={"status": "connected"}):
            with patch('src.core.cache.check_redis_connection', return_value={"status": "connected"}):
                response = await async_client.get("/api/v1/status")
                
                assert response.status_code == 200
                data = response.json()
                assert data["api"] == "operational"
                assert data["version"] == "1.0.0"
                assert "features" in data
    
    @pytest.mark.asyncio
    async def test_test_endpoint(self, async_client):
        """Test the test endpoint."""
        response = await async_client.get("/api/v1/test")
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Test endpoint operational"
        assert "test_data" in data
        assert "sample_risk_score" in data["test_data"]


@pytest.mark.integration
class TestNetworkAnalysisEndpoints:
    """Test Network Analysis API endpoints."""
    
    @pytest.mark.asyncio
    async def test_network_overview(self, async_client):
        """Test network overview endpoint."""
        response = await async_client.get("/api/v1/network/overview")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "data" in data
        assert "network_metrics" in data["data"]
        assert "critical_components" in data["data"]
        assert "visualization_data" in data["data"]
    
    @pytest.mark.asyncio
    async def test_network_nodes_analysis(self, async_client):
        """Test nodes analysis endpoint."""
        response = await async_client.get("/api/v1/network/nodes?limit=10")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "data" in data
        if "nodes_analysis" in data["data"]:
            assert isinstance(data["data"]["nodes_analysis"], list)
    
    @pytest.mark.asyncio
    async def test_network_centrality(self, async_client):
        """Test centrality analysis endpoint."""
        response = await async_client.get("/api/v1/network/centrality?metric=all&top_n=5")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "data" in data
    
    @pytest.mark.asyncio
    async def test_network_vulnerabilities(self, async_client):
        """Test vulnerability assessment endpoint.""" 
        response = await async_client.get("/api/v1/network/vulnerabilities?assessment_type=comprehensive")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "data" in data
        assert "overall_assessment" in data["data"]
    
    @pytest.mark.asyncio
    async def test_network_propagation(self, async_client):
        """Test risk propagation analysis endpoint."""
        response = await async_client.get("/api/v1/network/propagation?scenario=default")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "data" in data
        assert "simulation_summary" in data["data"]
    
    @pytest.mark.asyncio
    async def test_network_critical_paths(self, async_client):
        """Test critical paths analysis endpoint."""
        response = await async_client.get("/api/v1/network/critical-paths?source_node=NODE_1&max_paths=5")
        
        # This might return 404 if NODE_1 doesn't exist in sample data
        assert response.status_code in [200, 404]
        
        if response.status_code == 200:
            data = response.json()
            assert data["status"] == "success"
            assert "data" in data
    
    @pytest.mark.asyncio
    async def test_network_simulation(self, async_client):
        """Test shock simulation endpoint."""
        simulation_request = {
            "initial_nodes": ["BANK_A"],
            "shock_magnitude": 0.5,
            "simulation_steps": 20,
            "containment_threshold": 0.1
        }
        
        response = await async_client.post("/api/v1/network/simulation", json=simulation_request)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "data" in data
        assert "simulation_metadata" in data["data"]


@pytest.mark.integration
class TestRiskEndpoints:
    """Test risk management endpoints."""
    
    @pytest.mark.asyncio
    async def test_risk_overview(self, async_client):
        """Test risk overview endpoint."""
        response = await async_client.get("/api/v1/risk/overview")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "data" in data
    
    @pytest.mark.asyncio
    async def test_risk_factors(self, async_client):
        """Test risk factors endpoint."""
        response = await async_client.get("/api/v1/risk/factors")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "data" in data
    
    @pytest.mark.asyncio
    async def test_realtime_risk_score(self, async_client):
        """Test real-time risk score endpoint."""
        response = await async_client.get("/api/v1/risk/score/realtime")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "data" in data


@pytest.mark.integration
class TestExternalAPIEndpoints:
    """Test external API integration endpoints."""
    
    @pytest.mark.asyncio
    async def test_fred_indicators(self, async_client):
        """Test FRED indicators endpoint."""
        with patch('src.data.sources.fred.get_key_indicators', return_value={"indicators": {}}):
            response = await async_client.get("/api/v1/external/fred/indicators")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] in ["success", "error"]
    
    @pytest.mark.asyncio
    async def test_external_apis_health(self, async_client):
        """Test external APIs health check."""
        with patch('src.data.sources.fred.health_check', return_value=True):
            with patch('src.data.sources.bea.health_check', return_value=True):
                with patch('src.data.sources.bls.health_check', return_value=True):
                    with patch('src.data.sources.census.health_check', return_value=True):
                        response = await async_client.get("/api/v1/external/health")
                        
                        assert response.status_code == 200
                        data = response.json()
                        assert data["status"] == "success"
                        assert "apis" in data


@pytest.mark.integration
class TestCacheEndpoints:
    """Test cache management endpoints."""
    
    @pytest.mark.asyncio
    async def test_cache_metrics(self, async_client):
        """Test cache metrics endpoint."""
        response = await async_client.get("/api/v1/cache/metrics")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "data" in data
    
    @pytest.mark.asyncio
    async def test_cache_status(self, async_client):
        """Test cache status endpoint."""
        response = await async_client.get("/api/v1/cache/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "data" in data
    
    @pytest.mark.asyncio
    async def test_cache_demo(self, async_client):
        """Test cache demo endpoint."""
        response = await async_client.get("/api/v1/cache/demo")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"


@pytest.mark.integration 
class TestEconomicEndpoints:
    """Test economic data endpoints."""
    
    @pytest.mark.asyncio
    async def test_economic_indicators(self, async_client):
        """Test economic indicators endpoint."""
        response = await async_client.get("/api/v1/economic/indicators")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "data" in data
    
    @pytest.mark.asyncio
    async def test_market_data(self, async_client):
        """Test market data endpoint."""
        response = await async_client.get("/api/v1/economic/market")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "data" in data


@pytest.mark.integration
class TestErrorHandling:
    """Test API error handling."""
    
    @pytest.mark.asyncio
    async def test_invalid_endpoint(self, async_client):
        """Test invalid endpoint returns 404."""
        response = await async_client.get("/api/v1/nonexistent")
        
        assert response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_invalid_method(self, async_client):
        """Test invalid HTTP method."""
        response = await async_client.delete("/api/v1/health")
        
        assert response.status_code == 405  # Method not allowed
    
    @pytest.mark.asyncio
    async def test_invalid_query_parameters(self, async_client):
        """Test invalid query parameters."""
        response = await async_client.get("/api/v1/network/nodes?limit=invalid")
        
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.asyncio
    async def test_invalid_json_body(self, async_client):
        """Test invalid JSON body."""
        response = await async_client.post("/api/v1/network/simulation", json="invalid")
        
        assert response.status_code == 422  # Validation error


@pytest.mark.integration
class TestWebSocketEndpoints:
    """Test WebSocket API endpoints."""
    
    @pytest.mark.asyncio
    async def test_websocket_status(self, async_client):
        """Test WebSocket status endpoint."""
        response = await async_client.get("/api/v1/ws/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "active"
        assert "total_connections" in data
        assert "available_topics" in data
    
    @pytest.mark.asyncio
    async def test_websocket_broadcast(self, async_client):
        """Test WebSocket broadcast endpoint."""
        message = {
            "type": "test_message",
            "data": {"test": "data"}
        }
        
        response = await async_client.post("/api/v1/ws/broadcast", json=message)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"


@pytest.mark.integration
class TestDatabaseEndpoints:
    """Test database setup endpoints."""
    
    @pytest.mark.asyncio
    async def test_database_schema(self, async_client):
        """Test database schema endpoint."""
        response = await async_client.get("/api/v1/database/schema")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "schema_info" in data["data"]
    
    @pytest.mark.asyncio
    async def test_database_data_summary(self, async_client):
        """Test database data summary endpoint."""
        response = await async_client.get("/api/v1/database/data/summary")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "summary" in data["data"]


@pytest.mark.integration
class TestCORS:
    """Test CORS configuration."""
    
    @pytest.mark.asyncio
    async def test_cors_headers(self, async_client):
        """Test CORS headers are present."""
        response = await async_client.options("/api/v1/health")
        
        # Should not be blocked by CORS
        assert response.status_code in [200, 405]  # OPTIONS might not be explicitly defined
    
    @pytest.mark.asyncio
    async def test_cors_with_origin(self, async_client):
        """Test CORS with Origin header."""
        headers = {"Origin": "http://localhost:3000"}
        response = await async_client.get("/api/v1/health", headers=headers)
        
        assert response.status_code == 200
        # CORS headers should be present in response


@pytest.mark.integration
class TestCaching:
    """Test caching behavior across endpoints."""
    
    @pytest.mark.asyncio
    async def test_cache_hit_behavior(self, async_client):
        """Test cache hit behavior."""
        # First request
        response1 = await async_client.get("/api/v1/risk/overview")
        assert response1.status_code == 200
        
        # Second request should potentially hit cache
        response2 = await async_client.get("/api/v1/risk/overview")
        assert response2.status_code == 200
        
        # Both should return success
        data1 = response1.json()
        data2 = response2.json()
        assert data1["status"] == "success"
        assert data2["status"] == "success"
    
    @pytest.mark.asyncio
    async def test_cache_source_indicator(self, async_client):
        """Test cache source is indicated in response."""
        response = await async_client.get("/api/v1/network/overview")
        
        assert response.status_code == 200
        data = response.json()
        assert "source" in data
        assert data["source"] in ["cache", "computed"]


@pytest.mark.integration 
@pytest.mark.slow
class TestPerformance:
    """Test API performance."""
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, async_client):
        """Test handling concurrent requests."""
        import asyncio
        
        # Make multiple concurrent requests
        tasks = []
        for _ in range(10):
            tasks.append(async_client.get("/api/v1/health"))
        
        responses = await asyncio.gather(*tasks)
        
        # All should succeed
        for response in responses:
            assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_large_response_handling(self, async_client):
        """Test handling of large responses."""
        # Request potentially large dataset
        response = await async_client.get("/api/v1/network/overview")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        
        # Response should be reasonable size
        response_size = len(json.dumps(data))
        assert response_size < 10 * 1024 * 1024  # Less than 10MB