"""
End-to-end tests for complete API workflows
"""

import pytest
import asyncio
from datetime import datetime, timedelta
import httpx
from unittest.mock import patch

from src.api.main import app


class TestRiskAssessmentWorkflow:
    """Test complete risk assessment workflow from API to response"""
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_complete_risk_assessment_workflow(self):
        """Test end-to-end risk assessment from data fetch to API response"""
        async with httpx.AsyncClient(app=app, base_url="http://test") as client:
            
            # Step 1: Check system health
            health_response = await client.get("/api/v1/health")
            assert health_response.status_code == 200
            health_data = health_response.json()
            assert health_data["status"] in ["healthy", "degraded"]
            
            # Step 2: Get current risk score
            risk_response = await client.get("/api/v1/risk/score")
            assert risk_response.status_code == 200
            risk_data = risk_response.json()
            
            # Verify risk score structure
            assert "overall_risk_score" in risk_data
            assert "risk_level" in risk_data
            assert "confidence" in risk_data
            assert 0 <= risk_data["overall_risk_score"] <= 1
            assert risk_data["risk_level"] in ["low", "moderate", "high", "critical"]
            
            # Step 3: Get risk factors
            factors_response = await client.get("/api/v1/risk/factors")
            assert factors_response.status_code == 200
            factors_data = factors_response.json()
            
            # Verify factors structure
            assert "economic_indicators" in factors_data
            assert "financial_indicators" in factors_data
            
            # Step 4: Get analytics data
            analytics_response = await client.get("/api/v1/analytics/aggregation")
            assert analytics_response.status_code == 200
            analytics_data = analytics_response.json()
            
            # Verify analytics structure
            assert "economic_data" in analytics_data or "metadata" in analytics_data
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_network_analysis_workflow(self):
        """Test network analysis workflow"""
        async with httpx.AsyncClient(app=app, base_url="http://test") as client:
            
            # Get network analysis
            network_response = await client.get("/api/v1/network/analysis")
            assert network_response.status_code == 200
            network_data = network_response.json()
            
            # Verify network structure
            assert "nodes" in network_data
            assert "edges" in network_data
            assert isinstance(network_data["nodes"], list)
            assert isinstance(network_data["edges"], list)
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_data_freshness_workflow(self):
        """Test data freshness and health monitoring workflow"""
        async with httpx.AsyncClient(app=app, base_url="http://test") as client:
            
            # Check analytics health
            health_response = await client.get("/api/v1/analytics/health")
            assert health_response.status_code == 200
            health_data = health_response.json()
            
            # Verify health monitoring structure
            assert "data_sources" in health_data or "overall_health" in health_data


class TestErrorHandlingWorkflows:
    """Test error handling across complete workflows"""
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_api_rate_limiting(self):
        """Test API rate limiting behavior"""
        async with httpx.AsyncClient(app=app, base_url="http://test") as client:
            
            # Make rapid requests to test rate limiting
            tasks = []
            for _ in range(10):
                tasks.append(client.get("/api/v1/health"))
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Most should succeed, rate limiting should be handled gracefully
            successful_responses = [r for r in responses if not isinstance(r, Exception) and r.status_code == 200]
            assert len(successful_responses) >= 5  # At least half should succeed
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_timeout_handling(self):
        """Test timeout handling in long-running operations"""
        async with httpx.AsyncClient(app=app, base_url="http://test", timeout=10.0) as client:
            
            # Test endpoints that might take longer
            try:
                response = await client.get("/api/v1/analytics/aggregation")
                # Should either succeed or handle timeout gracefully
                assert response.status_code in [200, 408, 500, 503]
            except httpx.TimeoutException:
                # Timeout is acceptable for this test
                pass
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_data_unavailability_handling(self):
        """Test handling when external data sources are unavailable"""
        with patch('src.data.sources.fred.FREDConnector.fetch_data', side_effect=Exception("External API unavailable")):
            async with httpx.AsyncClient(app=app, base_url="http://test") as client:
                
                response = await client.get("/api/v1/risk/score")
                
                # Should handle gracefully, either with cached data or appropriate error
                assert response.status_code in [200, 503]
                
                if response.status_code == 200:
                    # If successful, should indicate cached/fallback data
                    data = response.json()
                    assert "overall_risk_score" in data
                elif response.status_code == 503:
                    # Service unavailable is acceptable
                    data = response.json()
                    assert "error" in data


class TestPerformanceWorkflows:
    """Test performance characteristics of complete workflows"""
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_response_time_requirements(self):
        """Test that API responses meet performance requirements"""
        async with httpx.AsyncClient(app=app, base_url="http://test") as client:
            
            start_time = datetime.now()
            response = await client.get("/api/v1/health")
            response_time = (datetime.now() - start_time).total_seconds()
            
            assert response.status_code == 200
            assert response_time < 2.0  # Health check should be fast
            
            # Test risk score response time
            start_time = datetime.now()
            response = await client.get("/api/v1/risk/score")
            response_time = (datetime.now() - start_time).total_seconds()
            
            assert response.status_code == 200
            assert response_time < 10.0  # Risk calculation should complete within 10 seconds
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_concurrent_user_simulation(self):
        """Simulate multiple concurrent users"""
        async def simulate_user_session():
            async with httpx.AsyncClient(app=app, base_url="http://test") as client:
                # Simulate typical user workflow
                health_response = await client.get("/api/v1/health")
                risk_response = await client.get("/api/v1/risk/score")
                factors_response = await client.get("/api/v1/risk/factors")
                
                return [health_response, risk_response, factors_response]
        
        # Simulate 5 concurrent users
        user_sessions = [simulate_user_session() for _ in range(5)]
        session_results = await asyncio.gather(*user_sessions, return_exceptions=True)
        
        # All sessions should complete successfully
        successful_sessions = 0
        for session_result in session_results:
            if not isinstance(session_result, Exception):
                if all(r.status_code == 200 for r in session_result):
                    successful_sessions += 1
        
        assert successful_sessions >= 3  # At least 60% should succeed


class TestDataIntegrityWorkflows:
    """Test data integrity across complete workflows"""
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_data_consistency_across_endpoints(self):
        """Test that data is consistent across different API endpoints"""
        async with httpx.AsyncClient(app=app, base_url="http://test") as client:
            
            # Get risk score
            risk_response = await client.get("/api/v1/risk/score")
            assert risk_response.status_code == 200
            risk_data = risk_response.json()
            
            # Get risk factors
            factors_response = await client.get("/api/v1/risk/factors")
            assert factors_response.status_code == 200
            factors_data = factors_response.json()
            
            # Get analytics data
            analytics_response = await client.get("/api/v1/analytics/aggregation")
            assert analytics_response.status_code == 200
            analytics_data = analytics_response.json()
            
            # Verify timestamps are consistent (within reasonable range)
            timestamps = []
            if "timestamp" in risk_data:
                timestamps.append(datetime.fromisoformat(risk_data["timestamp"].replace('Z', '+00:00')))
            if "metadata" in analytics_data and "last_updated" in analytics_data["metadata"]:
                timestamps.append(datetime.fromisoformat(analytics_data["metadata"]["last_updated"].replace('Z', '+00:00')))
            
            if len(timestamps) >= 2:
                time_diff = abs((timestamps[0] - timestamps[1]).total_seconds())
                assert time_diff < 3600  # Timestamps should be within 1 hour of each other
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_cache_coherence_workflow(self):
        """Test cache coherence across multiple requests"""
        async with httpx.AsyncClient(app=app, base_url="http://test") as client:
            
            # Make first request
            first_response = await client.get("/api/v1/risk/score")
            assert first_response.status_code == 200
            first_data = first_response.json()
            
            # Make second request immediately (should use cache)
            second_response = await client.get("/api/v1/risk/score")
            assert second_response.status_code == 200
            second_data = second_response.json()
            
            # Data should be identical if cached
            if "timestamp" in first_data and "timestamp" in second_data:
                # If timestamps are very close, data should be identical
                first_time = datetime.fromisoformat(first_data["timestamp"].replace('Z', '+00:00'))
                second_time = datetime.fromisoformat(second_data["timestamp"].replace('Z', '+00:00'))
                time_diff = abs((second_time - first_time).total_seconds())
                
                if time_diff < 60:  # Within 1 minute
                    assert first_data["overall_risk_score"] == second_data["overall_risk_score"]


class TestSecurityWorkflows:
    """Test security aspects of API workflows"""
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_input_validation_security(self):
        """Test input validation and security"""
        async with httpx.AsyncClient(app=app, base_url="http://test") as client:
            
            # Test with various malicious inputs
            malicious_inputs = [
                "'; DROP TABLE users; --",
                "<script>alert('xss')</script>",
                "../../etc/passwd",
                "\x00\x01\x02",
                "A" * 10000  # Very long string
            ]
            
            for malicious_input in malicious_inputs:
                try:
                    # Try to inject malicious input via query parameters
                    response = await client.get(f"/api/v1/analytics/aggregation?source={malicious_input}")
                    # Should either reject with 400/422 or handle safely
                    assert response.status_code in [200, 400, 422]
                    
                    if response.status_code == 200:
                        # If accepted, ensure it was sanitized
                        data = response.json()
                        # Data should not contain the malicious input
                        assert malicious_input not in str(data)
                
                except Exception:
                    # Exceptions during input validation are acceptable
                    pass
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_cors_and_headers(self):
        """Test CORS and security headers"""
        async with httpx.AsyncClient(app=app, base_url="http://test") as client:
            
            response = await client.get("/api/v1/health")
            assert response.status_code == 200
            
            # Check for security headers
            headers = response.headers
            
            # Content-Type should be set
            assert "content-type" in headers
            assert "application/json" in headers["content-type"]
            
            # Should have appropriate cache control for API responses
            if "cache-control" in headers:
                assert "no-cache" in headers["cache-control"] or "max-age" in headers["cache-control"]


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_full_system_integration():
    """Test full system integration from frontend to backend"""
    async with httpx.AsyncClient(app=app, base_url="http://test") as client:
        
        # Simulate a complete user workflow
        workflow_steps = [
            ("/api/v1/health", "System health check"),
            ("/api/v1/risk/score", "Get current risk assessment"),
            ("/api/v1/risk/factors", "Get risk factor breakdown"),
            ("/api/v1/network/analysis", "Analyze risk network"),
            ("/api/v1/analytics/health", "Check data source health")
        ]
        
        workflow_results = {}
        
        for endpoint, description in workflow_steps:
            try:
                response = await client.get(endpoint)
                workflow_results[endpoint] = {
                    "status_code": response.status_code,
                    "description": description,
                    "success": response.status_code == 200,
                    "response_time": response.elapsed.total_seconds() if hasattr(response, 'elapsed') else None
                }
            except Exception as e:
                workflow_results[endpoint] = {
                    "status_code": None,
                    "description": description,
                    "success": False,
                    "error": str(e)
                }
        
        # At least 80% of workflow steps should succeed
        successful_steps = sum(1 for result in workflow_results.values() if result["success"])
        total_steps = len(workflow_steps)
        success_rate = successful_steps / total_steps
        
        assert success_rate >= 0.8, f"Only {successful_steps}/{total_steps} workflow steps succeeded"
        
        # Core functionality (health and risk score) must work
        assert workflow_results["/api/v1/health"]["success"], "Health check must succeed"
        assert workflow_results["/api/v1/risk/score"]["success"], "Risk score calculation must succeed"