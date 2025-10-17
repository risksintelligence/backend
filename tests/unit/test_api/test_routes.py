"""
Tests for API route handlers
"""

import pytest
import json
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI

from src.api.main import app
from src.api.routes import risk, analytics, health


class TestHealthRoutes:
    """Test health check endpoints"""
    
    def setup_method(self):
        self.client = TestClient(app)
    
    def test_health_check_endpoint(self):
        """Test basic health check"""
        response = self.client.get("/api/v1/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
    
    @patch('src.cache.cache_manager.CacheManager.health_check')
    async def test_health_check_with_dependencies(self, mock_cache_health):
        """Test health check with dependency status"""
        mock_cache_health.return_value = {
            "redis": {"status": "healthy"},
            "postgres": {"status": "healthy"}
        }
        
        response = self.client.get("/api/v1/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"


class TestRiskRoutes:
    """Test risk assessment endpoints"""
    
    def setup_method(self):
        self.client = TestClient(app)
    
    @patch('src.ml.models.risk_scorer.RiskScorer.calculate_risk_score')
    def test_risk_score_endpoint(self, mock_risk_scorer):
        """Test risk score calculation endpoint"""
        mock_risk_scorer.return_value = {
            "overall_risk_score": 0.65,
            "confidence": 0.87,
            "risk_level": "moderate",
            "contributing_factors": {
                "economic": 0.4,
                "financial": 0.3,
                "trade": 0.2,
                "external": 0.1
            }
        }
        
        response = self.client.get("/api/v1/risk/score")
        assert response.status_code == 200
        
        data = response.json()
        assert "overall_risk_score" in data
        assert "confidence" in data
        assert "risk_level" in data
        assert data["overall_risk_score"] == 0.65
    
    @patch('src.ml.models.risk_scorer.RiskScorer.get_risk_factors')
    def test_risk_factors_endpoint(self, mock_risk_factors):
        """Test risk factors endpoint"""
        mock_risk_factors.return_value = {
            "economic_indicators": {
                "gdp_growth": {"value": 2.1, "risk_contribution": 0.15},
                "unemployment": {"value": 3.8, "risk_contribution": 0.12}
            },
            "financial_indicators": {
                "market_volatility": {"value": 0.18, "risk_contribution": 0.10}
            }
        }
        
        response = self.client.get("/api/v1/risk/factors")
        assert response.status_code == 200
        
        data = response.json()
        assert "economic_indicators" in data
        assert "financial_indicators" in data
    
    def test_risk_categories_endpoint(self):
        """Test risk categories endpoint"""
        response = self.client.get("/api/v1/risk/categories")
        assert response.status_code == 200
        
        data = response.json()
        assert "categories" in data
        assert isinstance(data["categories"], list)
    
    @patch('src.ml.models.network_analyzer.RiskNetworkAnalyzer.analyze_network')
    def test_network_analysis_endpoint(self, mock_network_analyzer):
        """Test network analysis endpoint"""
        mock_network_analyzer.return_value = {
            "nodes": [
                {"id": "bank_1", "type": "financial", "risk_score": 0.3},
                {"id": "supplier_1", "type": "supply_chain", "risk_score": 0.5}
            ],
            "edges": [
                {"source": "bank_1", "target": "supplier_1", "weight": 0.7}
            ],
            "network_metrics": {
                "clustering_coefficient": 0.45,
                "average_path_length": 2.3
            }
        }
        
        response = self.client.get("/api/v1/network/analysis")
        assert response.status_code == 200
        
        data = response.json()
        assert "nodes" in data
        assert "edges" in data
        assert "network_metrics" in data


class TestAnalyticsRoutes:
    """Test analytics endpoints"""
    
    def setup_method(self):
        self.client = TestClient(app)
    
    @patch('src.data.processors.indicator_aggregator.IndicatorAggregator.get_aggregated_data')
    def test_analytics_aggregation_endpoint(self, mock_aggregator):
        """Test analytics data aggregation"""
        mock_aggregator.return_value = {
            "economic_data": {
                "gdp": [{"date": "2023-12-01", "value": 21000.5}],
                "unemployment": [{"date": "2023-12-01", "value": 3.8}]
            },
            "metadata": {
                "last_updated": "2024-01-15T10:00:00Z",
                "data_sources": ["fred", "bls"]
            }
        }
        
        response = self.client.get("/api/v1/analytics/aggregation")
        assert response.status_code == 200
        
        data = response.json()
        assert "economic_data" in data
        assert "metadata" in data
    
    @patch('src.data.sources.fred.FREDConnector.health_check')
    @patch('src.data.sources.bea.BEAConnector.health_check')
    async def test_analytics_health_endpoint(self, mock_bea_health, mock_fred_health):
        """Test analytics health check"""
        mock_fred_health.return_value = {"status": "healthy", "response_time": 150}
        mock_bea_health.return_value = {"status": "healthy", "response_time": 200}
        
        response = self.client.get("/api/v1/analytics/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "data_sources" in data
        assert "overall_health" in data


class TestAPIErrorHandling:
    """Test API error handling"""
    
    def setup_method(self):
        self.client = TestClient(app)
    
    def test_404_endpoint(self):
        """Test 404 handling for non-existent endpoints"""
        response = self.client.get("/api/v1/nonexistent")
        assert response.status_code == 404
    
    def test_method_not_allowed(self):
        """Test 405 handling for wrong HTTP methods"""
        response = self.client.post("/api/v1/health")
        assert response.status_code == 405
    
    @patch('src.ml.models.risk_scorer.RiskScorer.calculate_risk_score')
    def test_internal_server_error_handling(self, mock_risk_scorer):
        """Test 500 error handling"""
        mock_risk_scorer.side_effect = Exception("Test error")
        
        response = self.client.get("/api/v1/risk/score")
        assert response.status_code == 500
        
        data = response.json()
        assert "error" in data
        assert "Internal server error" in data["error"]


class TestAPIValidation:
    """Test API input validation"""
    
    def setup_method(self):
        self.client = TestClient(app)
    
    def test_query_parameter_validation(self):
        """Test validation of query parameters"""
        # Test with invalid date format
        response = self.client.get("/api/v1/analytics/aggregation?start_date=invalid")
        assert response.status_code == 422  # Validation error
    
    def test_request_body_validation(self):
        """Test validation of request body"""
        # Test POST endpoint with invalid JSON
        invalid_json = {"invalid": "data"}
        response = self.client.post("/api/v1/risk/simulate", json=invalid_json)
        
        # Should handle validation appropriately
        assert response.status_code in [400, 422]


class TestAPIPerformance:
    """Test API performance characteristics"""
    
    def setup_method(self):
        self.client = TestClient(app)
    
    @patch('src.cache.cache_manager.CacheManager.get')
    def test_caching_behavior(self, mock_cache_get):
        """Test that endpoints use caching appropriately"""
        mock_cache_get.return_value = {
            "cached_data": True,
            "timestamp": "2024-01-15T10:00:00Z"
        }
        
        response = self.client.get("/api/v1/risk/score")
        
        # Verify cache was checked
        mock_cache_get.assert_called()
    
    def test_response_headers(self):
        """Test that appropriate response headers are set"""
        response = self.client.get("/api/v1/health")
        
        assert "content-type" in response.headers
        assert response.headers["content-type"] == "application/json"
    
    def test_cors_headers(self):
        """Test CORS headers are present"""
        response = self.client.get("/api/v1/health")
        
        # Should have CORS headers in production
        # This test might need adjustment based on actual CORS configuration


@pytest.mark.asyncio
class TestAsyncAPIBehavior:
    """Test async behavior of API endpoints"""
    
    async def test_concurrent_requests(self):
        """Test handling of concurrent requests"""
        import asyncio
        import httpx
        
        async with httpx.AsyncClient(app=app, base_url="http://test") as client:
            # Make multiple concurrent requests
            tasks = [
                client.get("/api/v1/health"),
                client.get("/api/v1/risk/score"),
                client.get("/api/v1/analytics/health")
            ]
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # All requests should complete successfully
            for response in responses:
                assert not isinstance(response, Exception)
                assert response.status_code == 200
    
    @patch('src.data.sources.fred.FREDConnector.fetch_data')
    async def test_timeout_handling(self, mock_fetch_data):
        """Test handling of timeouts in async operations"""
        # Mock a slow operation
        async def slow_operation():
            await asyncio.sleep(10)
            return {"data": "test"}
        
        mock_fetch_data.side_effect = slow_operation
        
        async with httpx.AsyncClient(app=app, base_url="http://test", timeout=1.0) as client:
            # This should timeout
            with pytest.raises(httpx.TimeoutException):
                await client.get("/api/v1/analytics/aggregation")