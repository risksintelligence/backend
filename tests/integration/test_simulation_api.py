"""
Integration tests for Simulation API endpoints.
"""

import pytest
import json
from unittest.mock import patch, AsyncMock


@pytest.mark.integration
class TestSimulationEndpoints:
    """Test simulation API endpoints."""
    
    @pytest.mark.asyncio
    async def test_get_scenario_templates(self, async_client):
        """Test getting available scenario templates."""
        response = await async_client.get("/api/v1/simulation/templates")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "data" in data
        assert "templates" in data["data"]
        
        templates = data["data"]["templates"]
        assert len(templates) > 0
        
        # Check template structure
        template = templates[0]
        assert "template_type" in template
        assert "name" in template
        assert "description" in template
        assert "parameters" in template
    
    @pytest.mark.asyncio
    async def test_create_scenario_from_template(self, async_client):
        """Test creating scenario from template."""
        template_request = {
            "template_name": "test_portfolio",
            "template_type": "financial_portfolio"
        }
        
        response = await async_client.post(
            "/api/v1/simulation/templates/financial_portfolio",
            json=template_request
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "data" in data
        assert "scenario" in data["data"]
        
        scenario = data["data"]["scenario"]
        assert scenario["scenario_id"] == "portfolio_test_portfolio"
        assert scenario["name"] == "Portfolio Risk: test_portfolio"
        assert len(scenario["parameters"]) == 4
    
    @pytest.mark.asyncio
    async def test_create_scenario_invalid_template(self, async_client):
        """Test creating scenario with invalid template type."""
        template_request = {
            "template_name": "test",
            "template_type": "invalid_template"
        }
        
        response = await async_client.post(
            "/api/v1/simulation/templates/invalid_type",
            json=template_request
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "Invalid template type" in data["detail"]
    
    @pytest.mark.asyncio
    async def test_validate_scenario(self, async_client):
        """Test scenario validation endpoint."""
        scenario_request = {
            "scenario_id": "test_scenario",
            "name": "Test Scenario",
            "description": "Test scenario for validation",
            "parameters": [
                {
                    "name": "asset_return",
                    "distribution": "normal",
                    "parameters": {
                        "mean": 0.08,
                        "std": 0.15
                    },
                    "description": "Asset return"
                }
            ],
            "simulation_steps": 5000,
            "confidence_levels": [0.90, 0.95, 0.99]
        }
        
        response = await async_client.post(
            "/api/v1/simulation/validate",
            json=scenario_request
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "data" in data
        
        validation = data["data"]
        assert validation["valid"] is True
        assert isinstance(validation["errors"], list)
        assert isinstance(validation["warnings"], list)
    
    @pytest.mark.asyncio
    async def test_validate_scenario_invalid_parameters(self, async_client):
        """Test scenario validation with invalid parameters."""
        scenario_request = {
            "scenario_id": "invalid_scenario",
            "name": "Invalid Scenario",
            "description": "Test scenario with invalid parameters",
            "parameters": [
                {
                    "name": "invalid_param",
                    "distribution": "normal",
                    "parameters": {
                        "mean": 0.08
                        # Missing 'std' parameter
                    }
                }
            ],
            "simulation_steps": 1000
        }
        
        response = await async_client.post(
            "/api/v1/simulation/validate",
            json=scenario_request
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        
        validation = data["data"]
        assert validation["valid"] is False
        assert len(validation["errors"]) > 0
    
    @pytest.mark.asyncio
    async def test_validate_scenario_invalid_correlation_matrix(self, async_client):
        """Test scenario validation with invalid correlation matrix."""
        scenario_request = {
            "scenario_id": "correlation_test",
            "name": "Correlation Test",
            "description": "Test with invalid correlation matrix",
            "parameters": [
                {
                    "name": "param1",
                    "distribution": "normal",
                    "parameters": {"mean": 0.0, "std": 1.0}
                },
                {
                    "name": "param2", 
                    "distribution": "normal",
                    "parameters": {"mean": 0.0, "std": 1.0}
                }
            ],
            "correlation_matrix": [
                [1.0, 1.5],  # Invalid correlation > 1
                [1.5, 1.0]
            ],
            "simulation_steps": 1000
        }
        
        response = await async_client.post(
            "/api/v1/simulation/validate",
            json=scenario_request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        validation = data["data"]
        assert validation["valid"] is False
        assert any("between -1 and 1" in error for error in validation["errors"])
    
    @pytest.mark.asyncio
    async def test_run_simulation(self, async_client):
        """Test running a Monte Carlo simulation."""
        simulation_request = {
            "scenario": {
                "scenario_id": "test_simulation",
                "name": "Test Simulation",
                "description": "Test Monte Carlo simulation",
                "parameters": [
                    {
                        "name": "asset_return",
                        "distribution": "normal",
                        "parameters": {
                            "mean": 0.08,
                            "std": 0.15
                        },
                        "description": "Asset return"
                    },
                    {
                        "name": "volatility",
                        "distribution": "lognormal",
                        "parameters": {
                            "mean": -1.5,
                            "sigma": 0.3
                        },
                        "description": "Volatility"
                    }
                ],
                "simulation_steps": 1000,
                "confidence_levels": [0.90, 0.95, 0.99]
            },
            "run_id": "test_run_001"
        }
        
        response = await async_client.post(
            "/api/v1/simulation/run",
            json=simulation_request
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "data" in data
        
        result = data["data"]
        assert result["scenario_id"] == "test_simulation"
        assert result["run_id"] == "test_run_001"
        assert result["sample_size"] == 1000
        assert result["execution_time"] > 0
        
        # Check result structure
        assert "statistics" in result
        assert "risk_metrics" in result
        assert "confidence_intervals" in result
        assert "convergence_metrics" in result
        assert "summary" in result
    
    @pytest.mark.asyncio
    async def test_run_simulation_cached_result(self, async_client, mock_cache_manager):
        """Test simulation with cached result."""
        cached_result = {
            "scenario_id": "cached_simulation",
            "run_id": "cached_run",
            "execution_time": 1.5,
            "sample_size": 1000,
            "statistics": {"asset_return": {"mean": 0.08}},
            "risk_metrics": {"var_95": -150000},
            "confidence_intervals": {},
            "convergence_metrics": {"converged": True}
        }
        
        mock_cache_manager.get.return_value = cached_result
        
        simulation_request = {
            "scenario": {
                "scenario_id": "cached_simulation",
                "name": "Cached Simulation",
                "description": "Test cached simulation result",
                "parameters": [
                    {
                        "name": "asset_return",
                        "distribution": "normal",
                        "parameters": {"mean": 0.08, "std": 0.15}
                    }
                ],
                "simulation_steps": 1000
            }
        }
        
        response = await async_client.post(
            "/api/v1/simulation/run",
            json=simulation_request
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["source"] == "cache"
        assert data["data"] == cached_result
    
    @pytest.mark.asyncio
    async def test_run_simulation_invalid_distribution(self, async_client):
        """Test simulation with invalid distribution type."""
        simulation_request = {
            "scenario": {
                "scenario_id": "invalid_distribution",
                "name": "Invalid Distribution Test",
                "description": "Test with invalid distribution",
                "parameters": [
                    {
                        "name": "invalid_param",
                        "distribution": "unknown_distribution",
                        "parameters": {"param1": 1.0}
                    }
                ],
                "simulation_steps": 1000
            }
        }
        
        response = await async_client.post(
            "/api/v1/simulation/run",
            json=simulation_request
        )
        
        assert response.status_code == 500
        data = response.json()
        assert "Simulation execution failed" in data["detail"]
    
    @pytest.mark.asyncio
    async def test_get_simulation_result(self, async_client):
        """Test retrieving simulation result by run ID."""
        # First run a simulation
        simulation_request = {
            "scenario": {
                "scenario_id": "result_test",
                "name": "Result Test",
                "description": "Test for result retrieval",
                "parameters": [
                    {
                        "name": "test_param",
                        "distribution": "normal",
                        "parameters": {"mean": 0.0, "std": 1.0}
                    }
                ],
                "simulation_steps": 500
            },
            "run_id": "result_test_run"
        }
        
        run_response = await async_client.post(
            "/api/v1/simulation/run",
            json=simulation_request
        )
        assert run_response.status_code == 200
        
        # Then retrieve the result
        response = await async_client.get("/api/v1/simulation/results/result_test_run")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "data" in data
        
        result = data["data"]
        assert result["run_id"] == "result_test_run"
        assert result["scenario_id"] == "result_test"
    
    @pytest.mark.asyncio
    async def test_get_simulation_result_not_found(self, async_client):
        """Test retrieving non-existent simulation result."""
        response = await async_client.get("/api/v1/simulation/results/nonexistent_run")
        
        assert response.status_code == 404
        data = response.json()
        assert "Simulation result not found" in data["detail"]
    
    @pytest.mark.asyncio
    async def test_get_simulation_history(self, async_client):
        """Test getting simulation execution history."""
        # Run a few simulations first
        for i in range(3):
            simulation_request = {
                "scenario": {
                    "scenario_id": f"history_test_{i}",
                    "name": f"History Test {i}",
                    "description": "Test for history",
                    "parameters": [
                        {
                            "name": "test_param",
                            "distribution": "normal",
                            "parameters": {"mean": 0.0, "std": 1.0}
                        }
                    ],
                    "simulation_steps": 500
                },
                "run_id": f"history_run_{i}"
            }
            
            run_response = await async_client.post(
                "/api/v1/simulation/run",
                json=simulation_request
            )
            assert run_response.status_code == 200
        
        # Get history
        response = await async_client.get("/api/v1/simulation/history?limit=5&offset=0")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "data" in data
        
        history_data = data["data"]
        assert "simulations" in history_data
        assert "pagination" in history_data
        
        simulations = history_data["simulations"]
        assert len(simulations) >= 3
        
        # Check simulation summary structure
        simulation = simulations[0]
        assert "run_id" in simulation
        assert "scenario_id" in simulation
        assert "timestamp" in simulation
        assert "execution_time" in simulation
        assert "sample_size" in simulation
        assert "convergence_status" in simulation
    
    @pytest.mark.asyncio
    async def test_get_supported_distributions(self, async_client):
        """Test getting supported probability distributions."""
        response = await async_client.get("/api/v1/simulation/distributions")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "data" in data
        
        distributions_data = data["data"]
        assert "distributions" in distributions_data
        assert "total_distributions" in distributions_data
        
        distributions = distributions_data["distributions"]
        assert len(distributions) > 0
        
        # Check that common distributions are included
        assert "normal" in distributions
        assert "uniform" in distributions
        assert "lognormal" in distributions
        
        # Check distribution info structure
        normal_dist = distributions["normal"]
        assert "name" in normal_dist
        assert "parameters" in normal_dist
        assert "description" in normal_dist
        assert "use_cases" in normal_dist
        assert "mean" in normal_dist["parameters"]
        assert "std" in normal_dist["parameters"]
    
    @pytest.mark.asyncio
    async def test_simulation_with_correlation_matrix(self, async_client):
        """Test simulation with correlation matrix."""
        simulation_request = {
            "scenario": {
                "scenario_id": "correlation_simulation",
                "name": "Correlation Simulation",
                "description": "Test simulation with correlated parameters",
                "parameters": [
                    {
                        "name": "param1",
                        "distribution": "normal",
                        "parameters": {"mean": 0.0, "std": 1.0}
                    },
                    {
                        "name": "param2",
                        "distribution": "normal",
                        "parameters": {"mean": 0.0, "std": 1.0}
                    }
                ],
                "correlation_matrix": [
                    [1.0, 0.5],
                    [0.5, 1.0]
                ],
                "simulation_steps": 1000,
                "confidence_levels": [0.90, 0.95]
            }
        }
        
        response = await async_client.post(
            "/api/v1/simulation/run",
            json=simulation_request
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        
        result = data["data"]
        assert "param1" in result["statistics"]
        assert "param2" in result["statistics"]
    
    @pytest.mark.asyncio
    async def test_simulation_warning_for_small_sample_size(self, async_client):
        """Test validation warning for small sample size."""
        scenario_request = {
            "scenario_id": "small_sample",
            "name": "Small Sample Test",
            "description": "Test with small sample size",
            "parameters": [
                {
                    "name": "test_param",
                    "distribution": "normal",
                    "parameters": {"mean": 0.0, "std": 1.0}
                }
            ],
            "simulation_steps": 500  # Small sample size
        }
        
        response = await async_client.post(
            "/api/v1/simulation/validate",
            json=scenario_request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        validation = data["data"]
        assert validation["valid"] is True
        assert any("unreliable results" in warning for warning in validation["warnings"])
    
    @pytest.mark.asyncio
    async def test_simulation_warning_for_large_sample_size(self, async_client):
        """Test validation warning for large sample size."""
        scenario_request = {
            "scenario_id": "large_sample",
            "name": "Large Sample Test", 
            "description": "Test with large sample size",
            "parameters": [
                {
                    "name": "test_param",
                    "distribution": "normal",
                    "parameters": {"mean": 0.0, "std": 1.0}
                }
            ],
            "simulation_steps": 150000  # Large sample size
        }
        
        response = await async_client.post(
            "/api/v1/simulation/validate",
            json=scenario_request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        validation = data["data"]
        assert validation["valid"] is True
        assert any("long execution times" in warning for warning in validation["warnings"])