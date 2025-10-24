"""
Unit tests for the Monte Carlo Simulation Engine.
"""

import pytest
import numpy as np
import asyncio
from unittest.mock import patch, MagicMock
from datetime import datetime

from src.ml.models.simulation_engine import (
    MonteCarloEngine,
    SimulationScenario,
    SimulationParameter,
    DistributionType,
    SimulationResult
)


@pytest.mark.unit
class TestMonteCarloEngine:
    """Test cases for MonteCarloEngine."""
    
    @pytest.fixture
    def engine(self):
        """Create a MonteCarloEngine instance."""
        return MonteCarloEngine(random_seed=42)
    
    @pytest.fixture
    def sample_scenario(self):
        """Sample simulation scenario for testing."""
        parameters = [
            SimulationParameter(
                name="asset_return",
                distribution=DistributionType.NORMAL,
                parameters={"mean": 0.08, "std": 0.15},
                description="Annual asset return"
            ),
            SimulationParameter(
                name="volatility",
                distribution=DistributionType.LOGNORMAL,
                parameters={"mean": -1.5, "sigma": 0.3},
                description="Return volatility"
            )
        ]
        
        return SimulationScenario(
            scenario_id="test_scenario",
            name="Test Scenario",
            description="Test scenario for unit testing",
            parameters=parameters,
            simulation_steps=1000,
            confidence_levels=[0.90, 0.95, 0.99]
        )
    
    def test_engine_initialization(self, engine):
        """Test MonteCarloEngine initialization."""
        assert isinstance(engine, MonteCarloEngine)
        assert engine.random_seed == 42
        assert engine.simulation_history == []
    
    def test_sample_normal_distribution(self, engine):
        """Test normal distribution sampling."""
        samples = engine._sample_distribution(
            DistributionType.NORMAL,
            {"mean": 0.0, "std": 1.0},
            1000
        )
        
        assert len(samples) == 1000
        assert abs(np.mean(samples)) < 0.1  # Should be close to 0
        assert abs(np.std(samples) - 1.0) < 0.1  # Should be close to 1
    
    def test_sample_lognormal_distribution(self, engine):
        """Test log-normal distribution sampling."""
        samples = engine._sample_distribution(
            DistributionType.LOGNORMAL,
            {"mean": 0.0, "sigma": 1.0},
            1000
        )
        
        assert len(samples) == 1000
        assert np.all(samples > 0)  # Log-normal is always positive
    
    def test_sample_uniform_distribution(self, engine):
        """Test uniform distribution sampling."""
        samples = engine._sample_distribution(
            DistributionType.UNIFORM,
            {"low": 0.0, "high": 1.0},
            1000
        )
        
        assert len(samples) == 1000
        assert np.all(samples >= 0.0)
        assert np.all(samples <= 1.0)
        assert abs(np.mean(samples) - 0.5) < 0.1  # Should be around 0.5
    
    def test_sample_triangular_distribution(self, engine):
        """Test triangular distribution sampling."""
        samples = engine._sample_distribution(
            DistributionType.TRIANGULAR,
            {"left": 0.0, "mode": 0.5, "right": 1.0},
            1000
        )
        
        assert len(samples) == 1000
        assert np.all(samples >= 0.0)
        assert np.all(samples <= 1.0)
    
    def test_sample_beta_distribution(self, engine):
        """Test beta distribution sampling."""
        samples = engine._sample_distribution(
            DistributionType.BETA,
            {"alpha": 2.0, "beta": 2.0},
            1000
        )
        
        assert len(samples) == 1000
        assert np.all(samples >= 0.0)
        assert np.all(samples <= 1.0)
    
    def test_sample_exponential_distribution(self, engine):
        """Test exponential distribution sampling."""
        samples = engine._sample_distribution(
            DistributionType.EXPONENTIAL,
            {"scale": 1.0},
            1000
        )
        
        assert len(samples) == 1000
        assert np.all(samples >= 0)  # Exponential is always positive
    
    def test_sample_gamma_distribution(self, engine):
        """Test gamma distribution sampling."""
        samples = engine._sample_distribution(
            DistributionType.GAMMA,
            {"shape": 2.0, "scale": 1.0},
            1000
        )
        
        assert len(samples) == 1000
        assert np.all(samples >= 0)  # Gamma is always positive
    
    def test_sample_weibull_distribution(self, engine):
        """Test Weibull distribution sampling."""
        samples = engine._sample_distribution(
            DistributionType.WEIBULL,
            {"a": 1.5, "scale": 1.0},
            1000
        )
        
        assert len(samples) == 1000
        assert np.all(samples >= 0)  # Weibull is always positive
    
    def test_unsupported_distribution(self, engine):
        """Test error handling for unsupported distribution."""
        with pytest.raises(ValueError, match="Unsupported distribution"):
            engine._sample_distribution(
                "unsupported_distribution",
                {},
                100
            )
    
    @pytest.mark.asyncio
    async def test_generate_samples(self, engine, sample_scenario):
        """Test sample generation for scenario parameters."""
        samples = await engine._generate_samples(sample_scenario)
        
        assert "asset_return" in samples
        assert "volatility" in samples
        assert len(samples["asset_return"]) == 1000
        assert len(samples["volatility"]) == 1000
        assert np.all(samples["volatility"] > 0)  # Log-normal is positive
    
    def test_apply_correlation_invalid_matrix(self, engine):
        """Test correlation application with invalid matrix."""
        samples = {
            "param1": np.random.normal(0, 1, 100),
            "param2": np.random.normal(0, 1, 100)
        }
        
        # Wrong size correlation matrix
        correlation_matrix = np.array([[1.0, 0.5], [0.5, 1.0], [0.3, 0.7, 1.0]])
        
        # Should return original samples due to dimension mismatch
        result = engine._apply_correlation(samples, correlation_matrix)
        assert result == samples
    
    def test_apply_correlation_valid_matrix(self, engine):
        """Test correlation application with valid matrix."""
        samples = {
            "param1": np.random.normal(0, 1, 100),
            "param2": np.random.normal(0, 1, 100)
        }
        
        # Valid correlation matrix
        correlation_matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
        
        result = engine._apply_correlation(samples, correlation_matrix)
        
        assert "param1" in result
        assert "param2" in result
        assert len(result["param1"]) == 100
        assert len(result["param2"]) == 100
    
    @pytest.mark.asyncio
    async def test_calculate_scenario_results(self, engine, sample_scenario):
        """Test scenario results calculation."""
        samples = {
            "asset_return": np.random.normal(0.08, 0.15, 1000),
            "volatility": np.random.lognormal(-1.5, 0.3, 1000),
            "risk_free_rate": np.random.normal(0.03, 0.01, 1000)
        }
        
        results = await engine._calculate_scenario_results(sample_scenario, samples)
        
        # Should include original samples
        assert "asset_return" in results
        assert "volatility" in results
        assert "risk_free_rate" in results
        
        # Should include calculated metrics
        assert "portfolio_value" in results
        assert "risk_adjusted_return" in results
        assert "max_drawdown" in results
        assert "var_95" in results
        assert "var_99" in results
        assert "expected_shortfall_95" in results
    
    def test_calculate_statistics(self, engine):
        """Test statistics calculation."""
        results = {
            "test_variable": np.random.normal(100, 15, 1000)
        }
        
        statistics = engine._calculate_statistics(results)
        
        assert "test_variable" in statistics
        stats = statistics["test_variable"]
        
        assert "mean" in stats
        assert "median" in stats
        assert "std" in stats
        assert "variance" in stats
        assert "min" in stats
        assert "max" in stats
        assert "skewness" in stats
        assert "kurtosis" in stats
        assert "q25" in stats
        assert "q75" in stats
        assert "iqr" in stats
        
        # Verify statistical properties
        assert abs(stats["mean"] - 100) < 5  # Should be close to 100
        assert abs(stats["std"] - 15) < 3   # Should be close to 15
    
    def test_calculate_risk_metrics(self, engine):
        """Test risk metrics calculation."""
        results = {
            "portfolio_value": np.random.normal(1000000, 150000, 10000)
        }
        confidence_levels = [0.90, 0.95, 0.99]
        
        risk_metrics = engine._calculate_risk_metrics(results, confidence_levels)
        
        # Check VaR metrics
        assert "var_90" in risk_metrics
        assert "var_95" in risk_metrics
        assert "var_99" in risk_metrics
        
        # Check Expected Shortfall
        assert "expected_shortfall_90" in risk_metrics
        assert "expected_shortfall_95" in risk_metrics
        assert "expected_shortfall_99" in risk_metrics
        
        # Check risk ratios
        assert "sharpe_ratio" in risk_metrics
        assert "information_ratio" in risk_metrics
        
        # VaR should be ordered (95% VaR < 90% VaR)
        assert risk_metrics["var_95"] < risk_metrics["var_90"]
        assert risk_metrics["var_99"] < risk_metrics["var_95"]
    
    def test_calculate_confidence_intervals(self, engine):
        """Test confidence intervals calculation."""
        results = {
            "test_variable": np.random.normal(100, 15, 1000)
        }
        confidence_levels = [0.90, 0.95]
        
        confidence_intervals = engine._calculate_confidence_intervals(
            results, confidence_levels
        )
        
        assert "test_variable" in confidence_intervals
        intervals = confidence_intervals["test_variable"]
        
        assert 0.90 in intervals
        assert 0.95 in intervals
        
        # Each interval should be a tuple (lower, upper)
        lower_90, upper_90 = intervals[0.90]
        lower_95, upper_95 = intervals[0.95]
        
        assert lower_90 < upper_90
        assert lower_95 < upper_95
        
        # 95% interval should be wider than 90% interval
        assert (upper_95 - lower_95) > (upper_90 - lower_90)
    
    def test_assess_convergence(self, engine):
        """Test convergence assessment."""
        results = {
            "portfolio_value": np.random.normal(1000000, 150000, 10000)
        }
        
        convergence_metrics = engine._assess_convergence(results)
        
        assert "portfolio_value_mc_standard_error" in convergence_metrics
        assert "portfolio_value_relative_precision" in convergence_metrics
        assert "portfolio_value_converged" in convergence_metrics
        
        # Monte Carlo standard error should be positive
        assert convergence_metrics["portfolio_value_mc_standard_error"] > 0
        
        # Relative precision should be reasonable for large sample
        assert convergence_metrics["portfolio_value_relative_precision"] < 0.1
    
    @pytest.mark.asyncio
    async def test_full_simulation_run(self, engine, sample_scenario):
        """Test complete simulation execution."""
        result = await engine.run_simulation(sample_scenario, "test_run")
        
        assert isinstance(result, SimulationResult)
        assert result.scenario_id == "test_scenario"
        assert result.run_id == "test_run"
        assert isinstance(result.timestamp, datetime)
        assert result.execution_time > 0
        assert result.sample_size == 1000
        
        # Check results structure
        assert isinstance(result.results, dict)
        assert isinstance(result.statistics, dict)
        assert isinstance(result.risk_metrics, dict)
        assert isinstance(result.confidence_intervals, dict)
        assert isinstance(result.convergence_metrics, dict)
        
        # Verify simulation was added to history
        assert len(engine.simulation_history) == 1
        assert engine.simulation_history[0] == result
    
    def test_create_financial_portfolio_template(self, engine):
        """Test financial portfolio template creation."""
        template = engine.create_scenario_template("test", "financial_portfolio")
        
        assert template.scenario_id == "portfolio_test"
        assert template.name == "Portfolio Risk: test"
        assert len(template.parameters) == 4
        
        param_names = [p.name for p in template.parameters]
        assert "asset_return" in param_names
        assert "volatility" in param_names
        assert "risk_free_rate" in param_names
        assert "correlation_factor" in param_names
    
    def test_create_credit_risk_template(self, engine):
        """Test credit risk template creation."""
        template = engine.create_scenario_template("test", "credit_risk")
        
        assert template.scenario_id == "credit_test"
        assert template.name == "Credit Risk: test"
        assert len(template.parameters) == 3
        
        param_names = [p.name for p in template.parameters]
        assert "default_probability" in param_names
        assert "loss_given_default" in param_names
        assert "exposure_at_default" in param_names
    
    def test_create_operational_risk_template(self, engine):
        """Test operational risk template creation."""
        template = engine.create_scenario_template("test", "operational_risk")
        
        assert template.scenario_id == "operational_test"
        assert template.name == "Operational Risk: test"
        assert len(template.parameters) == 3
        
        param_names = [p.name for p in template.parameters]
        assert "frequency" in param_names
        assert "severity" in param_names
        assert "recovery_rate" in param_names
    
    def test_create_systemic_risk_template(self, engine):
        """Test systemic risk template creation."""
        template = engine.create_scenario_template("test", "systemic_risk")
        
        assert template.scenario_id == "systemic_test"
        assert template.name == "Systemic Risk: test"
        assert len(template.parameters) == 4
        assert template.correlation_matrix is not None
        assert template.correlation_matrix.shape == (4, 4)
        
        param_names = [p.name for p in template.parameters]
        assert "contagion_probability" in param_names
        assert "system_shock" in param_names
        assert "recovery_time" in param_names
        assert "interconnectedness" in param_names
    
    def test_create_unknown_template_type(self, engine):
        """Test error handling for unknown template type."""
        with pytest.raises(ValueError, match="Unknown template type"):
            engine.create_scenario_template("test", "unknown_type")
    
    def test_export_results_json(self, engine):
        """Test JSON export of simulation results."""
        # Create a simple result for testing
        from datetime import datetime
        result = SimulationResult(
            scenario_id="test",
            run_id="test_run",
            timestamp=datetime.utcnow(),
            results={"test": np.array([1, 2, 3])},
            statistics={"test": {"mean": 2.0}},
            risk_metrics={"var_95": -100},
            confidence_intervals={"test": {0.95: (1.0, 3.0)}},
            convergence_metrics={"test_converged": 1.0},
            execution_time=1.5,
            sample_size=1000
        )
        
        json_export = engine.export_results(result, "json")
        
        assert isinstance(json_export, str)
        import json
        data = json.loads(json_export)
        
        assert data["scenario_id"] == "test"
        assert data["run_id"] == "test_run"
        assert data["execution_time"] == 1.5
        assert data["sample_size"] == 1000
    
    def test_export_results_summary(self, engine):
        """Test summary export of simulation results."""
        from datetime import datetime
        result = SimulationResult(
            scenario_id="test",
            run_id="test_run",
            timestamp=datetime.utcnow(),
            results={"portfolio_value": np.array([1000, 1100, 900])},
            statistics={"portfolio_value": {"mean": 1000, "std": 100}},
            risk_metrics={"var_95": 850, "expected_shortfall_95": 800},
            confidence_intervals={},
            convergence_metrics={"portfolio_value_converged": 1.0},
            execution_time=1.5,
            sample_size=1000
        )
        
        summary = engine.export_results(result, "summary")
        
        assert isinstance(summary, dict)
        assert summary["scenario"] == "test"
        assert "execution_time" in summary
        assert "sample_size" in summary
        assert "key_metrics" in summary
        assert "convergence" in summary
    
    def test_export_results_unsupported_format(self, engine):
        """Test error handling for unsupported export format."""
        from datetime import datetime
        result = SimulationResult(
            scenario_id="test",
            run_id="test_run",
            timestamp=datetime.utcnow(),
            results={},
            statistics={},
            risk_metrics={},
            confidence_intervals={},
            convergence_metrics={},
            execution_time=1.5,
            sample_size=1000
        )
        
        with pytest.raises(ValueError, match="Unsupported export format"):
            engine.export_results(result, "unsupported_format")
    
    def test_calculate_portfolio_value(self, engine):
        """Test portfolio value calculation."""
        returns = np.array([0.05, 0.08, 0.03])
        volatility = np.array([0.15, 0.20, 0.12])
        initial_value = 1000000.0
        
        portfolio_values = engine._calculate_portfolio_value(
            returns, volatility, initial_value
        )
        
        assert len(portfolio_values) == 3
        assert np.all(portfolio_values > 0)  # Values should be positive
    
    def test_calculate_max_drawdown(self, engine):
        """Test maximum drawdown calculation."""
        values = np.array([1000, 1100, 900, 950, 1050])
        
        max_drawdown = engine._calculate_max_drawdown(values)
        
        assert len(max_drawdown) == len(values)
        assert np.all(max_drawdown <= 0)  # Drawdown should be non-positive