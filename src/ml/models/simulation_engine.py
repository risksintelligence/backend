"""
Monte Carlo Simulation Engine for RiskX Platform.

This module provides comprehensive Monte Carlo simulation capabilities for risk assessment,
scenario analysis, and uncertainty quantification in financial and systemic risk models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import asyncio
from datetime import datetime, timedelta
import logging
from scipy import stats
import json

logger = logging.getLogger(__name__)


class DistributionType(Enum):
    """Supported probability distributions for Monte Carlo simulations."""
    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    UNIFORM = "uniform"
    TRIANGULAR = "triangular"
    BETA = "beta"
    EXPONENTIAL = "exponential"
    GAMMA = "gamma"
    WEIBULL = "weibull"


@dataclass
class SimulationParameter:
    """Individual parameter for Monte Carlo simulation."""
    name: str
    distribution: DistributionType
    parameters: Dict[str, float]
    correlation_group: Optional[str] = None
    description: Optional[str] = None


@dataclass
class SimulationScenario:
    """Complete simulation scenario configuration."""
    scenario_id: str
    name: str
    description: str
    parameters: List[SimulationParameter]
    correlation_matrix: Optional[np.ndarray] = None
    simulation_steps: int = 10000
    confidence_levels: List[float] = None
    time_horizon: Optional[int] = None
    
    def __post_init__(self):
        if self.confidence_levels is None:
            self.confidence_levels = [0.90, 0.95, 0.99]


@dataclass
class SimulationResult:
    """Results from Monte Carlo simulation."""
    scenario_id: str
    run_id: str
    timestamp: datetime
    results: Dict[str, np.ndarray]
    statistics: Dict[str, Dict[str, float]]
    risk_metrics: Dict[str, float]
    confidence_intervals: Dict[str, Dict[float, Tuple[float, float]]]
    convergence_metrics: Dict[str, float]
    execution_time: float
    sample_size: int


class MonteCarloEngine:
    """
    Advanced Monte Carlo simulation engine for risk modeling.
    
    Supports multiple probability distributions, correlation modeling,
    scenario analysis, and comprehensive risk metrics calculation.
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        """Initialize Monte Carlo engine."""
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
        
        self.simulation_history: List[SimulationResult] = []
        
    async def run_simulation(
        self,
        scenario: SimulationScenario,
        run_id: Optional[str] = None
    ) -> SimulationResult:
        """
        Execute Monte Carlo simulation for given scenario.
        
        Args:
            scenario: Simulation scenario configuration
            run_id: Optional identifier for this simulation run
            
        Returns:
            Complete simulation results with statistics and risk metrics
        """
        start_time = datetime.utcnow()
        execution_start = datetime.now()
        
        if run_id is None:
            run_id = f"simulation_{int(start_time.timestamp())}"
        
        logger.info(f"Starting Monte Carlo simulation: {scenario.scenario_id}")
        
        try:
            # Generate random samples for all parameters
            samples = await self._generate_samples(scenario)
            
            # Calculate derived metrics and risk measures
            results = await self._calculate_scenario_results(scenario, samples)
            
            # Compute statistical metrics
            statistics = self._calculate_statistics(results)
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(results, scenario.confidence_levels)
            
            # Calculate confidence intervals
            confidence_intervals = self._calculate_confidence_intervals(
                results, scenario.confidence_levels
            )
            
            # Assess convergence
            convergence_metrics = self._assess_convergence(results)
            
            execution_time = (datetime.now() - execution_start).total_seconds()
            
            # Create simulation result
            result = SimulationResult(
                scenario_id=scenario.scenario_id,
                run_id=run_id,
                timestamp=start_time,
                results=results,
                statistics=statistics,
                risk_metrics=risk_metrics,
                confidence_intervals=confidence_intervals,
                convergence_metrics=convergence_metrics,
                execution_time=execution_time,
                sample_size=scenario.simulation_steps
            )
            
            self.simulation_history.append(result)
            logger.info(f"Simulation completed: {run_id} in {execution_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Simulation failed for {scenario.scenario_id}: {e}")
            raise
    
    async def _generate_samples(
        self,
        scenario: SimulationScenario
    ) -> Dict[str, np.ndarray]:
        """Generate random samples for all scenario parameters."""
        samples = {}
        
        # Generate independent samples for each parameter
        for param in scenario.parameters:
            samples[param.name] = self._sample_distribution(
                param.distribution,
                param.parameters,
                scenario.simulation_steps
            )
        
        # Apply correlation structure if specified
        if scenario.correlation_matrix is not None:
            samples = self._apply_correlation(samples, scenario.correlation_matrix)
        
        return samples
    
    def _sample_distribution(
        self,
        distribution: DistributionType,
        parameters: Dict[str, float],
        n_samples: int
    ) -> np.ndarray:
        """Generate samples from specified probability distribution."""
        if distribution == DistributionType.NORMAL:
            return np.random.normal(
                parameters['mean'],
                parameters['std'],
                n_samples
            )
        
        elif distribution == DistributionType.LOGNORMAL:
            return np.random.lognormal(
                parameters['mean'],
                parameters['sigma'],
                n_samples
            )
        
        elif distribution == DistributionType.UNIFORM:
            return np.random.uniform(
                parameters['low'],
                parameters['high'],
                n_samples
            )
        
        elif distribution == DistributionType.TRIANGULAR:
            return np.random.triangular(
                parameters['left'],
                parameters['mode'],
                parameters['right'],
                n_samples
            )
        
        elif distribution == DistributionType.BETA:
            return np.random.beta(
                parameters['alpha'],
                parameters['beta'],
                n_samples
            )
        
        elif distribution == DistributionType.EXPONENTIAL:
            return np.random.exponential(
                parameters['scale'],
                n_samples
            )
        
        elif distribution == DistributionType.GAMMA:
            return np.random.gamma(
                parameters['shape'],
                parameters['scale'],
                n_samples
            )
        
        elif distribution == DistributionType.WEIBULL:
            return np.random.weibull(
                parameters['a'],
                n_samples
            ) * parameters.get('scale', 1.0)
        
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")
    
    def _apply_correlation(
        self,
        samples: Dict[str, np.ndarray],
        correlation_matrix: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Apply correlation structure to independent samples."""
        param_names = list(samples.keys())
        n_params = len(param_names)
        
        if correlation_matrix.shape != (n_params, n_params):
            logger.warning("Correlation matrix dimensions mismatch, using independent samples")
            return samples
        
        # Convert to standard normal
        standard_samples = np.zeros((len(samples[param_names[0]]), n_params))
        for i, param in enumerate(param_names):
            standard_samples[:, i] = stats.norm.ppf(
                stats.rankdata(samples[param]) / (len(samples[param]) + 1)
            )
        
        # Apply correlation using Cholesky decomposition
        try:
            chol = np.linalg.cholesky(correlation_matrix)
            correlated_samples = standard_samples @ chol.T
            
            # Convert back to original distributions
            correlated_dict = {}
            for i, param in enumerate(param_names):
                uniform_samples = stats.norm.cdf(correlated_samples[:, i])
                # Use inverse transform to get back to original distribution
                correlated_dict[param] = self._inverse_transform_to_original(
                    uniform_samples, samples[param]
                )
            
            return correlated_dict
            
        except np.linalg.LinAlgError:
            logger.warning("Correlation matrix not positive definite, using independent samples")
            return samples
    
    def _inverse_transform_to_original(
        self,
        uniform_samples: np.ndarray,
        original_samples: np.ndarray
    ) -> np.ndarray:
        """Transform uniform samples back to original distribution using inverse CDF."""
        sorted_original = np.sort(original_samples)
        indices = (uniform_samples * (len(sorted_original) - 1)).astype(int)
        indices = np.clip(indices, 0, len(sorted_original) - 1)
        return sorted_original[indices]
    
    async def _calculate_scenario_results(
        self,
        scenario: SimulationScenario,
        samples: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Calculate scenario-specific results from parameter samples."""
        results = {}
        
        # Include all parameter samples
        results.update(samples)
        
        # Calculate portfolio value (example composite metric)
        if all(param in samples for param in ['asset_return', 'volatility']):
            results['portfolio_value'] = self._calculate_portfolio_value(
                samples['asset_return'],
                samples['volatility']
            )
        
        # Calculate risk-adjusted return
        if 'portfolio_value' in results and 'risk_free_rate' in samples:
            results['risk_adjusted_return'] = (
                results['portfolio_value'] - samples['risk_free_rate']
            )
        
        # Calculate maximum drawdown
        if 'portfolio_value' in results:
            results['max_drawdown'] = self._calculate_max_drawdown(
                results['portfolio_value']
            )
        
        # Calculate Value at Risk (VaR)
        if 'portfolio_value' in results:
            results['var_95'] = np.percentile(results['portfolio_value'], 5)
            results['var_99'] = np.percentile(results['portfolio_value'], 1)
        
        # Calculate Expected Shortfall (Conditional VaR)
        if 'portfolio_value' in results:
            var_95 = results['var_95']
            results['expected_shortfall_95'] = np.mean(
                results['portfolio_value'][results['portfolio_value'] <= var_95]
            )
        
        return results
    
    def _calculate_portfolio_value(
        self,
        returns: np.ndarray,
        volatility: np.ndarray,
        initial_value: float = 1000000.0
    ) -> np.ndarray:
        """Calculate portfolio value evolution."""
        # Simple geometric return calculation
        portfolio_returns = returns - 0.5 * volatility**2 + volatility * np.random.normal(0, 1, len(returns))
        return initial_value * np.exp(portfolio_returns)
    
    def _calculate_max_drawdown(self, values: np.ndarray) -> np.ndarray:
        """Calculate maximum drawdown for each simulation path."""
        # For simplicity, calculate single drawdown per simulation
        return np.minimum(values - np.max(values), 0) / np.max(values)
    
    def _calculate_statistics(
        self,
        results: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate comprehensive statistics for all result variables."""
        statistics = {}
        
        for var_name, values in results.items():
            if isinstance(values, np.ndarray) and values.size > 0:
                statistics[var_name] = {
                    'mean': float(np.mean(values)),
                    'median': float(np.median(values)),
                    'std': float(np.std(values)),
                    'variance': float(np.var(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'skewness': float(stats.skew(values)),
                    'kurtosis': float(stats.kurtosis(values)),
                    'q25': float(np.percentile(values, 25)),
                    'q75': float(np.percentile(values, 75)),
                    'iqr': float(np.percentile(values, 75) - np.percentile(values, 25))
                }
        
        return statistics
    
    def _calculate_risk_metrics(
        self,
        results: Dict[str, np.ndarray],
        confidence_levels: List[float]
    ) -> Dict[str, float]:
        """Calculate comprehensive risk metrics."""
        risk_metrics = {}
        
        if 'portfolio_value' in results:
            portfolio_values = results['portfolio_value']
            
            # Value at Risk for different confidence levels
            for cl in confidence_levels:
                percentile = (1 - cl) * 100
                risk_metrics[f'var_{int(cl*100)}'] = float(np.percentile(portfolio_values, percentile))
            
            # Expected Shortfall (Conditional VaR)
            for cl in confidence_levels:
                percentile = (1 - cl) * 100
                var = np.percentile(portfolio_values, percentile)
                tail_values = portfolio_values[portfolio_values <= var]
                if len(tail_values) > 0:
                    risk_metrics[f'expected_shortfall_{int(cl*100)}'] = float(np.mean(tail_values))
            
            # Risk ratios
            mean_return = np.mean(portfolio_values)
            std_return = np.std(portfolio_values)
            
            if std_return > 0:
                risk_metrics['sharpe_ratio'] = mean_return / std_return
                risk_metrics['information_ratio'] = mean_return / std_return
            
            # Downside deviation and Sortino ratio
            downside_returns = portfolio_values[portfolio_values < mean_return]
            if len(downside_returns) > 0:
                downside_deviation = np.std(downside_returns)
                if downside_deviation > 0:
                    risk_metrics['sortino_ratio'] = mean_return / downside_deviation
                risk_metrics['downside_deviation'] = downside_deviation
            
            # Maximum drawdown
            if 'max_drawdown' in results:
                risk_metrics['max_drawdown'] = float(np.min(results['max_drawdown']))
                risk_metrics['avg_drawdown'] = float(np.mean(results['max_drawdown']))
        
        return risk_metrics
    
    def _calculate_confidence_intervals(
        self,
        results: Dict[str, np.ndarray],
        confidence_levels: List[float]
    ) -> Dict[str, Dict[float, Tuple[float, float]]]:
        """Calculate confidence intervals for all variables."""
        confidence_intervals = {}
        
        for var_name, values in results.items():
            if isinstance(values, np.ndarray) and values.size > 0:
                confidence_intervals[var_name] = {}
                
                for cl in confidence_levels:
                    alpha = 1 - cl
                    lower_percentile = (alpha / 2) * 100
                    upper_percentile = (1 - alpha / 2) * 100
                    
                    lower = float(np.percentile(values, lower_percentile))
                    upper = float(np.percentile(values, upper_percentile))
                    
                    confidence_intervals[var_name][cl] = (lower, upper)
        
        return confidence_intervals
    
    def _assess_convergence(
        self,
        results: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """Assess simulation convergence using various metrics."""
        convergence_metrics = {}
        
        # Check convergence for key variables
        key_variables = ['portfolio_value'] if 'portfolio_value' in results else list(results.keys())[:1]
        
        for var_name in key_variables:
            if var_name in results:
                values = results[var_name]
                
                # Monte Carlo standard error
                mc_se = np.std(values) / np.sqrt(len(values))
                convergence_metrics[f'{var_name}_mc_standard_error'] = float(mc_se)
                
                # Relative precision
                mean_val = np.mean(values)
                if mean_val != 0:
                    convergence_metrics[f'{var_name}_relative_precision'] = float(mc_se / abs(mean_val))
                
                # Check if we have enough samples for stable estimates
                # Rule of thumb: MC SE should be < 1% of estimate
                if mean_val != 0:
                    convergence_ratio = mc_se / abs(mean_val)
                    convergence_metrics[f'{var_name}_converged'] = float(convergence_ratio < 0.01)
        
        return convergence_metrics
    
    def create_scenario_template(
        self,
        template_name: str,
        template_type: str = "financial_portfolio"
    ) -> SimulationScenario:
        """Create predefined scenario templates for common use cases."""
        if template_type == "financial_portfolio":
            return self._create_financial_portfolio_template(template_name)
        elif template_type == "credit_risk":
            return self._create_credit_risk_template(template_name)
        elif template_type == "operational_risk":
            return self._create_operational_risk_template(template_name)
        elif template_type == "systemic_risk":
            return self._create_systemic_risk_template(template_name)
        else:
            raise ValueError(f"Unknown template type: {template_type}")
    
    def _create_financial_portfolio_template(self, name: str) -> SimulationScenario:
        """Create financial portfolio risk scenario template."""
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
            ),
            SimulationParameter(
                name="risk_free_rate",
                distribution=DistributionType.NORMAL,
                parameters={"mean": 0.03, "std": 0.01},
                description="Risk-free rate"
            ),
            SimulationParameter(
                name="correlation_factor",
                distribution=DistributionType.BETA,
                parameters={"alpha": 2, "beta": 2},
                description="Market correlation factor"
            )
        ]
        
        return SimulationScenario(
            scenario_id=f"portfolio_{name}",
            name=f"Portfolio Risk: {name}",
            description="Financial portfolio risk assessment with Monte Carlo simulation",
            parameters=parameters,
            simulation_steps=10000,
            confidence_levels=[0.90, 0.95, 0.99]
        )
    
    def _create_credit_risk_template(self, name: str) -> SimulationScenario:
        """Create credit risk scenario template."""
        parameters = [
            SimulationParameter(
                name="default_probability",
                distribution=DistributionType.BETA,
                parameters={"alpha": 1, "beta": 20},
                description="Probability of default"
            ),
            SimulationParameter(
                name="loss_given_default",
                distribution=DistributionType.TRIANGULAR,
                parameters={"left": 0.3, "mode": 0.6, "right": 0.9},
                description="Loss given default rate"
            ),
            SimulationParameter(
                name="exposure_at_default",
                distribution=DistributionType.LOGNORMAL,
                parameters={"mean": 12.0, "sigma": 0.5},
                description="Exposure at default"
            )
        ]
        
        return SimulationScenario(
            scenario_id=f"credit_{name}",
            name=f"Credit Risk: {name}",
            description="Credit risk assessment using Monte Carlo simulation",
            parameters=parameters,
            simulation_steps=10000
        )
    
    def _create_operational_risk_template(self, name: str) -> SimulationScenario:
        """Create operational risk scenario template."""
        parameters = [
            SimulationParameter(
                name="frequency",
                distribution=DistributionType.EXPONENTIAL,
                parameters={"scale": 2.0},
                description="Event frequency (events per year)"
            ),
            SimulationParameter(
                name="severity",
                distribution=DistributionType.WEIBULL,
                parameters={"a": 1.5, "scale": 100000},
                description="Loss severity"
            ),
            SimulationParameter(
                name="recovery_rate",
                distribution=DistributionType.BETA,
                parameters={"alpha": 3, "beta": 2},
                description="Recovery rate"
            )
        ]
        
        return SimulationScenario(
            scenario_id=f"operational_{name}",
            name=f"Operational Risk: {name}",
            description="Operational risk assessment using frequency-severity modeling",
            parameters=parameters,
            simulation_steps=10000
        )
    
    def _create_systemic_risk_template(self, name: str) -> SimulationScenario:
        """Create systemic risk scenario template."""
        parameters = [
            SimulationParameter(
                name="contagion_probability",
                distribution=DistributionType.BETA,
                parameters={"alpha": 2, "beta": 8},
                description="Probability of contagion spread"
            ),
            SimulationParameter(
                name="system_shock",
                distribution=DistributionType.NORMAL,
                parameters={"mean": 0.0, "std": 0.3},
                description="System-wide shock magnitude"
            ),
            SimulationParameter(
                name="recovery_time",
                distribution=DistributionType.GAMMA,
                parameters={"shape": 2, "scale": 30},
                description="Recovery time (days)"
            ),
            SimulationParameter(
                name="interconnectedness",
                distribution=DistributionType.UNIFORM,
                parameters={"low": 0.1, "high": 0.9},
                description="System interconnectedness level"
            )
        ]
        
        # Create correlation matrix for systemic dependencies
        correlation_matrix = np.array([
            [1.0, 0.6, -0.3, 0.7],
            [0.6, 1.0, -0.2, 0.5],
            [-0.3, -0.2, 1.0, -0.4],
            [0.7, 0.5, -0.4, 1.0]
        ])
        
        return SimulationScenario(
            scenario_id=f"systemic_{name}",
            name=f"Systemic Risk: {name}",
            description="Systemic risk propagation analysis with correlated factors",
            parameters=parameters,
            correlation_matrix=correlation_matrix,
            simulation_steps=15000
        )
    
    def export_results(
        self,
        result: SimulationResult,
        format_type: str = "json"
    ) -> Union[str, Dict[str, Any]]:
        """Export simulation results in specified format."""
        if format_type == "json":
            return self._export_json(result)
        elif format_type == "summary":
            return self._export_summary(result)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    def _export_json(self, result: SimulationResult) -> str:
        """Export results as JSON string."""
        export_data = {
            "scenario_id": result.scenario_id,
            "run_id": result.run_id,
            "timestamp": result.timestamp.isoformat(),
            "execution_time": result.execution_time,
            "sample_size": result.sample_size,
            "statistics": result.statistics,
            "risk_metrics": result.risk_metrics,
            "confidence_intervals": {
                var: {str(cl): interval for cl, interval in intervals.items()}
                for var, intervals in result.confidence_intervals.items()
            },
            "convergence_metrics": result.convergence_metrics
        }
        
        return json.dumps(export_data, indent=2, default=str)
    
    def _export_summary(self, result: SimulationResult) -> Dict[str, Any]:
        """Export concise summary of results."""
        summary = {
            "scenario": result.scenario_id,
            "execution_time": f"{result.execution_time:.2f} seconds",
            "sample_size": result.sample_size,
            "key_metrics": {}
        }
        
        # Extract key risk metrics
        if 'portfolio_value' in result.statistics:
            portfolio_stats = result.statistics['portfolio_value']
            summary["key_metrics"]["portfolio"] = {
                "expected_value": portfolio_stats['mean'],
                "volatility": portfolio_stats['std'],
                "var_95": result.risk_metrics.get('var_95'),
                "expected_shortfall_95": result.risk_metrics.get('expected_shortfall_95')
            }
        
        # Add convergence assessment
        convergence_vars = [k for k in result.convergence_metrics.keys() if k.endswith('_converged')]
        if convergence_vars:
            summary["convergence"] = {
                var: bool(result.convergence_metrics[var]) 
                for var in convergence_vars
            }
        
        return summary