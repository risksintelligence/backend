"""
Monte Carlo Simulation API endpoints for RiskX Platform.

Provides comprehensive simulation capabilities including scenario analysis,
risk modeling, and uncertainty quantification.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime
import logging
import json

from src.ml.models.simulation_engine import (
    MonteCarloEngine, 
    SimulationScenario, 
    SimulationParameter, 
    DistributionType,
    SimulationResult
)
from src.core.dependencies import get_cache_manager
from src.cache.cache_manager import IntelligentCacheManager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/simulation", tags=["simulation"])

# Global simulation engine instance
simulation_engine = MonteCarloEngine(random_seed=42)


class DistributionParametersModel(BaseModel):
    """Probability distribution parameters."""
    mean: Optional[float] = None
    std: Optional[float] = None
    sigma: Optional[float] = None
    low: Optional[float] = None
    high: Optional[float] = None
    left: Optional[float] = None
    mode: Optional[float] = None
    right: Optional[float] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None
    scale: Optional[float] = None
    shape: Optional[float] = None
    a: Optional[float] = None


class SimulationParameterModel(BaseModel):
    """Individual simulation parameter definition."""
    name: str = Field(..., description="Parameter name")
    distribution: str = Field(..., description="Distribution type")
    parameters: DistributionParametersModel = Field(..., description="Distribution parameters")
    correlation_group: Optional[str] = Field(None, description="Correlation group identifier")
    description: Optional[str] = Field(None, description="Parameter description")


class SimulationScenarioRequest(BaseModel):
    """Request model for creating simulation scenario."""
    scenario_id: str = Field(..., description="Unique scenario identifier")
    name: str = Field(..., description="Scenario name")
    description: str = Field(..., description="Scenario description")
    parameters: List[SimulationParameterModel] = Field(..., description="Simulation parameters")
    correlation_matrix: Optional[List[List[float]]] = Field(None, description="Correlation matrix")
    simulation_steps: int = Field(10000, description="Number of simulation steps")
    confidence_levels: List[float] = Field([0.90, 0.95, 0.99], description="Confidence levels")
    time_horizon: Optional[int] = Field(None, description="Time horizon in periods")


class SimulationRunRequest(BaseModel):
    """Request model for running simulation."""
    scenario: SimulationScenarioRequest = Field(..., description="Simulation scenario")
    run_id: Optional[str] = Field(None, description="Optional run identifier")


class TemplateRequest(BaseModel):
    """Request model for creating scenario template."""
    template_name: str = Field(..., description="Template name")
    template_type: str = Field("financial_portfolio", description="Template type")


@router.post("/run")
async def run_simulation(
    request: SimulationRunRequest,
    cache: IntelligentCacheManager = Depends(get_cache_manager)
) -> Dict[str, Any]:
    """
    Execute Monte Carlo simulation for given scenario.
    
    Runs comprehensive risk simulation with statistical analysis,
    confidence intervals, and risk metric calculations.
    """
    try:
        # Convert request to scenario object
        scenario = _convert_request_to_scenario(request.scenario)
        
        # Check cache for existing results
        cache_key = f"simulation:{scenario.scenario_id}:{hash(str(request.dict()))}"
        cached_result = await cache.get(cache_key, max_age_seconds=3600)
        
        if cached_result:
            logger.info(f"Returning cached simulation result: {scenario.scenario_id}")
            return {
                "status": "success",
                "source": "cache",
                "data": cached_result,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Run simulation
        logger.info(f"Running Monte Carlo simulation: {scenario.scenario_id}")
        result = await simulation_engine.run_simulation(scenario, request.run_id)
        
        # Convert result to serializable format
        result_data = _convert_result_to_dict(result)
        
        # Cache the result
        await cache.set(cache_key, result_data, ttl_seconds=3600)
        
        return {
            "status": "success",
            "source": "computed",
            "data": result_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Simulation execution failed: {str(e)}"
        )


@router.get("/templates")
async def get_scenario_templates() -> Dict[str, Any]:
    """
    Get available simulation scenario templates.
    
    Returns predefined templates for common risk modeling scenarios
    including financial portfolios, credit risk, and operational risk.
    """
    templates = [
        {
            "template_type": "financial_portfolio",
            "name": "Financial Portfolio Risk",
            "description": "Portfolio risk assessment with asset returns, volatility, and correlations",
            "parameters": ["asset_return", "volatility", "risk_free_rate", "correlation_factor"]
        },
        {
            "template_type": "credit_risk",
            "name": "Credit Risk Assessment",
            "description": "Credit loss modeling with default probability and loss given default",
            "parameters": ["default_probability", "loss_given_default", "exposure_at_default"]
        },
        {
            "template_type": "operational_risk",
            "name": "Operational Risk Modeling",
            "description": "Frequency-severity modeling for operational loss events",
            "parameters": ["frequency", "severity", "recovery_rate"]
        },
        {
            "template_type": "systemic_risk",
            "name": "Systemic Risk Analysis",
            "description": "System-wide risk with contagion and interconnectedness modeling",
            "parameters": ["contagion_probability", "system_shock", "recovery_time", "interconnectedness"]
        }
    ]
    
    return {
        "status": "success",
        "data": {
            "templates": templates,
            "total_templates": len(templates)
        },
        "timestamp": datetime.utcnow().isoformat()
    }


@router.post("/templates/{template_type}")
async def create_scenario_from_template(
    template_type: str,
    request: TemplateRequest
) -> Dict[str, Any]:
    """
    Create simulation scenario from predefined template.
    
    Generates a complete scenario configuration based on the specified
    template type with appropriate parameter distributions and correlations.
    """
    try:
        # Create scenario from template
        scenario = simulation_engine.create_scenario_template(
            request.template_name,
            template_type
        )
        
        # Convert to response format
        scenario_data = _convert_scenario_to_dict(scenario)
        
        return {
            "status": "success",
            "data": {
                "scenario": scenario_data,
                "template_type": template_type,
                "created_at": datetime.utcnow().isoformat()
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid template type: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Template creation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Template creation failed: {str(e)}"
        )


@router.get("/results/{run_id}")
async def get_simulation_result(
    run_id: str,
    cache: IntelligentCacheManager = Depends(get_cache_manager)
) -> Dict[str, Any]:
    """
    Retrieve simulation results by run ID.
    
    Returns complete simulation results including statistics,
    risk metrics, and confidence intervals.
    """
    try:
        # Check cache for results
        cache_key = f"simulation_result:{run_id}"
        cached_result = await cache.get(cache_key)
        
        if cached_result:
            return {
                "status": "success",
                "source": "cache",
                "data": cached_result,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Check simulation engine history
        for result in simulation_engine.simulation_history:
            if result.run_id == run_id:
                result_data = _convert_result_to_dict(result)
                
                # Cache the result
                await cache.set(cache_key, result_data, ttl_seconds=7200)
                
                return {
                    "status": "success",
                    "source": "history",
                    "data": result_data,
                    "timestamp": datetime.utcnow().isoformat()
                }
        
        raise HTTPException(
            status_code=404,
            detail=f"Simulation result not found for run_id: {run_id}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving simulation result: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve simulation result: {str(e)}"
        )


@router.get("/history")
async def get_simulation_history(
    limit: int = 10,
    offset: int = 0
) -> Dict[str, Any]:
    """
    Get simulation execution history.
    
    Returns a paginated list of recent simulation runs with
    basic metadata and summary statistics.
    """
    try:
        total_simulations = len(simulation_engine.simulation_history)
        
        # Apply pagination
        start_idx = offset
        end_idx = min(offset + limit, total_simulations)
        
        history_slice = simulation_engine.simulation_history[start_idx:end_idx]
        
        # Convert to summary format
        history_data = []
        for result in history_slice:
            summary = {
                "run_id": result.run_id,
                "scenario_id": result.scenario_id,
                "timestamp": result.timestamp.isoformat(),
                "execution_time": result.execution_time,
                "sample_size": result.sample_size,
                "convergence_status": _assess_overall_convergence(result.convergence_metrics)
            }
            
            # Add key risk metrics if available
            if result.risk_metrics:
                summary["key_risk_metrics"] = {
                    "var_95": result.risk_metrics.get('var_95'),
                    "expected_shortfall_95": result.risk_metrics.get('expected_shortfall_95'),
                    "sharpe_ratio": result.risk_metrics.get('sharpe_ratio')
                }
            
            history_data.append(summary)
        
        return {
            "status": "success",
            "data": {
                "simulations": history_data,
                "pagination": {
                    "total": total_simulations,
                    "limit": limit,
                    "offset": offset,
                    "has_more": end_idx < total_simulations
                }
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error retrieving simulation history: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve simulation history: {str(e)}"
        )


@router.post("/validate")
async def validate_scenario(
    scenario: SimulationScenarioRequest
) -> Dict[str, Any]:
    """
    Validate simulation scenario configuration.
    
    Checks parameter definitions, distribution parameters,
    and correlation matrix validity without running simulation.
    """
    try:
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Validate parameters
        for param in scenario.parameters:
            param_validation = _validate_parameter(param)
            if not param_validation["valid"]:
                validation_results["valid"] = False
                validation_results["errors"].extend(param_validation["errors"])
            validation_results["warnings"].extend(param_validation["warnings"])
        
        # Validate correlation matrix if provided
        if scenario.correlation_matrix:
            correlation_validation = _validate_correlation_matrix(
                scenario.correlation_matrix,
                len(scenario.parameters)
            )
            if not correlation_validation["valid"]:
                validation_results["valid"] = False
                validation_results["errors"].extend(correlation_validation["errors"])
            validation_results["warnings"].extend(correlation_validation["warnings"])
        
        # Validate simulation settings
        if scenario.simulation_steps < 1000:
            validation_results["warnings"].append(
                "Simulation steps < 1000 may produce unreliable results"
            )
        
        if scenario.simulation_steps > 100000:
            validation_results["warnings"].append(
                "Large simulation steps may result in long execution times"
            )
        
        return {
            "status": "success",
            "data": validation_results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Scenario validation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Scenario validation failed: {str(e)}"
        )


@router.get("/distributions")
async def get_supported_distributions() -> Dict[str, Any]:
    """
    Get supported probability distributions and their parameters.
    
    Returns information about available distributions for simulation
    parameters including required parameters and typical use cases.
    """
    distributions = {
        "normal": {
            "name": "Normal Distribution",
            "parameters": ["mean", "std"],
            "description": "Symmetric bell-shaped distribution for returns and rates",
            "use_cases": ["asset returns", "interest rates", "error terms"]
        },
        "lognormal": {
            "name": "Log-Normal Distribution",
            "parameters": ["mean", "sigma"],
            "description": "Right-skewed distribution for positive-only variables",
            "use_cases": ["asset prices", "volatility", "exposure amounts"]
        },
        "uniform": {
            "name": "Uniform Distribution",
            "parameters": ["low", "high"],
            "description": "Constant probability over specified range",
            "use_cases": ["bounded parameters", "scenario weights"]
        },
        "triangular": {
            "name": "Triangular Distribution",
            "parameters": ["left", "mode", "right"],
            "description": "Three-point estimation with mode and bounds",
            "use_cases": ["expert estimates", "project parameters"]
        },
        "beta": {
            "name": "Beta Distribution",
            "parameters": ["alpha", "beta"],
            "description": "Bounded distribution between 0 and 1",
            "use_cases": ["probabilities", "correlation factors", "recovery rates"]
        },
        "exponential": {
            "name": "Exponential Distribution",
            "parameters": ["scale"],
            "description": "Memoryless distribution for time intervals",
            "use_cases": ["failure times", "event frequencies"]
        },
        "gamma": {
            "name": "Gamma Distribution",
            "parameters": ["shape", "scale"],
            "description": "Flexible positive distribution",
            "use_cases": ["loss severities", "waiting times"]
        },
        "weibull": {
            "name": "Weibull Distribution",
            "parameters": ["a", "scale"],
            "description": "Reliability and survival analysis distribution",
            "use_cases": ["failure rates", "extreme events"]
        }
    }
    
    return {
        "status": "success",
        "data": {
            "distributions": distributions,
            "total_distributions": len(distributions)
        },
        "timestamp": datetime.utcnow().isoformat()
    }


def _convert_request_to_scenario(request: SimulationScenarioRequest) -> SimulationScenario:
    """Convert API request to simulation scenario object."""
    import numpy as np
    
    # Convert parameters
    parameters = []
    for param_req in request.parameters:
        # Convert distribution parameters to dict
        param_dict = param_req.parameters.dict(exclude_unset=True)
        
        # Validate distribution type
        try:
            distribution = DistributionType(param_req.distribution)
        except ValueError:
            raise ValueError(f"Unsupported distribution type: {param_req.distribution}")
        
        parameter = SimulationParameter(
            name=param_req.name,
            distribution=distribution,
            parameters=param_dict,
            correlation_group=param_req.correlation_group,
            description=param_req.description
        )
        parameters.append(parameter)
    
    # Convert correlation matrix
    correlation_matrix = None
    if request.correlation_matrix:
        correlation_matrix = np.array(request.correlation_matrix)
    
    return SimulationScenario(
        scenario_id=request.scenario_id,
        name=request.name,
        description=request.description,
        parameters=parameters,
        correlation_matrix=correlation_matrix,
        simulation_steps=request.simulation_steps,
        confidence_levels=request.confidence_levels,
        time_horizon=request.time_horizon
    )


def _convert_result_to_dict(result: SimulationResult) -> Dict[str, Any]:
    """Convert simulation result to serializable dictionary."""
    return {
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
        "convergence_metrics": result.convergence_metrics,
        "summary": simulation_engine.export_results(result, "summary")
    }


def _convert_scenario_to_dict(scenario: SimulationScenario) -> Dict[str, Any]:
    """Convert scenario to serializable dictionary."""
    parameters_data = []
    for param in scenario.parameters:
        param_data = {
            "name": param.name,
            "distribution": param.distribution.value,
            "parameters": param.parameters,
            "correlation_group": param.correlation_group,
            "description": param.description
        }
        parameters_data.append(param_data)
    
    scenario_data = {
        "scenario_id": scenario.scenario_id,
        "name": scenario.name,
        "description": scenario.description,
        "parameters": parameters_data,
        "simulation_steps": scenario.simulation_steps,
        "confidence_levels": scenario.confidence_levels,
        "time_horizon": scenario.time_horizon
    }
    
    if scenario.correlation_matrix is not None:
        scenario_data["correlation_matrix"] = scenario.correlation_matrix.tolist()
    
    return scenario_data


def _validate_parameter(param: SimulationParameterModel) -> Dict[str, Any]:
    """Validate individual simulation parameter."""
    validation = {
        "valid": True,
        "errors": [],
        "warnings": []
    }
    
    # Check distribution type
    try:
        DistributionType(param.distribution)
    except ValueError:
        validation["valid"] = False
        validation["errors"].append(f"Unsupported distribution: {param.distribution}")
        return validation
    
    # Validate distribution parameters
    param_dict = param.parameters.dict(exclude_unset=True)
    
    if param.distribution == "normal":
        if "mean" not in param_dict or "std" not in param_dict:
            validation["valid"] = False
            validation["errors"].append("Normal distribution requires 'mean' and 'std' parameters")
        elif param_dict.get("std", 0) <= 0:
            validation["valid"] = False
            validation["errors"].append("Normal distribution 'std' must be positive")
    
    elif param.distribution == "uniform":
        if "low" not in param_dict or "high" not in param_dict:
            validation["valid"] = False
            validation["errors"].append("Uniform distribution requires 'low' and 'high' parameters")
        elif param_dict.get("low", 0) >= param_dict.get("high", 1):
            validation["valid"] = False
            validation["errors"].append("Uniform distribution 'low' must be less than 'high'")
    
    # Add more distribution-specific validations as needed
    
    return validation


def _validate_correlation_matrix(
    correlation_matrix: List[List[float]],
    n_parameters: int
) -> Dict[str, Any]:
    """Validate correlation matrix properties."""
    import numpy as np
    
    validation = {
        "valid": True,
        "errors": [],
        "warnings": []
    }
    
    matrix = np.array(correlation_matrix)
    
    # Check dimensions
    if matrix.shape != (n_parameters, n_parameters):
        validation["valid"] = False
        validation["errors"].append(
            f"Correlation matrix must be {n_parameters}x{n_parameters}, got {matrix.shape}"
        )
        return validation
    
    # Check symmetry
    if not np.allclose(matrix, matrix.T):
        validation["valid"] = False
        validation["errors"].append("Correlation matrix must be symmetric")
    
    # Check diagonal elements
    if not np.allclose(np.diag(matrix), 1.0):
        validation["valid"] = False
        validation["errors"].append("Correlation matrix diagonal elements must be 1.0")
    
    # Check bounds
    if np.any(matrix < -1) or np.any(matrix > 1):
        validation["valid"] = False
        validation["errors"].append("Correlation matrix elements must be between -1 and 1")
    
    # Check positive semi-definite
    try:
        eigenvalues = np.linalg.eigvals(matrix)
        if np.any(eigenvalues < -1e-8):  # Allow small numerical errors
            validation["valid"] = False
            validation["errors"].append("Correlation matrix must be positive semi-definite")
    except np.linalg.LinAlgError:
        validation["valid"] = False
        validation["errors"].append("Correlation matrix is not valid")
    
    return validation


def _assess_overall_convergence(convergence_metrics: Dict[str, float]) -> str:
    """Assess overall simulation convergence status."""
    convergence_flags = [
        k for k, v in convergence_metrics.items() 
        if k.endswith('_converged') and v > 0.5
    ]
    
    if convergence_flags:
        return "converged"
    
    # Check relative precision
    precision_metrics = [
        v for k, v in convergence_metrics.items()
        if k.endswith('_relative_precision')
    ]
    
    if precision_metrics and max(precision_metrics) < 0.05:
        return "acceptable"
    else:
        return "needs_more_samples"