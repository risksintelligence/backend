"""
Simulation API endpoints for policy impact analysis and scenario modeling.
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field
import json

from ...data.storage.cache import get_cache_instance
from ...ml.models.risk_scorer import RiskScorer

logger = logging.getLogger('riskx.api.routes.simulation')

router = APIRouter()


class PolicySimulationRequest(BaseModel):
    """Request model for policy impact simulations."""
    policy_name: str = Field(..., description="Name of the policy being simulated")
    policy_type: str = Field(..., description="Type of policy (monetary, fiscal, regulatory, trade)")
    parameters: Dict[str, Any] = Field(..., description="Policy parameters and settings")
    duration_months: int = Field(default=12, ge=1, le=60, description="Simulation duration in months")
    confidence_level: float = Field(default=0.95, ge=0.5, le=0.99, description="Confidence level")


class SimulationResponse(BaseModel):
    """Response model for simulation results."""
    simulation_id: str
    timestamp: datetime
    policy_name: str
    duration_months: int
    baseline_scenario: Dict
    policy_scenario: Dict
    impact_analysis: Dict
    recommendations: List[str]


class MonteCarloRequest(BaseModel):
    """Request model for Monte Carlo simulations."""
    scenario_name: str = Field(..., description="Name of the scenario")
    variables: Dict[str, Dict] = Field(..., description="Variable definitions with distributions")
    iterations: int = Field(default=1000, ge=100, le=10000, description="Number of Monte Carlo iterations")
    confidence_intervals: List[float] = Field(default=[0.05, 0.25, 0.75, 0.95], description="Confidence intervals")


@router.post("/policy/simulate", response_model=SimulationResponse)
async def simulate_policy_impact(request: PolicySimulationRequest):
    """
    Simulate the impact of policy changes on economic and financial stability.
    
    Models how monetary, fiscal, regulatory, or trade policies would affect
    risk levels across different sectors and time horizons.
    """
    try:
        # Initialize risk scorer for baseline
        risk_scorer = RiskScorer()
        baseline_score = await risk_scorer.calculate_risk_score()
        
        current_time = datetime.utcnow()
        simulation_id = f"policy_sim_{current_time.strftime('%Y%m%d_%H%M%S')}"
        
        # Define policy impact models
        policy_models = {
            "monetary": {
                "interest_rate_change": {
                    "financial_sector": lambda x: 1 + (x * 0.15),  # 15% impact per percentage point
                    "economic_growth": lambda x: 1 - (x * 0.08),   # Negative impact on growth
                    "inflation_pressure": lambda x: 1 - (x * 0.12)  # Reduces inflation pressure
                },
                "quantitative_easing": {
                    "financial_sector": lambda x: 1 - (x * 0.1),    # Reduces financial risk
                    "asset_bubble_risk": lambda x: 1 + (x * 0.2),   # Increases bubble risk
                    "currency_stability": lambda x: 1 + (x * 0.05)  # Minor currency impact
                }
            },
            "fiscal": {
                "government_spending": {
                    "economic_growth": lambda x: 1 + (x * 0.1),     # Stimulative effect
                    "debt_sustainability": lambda x: 1 + (x * 0.08), # Increases debt risk
                    "inflation_pressure": lambda x: 1 + (x * 0.06)   # Inflationary pressure
                },
                "tax_policy": {
                    "economic_growth": lambda x: 1 - (x * 0.12),    # Negative growth impact
                    "government_revenue": lambda x: 1 + (x * 0.8),   # Direct revenue impact
                    "income_inequality": lambda x: 1 - (x * 0.15)    # Redistributive effect
                }
            },
            "regulatory": {
                "banking_regulation": {
                    "financial_stability": lambda x: 1 - (x * 0.2), # Improves stability
                    "credit_availability": lambda x: 1 - (x * 0.1), # Reduces credit
                    "compliance_costs": lambda x: 1 + (x * 0.05)     # Increases costs
                },
                "environmental_regulation": {
                    "transition_costs": lambda x: 1 + (x * 0.15),    # Short-term costs
                    "long_term_stability": lambda x: 1 - (x * 0.25), # Long-term benefits
                    "innovation_incentives": lambda x: 1 + (x * 0.1) # Innovation boost
                }
            },
            "trade": {
                "tariff_policy": {
                    "import_costs": lambda x: 1 + (x * 0.3),         # Direct cost impact
                    "supply_chain_risk": lambda x: 1 + (x * 0.2),    # Supply disruption
                    "domestic_production": lambda x: 1 + (x * 0.1)   # Domestic boost
                },
                "trade_agreement": {
                    "trade_efficiency": lambda x: 1 + (x * 0.15),    # Efficiency gains
                    "regulatory_complexity": lambda x: 1 + (x * 0.05), # Complexity increase
                    "economic_integration": lambda x: 1 + (x * 0.2)   # Integration benefits
                }
            }
        }
        
        # Get policy model
        policy_type = request.policy_type.lower()
        if policy_type not in policy_models:
            raise HTTPException(status_code=400, detail=f"Unsupported policy type: {policy_type}")
        
        # Extract policy parameters
        policy_intensity = request.parameters.get("intensity", 1.0)  # Policy strength multiplier
        implementation_speed = request.parameters.get("implementation_speed", "gradual")  # fast, gradual, slow
        
        # Generate baseline scenario
        baseline_timeline = []
        for month in range(request.duration_months):
            date = current_time + timedelta(days=month * 30)
            baseline_timeline.append({
                "date": date.isoformat(),
                "risk_score": baseline_score,
                "economic_growth": 2.1,
                "inflation_rate": 2.5,
                "unemployment_rate": 4.2,
                "financial_stability_index": 75.0
            })
        
        # Generate policy scenario
        policy_timeline = []
        implementation_curves = {
            "fast": lambda t: min(1.0, t * 0.5),      # Full effect in 2 months
            "gradual": lambda t: min(1.0, t * 0.2),   # Full effect in 5 months
            "slow": lambda t: min(1.0, t * 0.1)       # Full effect in 10 months
        }
        
        curve = implementation_curves.get(implementation_speed, implementation_curves["gradual"])
        
        for month in range(request.duration_months):
            implementation_factor = curve(month + 1)
            adjusted_intensity = policy_intensity * implementation_factor
            
            # Apply policy effects based on type
            policy_effects = policy_models[policy_type]
            adjusted_risk_score = baseline_score
            
            # Apply each effect in the policy model
            for effect_name, effect_func in policy_effects.get(list(policy_effects.keys())[0], {}).items():
                factor = effect_func(adjusted_intensity)
                adjusted_risk_score *= factor
            
            adjusted_risk_score = min(max(adjusted_risk_score, 0), 100)
            
            date = current_time + timedelta(days=month * 30)
            policy_timeline.append({
                "date": date.isoformat(),
                "risk_score": round(adjusted_risk_score, 2),
                "implementation_progress": round(implementation_factor * 100, 1),
                "policy_intensity": round(adjusted_intensity, 2),
                "economic_growth": round(2.1 * (1 + (adjusted_intensity - 1) * 0.1), 2),
                "inflation_rate": round(2.5 * (1 + (adjusted_intensity - 1) * 0.05), 2),
                "unemployment_rate": round(4.2 * (1 - (adjusted_intensity - 1) * 0.03), 2),
                "financial_stability_index": round(75.0 * (2 - factor), 1)
            })
        
        # Calculate impact analysis
        final_baseline = baseline_timeline[-1]["risk_score"]
        final_policy = policy_timeline[-1]["risk_score"]
        risk_change = final_policy - final_baseline
        risk_change_percent = (risk_change / final_baseline) * 100
        
        # Sector-specific impacts
        sector_impacts = {
            "financial_services": round(risk_change * 0.8, 2),
            "manufacturing": round(risk_change * 0.6, 2),
            "technology": round(risk_change * 0.4, 2),
            "energy": round(risk_change * 0.7, 2),
            "healthcare": round(risk_change * 0.3, 2),
            "real_estate": round(risk_change * 0.9, 2)
        }
        
        # Generate recommendations
        recommendations = []
        if abs(risk_change_percent) > 10:
            if risk_change_percent > 0:
                recommendations.extend([
                    f"Consider gradual implementation to reduce {abs(risk_change_percent):.1f}% risk increase",
                    "Monitor financial sector stress indicators closely",
                    "Prepare contingency measures for supply chain disruptions"
                ])
            else:
                recommendations.extend([
                    f"Policy shows positive {abs(risk_change_percent):.1f}% risk reduction",
                    "Consider accelerated implementation timeline",
                    "Monitor for unintended consequences in other sectors"
                ])
        else:
            recommendations.append("Policy impact appears manageable with current implementation plan")
        
        # Cache simulation results
        cache_manager = get_cache_instance()
        if cache_manager:
            cache_key = f"simulation:{simulation_id}"
            simulation_data = {
                "baseline": baseline_timeline,
                "policy": policy_timeline,
                "parameters": request.parameters
            }
            cache_manager.set(cache_key, simulation_data, ttl=7200)  # 2 hours
        
        response = SimulationResponse(
            simulation_id=simulation_id,
            timestamp=current_time,
            policy_name=request.policy_name,
            duration_months=request.duration_months,
            baseline_scenario={
                "timeline": baseline_timeline[:6],  # First 6 months
                "average_risk_score": baseline_score,
                "stability_metrics": {
                    "volatility": 0.12,
                    "trend": "stable",
                    "confidence": 0.85
                }
            },
            policy_scenario={
                "timeline": policy_timeline[:6],  # First 6 months
                "average_risk_score": round(sum([p["risk_score"] for p in policy_timeline]) / len(policy_timeline), 2),
                "implementation_timeline": implementation_speed,
                "stability_metrics": {
                    "volatility": round(0.12 * (1 + abs(policy_intensity - 1) * 0.2), 3),
                    "trend": "improving" if risk_change < 0 else "deteriorating" if risk_change > 0 else "stable",
                    "confidence": 0.75
                }
            },
            impact_analysis={
                "overall_risk_change": round(risk_change, 2),
                "risk_change_percent": round(risk_change_percent, 1),
                "peak_impact_month": max(range(len(policy_timeline)), key=lambda i: abs(policy_timeline[i]["risk_score"] - baseline_score)) + 1,
                "sector_impacts": sector_impacts,
                "confidence_interval": {
                    "lower": round(risk_change * 0.8, 2),
                    "upper": round(risk_change * 1.2, 2),
                    "confidence_level": request.confidence_level
                }
            },
            recommendations=recommendations
        )
        
        logger.info(f"Generated policy simulation: {request.policy_name}, impact: {risk_change_percent:.1f}%")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error simulating policy impact: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to simulate policy: {str(e)}")


@router.post("/monte-carlo/run")
async def run_monte_carlo_simulation(request: MonteCarloRequest):
    """
    Run Monte Carlo simulations for risk assessment under uncertainty.
    
    Performs probabilistic analysis using specified variable distributions
    to assess range of possible outcomes and their probabilities.
    """
    try:
        import random
        import numpy as np
        from statistics import mean, stdev
        
        current_time = datetime.utcnow()
        simulation_id = f"monte_carlo_{current_time.strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize results storage
        results = []
        
        # Run Monte Carlo iterations
        for iteration in range(request.iterations):
            iteration_result = {"iteration": iteration + 1}
            
            # Sample from each variable distribution
            for var_name, var_config in request.variables.items():
                distribution = var_config.get("distribution", "normal")
                params = var_config.get("parameters", {})
                
                if distribution == "normal":
                    mean_val = params.get("mean", 0)
                    std_val = params.get("std", 1)
                    sample = random.gauss(mean_val, std_val)
                elif distribution == "uniform":
                    low = params.get("low", 0)
                    high = params.get("high", 1)
                    sample = random.uniform(low, high)
                elif distribution == "triangular":
                    low = params.get("low", 0)
                    high = params.get("high", 1)
                    mode = params.get("mode", 0.5)
                    sample = random.triangular(low, high, mode)
                else:
                    # Default to normal distribution
                    sample = random.gauss(0, 1)
                
                iteration_result[var_name] = sample
            
            # Calculate risk score for this iteration
            # This is a simplified model - in practice would use trained models
            base_risk = 35.0
            risk_factors = {
                "gdp_growth": iteration_result.get("gdp_growth", 0) * -5,  # Negative correlation
                "inflation": iteration_result.get("inflation", 0) * 3,     # Positive correlation
                "unemployment": iteration_result.get("unemployment", 0) * 4, # Positive correlation
                "credit_spread": iteration_result.get("credit_spread", 0) * 8, # Strong positive correlation
                "supply_chain_stress": iteration_result.get("supply_chain_stress", 0) * 6
            }
            
            iteration_risk = base_risk + sum(risk_factors.values())
            iteration_risk = max(0, min(100, iteration_risk))  # Bound between 0-100
            
            iteration_result["risk_score"] = iteration_risk
            results.append(iteration_result)
        
        # Calculate statistics
        risk_scores = [r["risk_score"] for r in results]
        risk_mean = mean(risk_scores)
        risk_std = stdev(risk_scores) if len(risk_scores) > 1 else 0
        
        # Calculate percentiles
        sorted_scores = sorted(risk_scores)
        percentiles = {}
        for conf in request.confidence_intervals:
            index = int(conf * len(sorted_scores))
            percentiles[f"p{int(conf*100)}"] = sorted_scores[min(index, len(sorted_scores)-1)]
        
        # Analyze variable sensitivity
        variable_correlations = {}
        for var_name in request.variables.keys():
            if var_name in results[0]:
                var_values = [r[var_name] for r in results]
                # Simple correlation calculation
                correlation = np.corrcoef(var_values, risk_scores)[0, 1] if len(var_values) > 1 else 0
                variable_correlations[var_name] = round(correlation, 3)
        
        # Generate risk distribution summary
        risk_distribution = {
            "very_low": len([s for s in risk_scores if s < 20]) / len(risk_scores),
            "low": len([s for s in risk_scores if 20 <= s < 40]) / len(risk_scores),
            "moderate": len([s for s in risk_scores if 40 <= s < 60]) / len(risk_scores),
            "high": len([s for s in risk_scores if 60 <= s < 80]) / len(risk_scores),
            "very_high": len([s for s in risk_scores if s >= 80]) / len(risk_scores)
        }
        
        # Cache results
        cache_manager = get_cache_instance()
        if cache_manager:
            cache_key = f"monte_carlo:{simulation_id}"
            cache_manager.set(cache_key, {
                "results": results[:100],  # Store first 100 iterations
                "statistics": {
                    "mean": risk_mean,
                    "std": risk_std,
                    "percentiles": percentiles
                }
            }, ttl=3600)
        
        response = {
            "simulation_id": simulation_id,
            "timestamp": current_time.isoformat(),
            "scenario_name": request.scenario_name,
            "iterations": request.iterations,
            "statistics": {
                "risk_score": {
                    "mean": round(risk_mean, 2),
                    "std": round(risk_std, 2),
                    "min": round(min(risk_scores), 2),
                    "max": round(max(risk_scores), 2)
                },
                "percentiles": {k: round(v, 2) for k, v in percentiles.items()},
                "distribution": {k: round(v, 3) for k, v in risk_distribution.items()}
            },
            "sensitivity_analysis": variable_correlations,
            "sample_results": results[:10],  # First 10 iterations as examples
            "interpretation": {
                "most_likely_outcome": f"Risk score between {percentiles.get('p25', 0):.1f} and {percentiles.get('p75', 0):.1f}",
                "worst_case_scenario": f"95% confidence that risk won't exceed {percentiles.get('p95', 0):.1f}",
                "best_case_scenario": f"5% chance of risk below {percentiles.get('p5', 0):.1f}",
                "key_drivers": sorted(variable_correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
            }
        }
        
        logger.info(f"Completed Monte Carlo simulation: {request.iterations} iterations, mean risk: {risk_mean:.2f}")
        return response
        
    except Exception as e:
        logger.error(f"Error running Monte Carlo simulation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to run simulation: {str(e)}")


@router.get("/simulations/{simulation_id}")
async def get_simulation_results(simulation_id: str):
    """
    Retrieve results from a previous simulation.
    
    Returns the full results and analysis from a completed policy
    simulation or Monte Carlo analysis.
    """
    try:
        cache_manager = get_cache_instance()
        if not cache_manager:
            raise HTTPException(status_code=503, detail="Cache service unavailable")
        
        # Try both simulation types
        cache_key = f"simulation:{simulation_id}"
        results = cache_manager.get(cache_key)
        
        if not results:
            cache_key = f"monte_carlo:{simulation_id}"
            results = cache_manager.get(cache_key)
        
        if not results:
            raise HTTPException(status_code=404, detail="Simulation not found")
        
        return {
            "simulation_id": simulation_id,
            "retrieved_at": datetime.utcnow().isoformat(),
            "results": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving simulation results: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve results: {str(e)}")


@router.get("/templates/policies")
async def get_policy_templates():
    """
    Get pre-defined policy simulation templates.
    
    Returns standardized policy configurations for common
    economic and regulatory scenarios.
    """
    templates = {
        "interest_rate_hike": {
            "name": "Federal Reserve Interest Rate Increase",
            "type": "monetary",
            "description": "Simulate impact of Fed rate increases on economic stability",
            "parameters": {
                "intensity": {"default": 0.25, "range": [0.25, 2.0], "unit": "percentage_points"},
                "implementation_speed": {"default": "gradual", "options": ["fast", "gradual", "slow"]},
                "market_expectations": {"default": "anticipated", "options": ["anticipated", "surprise"]}
            },
            "expected_impacts": ["financial_sector_stress", "reduced_lending", "currency_strengthening"],
            "monitoring_indicators": ["credit_spreads", "banking_stability", "housing_market"]
        },
        "fiscal_stimulus": {
            "name": "Government Spending Stimulus Package",
            "type": "fiscal",
            "description": "Analyze effects of increased government spending on economic growth",
            "parameters": {
                "intensity": {"default": 2.0, "range": [0.5, 5.0], "unit": "percent_of_gdp"},
                "implementation_speed": {"default": "gradual", "options": ["fast", "gradual", "slow"]},
                "spending_focus": {"default": "infrastructure", "options": ["infrastructure", "social", "defense", "technology"]}
            },
            "expected_impacts": ["economic_growth", "inflation_pressure", "debt_increase"],
            "monitoring_indicators": ["gdp_growth", "inflation_rate", "debt_to_gdp"]
        },
        "trade_tariffs": {
            "name": "Import Tariff Implementation",
            "type": "trade", 
            "description": "Model consequences of new import tariffs on specific sectors",
            "parameters": {
                "intensity": {"default": 10.0, "range": [5.0, 50.0], "unit": "percent_tariff"},
                "implementation_speed": {"default": "fast", "options": ["fast", "gradual"]},
                "sector_scope": {"default": "manufacturing", "options": ["manufacturing", "agriculture", "technology", "all"]}
            },
            "expected_impacts": ["import_cost_increase", "supply_chain_disruption", "domestic_production_boost"],
            "monitoring_indicators": ["import_prices", "supply_chain_stress", "domestic_capacity"]
        },
        "banking_regulation": {
            "name": "Enhanced Banking Capital Requirements",
            "type": "regulatory",
            "description": "Assess impact of stricter bank capital and liquidity requirements",
            "parameters": {
                "intensity": {"default": 1.5, "range": [1.1, 3.0], "unit": "multiplier"},
                "implementation_speed": {"default": "gradual", "options": ["gradual", "slow"]},
                "scope": {"default": "large_banks", "options": ["large_banks", "all_banks", "systemically_important"]}
            },
            "expected_impacts": ["banking_stability", "reduced_lending", "compliance_costs"],
            "monitoring_indicators": ["bank_capital_ratios", "credit_growth", "financial_stability"]
        }
    }
    
    return {
        "templates": templates,
        "usage_instructions": {
            "selection": "Choose a template based on the policy scenario you want to analyze",
            "customization": "Adjust parameters within recommended ranges for your specific case",
            "validation": "Review expected impacts and monitoring indicators before running simulation"
        },
        "best_practices": [
            "Start with default parameter values and adjust incrementally",
            "Consider implementation speed based on political and economic feasibility",
            "Monitor recommended indicators during and after policy implementation",
            "Run sensitivity analysis for key uncertain parameters"
        ]
    }