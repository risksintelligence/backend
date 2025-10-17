"""
Simulation-related API models for RiskX platform.
Pydantic models for scenario simulation, policy analysis, and stress testing endpoints.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime, date
from enum import Enum
from pydantic import BaseModel, Field, validator


class SimulationType(str, Enum):
    """Simulation type enumeration."""
    STRESS_TEST = "stress_test"
    SCENARIO_ANALYSIS = "scenario_analysis"
    POLICY_IMPACT = "policy_impact"
    MONTE_CARLO = "monte_carlo"
    SENSITIVITY_ANALYSIS = "sensitivity_analysis"


class ScenarioType(str, Enum):
    """Scenario type enumeration."""
    ECONOMIC_SHOCK = "economic_shock"
    SUPPLY_DISRUPTION = "supply_disruption"
    FINANCIAL_CRISIS = "financial_crisis"
    POLICY_CHANGE = "policy_change"
    NATURAL_DISASTER = "natural_disaster"
    CYBER_ATTACK = "cyber_attack"
    CUSTOM = "custom"


class ImpactSeverity(str, Enum):
    """Impact severity enumeration."""
    MINIMAL = "minimal"
    MODERATE = "moderate"
    SEVERE = "severe"
    CATASTROPHIC = "catastrophic"


class ScenarioParameters(BaseModel):
    """Parameters for scenario definition."""
    
    scenario_name: str = Field(..., description="Name of the scenario")
    scenario_type: ScenarioType = Field(..., description="Type of scenario")
    severity: ImpactSeverity = Field(..., description="Impact severity level")
    duration_days: int = Field(..., ge=1, le=1095, description="Scenario duration in days")
    affected_sectors: List[str] = Field(..., description="List of affected economic sectors")
    parameter_changes: Dict[str, float] = Field(..., description="Parameter changes from baseline")
    trigger_conditions: Optional[Dict[str, Any]] = Field(
        None, description="Conditions that trigger the scenario"
    )
    recovery_profile: Optional[str] = Field(
        None, description="Recovery profile shape", regex="^(linear|exponential|step|custom)$"
    )
    
    @validator('affected_sectors')
    def validate_sectors(cls, v):
        """Validate affected sectors are not empty."""
        if not v:
            raise ValueError("At least one affected sector must be specified")
        return v
    
    @validator('parameter_changes')
    def validate_parameter_changes(cls, v):
        """Validate parameter changes are reasonable."""
        for param, change in v.items():
            if abs(change) > 10.0:  # Arbitrary large change threshold
                raise ValueError(f"Parameter change for {param} seems unrealistic: {change}")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "scenario_name": "Supply Chain Disruption - Port Closure",
                "scenario_type": "supply_disruption",
                "severity": "severe",
                "duration_days": 45,
                "affected_sectors": ["transportation", "manufacturing", "retail"],
                "parameter_changes": {
                    "supply_chain_efficiency": -0.35,
                    "transportation_costs": 0.45,
                    "inventory_levels": -0.25
                },
                "trigger_conditions": {
                    "port_closure_probability": 0.8,
                    "alternative_routes_available": False
                },
                "recovery_profile": "exponential"
            }
        }


class PolicyParameters(BaseModel):
    """Parameters for policy intervention simulation."""
    
    policy_name: str = Field(..., description="Name of the policy intervention")
    policy_type: str = Field(..., description="Type of policy intervention")
    implementation_date: date = Field(..., description="Policy implementation date")
    implementation_speed: str = Field(
        "gradual", description="Implementation speed", regex="^(immediate|gradual|phased)$"
    )
    target_sectors: List[str] = Field(..., description="Sectors targeted by policy")
    policy_instruments: Dict[str, float] = Field(..., description="Policy instruments and magnitudes")
    effectiveness_assumption: float = Field(
        ..., ge=0, le=1, description="Assumed policy effectiveness (0-1)"
    )
    side_effects: Optional[Dict[str, float]] = Field(
        None, description="Potential negative side effects"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "policy_name": "Emergency Infrastructure Investment",
                "policy_type": "fiscal_stimulus",
                "implementation_date": "2024-02-01",
                "implementation_speed": "gradual",
                "target_sectors": ["infrastructure", "construction", "transportation"],
                "policy_instruments": {
                    "infrastructure_spending": 50000000000,  # $50B
                    "tax_incentives": 0.15,
                    "regulatory_relaxation": 0.2
                },
                "effectiveness_assumption": 0.75,
                "side_effects": {
                    "inflation_pressure": 0.1,
                    "debt_increase": 0.05
                }
            }
        }


class SimulationRequest(BaseModel):
    """Request model for scenario simulation."""
    
    simulation_type: SimulationType = Field(..., description="Type of simulation to run")
    simulation_name: str = Field(..., description="Name for this simulation run")
    baseline_date: date = Field(..., description="Baseline date for simulation")
    simulation_horizon_days: int = Field(..., ge=1, le=1095, description="Simulation horizon in days")
    
    # Scenario definition
    scenarios: List[ScenarioParameters] = Field(..., description="Scenarios to simulate")
    
    # Policy interventions (optional)
    policy_interventions: Optional[List[PolicyParameters]] = Field(
        None, description="Policy interventions to test"
    )
    
    # Monte Carlo settings (if applicable)
    monte_carlo_runs: Optional[int] = Field(
        None, ge=100, le=10000, description="Number of Monte Carlo runs"
    )
    confidence_levels: Optional[List[float]] = Field(
        None, description="Confidence levels for output intervals"
    )
    
    # Output options
    include_detailed_breakdown: bool = Field(True, description="Include detailed sector breakdown")
    include_uncertainty_analysis: bool = Field(True, description="Include uncertainty analysis")
    output_frequency: str = Field(
        "daily", description="Output frequency", regex="^(daily|weekly|monthly)$"
    )
    
    @validator('scenarios')
    def validate_scenarios_not_empty(cls, v):
        """Validate at least one scenario is provided."""
        if not v:
            raise ValueError("At least one scenario must be provided")
        return v
    
    @validator('confidence_levels')
    def validate_confidence_levels(cls, v):
        """Validate confidence levels are between 0 and 1."""
        if v:
            for level in v:
                if not 0 < level < 1:
                    raise ValueError(f"Confidence level {level} must be between 0 and 1")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "simulation_type": "scenario_analysis",
                "simulation_name": "Q1 2024 Supply Chain Risk Assessment",
                "baseline_date": "2024-01-15",
                "simulation_horizon_days": 90,
                "scenarios": [
                    {
                        "scenario_name": "Port Closure",
                        "scenario_type": "supply_disruption",
                        "severity": "severe",
                        "duration_days": 45,
                        "affected_sectors": ["transportation", "manufacturing"],
                        "parameter_changes": {
                            "supply_chain_efficiency": -0.35
                        }
                    }
                ],
                "policy_interventions": [
                    {
                        "policy_name": "Emergency Infrastructure Investment",
                        "policy_type": "fiscal_stimulus",
                        "implementation_date": "2024-02-01",
                        "target_sectors": ["infrastructure"],
                        "policy_instruments": {"infrastructure_spending": 50000000000}
                    }
                ],
                "include_detailed_breakdown": True,
                "include_uncertainty_analysis": True,
                "output_frequency": "daily"
            }
        }


class SimulationResult(BaseModel):
    """Individual simulation result point."""
    
    date: date = Field(..., description="Result date")
    overall_risk_score: float = Field(..., ge=0, le=100, description="Overall risk score")
    sector_scores: Dict[str, float] = Field(..., description="Risk scores by sector")
    confidence_intervals: Optional[Dict[str, Dict[str, float]]] = Field(
        None, description="Confidence intervals for scores"
    )
    contributing_factors: List[str] = Field(..., description="Top contributing risk factors")
    
    class Config:
        json_schema_extra = {
            "example": {
                "date": "2024-02-15",
                "overall_risk_score": 78.5,
                "sector_scores": {
                    "transportation": 85.2,
                    "manufacturing": 72.8,
                    "financial": 65.1
                },
                "confidence_intervals": {
                    "overall_risk_score": {"lower": 74.2, "upper": 82.8},
                    "transportation": {"lower": 80.1, "upper": 90.3}
                },
                "contributing_factors": [
                    "supply_chain_disruption",
                    "transportation_bottlenecks",
                    "inventory_shortages"
                ]
            }
        }


class PolicyImpact(BaseModel):
    """Policy intervention impact assessment."""
    
    policy_name: str = Field(..., description="Name of policy intervention")
    implementation_date: date = Field(..., description="Policy implementation date")
    impact_magnitude: float = Field(..., description="Overall impact magnitude")
    impact_direction: str = Field(..., description="Impact direction", regex="^(positive|negative|neutral)$")
    affected_sectors: Dict[str, float] = Field(..., description="Impact by sector")
    effectiveness_score: float = Field(..., ge=0, le=1, description="Policy effectiveness score")
    unintended_consequences: Optional[List[str]] = Field(
        None, description="Identified unintended consequences"
    )
    cost_benefit_ratio: Optional[float] = Field(None, description="Cost-benefit ratio")
    
    class Config:
        json_schema_extra = {
            "example": {
                "policy_name": "Emergency Infrastructure Investment",
                "implementation_date": "2024-02-01",
                "impact_magnitude": -15.2,
                "impact_direction": "positive",
                "affected_sectors": {
                    "transportation": -20.5,
                    "construction": -18.3,
                    "manufacturing": -12.7
                },
                "effectiveness_score": 0.78,
                "unintended_consequences": [
                    "Temporary inflation in construction materials",
                    "Labor shortage in related sectors"
                ],
                "cost_benefit_ratio": 2.3
            }
        }


class UncertaintyAnalysis(BaseModel):
    """Uncertainty analysis results."""
    
    monte_carlo_runs: int = Field(..., description="Number of Monte Carlo runs performed")
    confidence_intervals: Dict[str, Dict[str, float]] = Field(
        ..., description="Confidence intervals for key metrics"
    )
    sensitivity_analysis: Dict[str, float] = Field(
        ..., description="Parameter sensitivity analysis"
    )
    uncertainty_sources: List[Dict[str, Any]] = Field(
        ..., description="Identified sources of uncertainty"
    )
    robustness_metrics: Dict[str, float] = Field(
        ..., description="Robustness metrics for results"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "monte_carlo_runs": 1000,
                "confidence_intervals": {
                    "overall_risk_score": {
                        "50%": {"lower": 75.2, "upper": 82.1},
                        "95%": {"lower": 68.5, "upper": 89.3}
                    }
                },
                "sensitivity_analysis": {
                    "supply_chain_efficiency": 0.45,
                    "transportation_costs": 0.32,
                    "policy_effectiveness": 0.28
                },
                "uncertainty_sources": [
                    {
                        "source": "parameter_estimation",
                        "contribution": 0.35,
                        "description": "Uncertainty in parameter estimates"
                    }
                ],
                "robustness_metrics": {
                    "variance_ratio": 0.15,
                    "tail_risk_measure": 0.08
                }
            }
        }


class SimulationResponse(BaseModel):
    """Response model for scenario simulation."""
    
    simulation_id: str = Field(..., description="Unique simulation identifier")
    simulation_name: str = Field(..., description="Simulation name")
    simulation_type: SimulationType = Field(..., description="Type of simulation")
    
    # Simulation results
    baseline_results: List[SimulationResult] = Field(..., description="Baseline scenario results")
    scenario_results: Dict[str, List[SimulationResult]] = Field(
        ..., description="Results for each scenario"
    )
    
    # Policy analysis (if applicable)
    policy_impacts: Optional[List[PolicyImpact]] = Field(
        None, description="Policy intervention impacts"
    )
    
    # Uncertainty analysis (if requested)
    uncertainty_analysis: Optional[UncertaintyAnalysis] = Field(
        None, description="Uncertainty analysis results"
    )
    
    # Summary metrics
    summary_metrics: Dict[str, Any] = Field(..., description="Summary metrics and insights")
    
    # Metadata
    simulation_metadata: Dict[str, Any] = Field(..., description="Simulation metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "simulation_id": "sim_12345",
                "simulation_name": "Q1 2024 Supply Chain Risk Assessment",
                "simulation_type": "scenario_analysis",
                "baseline_results": [
                    {
                        "date": "2024-01-15",
                        "overall_risk_score": 65.2,
                        "sector_scores": {"transportation": 62.1, "manufacturing": 68.3},
                        "contributing_factors": ["normal_operations"]
                    }
                ],
                "scenario_results": {
                    "port_closure": [
                        {
                            "date": "2024-02-15",
                            "overall_risk_score": 78.5,
                            "sector_scores": {"transportation": 85.2, "manufacturing": 72.8},
                            "contributing_factors": ["supply_chain_disruption"]
                        }
                    ]
                },
                "summary_metrics": {
                    "max_risk_increase": 13.3,
                    "recovery_time_days": 65,
                    "most_vulnerable_sector": "transportation"
                },
                "simulation_metadata": {
                    "created_at": "2024-01-15T10:30:00Z",
                    "computation_time_seconds": 45.7,
                    "model_versions": {"risk_model": "v2.1.0", "simulation_engine": "v1.3.0"}
                }
            }
        }


class StressTestRequest(BaseModel):
    """Request model for stress testing."""
    
    test_name: str = Field(..., description="Name of stress test")
    test_type: str = Field(..., description="Type of stress test")
    severity_levels: List[str] = Field(..., description="Severity levels to test")
    stress_parameters: Dict[str, List[float]] = Field(..., description="Parameters to stress")
    correlation_adjustments: Optional[Dict[str, float]] = Field(
        None, description="Correlation adjustments under stress"
    )
    include_tail_risk: bool = Field(True, description="Include tail risk analysis")
    
    class Config:
        json_schema_extra = {
            "example": {
                "test_name": "Financial Sector Resilience Test",
                "test_type": "systematic_risk",
                "severity_levels": ["moderate", "severe", "extreme"],
                "stress_parameters": {
                    "interest_rate_shock": [0.02, 0.05, 0.10],
                    "credit_spread_widening": [0.5, 1.0, 2.0],
                    "equity_market_decline": [-0.2, -0.4, -0.6]
                },
                "correlation_adjustments": {
                    "flight_to_quality": 0.3,
                    "contagion_factor": 0.25
                },
                "include_tail_risk": True
            }
        }


class StressTestResponse(BaseModel):
    """Response model for stress testing."""
    
    test_id: str = Field(..., description="Unique test identifier")
    test_name: str = Field(..., description="Stress test name")
    results_by_severity: Dict[str, Dict[str, float]] = Field(
        ..., description="Results organized by severity level"
    )
    vulnerability_ranking: List[Dict[str, Any]] = Field(
        ..., description="Ranking of most vulnerable components"
    )
    systemic_risk_indicators: Dict[str, float] = Field(
        ..., description="Systemic risk indicators"
    )
    tail_risk_analysis: Optional[Dict[str, Any]] = Field(
        None, description="Tail risk analysis results"
    )
    recommendations: List[str] = Field(..., description="Risk mitigation recommendations")
    
    class Config:
        json_schema_extra = {
            "example": {
                "test_id": "stress_12345",
                "test_name": "Financial Sector Resilience Test",
                "results_by_severity": {
                    "moderate": {"overall_impact": 15.2, "capital_adequacy": 0.87},
                    "severe": {"overall_impact": 28.5, "capital_adequacy": 0.72},
                    "extreme": {"overall_impact": 45.3, "capital_adequacy": 0.58}
                },
                "vulnerability_ranking": [
                    {"component": "regional_banks", "vulnerability_score": 78.5},
                    {"component": "insurance_sector", "vulnerability_score": 65.2}
                ],
                "systemic_risk_indicators": {
                    "contagion_risk": 0.35,
                    "interconnectedness": 0.67,
                    "concentration_risk": 0.42
                },
                "recommendations": [
                    "Increase capital buffers for regional banks",
                    "Enhance liquidity risk management",
                    "Improve stress testing frequency"
                ]
            }
        }