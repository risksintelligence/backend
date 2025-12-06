#!/usr/bin/env python3
"""
Simulation API endpoints for Monte Carlo simulation and stress testing.
"""

import logging
import numpy as np
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Dict, List, Optional, Any
from pydantic import BaseModel

from app.core.auth import optional_auth
from app.core.security import require_system_rate_limit

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/simulation", tags=["simulation"])

class MonteCarloRequest(BaseModel):
    iterations: int = 10000
    time_horizon: int = 1  # days
    confidence_levels: List[float] = [0.95, 0.99]
    random_seed: Optional[int] = None

class StressTestRequest(BaseModel):
    scenario_id: str
    custom_shocks: Optional[Dict[str, float]] = None

@router.post("/monte-carlo")
async def run_monte_carlo_simulation(
    request: MonteCarloRequest,
    _auth: dict = Depends(optional_auth),
    _rate_limit: bool = Depends(require_system_rate_limit)
) -> Dict[str, Any]:
    """Run Monte Carlo simulation using current GRII baseline and forecast data."""
    try:
        # Get current GRII and forecast data
        from app.main import _get_observations
        from app.services.geri import compute_geri_score
        from app.ml.regime import classify_regime
        
        observations = _get_observations()
        if not observations:
            raise HTTPException(status_code=503, detail="No observation data available for simulation")
        
        # Compute current baseline
        geri_result = compute_geri_score(observations)
        base_score = geri_result.get("score", 50)
        
        # Get regime-based volatility estimation
        regime_probs = classify_regime(observations)
        regime_confidence = max(regime_probs.values()) if regime_probs else 0.5
        
        # Estimate volatility from recent GERI score variations
        from app.db import SessionLocal
        from app.models import ObservationModel
        from datetime import timedelta
        
        db = SessionLocal()
        try:
            # Get recent observations for volatility estimation
            recent_date = datetime.utcnow() - timedelta(days=30)
            recent_obs = db.query(ObservationModel).filter(
                ObservationModel.observed_at >= recent_date
            ).order_by(ObservationModel.observed_at.desc()).limit(100).all()
            
            # Calculate daily GERI variations for volatility
            if len(recent_obs) > 5:
                # Group by date and calculate daily GERI scores
                daily_scores = []
                from collections import defaultdict
                obs_by_date = defaultdict(lambda: defaultdict(list))
                
                for obs in recent_obs:
                    date_key = obs.observed_at.date()
                    obs_by_date[date_key][obs.series_id].append(obs.value)
                
                # Calculate GERI for each date with sufficient data
                for date_key in sorted(obs_by_date.keys()):
                    daily_obs = obs_by_date[date_key]
                    if len(daily_obs) >= 3:  # Need minimum data
                        try:
                            from app.services.ingestion import Observation
                            formatted_obs = {}
                            for series_id, values in daily_obs.items():
                                if values:  # Use latest value for that date
                                    formatted_obs[series_id] = [Observation(
                                        series_id=series_id,
                                        observed_at=datetime.combine(date_key, datetime.min.time()),
                                        value=float(values[-1])
                                    )]
                            
                            if formatted_obs:
                                daily_geri = compute_geri_score(formatted_obs)
                                daily_scores.append(daily_geri.get("score", 50))
                        except Exception:
                            continue
                
                # Calculate volatility from daily score changes
                if len(daily_scores) >= 3:
                    score_changes = [daily_scores[i] - daily_scores[i-1] for i in range(1, len(daily_scores))]
                    estimated_vol = np.std(score_changes) if len(score_changes) > 1 else 2.0
                else:
                    estimated_vol = 2.0  # Default volatility
            else:
                estimated_vol = 2.0  # Default volatility
                
        finally:
            db.close()
        
        # Set random seed for reproducibility
        if request.random_seed:
            np.random.seed(request.random_seed)
        
        # Run Monte Carlo simulation
        mu = 0  # Drift - can be enhanced with forecast data
        sigma = estimated_vol
        dt = request.time_horizon / 252  # Convert days to years for financial modeling
        
        # Generate correlated random shocks
        iterations = min(request.iterations, 50000)  # Limit for performance
        random_shocks = np.random.normal(mu * dt, sigma * np.sqrt(dt), iterations)
        
        # Apply shocks to base score
        simulated_scores = base_score + random_shocks
        
        # Ensure scores stay within valid range [0, 100]
        simulated_scores = np.clip(simulated_scores, 0, 100)
        
        # Calculate statistics
        mean_score = float(np.mean(simulated_scores))
        std_score = float(np.std(simulated_scores))
        
        # Calculate confidence intervals
        percentiles = {}
        for conf_level in request.confidence_levels:
            alpha = (1 - conf_level) / 2
            lower = float(np.percentile(simulated_scores, alpha * 100))
            upper = float(np.percentile(simulated_scores, (1 - alpha) * 100))
            percentiles[f"{conf_level:.0%}"] = {"lower": lower, "upper": upper}
        
        # Generate paths for visualization (sample subset)
        n_paths = min(100, iterations)
        sample_indices = np.random.choice(iterations, n_paths, replace=False)
        
        paths = []
        for i in range(n_paths):
            idx = sample_indices[i]
            paths.append({
                "path_id": i,
                "start_score": base_score,
                "end_score": float(simulated_scores[idx]),
                "delta": float(simulated_scores[idx] - base_score)
            })
        
        # Create distribution histogram
        hist, bin_edges = np.histogram(simulated_scores, bins=20)
        distribution = []
        for i in range(len(hist)):
            distribution.append({
                "range": f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}",
                "count": int(hist[i]),
                "probability": float(hist[i] / iterations)
            })
        
        logger.info(f"Monte Carlo simulation completed: {iterations} iterations, baseline {base_score:.1f}, vol {estimated_vol:.2f}")
        
        return {
            "simulation_id": f"mc_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "parameters": {
                "iterations": iterations,
                "time_horizon_days": request.time_horizon,
                "baseline_score": base_score,
                "estimated_volatility": estimated_vol,
                "random_seed": request.random_seed
            },
            "results": {
                "mean_score": mean_score,
                "std_score": std_score,
                "min_score": float(np.min(simulated_scores)),
                "max_score": float(np.max(simulated_scores)),
                "baseline_score": base_score,
                "mean_delta": mean_score - base_score
            },
            "confidence_intervals": percentiles,
            "distribution": distribution,
            "sample_paths": paths[:10],  # Return first 10 paths
            "risk_metrics": {
                "var_95": float(np.percentile(simulated_scores, 5)),
                "cvar_95": float(np.mean(simulated_scores[simulated_scores <= np.percentile(simulated_scores, 5)])),
                "prob_increase_gt_5": float(np.mean((simulated_scores - base_score) > 5)),
                "prob_decrease_gt_5": float(np.mean((simulated_scores - base_score) < -5))
            },
            "generated_at": datetime.utcnow().isoformat() + "Z"
        }
        
    except Exception as e:
        logger.error(f"Monte Carlo simulation failed: {e}")
        raise HTTPException(status_code=503, detail=f"Simulation service unavailable: {str(e)}")

@router.post("/stress-test")
async def run_stress_test(
    request: StressTestRequest,
    _auth: dict = Depends(optional_auth),
    _rate_limit: bool = Depends(require_system_rate_limit)
) -> Dict[str, Any]:
    """Run stress test using predefined or custom scenarios."""
    try:
        # Get current baseline data
        from app.main import _get_observations
        from app.services.geri import compute_geri_score
        
        observations = _get_observations()
        if not observations:
            raise HTTPException(status_code=503, detail="No observation data available for stress testing")
        
        # Compute baseline GERI
        baseline_result = compute_geri_score(observations)
        baseline_score = baseline_result.get("score", 50)
        baseline_contributions = baseline_result.get("contributions", {})
        
        # Define predefined stress scenarios
        stress_scenarios = {
            "rate_shock": {
                "name": "Interest Rate Shock +200bps",
                "description": "Sudden 200 basis point increase in interest rates",
                "shocks": {"YIELD_CURVE": 2.0, "CREDIT_SPREAD": 0.5}
            },
            "energy_crisis": {
                "name": "Energy Crisis +50%",
                "description": "Oil prices increase 50%, energy supply disruption",
                "shocks": {"WTI_OIL": 50.0, "FREIGHT_DIESEL": 25.0}
            },
            "macro_recession": {
                "name": "Macro Recession",
                "description": "PMI collapse, unemployment spike, market volatility",
                "shocks": {"PMI": -15.0, "UNEMPLOYMENT": 3.0, "VIX": 20.0}
            },
            "financial_crisis": {
                "name": "Financial Crisis",
                "description": "Credit spreads widen dramatically, market stress",
                "shocks": {"CREDIT_SPREAD": 3.0, "VIX": 30.0, "YIELD_CURVE": -1.0}
            }
        }
        
        # Get scenario definition
        if request.scenario_id in stress_scenarios:
            scenario = stress_scenarios[request.scenario_id]
            shocks = scenario["shocks"]
        elif request.custom_shocks:
            scenario = {
                "name": "Custom Stress Test",
                "description": "User-defined stress scenario",
                "shocks": request.custom_shocks
            }
            shocks = request.custom_shocks
        else:
            raise HTTPException(status_code=400, detail="Invalid scenario_id and no custom_shocks provided")
        
        # Apply shocks to current observations
        stressed_observations = {}
        component_impacts = {}
        
        for series_id, obs_list in observations.items():
            if obs_list:
                current_value = obs_list[-1].value
                shock_pct = shocks.get(series_id, 0.0)
                
                # Apply percentage shock
                if shock_pct != 0:
                    stressed_value = current_value * (1 + shock_pct / 100) if abs(shock_pct) > 10 else current_value + shock_pct
                    component_impacts[series_id] = {
                        "original_value": current_value,
                        "stressed_value": stressed_value,
                        "shock_applied": shock_pct,
                        "absolute_change": stressed_value - current_value
                    }
                else:
                    stressed_value = current_value
                    component_impacts[series_id] = {
                        "original_value": current_value,
                        "stressed_value": stressed_value,
                        "shock_applied": 0.0,
                        "absolute_change": 0.0
                    }
                
                # Create stressed observation
                from app.services.ingestion import Observation
                stressed_observations[series_id] = [Observation(
                    series_id=series_id,
                    observed_at=obs_list[-1].observed_at,
                    value=stressed_value
                )]
        
        # Compute stressed GERI
        stressed_result = compute_geri_score(stressed_observations)
        stressed_score = stressed_result.get("score", 50)
        stressed_contributions = stressed_result.get("contributions", {})
        
        # Calculate impact metrics
        total_impact = stressed_score - baseline_score
        
        # Contribution changes
        contribution_changes = {}
        for series_id in baseline_contributions:
            baseline_contrib = baseline_contributions.get(series_id, 0)
            stressed_contrib = stressed_contributions.get(series_id, 0)
            contribution_changes[series_id] = stressed_contrib - baseline_contrib
        
        logger.info(f"Stress test completed: {request.scenario_id}, baseline {baseline_score:.1f} → stressed {stressed_score:.1f}")
        
        return {
            "stress_test_id": f"stress_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "scenario": scenario,
            "baseline": {
                "score": baseline_score,
                "band": baseline_result.get("band", "moderate"),
                "contributions": baseline_contributions
            },
            "stressed": {
                "score": stressed_score,
                "band": stressed_result.get("band", "moderate"), 
                "contributions": stressed_contributions
            },
            "impact_analysis": {
                "total_impact": total_impact,
                "impact_percentage": (total_impact / baseline_score) * 100 if baseline_score > 0 else 0,
                "risk_band_change": f"{baseline_result.get('band', 'moderate')} → {stressed_result.get('band', 'moderate')}",
                "contribution_changes": contribution_changes
            },
            "component_impacts": component_impacts,
            "stress_metrics": {
                "severity": "extreme" if abs(total_impact) > 20 else "severe" if abs(total_impact) > 10 else "moderate",
                "components_shocked": len([k for k, v in component_impacts.items() if v["shock_applied"] != 0]),
                "max_component_impact": max([abs(v["absolute_change"]) for v in component_impacts.values()]) if component_impacts else 0
            },
            "generated_at": datetime.utcnow().isoformat() + "Z"
        }
        
    except Exception as e:
        logger.error(f"Stress test failed: {e}")
        raise HTTPException(status_code=503, detail=f"Stress test service unavailable: {str(e)}")

@router.get("/scenarios")
async def get_simulation_scenarios(
    _auth: dict = Depends(optional_auth),
    _rate_limit: bool = Depends(require_system_rate_limit)
) -> Dict[str, Any]:
    """Get available simulation scenarios for stress testing."""
    return {
        "stress_scenarios": {
            "rate_shock": {
                "name": "Interest Rate Shock +200bps", 
                "description": "Sudden 200 basis point increase in interest rates",
                "severity": "severe",
                "components": ["YIELD_CURVE", "CREDIT_SPREAD"]
            },
            "energy_crisis": {
                "name": "Energy Crisis +50%",
                "description": "Oil prices increase 50%, energy supply disruption", 
                "severity": "extreme",
                "components": ["WTI_OIL", "FREIGHT_DIESEL"]
            },
            "macro_recession": {
                "name": "Macro Recession",
                "description": "PMI collapse, unemployment spike, market volatility",
                "severity": "extreme", 
                "components": ["PMI", "UNEMPLOYMENT", "VIX"]
            },
            "financial_crisis": {
                "name": "Financial Crisis",
                "description": "Credit spreads widen dramatically, market stress",
                "severity": "extreme",
                "components": ["CREDIT_SPREAD", "VIX", "YIELD_CURVE"]
            }
        },
        "monte_carlo_params": {
            "max_iterations": 50000,
            "default_iterations": 10000,
            "max_time_horizon": 365,
            "confidence_levels": [0.90, 0.95, 0.99]
        },
        "available_components": list({
            "VIX": "Market Volatility Index",
            "YIELD_CURVE": "10Y-2Y Treasury Spread (bps)",
            "CREDIT_SPREAD": "Corporate Credit Spread (bps)", 
            "PMI": "PMI Manufacturing Index",
            "WTI_OIL": "WTI Oil Price ($/barrel)",
            "UNEMPLOYMENT": "Unemployment Rate (%)",
            "FREIGHT_DIESEL": "Freight Diesel Price ($/gallon)"
        }.items()),
        "updated_at": datetime.utcnow().isoformat() + "Z"
    }

@router.get("/volatility")
async def get_volatility_calibration(
    _auth: dict = Depends(optional_auth),
    _rate_limit: bool = Depends(require_system_rate_limit)
) -> Dict[str, Any]:
    """Get volatility calibration data for Monte Carlo simulations."""
    try:
        # This would normally fetch historical volatility from market data
        # For now, return reasonable estimates based on typical financial volatility
        return {
            "grii_volatility": {
                "daily_vol": 2.0,
                "annual_vol": 31.6,  # sqrt(252) * daily_vol
                "calibration_period": "30_days",
                "confidence": 0.85
            },
            "component_volatilities": {
                "VIX": {"daily_vol": 4.2, "annual_vol": 66.7},
                "YIELD_CURVE": {"daily_vol": 0.15, "annual_vol": 2.4},
                "CREDIT_SPREAD": {"daily_vol": 0.08, "annual_vol": 1.3},
                "PMI": {"daily_vol": 0.3, "annual_vol": 4.8},
                "WTI_OIL": {"daily_vol": 2.8, "annual_vol": 44.5},
                "UNEMPLOYMENT": {"daily_vol": 0.02, "annual_vol": 0.3}
            },
            "correlation_matrix": {
                "VIX": {"CREDIT_SPREAD": 0.65, "YIELD_CURVE": -0.45},
                "CREDIT_SPREAD": {"VIX": 0.65, "UNEMPLOYMENT": 0.35},
                "YIELD_CURVE": {"VIX": -0.45, "PMI": 0.25},
                "WTI_OIL": {"PMI": 0.40, "VIX": 0.15}
            },
            "updated_at": datetime.utcnow().isoformat() + "Z"
        }
        
    except Exception as e:
        logger.error(f"Volatility calibration failed: {e}")
        raise HTTPException(status_code=503, detail=f"Volatility service unavailable: {str(e)}")