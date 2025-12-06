"""
Predictive Analysis API

Provides predictive analytics endpoints for supply chain disruption forecasting,
risk modeling, and early warning systems.
"""

import logging
from datetime import datetime
from fastapi import APIRouter, Depends, Query
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

from app.api.schemas import (
    DisruptionPredictionsResponse,
    CascadeImpactResponse,
    EarlyWarningResponse,
)
from app.core.security import require_system_rate_limit
from app.services.predictive_analytics import get_predictive_analytics, RiskLevel, DisruptionType

router = APIRouter(prefix="/api/v1/predictive", tags=["predictive"])


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


@router.get("/disruption-forecast", response_model=DisruptionPredictionsResponse)
async def get_disruption_forecast(
    time_horizon: int = Query(30, description="Forecast horizon in days", ge=1, le=365),
    risk_level_filter: Optional[str] = Query(None, description="Filter by risk level (low, medium, high, critical)"),
    include_cascade: bool = Query(True, description="Include cascade impact modeling"),
    _rate_limit: bool = Depends(require_system_rate_limit),
) -> Dict[str, Any]:
    """Get predictive disruption forecasting analysis."""
    as_of = _now_iso()
    
    try:
        analytics_service = get_predictive_analytics()
        
        # Get predictions with or without cascade effects
        if include_cascade:
            predictions, cascade_models = analytics_service.predict_disruptions(
                time_horizon_days=time_horizon,
                include_cascade_effects=True
            )
        else:
            predictions = analytics_service.predict_disruptions(
                time_horizon_days=time_horizon,
                include_cascade_effects=False
            )
            cascade_models = []
        
        # Filter predictions by risk level if specified
        if risk_level_filter:
            risk_level = RiskLevel(risk_level_filter.lower())
            predictions = [p for p in predictions if p.risk_level == risk_level]
        
        # Convert predictions to response format
        predictions_data = []
        for prediction in predictions:
            predictions_data.append({
                "disruption_type": prediction.disruption_type.value,
                "probability": round(prediction.probability, 3),
                "risk_level": prediction.risk_level.value,
                "estimated_impact_usd": prediction.estimated_impact_usd,
                "confidence_score": round(prediction.confidence_score, 3),
                "time_horizon_days": prediction.time_horizon_days,
                "affected_regions": prediction.affected_regions,
                "affected_commodities": prediction.affected_commodities,
                "risk_triggers": prediction.triggers,
                "mitigation_strategies": prediction.mitigation_recommendations
            })
        
        # Convert cascade models to response format
        cascade_data = []
        for model in cascade_models[:10]:  # Top 10 most at-risk nodes
            cascade_data.append({
                "node_id": model.node_id,
                "node_name": model.node_name,
                "direct_impact_probability": round(model.direct_impact_probability, 3),
                "indirect_impact_probability": round(model.indirect_impact_probability, 3),
                "cascade_delay_hours": round(model.cascade_delay_hours, 1),
                "economic_impact_usd": model.economic_impact_usd,
                "recovery_time_days": model.recovery_time_days
            })
        
        # Calculate summary statistics
        total_economic_risk = sum(p["estimated_impact_usd"] for p in predictions_data)
        high_risk_count = len([p for p in predictions_data if p["risk_level"] in ["high", "critical"]])
        avg_confidence = sum(p["confidence_score"] for p in predictions_data) / max(1, len(predictions_data))
        
        logger.info(f"Generated {len(predictions_data)} disruption predictions with total risk exposure: ${total_economic_risk:,.0f}")
        
        return {
            "as_of": as_of,
            "forecast_horizon_days": time_horizon,
            "predictions": predictions_data,
            "cascade_impacts": cascade_data if include_cascade else [],
            "summary": {
                "total_predictions": len(predictions_data),
                "high_risk_predictions": high_risk_count,
                "total_economic_risk_usd": total_economic_risk,
                "average_confidence": round(avg_confidence, 3),
                "cascade_nodes_analyzed": len(cascade_data)
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to generate disruption forecast: {e}")
        
        # Try to get cached predictions from successful previous runs
        from app.core.unified_cache import UnifiedCache
        cache = UnifiedCache("predictive_analysis")
        cached_result, cache_meta = cache.get("disruption_forecast")
        
        if cached_result and cache_meta and not cache_meta.is_stale_hard:
            logger.info("Using cached disruption forecast as fallback")
            cached_result["cache_fallback"] = True
            cached_result["cache_age_seconds"] = cache_meta.age_seconds
            return cached_result
        else:
            # Only if no valid cache exists, return error response
            logger.error("No valid cached disruption forecast data available")
            raise HTTPException(
                status_code=503,
                detail=f"Predictive analysis service unavailable and no cached data: {str(e)}"
            )


@router.get("/correlation-analysis", response_model=CascadeImpactResponse)
async def get_correlation_analysis(
    _rate_limit: bool = Depends(require_system_rate_limit),
) -> Dict[str, Any]:
    """Get disruption correlation and cascade amplification analysis."""
    as_of = _now_iso()
    
    try:
        analytics_service = get_predictive_analytics()
        correlation_data = analytics_service.analyze_disruption_correlation([])
        
        logger.info("Generated disruption correlation analysis")
        
        return {
            "as_of": as_of,
            "disruption_correlations": correlation_data["correlations"],
            "amplification_factors": correlation_data["amplification_factors"],
            "high_risk_combinations": correlation_data["high_risk_combinations"],
            "analysis_notes": {
                "correlation_threshold": 0.3,
                "amplification_model": "multiplicative",
                "confidence_level": 0.80
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to generate correlation analysis: {e}")
        
        # Fallback correlation data
        return {
            "as_of": as_of,
            "disruption_correlations": {
                "geopolitical_cyber": 0.65,
                "natural_disaster_maritime": 0.45,
                "trade_policy_geopolitical": 0.78
            },
            "amplification_factors": {
                "simultaneous_disruptions": {"2_types": 1.4, "3_types": 2.1},
                "critical_node_impact": 1.8
            },
            "high_risk_combinations": [
                "Geopolitical + Cyber attack",
                "Natural disaster + Maritime incident"
            ],
            "analysis_notes": {
                "note": "Fallback analysis - full service unavailable"
            }
        }


@router.get("/early-warning", response_model=EarlyWarningResponse)
async def get_early_warning_system(
    _rate_limit: bool = Depends(require_system_rate_limit),
) -> Dict[str, Any]:
    """Get early warning system configuration and current alert status."""
    as_of = _now_iso()
    
    try:
        analytics_service = get_predictive_analytics()
        warning_config = analytics_service.generate_early_warning_system()
        
        # Get current predictions to determine alert status
        predictions = analytics_service.predict_disruptions(time_horizon_days=7, include_cascade_effects=False)
        
        # Determine current alert level
        alert_level = "green"
        active_alerts = []
        
        for prediction in predictions:
            threshold_met = False
            
            # Check amber alert thresholds
            if (prediction.probability >= warning_config["thresholds"]["amber_alert"]["probability_threshold"] and
                prediction.estimated_impact_usd >= warning_config["thresholds"]["amber_alert"]["impact_threshold_usd"] and
                prediction.confidence_score >= warning_config["thresholds"]["amber_alert"]["confidence_threshold"]):
                
                alert_level = max(alert_level, "amber", key=lambda x: ["green", "amber", "red"].index(x))
                threshold_met = True
            
            # Check red alert thresholds
            if (prediction.probability >= warning_config["thresholds"]["red_alert"]["probability_threshold"] and
                prediction.estimated_impact_usd >= warning_config["thresholds"]["red_alert"]["impact_threshold_usd"] and
                prediction.confidence_score >= warning_config["thresholds"]["red_alert"]["confidence_threshold"]):
                
                alert_level = "red"
                threshold_met = True
            
            if threshold_met:
                active_alerts.append({
                    "disruption_type": prediction.disruption_type.value,
                    "risk_level": prediction.risk_level.value,
                    "probability": round(prediction.probability, 3),
                    "estimated_impact_usd": prediction.estimated_impact_usd,
                    "affected_regions": prediction.affected_regions[:3],  # Top 3 regions
                    "immediate_actions": prediction.mitigation_recommendations[:2]  # Top 2 actions
                })
        
        logger.info(f"Early warning system status: {alert_level} with {len(active_alerts)} active alerts")
        
        return {
            "as_of": as_of,
            "alert_level": alert_level,
            "active_alerts": active_alerts,
            "alert_thresholds": warning_config["thresholds"],
            "monitoring_indicators": warning_config["indicators"],
            "system_configuration": {
                "update_frequency": warning_config["update_frequency"],
                "escalation_protocols": warning_config["escalation_protocols"]
            },
            "recommendations": {
                "immediate_actions": [
                    "Monitor high-risk regions closely",
                    "Review contingency plans",
                    "Increase supplier communication"
                ] if alert_level != "green" else ["Continue normal monitoring"],
                "strategic_actions": [
                    "Diversify critical supply chains",
                    "Build emergency inventory buffers",
                    "Strengthen early warning capabilities"
                ]
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to generate early warning system data: {e}")
        
        # Fallback early warning data
        return {
            "as_of": as_of,
            "alert_level": "green",
            "active_alerts": [],
            "alert_thresholds": {
                "amber_alert": {"probability_threshold": 0.4, "impact_threshold_usd": 200_000_000},
                "red_alert": {"probability_threshold": 0.7, "impact_threshold_usd": 500_000_000}
            },
            "monitoring_indicators": {
                "geopolitical": ["Military activity", "Trade tensions"],
                "natural_disaster": ["Weather warnings", "Seismic activity"]
            },
            "system_configuration": {
                "update_frequency": "every_4_hours",
                "note": "Fallback configuration - full system unavailable"
            },
            "recommendations": {
                "immediate_actions": ["Continue monitoring"],
                "strategic_actions": ["Strengthen supply chain resilience"]
            }
        }


@router.get("/risk-scenarios")
async def get_risk_scenarios(
    scenario_type: Optional[str] = Query(None, description="Filter by scenario type"),
    _rate_limit: bool = Depends(require_system_rate_limit),
) -> Dict[str, Any]:
    """Get predefined risk scenarios for stress testing."""
    as_of = _now_iso()
    
    # Predefined stress test scenarios
    scenarios = {
        "suez_canal_blockage": {
            "name": "Suez Canal Extended Blockage",
            "description": "Major vessel blocking Suez Canal for 7+ days",
            "probability": 0.15,
            "duration_days": 10,
            "economic_impact_usd": 9_600_000_000,
            "affected_trade_percent": 12,
            "affected_commodities": ["crude_oil", "containers", "manufactured_goods"],
            "cascade_effects": {
                "europe": {"delay_days": 14, "cost_increase_percent": 15},
                "asia": {"delay_days": 12, "cost_increase_percent": 18}
            }
        },
        "china_port_shutdown": {
            "name": "Major Chinese Ports COVID Shutdown",
            "description": "Extended shutdown of Shanghai, Shenzhen, and Guangzhou ports",
            "probability": 0.25,
            "duration_days": 21,
            "economic_impact_usd": 15_000_000_000,
            "affected_trade_percent": 28,
            "affected_commodities": ["electronics", "manufactured_goods", "textiles"],
            "cascade_effects": {
                "global_electronics": {"delay_days": 45, "cost_increase_percent": 35},
                "automotive": {"delay_days": 60, "cost_increase_percent": 25}
            }
        },
        "middle_east_conflict": {
            "name": "Extended Middle East Regional Conflict",
            "description": "Military conflict affecting Strait of Hormuz and regional trade",
            "probability": 0.35,
            "duration_days": 60,
            "economic_impact_usd": 25_000_000_000,
            "affected_trade_percent": 20,
            "affected_commodities": ["crude_oil", "natural_gas", "petrochemicals"],
            "cascade_effects": {
                "energy_markets": {"price_increase_percent": 45},
                "global_inflation": {"impact_percent": 2.5}
            }
        },
        "cyber_attack_ports": {
            "name": "Coordinated Cyber Attack on Port Systems",
            "description": "Simultaneous cyber attacks on major US and European ports",
            "probability": 0.18,
            "duration_days": 5,
            "economic_impact_usd": 3_200_000_000,
            "affected_trade_percent": 8,
            "affected_commodities": ["containers", "automotive", "technology"],
            "cascade_effects": {
                "supply_chain_trust": {"recovery_days": 90},
                "cybersecurity_costs": {"increase_percent": 150}
            }
        }
    }
    
    # Filter scenarios if requested
    if scenario_type and scenario_type in scenarios:
        filtered_scenarios = {scenario_type: scenarios[scenario_type]}
    else:
        filtered_scenarios = scenarios
    
    # Calculate aggregate risk exposure
    total_economic_risk = sum(s["economic_impact_usd"] for s in filtered_scenarios.values())
    
    logger.info(f"Provided {len(filtered_scenarios)} risk scenarios with total exposure: ${total_economic_risk:,.0f}")
    
    return {
        "as_of": as_of,
        "scenarios": filtered_scenarios,
        "scenario_summary": {
            "total_scenarios": len(filtered_scenarios),
            "total_economic_exposure_usd": total_economic_risk,
            "average_probability": sum(s["probability"] for s in filtered_scenarios.values()) / len(filtered_scenarios),
            "highest_impact_scenario": max(filtered_scenarios.keys(), key=lambda k: filtered_scenarios[k]["economic_impact_usd"]),
            "most_likely_scenario": max(filtered_scenarios.keys(), key=lambda k: filtered_scenarios[k]["probability"])
        },
        "stress_test_recommendations": [
            "Test contingency plans against these scenarios",
            "Evaluate supplier diversification effectiveness",
            "Assess financial resilience under stress conditions",
            "Review crisis communication protocols"
        ]
    }