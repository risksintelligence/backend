from datetime import datetime, timedelta
from statistics import mean, pstdev
from typing import Dict, List, Optional
import logging

from app.services.ingestion import Observation

logger = logging.getLogger(__name__)

# GERI v1 component weights per methodology documentation
# Finance pillar (~33%), Supply chain pillar (~33%), Macro pillar (~33%)
BASE_WEIGHTS = {
    "VIX": 0.15,                    # Finance: Volatility
    "YIELD_CURVE": 0.10,            # Finance: 10Y-2Y spread
    "CREDIT_SPREAD": 0.08,          # Finance: BAA-10Y spread
    "FREIGHT_DIESEL": 0.12,         # Supply: Diesel prices
    "BALTIC_DRY": 0.10,             # Supply: Shipping costs
    "PMI": 0.15,                    # Macro: Manufacturing sentiment
    "WTI_OIL": 0.15,               # Macro: Energy prices
    "UNEMPLOYMENT": 0.15,           # Macro: Labor market
}

# Risk direction per component (1 = higher value = higher risk, -1 = inverted)
RISK_DIRECTIONS = {
    "VIX": 1,
    "YIELD_CURVE": -1,              # Inverted curve = risk
    "CREDIT_SPREAD": 1,
    "FREIGHT_DIESEL": 1,
    "BALTIC_DRY": 1,
    "PMI": -1,                      # Lower PMI = risk
    "WTI_OIL": 1,
    "UNEMPLOYMENT": 1,
}

WINDOW_YEARS = 5


def calculate_rolling_zscore(observations: List[Observation], component_id: str) -> Optional[float]:
    """Calculate 5-year rolling z-score with risk direction adjustment."""
    if not observations:
        return None
    
    # Filter to 5-year window
    cutoff = datetime.utcnow() - timedelta(days=WINDOW_YEARS * 365)
    windowed_obs = [obs for obs in observations if obs.observed_at >= cutoff]
    
    if len(windowed_obs) < 30:  # Need minimum data points
        logger.warning(f"Insufficient data for {component_id}: {len(windowed_obs)} points")
        return None
    
    values = [obs.value for obs in windowed_obs]
    latest_value = values[-1]
    
    # Calculate rolling statistics
    avg = mean(values)
    std_dev = pstdev(values) or 1.0
    
    # Calculate z-score
    z_score = (latest_value - avg) / std_dev
    
    # Apply risk direction
    risk_direction = RISK_DIRECTIONS.get(component_id, 1)
    adjusted_z = z_score * risk_direction
    
    logger.debug(f"{component_id}: value={latest_value:.3f}, z={z_score:.3f}, adjusted={adjusted_z:.3f}")
    return adjusted_z


def compute_geri_score(observations: Dict[str, List[Observation]], 
                      regime_weights: Optional[Dict[str, float]] = None,
                      regime_confidence: float = 0.0) -> Dict[str, any]:
    """
    Compute GERI score following v1 methodology.
    
    Args:
        observations: Time series data by component
        regime_weights: Optional regime-specific weights
        regime_confidence: Confidence level for regime weights (0-1)
    """
    contributions = {}
    component_scores = {}
    total_weight = 0.0
    weighted_sum = 0.0
    
    # Determine which weights to use
    use_regime_weights = regime_weights and regime_confidence >= 0.6
    active_weights = regime_weights if use_regime_weights else BASE_WEIGHTS
    
    logger.info(f"Computing GERI with {'regime' if use_regime_weights else 'base'} weights")
    
    for component_id in BASE_WEIGHTS.keys():
        obs_list = observations.get(component_id, [])
        
        # Calculate z-score for component
        z_score = calculate_rolling_zscore(obs_list, component_id)
        
        if z_score is not None:
            weight = active_weights.get(component_id, BASE_WEIGHTS[component_id])
            contribution = weight * z_score
            
            contributions[component_id] = round(contribution, 4)
            component_scores[component_id] = round(z_score, 3)
            
            weighted_sum += contribution
            total_weight += weight
        else:
            # Component missing - redistribute weight proportionally
            logger.warning(f"Missing data for {component_id}, excluding from calculation")
    
    if total_weight == 0:
        logger.error("No valid components for GERI calculation")
        return {
            "score": 50.0,
            "band": "moderate",
            "confidence": "low",
            "updated_at": datetime.utcnow().isoformat() + "Z",
            "contributions": {},
            "component_scores": {},
            "metadata": {"error": "No valid data", "total_weight": 0.0}
        }
    
    # Normalize if weights don't sum to 1 due to missing components
    if abs(total_weight - 1.0) > 0.01:
        weighted_sum = weighted_sum / total_weight
        logger.info(f"Normalized weights: {total_weight:.3f} -> 1.0")
    
    # Apply GERI formula: 50 + 10 * weighted_sum
    raw_score = 50.0 + (10.0 * weighted_sum)
    
    # Clip to 0-100 range
    final_score = max(0.0, min(100.0, raw_score))
    
    # Determine risk band
    band = determine_risk_band(final_score)
    
    # Calculate confidence based on data completeness and age
    confidence = calculate_confidence(observations, total_weight)
    
    result = {
        "score": round(final_score, 2),
        "band": band,
        "confidence": confidence,
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "contributions": contributions,
        "component_scores": component_scores,
        "metadata": {
            "total_weight": round(total_weight, 3),
            "raw_score": round(raw_score, 2),
            "regime_override": use_regime_weights,
            "regime_confidence": regime_confidence
        }
    }
    
    logger.info(f"GERI computed: {final_score:.2f} ({band}) with {confidence} confidence")
    return result


def determine_risk_band(score: float) -> str:
    """Map GERI score to risk band per methodology."""
    if score < 20:
        return "minimal"
    elif score < 40:
        return "low"
    elif score < 60:
        return "moderate"
    elif score < 80:
        return "high"
    else:
        return "critical"


def calculate_confidence(observations: Dict[str, List[Observation]], total_weight: float) -> str:
    """Calculate confidence level based on data completeness and freshness."""
    if total_weight < 0.7:
        return "low"  # Missing too many components
    
    # Check data freshness
    stale_count = 0
    for component_id, obs_list in observations.items():
        if obs_list:
            latest_obs = obs_list[-1]
            age_hours = (datetime.utcnow() - latest_obs.observed_at).total_seconds() / 3600
            if age_hours > 48:  # Consider stale after 48 hours
                stale_count += 1
    
    if stale_count > len(observations) // 2:
        return "low"
    elif stale_count > 0:
        return "medium"
    else:
        return "high"


# Backward compatibility alias
def compute_griscore(observations: Dict[str, List[Observation]]) -> Dict[str, float]:
    """Legacy function name - delegates to new implementation."""
    result = compute_geri_score(observations)
    
    # Convert to old format for backward compatibility
    return {
        "score": result["score"],
        "band": result["band"], 
        "updated_at": result["updated_at"],
        "contributions": result["contributions"]
    }


# Legacy function - use determine_risk_band instead
def _band(score: float) -> str:
    return determine_risk_band(score)
