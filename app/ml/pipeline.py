"""
ML Pipeline Integration Module

This module provides a unified interface to the trained ML models
that are loaded from disk. Individual models are handled by their
respective modules:
- app.ml.regime: Regime classification
- app.ml.forecast: 24-hour forecasting  
- app.ml.anomaly: Anomaly detection

The models are trained using app.services.training and saved to
the configured models directory.
"""

from typing import Dict, List
import logging

from app.services.ingestion import Observation
from app.ml.regime import classify_regime
from app.ml.forecast import forecast_delta  
from app.ml.anomaly import detect_anomalies

logger = logging.getLogger(__name__)


def run_complete_analysis(observations: Dict[str, List[Observation]]) -> Dict:
    """
    Run complete ML analysis pipeline on observations.
    
    Returns combined results from regime classification, 
    forecasting, and anomaly detection.
    """
    try:
        results = {}
        
        # Regime classification
        results["regime"] = classify_regime(observations)
        
        # 24-hour forecast  
        results["forecast"] = forecast_delta(observations)
        
        # Anomaly detection
        results["anomaly"] = detect_anomalies(observations)
        
        logger.info("Complete ML analysis pipeline executed successfully")
        return results
        
    except Exception as e:
        logger.error(f"ML pipeline failed: {e}")
        return {
            "regime": {"Calm": 0.25, "Inflationary_Stress": 0.25, "Supply_Shock": 0.25, "Financial_Stress": 0.25},
            "forecast": {"delta": 0.0, "p_gt_5": 0.5, "confidence_interval": [-2.0, 2.0]},
            "anomaly": {"score": 0.1, "classification": "normal"}
        }
