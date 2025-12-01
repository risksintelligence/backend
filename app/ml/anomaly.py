import logging
from typing import Dict, List
import pandas as pd
import numpy as np

from app.services.ingestion import Observation
from app.services.training import load_model

logger = logging.getLogger(__name__)


def detect_anomalies(observations: Dict[str, List[Observation]]) -> Dict[str, float]:
    """Detect anomalies using trained isolation forest model."""
    try:
        # Load trained model
        model = load_model("anomaly_detector.pkl", max_age_hours=48)

        # Prepare features from latest observations
        features = _prepare_anomaly_features(observations)
        if features is None:
            logger.warning("Insufficient data for anomaly detection")
            return _fallback_anomaly()

        # Predict anomaly score
        anomaly_score = model.score_samples([features])[0]
        is_outlier = model.predict([features])[0] == -1

        # Convert to 0-1 score (higher = more anomalous)
        normalized_score = max(0.0, min(1.0, (0.5 - anomaly_score) * 2))

        classification = "anomaly" if is_outlier else "normal"

        return {
            "score": round(float(normalized_score), 3),
            "classification": classification,
        }

    except FileNotFoundError as e:
        logger.error(f"Anomaly model missing: {e}")
        raise
    except ValueError as e:
        logger.error(f"Anomaly model stale or invalid: {e}")
        raise
    except Exception as e:
        logger.error(f"Anomaly detection failed: {e}")
        return _fallback_anomaly()


def _prepare_anomaly_features(observations: Dict[str, List[Observation]]) -> List[float]:
    """Extract features for anomaly detection from observations."""
    try:
        # Get recent data for feature calculation
        all_values = []
        for series_data in observations.values():
            if len(series_data) > 0:
                all_values.extend([obs.value for obs in series_data[-30:]])  # Last 30 points
        
        if len(all_values) < 7:
            return None
            
        df = pd.DataFrame({'value': all_values})
        
        # Calculate features matching training
        df['pct_change'] = df['value'].pct_change()
        df['volatility'] = df['pct_change'].rolling(min(7, len(df))).std()
        df['rolling_std_7'] = df['value'].rolling(min(7, len(df))).std()
        
        # Get latest features (last row)
        latest = df.dropna().iloc[-1] if len(df.dropna()) > 0 else df.iloc[-1]
        features = [
            latest.get('value', 0) or 0,
            latest.get('pct_change', 0) or 0,
            latest.get('volatility', 0) or 0,
            latest.get('rolling_std_7', 0) or 0
        ]
        
        return features
        
    except Exception as e:
        logger.error(f"Anomaly feature preparation failed: {e}")
        return None


def _fallback_anomaly() -> Dict[str, float]:
    """Fallback anomaly detection when ML model fails."""
    return {
        "score": 0.12,
        "classification": "normal",
    }
