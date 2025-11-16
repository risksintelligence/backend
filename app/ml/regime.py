import logging
from typing import Dict, List
import pandas as pd

from app.services.ingestion import Observation
from app.services.training import load_model

logger = logging.getLogger(__name__)


def classify_regime(observations: Dict[str, List[Observation]]) -> Dict[str, float]:
    """Classify current market regime using trained ML model."""
    try:
        # Load trained models
        model = load_model("regime_classifier.pkl")
        scaler = load_model("regime_scaler.pkl")
        
        # Prepare features from latest observations
        features = _prepare_regime_features(observations)
        if features is None:
            logger.warning("Insufficient data for regime classification")
            return _fallback_regime()
        
        # Scale and predict
        features_scaled = scaler.transform([features])
        probabilities = model.predict_proba(features_scaled)[0]
        
        # Map to regime names (matching training labels)
        regime_names = ["Calm", "Inflationary_Stress", "Supply_Shock"]
        result = {}
        for i, name in enumerate(regime_names):
            if i < len(probabilities):
                result[name] = round(float(probabilities[i]), 3)
        
        # Add Financial_Stress as residual
        result["Financial_Stress"] = round(1.0 - sum(result.values()), 3)
        
        return result
        
    except Exception as e:
        logger.error(f"Regime classification failed: {e}")
        return _fallback_regime()


def _prepare_regime_features(observations: Dict[str, List[Observation]]) -> List[float]:
    """Extract features for regime classification from observations."""
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
        df['value_lag1'] = df['value'].shift(1)
        df['value_lag7'] = df['value'].shift(7) if len(df) > 7 else df['value'].shift(1)
        df['rolling_mean_7'] = df['value'].rolling(min(7, len(df))).mean()
        df['rolling_std_7'] = df['value'].rolling(min(7, len(df))).std()
        df['pct_change'] = df['value'].pct_change()
        df['volatility'] = df['pct_change'].rolling(min(7, len(df))).std()
        
        # Get latest features (last row)
        latest = df.dropna().iloc[-1] if len(df.dropna()) > 0 else df.iloc[-1]
        features = [
            latest.get('value_lag1', 0) or 0,
            latest.get('value_lag7', 0) or 0,
            latest.get('rolling_mean_7', 0) or 0,
            latest.get('rolling_std_7', 0) or 0,
            latest.get('volatility', 0) or 0
        ]
        
        return features
        
    except Exception as e:
        logger.error(f"Feature preparation failed: {e}")
        return None


def _fallback_regime() -> Dict[str, float]:
    """Fallback regime probabilities when ML model fails."""
    return {
        "Calm": 0.2,
        "Inflationary_Stress": 0.4,
        "Supply_Shock": 0.25,
        "Financial_Stress": 0.15,
    }
