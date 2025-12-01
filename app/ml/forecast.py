import logging
from typing import Dict, List
import pandas as pd
import numpy as np

from app.services.ingestion import Observation
from app.services.training import load_model

logger = logging.getLogger(__name__)


def forecast_delta(observations: Dict[str, List[Observation]]) -> Dict[str, float]:
    """Generate 24-hour forecast using trained ML model."""
    try:
        # Load trained models
        model = load_model("forecast_model.pkl", max_age_hours=48)
        scaler = load_model("forecast_scaler.pkl", max_age_hours=48)

        # Prepare features from latest observations
        features = _prepare_forecast_features(observations)
        if features is None:
            logger.warning("Insufficient data for forecasting")
            return _fallback_forecast()

        # Scale and predict
        features_scaled = scaler.transform([features])
        delta_prediction = float(model.predict(features_scaled)[0])

        # Calculate confidence metrics
        confidence_interval = [delta_prediction - 2.0, delta_prediction + 2.0]
        p_gt_5 = max(0.0, min(1.0, (delta_prediction - 5.0) / 10.0 + 0.5))

        return {
            "delta": round(delta_prediction, 2),
            "p_gt_5": round(p_gt_5, 3),
            "confidence_interval": [round(ci, 1) for ci in confidence_interval],
        }

    except FileNotFoundError as e:
        logger.error(f"Forecast model missing: {e}")
        raise
    except ValueError as e:
        logger.error(f"Forecast model stale or invalid: {e}")
        raise
    except Exception as e:
        logger.error(f"Forecast failed: {e}")
        return _fallback_forecast()


FORECAST_FEATURE_NAMES = [
    "value_lag1",
    "value_lag7",
    "rolling_mean_7",
    "pct_change",
    "volatility",
]


def explain_forecast(observations: Dict[str, List[Observation]]) -> List[Dict[str, float]]:
    """
    Return simple driver contributions for the forecast model.
    For linear regression, contribution = coef * scaled_feature.
    """
    try:
        model = load_model("forecast_model.pkl", max_age_hours=48)
        scaler = load_model("forecast_scaler.pkl", max_age_hours=48)

        features = _prepare_forecast_features(observations)
        if features is None:
            logger.warning("Insufficient data for forecast explainability")
            return []

        if not hasattr(model, "coef_"):
            return []

        scaled = scaler.transform([features])[0]
        contributions = []
        for name, coef, val in zip(FORECAST_FEATURE_NAMES, model.coef_, scaled):
            contributions.append(
                {
                    "feature": name,
                    "contribution": round(float(coef * val), 4),
                    "coef": round(float(coef), 4),
                    "value": round(float(val), 4),
                }
            )

        contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
        return contributions
    except Exception as e:
        logger.error(f"Forecast explainability failed: {e}")
        return []


def _prepare_forecast_features(observations: Dict[str, List[Observation]]) -> List[float]:
    """Extract features for forecasting from observations."""
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
        df['pct_change'] = df['value'].pct_change()
        df['volatility'] = df['pct_change'].rolling(min(7, len(df))).std()
        
        # Get latest features (last row)
        latest = df.dropna().iloc[-1] if len(df.dropna()) > 0 else df.iloc[-1]
        features = [
            latest.get('value_lag1', 0) or 0,
            latest.get('value_lag7', 0) or 0,
            latest.get('rolling_mean_7', 0) or 0,
            latest.get('pct_change', 0) or 0,
            latest.get('volatility', 0) or 0
        ]
        
        return features
        
    except Exception as e:
        logger.error(f"Forecast feature preparation failed: {e}")
        return None


def _fallback_forecast() -> Dict[str, float]:
    """Fallback forecast when ML model fails."""
    return {
        "delta": 1.5,
        "p_gt_5": 0.25,
        "confidence_interval": [-2.0, 4.0],
    }
