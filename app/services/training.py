import joblib
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from app.db import SessionLocal
from app.models import ObservationModel, ModelMetadataModel

WINDOW_YEARS = 5
from app.core.config import get_settings

settings = get_settings()
MODEL_DIR = str(settings.models_dir)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_training_window() -> Dict[str, List[ObservationModel]]:
    """Fetch observations from the last 5 years, grouped by series."""
    db: Session = SessionLocal()
    cutoff = datetime.utcnow() - timedelta(days=WINDOW_YEARS * 365)
    observations = db.query(ObservationModel).filter(
        ObservationModel.observed_at >= cutoff
    ).order_by(ObservationModel.observed_at).all()
    
    result: Dict[str, List[ObservationModel]] = {}
    for obs in observations:
        result.setdefault(obs.series_id, []).append(obs)
    db.close()
    
    logger.info(f"Fetched {len(observations)} observations across {len(result)} series")
    return result


def prepare_features(data: Dict[str, List[ObservationModel]]) -> pd.DataFrame:
    """Convert observations to DataFrame with engineered features."""
    rows = []
    
    for series_id, observations in data.items():
        if len(observations) < 30:  # Skip series with insufficient data
            continue
            
        # Convert to pandas for easier manipulation
        df = pd.DataFrame([
            {
                'timestamp': obs.observed_at,
                'value': obs.value,
                'series_id': series_id
            }
            for obs in observations
        ])
        
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Feature engineering
        df['value_lag1'] = df['value'].shift(1)
        df['value_lag7'] = df['value'].shift(7)
        df['rolling_mean_7'] = df['value'].rolling(7).mean()
        df['rolling_std_7'] = df['value'].rolling(7).std()
        df['pct_change'] = df['value'].pct_change()
        df['volatility'] = df['pct_change'].rolling(7).std()
        
        # Remove rows with NaN values
        df = df.dropna()
        rows.append(df)
    
    if not rows:
        return pd.DataFrame()
    
    combined_df = pd.concat(rows, ignore_index=True)
    logger.info(f"Prepared {len(combined_df)} feature rows")
    return combined_df


def train_regime_classifier() -> Tuple[RandomForestClassifier, StandardScaler]:
    """Train regime classification model."""
    logger.info("Training regime classifier...")
    
    data = fetch_training_window()
    df = prepare_features(data)
    
    if df.empty:
        raise ValueError("No training data available")
    
    # Create regime labels based on volatility
    volatility_median = df['volatility'].median()
    volatility_75th = df['volatility'].quantile(0.75)
    
    df['regime'] = 0  # stable
    df.loc[df['volatility'] > volatility_median, 'regime'] = 1  # moderate
    df.loc[df['volatility'] > volatility_75th, 'regime'] = 2  # volatile
    
    # Features for training
    feature_cols = ['value_lag1', 'value_lag7', 'rolling_mean_7', 'rolling_std_7', 'volatility']
    X = df[feature_cols].fillna(0)
    y = df['regime']
    
    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    logger.info(f"Regime classifier performance:\n{classification_report(y_test, y_pred)}")
    
    return model, scaler


def train_forecast_model() -> Tuple[LinearRegression, StandardScaler]:
    """Train 24-hour forecast model."""
    logger.info("Training forecast model...")
    
    data = fetch_training_window()
    df = prepare_features(data)
    
    if df.empty:
        raise ValueError("No training data available")
    
    # Create target: next day's value change
    df['target'] = df.groupby('series_id')['value'].shift(-1) - df['value']
    df = df.dropna()
    
    # Features
    feature_cols = ['value_lag1', 'value_lag7', 'rolling_mean_7', 'pct_change', 'volatility']
    X = df[feature_cols].fillna(0)
    y = df['target']
    
    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    logger.info(f"Forecast model MSE: {mse:.4f}")
    
    return model, scaler


def train_anomaly_detector() -> IsolationForest:
    """Train anomaly detection model."""
    logger.info("Training anomaly detector...")
    
    data = fetch_training_window()
    df = prepare_features(data)
    
    if df.empty:
        raise ValueError("No training data available")
    
    # Features for anomaly detection
    feature_cols = ['value', 'pct_change', 'volatility', 'rolling_std_7']
    X = df[feature_cols].fillna(0)
    
    # Train isolation forest
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(X)
    
    # Log some stats
    anomaly_scores = model.score_samples(X)
    logger.info(f"Anomaly detector trained. Mean score: {anomaly_scores.mean():.4f}")
    
    return model


def save_model(model: Any, filename: str) -> None:
    """Save a trained model to disk."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    filepath = os.path.join(MODEL_DIR, filename)
    joblib.dump(model, filepath)
    logger.info(f"Model saved to {filepath}")


def save_model_metadata(model_name: str, model_type: str, metrics: Dict[str, Any]) -> None:
    """Upsert model metadata safely (handles concurrent workers)."""
    db: Session | None = None
    try:
        db = SessionLocal()
        now = datetime.utcnow()
        training_start = now - timedelta(days=5 * 365)

        metadata = db.query(ModelMetadataModel).filter(
            ModelMetadataModel.model_name == model_name
        ).first()

        if metadata is None:
            metadata = ModelMetadataModel(
                model_name=model_name,
                created_at=now,
            )
            db.add(metadata)

        metadata.version = "1.0"
        metadata.trained_at = now
        metadata.training_window_start = training_start
        metadata.training_window_end = now
        metadata.performance_metrics = {"model_type": model_type, **metrics}
        metadata.is_active = True
        metadata.file_path = f"models/{model_name}.pkl"

        db.commit()
        logger.info(f"Saved model metadata for {model_name}")
    except IntegrityError:
        if db:
            db.rollback()
            now = datetime.utcnow()
            training_start = now - timedelta(days=5 * 365)
            db.query(ModelMetadataModel).filter(
                ModelMetadataModel.model_name == model_name
            ).update(
                {
                    ModelMetadataModel.version: "1.0",
                    ModelMetadataModel.trained_at: now,
                    ModelMetadataModel.training_window_start: training_start,
                    ModelMetadataModel.training_window_end: now,
                    ModelMetadataModel.performance_metrics: {"model_type": model_type, **metrics},
                    ModelMetadataModel.is_active: True,
                    ModelMetadataModel.file_path: f"models/{model_name}.pkl",
                },
                synchronize_session=False,
            )
            db.commit()
            logger.info(f"Upserted model metadata after conflict for {model_name}")
    except Exception as e:
        if db:
            db.rollback()
        logger.error(f"Failed to save model metadata: {e}")
    finally:
        if db:
            db.close()


def load_model(filename: str) -> Any:
    """Load a trained model from disk."""
    filepath = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model not found: {filepath}")
    return joblib.load(filepath)


def train_all_models() -> None:
    """Train and save all ML models with deployment optimization."""
    logger.info("Starting complete model training pipeline...")
    
    # Add timeout for deployment contexts
    import signal
    deployment_mode = os.getenv('RENDER_SERVICE_TYPE') == 'background_worker'
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Model training timed out")
    
    if deployment_mode:
        # Set 8 minute timeout for deployment
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(480)  # 8 minutes
    
    try:
        # Train regime classifier
        regime_model, regime_scaler = train_regime_classifier()
        save_model(regime_model, "regime_classifier.pkl")
        save_model(regime_scaler, "regime_scaler.pkl")
        
        # Save regime classifier metadata
        save_model_metadata(
            model_name="regime_classifier",
            model_type="classification",
            metrics={"feature_count": 5, "classes": 3}
        )
        
        # Train forecast model
        forecast_model, forecast_scaler = train_forecast_model()
        save_model(forecast_model, "forecast_model.pkl")
        save_model(forecast_scaler, "forecast_scaler.pkl")
        
        # Save forecast model metadata
        save_model_metadata(
            model_name="forecast_model",
            model_type="regression",
            metrics={"feature_count": 5, "target": "value_change"}
        )
        
        # Train anomaly detector
        anomaly_model = train_anomaly_detector()
        save_model(anomaly_model, "anomaly_detector.pkl")
        
        # Save anomaly detector metadata
        save_model_metadata(
            model_name="anomaly_detector",
            model_type="anomaly_detection",
            metrics={"contamination": 0.1, "feature_count": 4}
        )
        
        logger.info("All models trained and saved successfully!")
        
    except TimeoutError:
        logger.warning("Model training timed out in deployment mode")
        if deployment_mode:
            return  # Don't fail deployment
        raise
    except Exception as e:
        logger.error(f"Training failed: {e}")
        if deployment_mode:
            logger.warning("Model training failed in deployment mode, continuing...")
            return
        raise
    finally:
        if deployment_mode:
            signal.alarm(0)  # Cancel timeout
