"""Inference service orchestrating ML predictions."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, TYPE_CHECKING

try:  # optional dependency in slim deployments
    import asyncpg  # type: ignore
except ImportError:  # pragma: no cover - environment without Postgres client
    asyncpg = None  # type: ignore[misc]

try:  # numpy is heavy; allow stub fallback if wheel missing
    import numpy as np
except ImportError:  # pragma: no cover - fallback to stub predictions
    np = None  # type: ignore[misc]

if TYPE_CHECKING:
    import numpy as _np
    NDArray = _np.ndarray
else:  # pragma: no cover - runtime fallback when numpy missing
    NDArray = object  # type: ignore[assignment]

try:  # allow running even if joblib unavailable (no trained artifacts)
    import joblib  # type: ignore
except ImportError:  # pragma: no cover - fallback to stub predictions
    joblib = None  # type: ignore[misc]

from src.ml.training.regime_training import RegimeArtifact
from src.ml.training.forecast_training import ForecastArtifact
from src.ml.training.anomaly_training import AnomalyArtifact


class MLInferenceService:
    def __init__(self) -> None:
        self._dsn = os.environ.get("RIS_POSTGRES_DSN")
        self._regime_artifact: RegimeArtifact | None = None
        self._forecast_artifact: ForecastArtifact | None = None
        self._anomaly_artifact: AnomalyArtifact | None = None

    async def _load_artifacts(self) -> None:
        if (
            self._regime_artifact is not None
            or not self._dsn
            or asyncpg is None
            or joblib is None
        ):
            return
        conn = await asyncpg.connect(self._dsn)
        try:
            async def latest(model_name: str) -> str:
                row = await conn.fetchrow(
                    "SELECT metadata ->> 'artifact_path' AS path FROM model_registry WHERE model_name=$1 ORDER BY created_at DESC LIMIT 1",
                    model_name,
                )
                if not row or not row["path"]:
                    raise RuntimeError(f"No artifact registered for {model_name}")
                return row["path"]

            self._regime_artifact = joblib.load(Path(await latest("regime_classifier")))
            self._forecast_artifact = joblib.load(Path(await latest("forecast_model")))
            self._anomaly_artifact = joblib.load(Path(await latest("anomaly_detector")))
        finally:
            await conn.close()

    async def predict_current_regime(self) -> Dict[str, object]:
        if self._regime_artifact is None:
            await self._load_artifacts()
        if self._regime_artifact is None or not self._dsn or np is None:
            return {"regime": "Calm", "probabilities": {}, "adaptive_weights": {}, "confidence": 0.5, "model_version": "stub"}
        features = await self._latest_regime_features()
        if features is None:
            return {"regime": "Calm", "probabilities": {}, "adaptive_weights": {}, "confidence": 0.5, "model_version": self._regime_artifact.created_at}
        X = self._regime_artifact.scaler.transform([features])
        probs = self._regime_artifact.model.predict_proba(X)[0]
        winner = int(np.argmax(probs))
        return {
            "regime": self._regime_artifact.regime_labels[winner],
            "probabilities": {self._regime_artifact.regime_labels[i]: float(prob) for i, prob in enumerate(probs)},
            "adaptive_weights": {},
            "confidence": float(probs[winner]),
            "model_version": self._regime_artifact.created_at,
        }

    async def forecast_change(self, horizon_hours: int) -> Dict[str, object]:
        if self._forecast_artifact is None:
            await self._load_artifacts()
        artifact = self._forecast_artifact
        # placeholder inference until feature extraction wired
        features = await self._latest_forecast_features()
        if artifact and features is not None and np is not None:
            delta = float(artifact.regression_model.predict(np.array([features]))[0])
            prob = float(artifact.classification_model.predict_proba(np.array([features]))[0][1])
            return {
                "delta_geri_prediction": delta,
                "threshold_exceedance_probability": prob,
                "confidence_interval": {"lower": delta - 1.0, "upper": delta + 1.0},
                "top_drivers": [],
                "model_version": artifact.created_at,
            }
        return {
            "delta_geri_prediction": 0.0,
            "threshold_exceedance_probability": 0.0,
            "confidence_interval": {"lower": -1.0, "upper": 1.0},
            "top_drivers": [],
            "model_version": artifact.created_at if artifact else "stub",
        }

    async def detect_anomalies(self, window_hours: int) -> Dict[str, object]:
        if self._anomaly_artifact is None:
            await self._load_artifacts()
        if self._anomaly_artifact is None or not self._dsn or np is None:
            return {"anomalies": [], "window_hours": window_hours, "total_detected": 0}
        feature = await self._latest_anomaly_features()
        if feature is None:
            return {"anomalies": [], "window_hours": window_hours, "total_detected": 0}
        score = float(self._anomaly_artifact.model.decision_function([feature])[0])
        return {
            "anomalies": [{"score": score, "feature": feature}],
            "window_hours": window_hours,
            "total_detected": 1 if score < 0 else 0,
        }

    async def _latest_regime_features(self) -> Optional[NDArray]:
        if not self._dsn or asyncpg is None or np is None:
            return None
        conn = await asyncpg.connect(self._dsn)
        try:
            row = await conn.fetchrow(
                "SELECT inputs FROM computed_indices ORDER BY ts_utc DESC LIMIT 1"
            )
        finally:
            await conn.close()
        if not row:
            return None
        inputs = row['inputs']
        try:
            return np.array([
                float(inputs.get('VIX', 0)),
                float(inputs.get('YC_10Y_2Y', 0)),
                float(inputs.get('CREDIT_SPREAD', 0)),
                float(inputs.get('FREIGHT_SHIPPING', 0)),
                float(inputs.get('PMI_MANUFACTURING', 0)),
                float(inputs.get('OIL_WTI', 0)),
                float(inputs.get('CPI_INDEX', 0)),
                float(inputs.get('UNEMPLOYMENT', 0)),
            ])
        except (TypeError, ValueError):
            return None

    async def _latest_forecast_features(self) -> Optional[NDArray]:
        if not self._dsn or asyncpg is None or np is None:
            return None
        conn = await asyncpg.connect(self._dsn)
        try:
            rows = await conn.fetch(
                "SELECT value, inputs FROM computed_indices ORDER BY ts_utc DESC LIMIT 200"
            )
        finally:
            await conn.close()
        if len(rows) < 169:
            return None
        rows = list(reversed(rows))
        values = [float(row['value']) for row in rows]
        current_inputs = rows[-1]['inputs']
        try:
            feature = [
                values[-1],
                values[-6],
                values[-24],
                values[-168],
                float(current_inputs.get('VIX', 0)),
                float(current_inputs.get('CREDIT_SPREAD', 0)),
                float(current_inputs.get('OIL_WTI', 0)),
            ]
            return np.array(feature)
        except (TypeError, ValueError):
            return None

    async def _latest_anomaly_features(self) -> Optional[NDArray]:
        if not self._dsn or asyncpg is None or np is None:
            return None
        conn = await asyncpg.connect(self._dsn)
        try:
            row = await conn.fetchrow(
                "SELECT inputs FROM computed_indices ORDER BY ts_utc DESC LIMIT 1"
            )
        finally:
            await conn.close()
        if not row:
            return None
        inputs = row['inputs']
        try:
            return np.array([
                float(inputs.get('VIX', 0)),
                float(inputs.get('CREDIT_SPREAD', 0)),
                float(inputs.get('FREIGHT_SHIPPING', 0)),
                float(inputs.get('PMI_MANUFACTURING', 0)),
                float(inputs.get('OIL_WTI', 0)),
                float(inputs.get('CPI_INDEX', 0)),
                float(inputs.get('UNEMPLOYMENT', 0)),
            ])
        except (TypeError, ValueError):
            return None
