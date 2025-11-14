"""Train GERI forecast model and persist artifacts."""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Tuple

import asyncpg
import joblib
import numpy as np
import xgboost as xgb

ARTIFACTS_DIR = Path(os.environ.get("ML_ARTIFACTS_DIR", "artifacts"))
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class ForecastArtifact:
    regression_model: xgb.XGBRegressor
    classification_model: xgb.XGBClassifier
    created_at: str


async def load_dataset(conn: asyncpg.Connection) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = await conn.fetch(
        """
        SELECT ts_utc, value, components, inputs
        FROM computed_indices
        ORDER BY ts_utc ASC
        """
    )
    values = [float(row['value']) for row in rows]
    features = []
    labels_reg = []
    labels_clf = []
    for idx in range(168, len(rows) - 24):
        current = rows[idx]
        lag1 = values[idx - 1]
        lag6 = values[idx - 6]
        lag24 = values[idx - 24]
        lag168 = values[idx - 168]
        feature_vector = [lag1, lag6, lag24, lag168]
        inputs = current['inputs']
        feature_vector.extend([
            float(inputs.get('VIX', 0)),
            float(inputs.get('CREDIT_SPREAD', 0)),
            float(inputs.get('OIL_WTI', 0)),
        ])
        features.append(feature_vector)
        future_value = values[idx + 24]
        delta = future_value - values[idx]
        labels_reg.append(delta)
        labels_clf.append(1 if abs(delta) > 5 else 0)
    if not features:
        raise RuntimeError("Not enough data for forecast training")
    return np.array(features), np.array(labels_reg), np.array(labels_clf)


async def main() -> None:
    conn = await asyncpg.connect(os.environ["RIS_POSTGRES_DSN"])
    try:
        X, y_reg, y_clf = await load_dataset(conn)
        reg_model = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1)
        reg_model.fit(X, y_reg)
        clf_model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1)
        clf_model.fit(X, y_clf)
        timestamp = datetime.utcnow().isoformat()
        artifact_path = ARTIFACTS_DIR / f"forecast_model_{timestamp}.joblib"
        joblib.dump(ForecastArtifact(regression_model=reg_model, classification_model=clf_model, created_at=timestamp), artifact_path)
        await conn.execute(
            "INSERT INTO model_registry (model_name, version, metadata, created_at) VALUES ($1, $2, $3, NOW())",
            "forecast_model",
            timestamp,
            {"artifact_path": str(artifact_path)},
        )
        print(f"Saved forecast model to {artifact_path}")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
