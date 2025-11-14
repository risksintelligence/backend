"""Train anomaly detector using IsolationForest."""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import asyncpg
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

ARTIFACTS_DIR = Path(os.environ.get("ML_ARTIFACTS_DIR", "artifacts"))
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class AnomalyArtifact:
    model: IsolationForest
    created_at: str


async def load_features(conn: asyncpg.Connection) -> np.ndarray:
    rows = await conn.fetch(
        "SELECT inputs FROM computed_indices ORDER BY ts_utc DESC LIMIT 20000"
    )
    matrix = []
    for row in rows:
        inputs = row['inputs']
        try:
            matrix.append([
                float(inputs.get('VIX', 0)),
                float(inputs.get('CREDIT_SPREAD', 0)),
                float(inputs.get('FREIGHT_SHIPPING', 0)),
                float(inputs.get('PMI_MANUFACTURING', 0)),
                float(inputs.get('OIL_WTI', 0)),
                float(inputs.get('CPI_INDEX', 0)),
                float(inputs.get('UNEMPLOYMENT', 0)),
            ])
        except (TypeError, ValueError):
            continue
    if not matrix:
        raise RuntimeError("Not enough inputs for anomaly training")
    return np.array(matrix)


async def main() -> None:
    conn = await asyncpg.connect(os.environ["RIS_POSTGRES_DSN"])
    try:
        X = await load_features(conn)
        model = IsolationForest(random_state=42, contamination=0.05)
        model.fit(X)
        timestamp = datetime.utcnow().isoformat()
        artifact_path = ARTIFACTS_DIR / f"anomaly_detector_{timestamp}.joblib"
        joblib.dump(AnomalyArtifact(model=model, created_at=timestamp), artifact_path)
        await conn.execute(
            "INSERT INTO model_registry (model_name, version, metadata, created_at) VALUES ($1, $2, $3, NOW())",
            "anomaly_detector",
            timestamp,
            {"artifact_path": str(artifact_path)},
        )
        print(f"Saved anomaly detector to {artifact_path}")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
