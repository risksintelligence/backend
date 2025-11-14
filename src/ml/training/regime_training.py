"""Train regime classifier and persist artifact/registry metadata."""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

import asyncpg
import joblib
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

ARTIFACTS_DIR = Path(os.environ.get("ML_ARTIFACTS_DIR", "artifacts"))
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class RegimeArtifact:
    scaler: StandardScaler
    model: GaussianMixture
    created_at: str
    regime_labels: List[str]


async def load_features(conn: asyncpg.Connection) -> np.ndarray:
    rows = await conn.fetch(
        """
        SELECT inputs ->> 'VIX' AS vix,
               inputs ->> 'YC_10Y_2Y' AS slope,
               inputs ->> 'CREDIT_SPREAD' AS spread,
               inputs ->> 'FREIGHT_SHIPPING' AS freight,
               inputs ->> 'PMI_MANUFACTURING' AS pmi,
               inputs ->> 'OIL_WTI' AS oil,
               inputs ->> 'CPI_INDEX' AS cpi,
               inputs ->> 'UNEMPLOYMENT' AS unemp
        FROM computed_indices
        ORDER BY ts_utc DESC
        LIMIT 40000
        """
    )
    matrix = []
    for row in rows:
        try:
            matrix.append([
                float(row['vix']),
                float(row['slope']),
                float(row['spread']),
                float(row['freight']),
                float(row['pmi']),
                float(row['oil']),
                float(row['cpi']),
                float(row['unemp']),
            ])
        except (TypeError, ValueError):
            continue
    if not matrix:
        raise RuntimeError("No features available for regime training")
    return np.array(matrix)


def _assign_labels(gmm: GaussianMixture) -> List[str]:
    labels = []
    for mean in gmm.means_:
        score_fin = mean[0] + mean[2] - mean[1]
        score_supply = mean[5] + mean[3] - mean[4]
        score_infl = mean[6] - mean[7]
        if max(score_fin, score_supply, score_infl) == score_fin:
            labels.append("Financial_Stress")
        elif max(score_fin, score_supply, score_infl) == score_supply:
            labels.append("Supply_Shock")
        elif max(score_fin, score_supply, score_infl) == score_infl:
            labels.append("Inflationary_Stress")
        else:
            labels.append("Calm")
    return labels


async def main() -> None:
    dsn = os.environ["RIS_POSTGRES_DSN"]
    conn = await asyncpg.connect(dsn)
    try:
        feats = await load_features(conn)
        scaler = StandardScaler().fit(feats)
        X_scaled = scaler.transform(feats)
        gmm = GaussianMixture(n_components=4, covariance_type="full", random_state=42)
        gmm.fit(X_scaled)
        labels = _assign_labels(gmm)
        silhouette = silhouette_score(X_scaled, gmm.predict(X_scaled))
        timestamp = datetime.utcnow().isoformat()
        artifact_path = ARTIFACTS_DIR / f"regime_classifier_{timestamp}.joblib"
        joblib.dump(RegimeArtifact(scaler=scaler, model=gmm, created_at=timestamp, regime_labels=labels), artifact_path)
        await conn.execute(
            "INSERT INTO model_registry (model_name, version, metadata, created_at) VALUES ($1, $2, $3, NOW())",
            "regime_classifier",
            timestamp,
            {"artifact_path": str(artifact_path), "silhouette": silhouette},
        )
        print(f"Saved regime classifier to {artifact_path}")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
