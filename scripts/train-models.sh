#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${RIS_POSTGRES_DSN:-}" ]]; then
  echo "RIS_POSTGRES_DSN not set"
  exit 1
fi

python backend/src/ml/training/regime_training.py
python backend/src/ml/training/forecast_training.py
python backend/src/ml/training/anomaly_training.py
