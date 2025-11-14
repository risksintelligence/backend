#!/bin/bash
set -e

echo "Installing dependencies with binary-only mode to avoid compilation errors..."

# Upgrade pip first
python -m pip install --upgrade pip

# Install packages preferring binary distributions
python -m pip install --only-binary=:all: --prefer-binary \
  fastapi==0.104.1 \
  uvicorn[standard]==0.24.0 \
  asyncpg==0.29.0 \
  psycopg2-binary==2.9.9 \
  redis==5.0.1 \
  aioredis==2.0.1 \
  aiohttp==3.9.1 \
  httpx==0.25.2

# Install ML packages one by one with binary preference
echo "Installing numpy..."
python -m pip install --only-binary=numpy numpy==1.24.4

echo "Installing scikit-learn..."
python -m pip install --only-binary=scikit-learn scikit-learn==1.3.2

echo "Installing remaining ML packages..."
python -m pip install --only-binary=:all: \
  xgboost==2.0.2 \
  pandas==2.1.4 \
  joblib==1.3.2 \
  shap==0.43.0

# Install remaining packages
python -m pip install --only-binary=:all: \
  psutil==5.9.6 \
  pydantic==2.5.3 \
  pydantic-settings==2.1.0 \
  prometheus-client==0.19.0 \
  python-dateutil==2.8.2 \
  pytz==2023.3 \
  boto3==1.34.0 \
  openpyxl==3.1.2 \
  pytest==7.4.3 \
  pytest-asyncio==0.21.1 \
  email-validator==2.1.0 \
  python-multipart==0.0.6

echo "All dependencies installed successfully!"