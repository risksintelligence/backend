#!/bin/bash

# RiskX Backend Production Startup Script
set -e

echo "Starting RiskX Backend..."

# Set Python path
export PYTHONPATH="${PYTHONPATH}:/opt/render/project/src"

# Set production environment
export ENVIRONMENT="production"
export DEBUG="false"

# Log startup information
echo "Environment: $ENVIRONMENT"
echo "Debug mode: $DEBUG"
echo "Python path: $PYTHONPATH"

# Validate critical environment variables (warn if missing, don't exit)
echo "Checking environment variables..."
if [[ -z "$DATABASE_URL" ]]; then
    echo "WARNING: DATABASE_URL not set - will use file cache"
fi

if [[ -z "$REDIS_URL" ]]; then
    echo "WARNING: REDIS_URL not set - will use file cache"
fi

# Start the FastAPI application
echo "Starting FastAPI server on port ${PORT:-8000}..."
exec uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --timeout-keep-alive 30