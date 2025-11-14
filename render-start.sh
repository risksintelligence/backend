#!/bin/bash
set -e

echo "=== RIS Backend Start Script ==="
echo "Starting FastAPI application..."

# Set environment
export PYTHONPATH=/opt/render/project/src

# Start the application
exec gunicorn -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:10000 src.api.app:app