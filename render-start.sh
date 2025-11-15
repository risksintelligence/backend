#!/bin/bash
set -e

echo "=== RIS Backend Start Script ==="
echo "Starting FastAPI application..."

# Set PYTHONPATH to current directory (already in backend on Render)
export PYTHONPATH=$(pwd)

# Start the application with correct port from environment
exec gunicorn -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT src.api.app:app