#!/usr/bin/env bash
set -euo pipefail

BASE_URL=${BASE_URL:-http://localhost:8000}

function check_endpoint() {
  local endpoint=$1
  echo "Checking $endpoint"
  http_code=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL$endpoint")
  if [[ "$http_code" != "200" ]]; then
    echo "Health check failed for $endpoint ($http_code)"
    exit 1
  fi
}

check_endpoint "/healthz"
check_endpoint "/api/v1/transparency"
check_endpoint "/api/v1/analytics/geri"

echo "All backend health checks passed"
