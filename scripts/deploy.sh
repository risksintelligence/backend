#!/usr/bin/env bash
set -euo pipefail

# Set default values
RENDER_API_KEY="${RENDER_API_KEY:-rnd_fC2WMrcAALB2FUpwcGWwsw6ubW7e}"
RENDER_BACKEND_SERVICE_ID="${RENDER_BACKEND_SERVICE_ID:-srv-backend-risksx}"

if [[ -z "${RENDER_BACKEND_SERVICE_ID}" || -z "${RENDER_API_KEY}" ]]; then
  echo "RENDER_BACKEND_SERVICE_ID and RENDER_API_KEY must be set"
  echo "RENDER_API_KEY: ${RENDER_API_KEY:+[SET]}${RENDER_API_KEY:-[NOT SET]}"
  echo "RENDER_BACKEND_SERVICE_ID: ${RENDER_BACKEND_SERVICE_ID:+[SET]}${RENDER_BACKEND_SERVICE_ID:-[NOT SET]}"
  exit 1
fi

echo "Triggering Render deployment for backend service: $RENDER_BACKEND_SERVICE_ID"

curl -X POST "https://api.render.com/v1/services/$RENDER_BACKEND_SERVICE_ID/deploys" \
  -H "Authorization: Bearer $RENDER_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{}' \
  -w "HTTP Status: %{http_code}\n" \
  -s
