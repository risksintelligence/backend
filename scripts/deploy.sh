#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${RENDER_BACKEND_SERVICE_ID:-}" || -z "${RENDER_API_KEY:-}" ]]; then
  echo "RENDER_BACKEND_SERVICE_ID and RENDER_API_KEY must be set"
  exit 1
fi

curl -X POST "https://api.render.com/v1/services/$RENDER_BACKEND_SERVICE_ID/deploys" \
  -H "Authorization: Bearer $RENDER_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{}'
