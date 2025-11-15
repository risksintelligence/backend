#!/usr/bin/env bash
set -euo pipefail

# Backend service configuration
RENDER_BACKEND_SERVICE_ID="srv-d4bb6vkhg0os73eqpngg"
DEPLOY_HOOK_KEY="FgzIcf6BY3I"

echo "Triggering backend deployment..."
curl -X POST "https://api.render.com/deploy/${RENDER_BACKEND_SERVICE_ID}?key=${DEPLOY_HOOK_KEY}" \
  -H 'Content-Type: application/json'

echo ""
echo "Backend deployment triggered successfully!"
echo "Check status at: https://dashboard.render.com/web/${RENDER_BACKEND_SERVICE_ID}"
