# RRIO Production Deployment Guide

## 🚀 Quick Deploy Checklist

### Required Environment Variables
```bash
# Database
DATABASE_URL="postgresql://user:pass@host:5432/rrio_prod"  # Replace SQLite

# ML Models  
MODEL_DIR="/app/models"  # Production model directory
MODEL_FRESHNESS_HOURS="48"  # Freshness policy (default: 48h)

# API Security
JWT_SECRET_KEY="your-256-bit-secret-key"
API_RATE_LIMIT="1000/hour"

# External APIs (Optional)
FRED_API_KEY="your-fred-api-key"
ALPHA_VANTAGE_API_KEY="your-alpha-vantage-key"

# Monitoring
SENTRY_DSN="your-sentry-dsn"  # Error tracking
REDIS_URL="redis://redis:6379"  # Caching (optional)
OTEL_EXPORTER_OTLP_ENDPOINT="https://otel-collector:4317"  # OTLP traces/metrics
OTEL_SERVICE_NAME="rrio-api"
OTEL_EXPORTER_OTLP_HEADERS="Authorization=Bearer your-token"  # if required

# Runtime
RENDER_SERVICE_TYPE="web_service"  # For Render.com
ENVIRONMENT="production"
```

### Pre-deployment Steps

1. **Database Migration**
```bash
pip install alembic
alembic upgrade head
```

2. **Model Artifacts**
```bash
# Copy versioned models to production MODEL_DIR
cp models_v20251123_183119/* /app/models/
# Verify checksums against manifest.json
```

3. **Verify Health**
```bash
curl https://your-domain.com/health
# Should return: {"status": "healthy", "models": {...}}
```

4. **Observability/Telemetry Dependencies**
```bash
pip install prometheus-client sentry-sdk structlog opentelemetry-sdk \
  opentelemetry-instrumentation-fastapi opentelemetry-instrumentation-requests \
  opentelemetry-exporter-otlp great_expectations
```

## 📊 Model Versioning

Current production models:
- **Version**: 20251123_183119
- **Training Data**: 77 years (1948-2025), 4,693 observations
- **Performance**: Regime 100% accuracy, Forecast MSE 1.53
- **Freshness**: All models < 1 hour old

## 🔍 Health Monitoring

Key endpoints to monitor:
- `/health` - Overall system health
- `/api/v1/ai/models` - ML model status and freshness
- `/api/v1/transparency/data-freshness` - Data pipeline health
- `/metrics` - Prometheus metrics (request latency/counts, drift/compliance counters)
- OTLP traces: ensure spans arrive with service name `rrio-api`

## 🎯 Contract Testing

Production validation:
```bash
pytest tests/test_contracts.py -v
# Should pass: 5/5 tests ✅
```

## 📡 Monitoring & Alerts
- Prometheus scrape: add job for `/metrics` on the API port; alert on:
  - `rrio_governance_drift_alerts_total` increases by drift_type/risk_level
  - `rrio_governance_compliance_failures_total` increases
  - High `rrio_request_latency_seconds` / elevated 5xx rates
- Sentry: set `SENTRY_DSN`, `SENTRY_ENVIRONMENT`, tune `SENTRY_TRACES_SAMPLE_RATE` (e.g., 0.1–0.2).
- OTLP: configure collector endpoint + headers; verify FastAPI instrumentation spans in your APM.

## 🔁 Daily Governance Snapshot
Optional daily summary:
```bash
python scripts/daily_governance_report.py > /tmp/governance_report.json
```
Contains overall compliance score, model count, alert count, timestamp for reporting/alerting.

Generated: 2025-11-23 18:31:19 UTC
