# RiskX Observatory - Backend API Documentation

FastAPI service providing comprehensive supply chain risk intelligence, real-time monitoring, and predictive analytics for global trade disruptions.

## Table of Contents
- [Quick Start](#quick-start)
- [Authentication & Configuration](#authentication--configuration)
- [Core Analytics](#core-analytics)
- [Real-time Monitoring](#real-time-monitoring)
- [ML Intelligence](#ml-intelligence)
- [Supply Chain Network](#supply-chain-network)
- [Health & Error Monitoring](#health--error-monitoring)
- [Cache Management](#cache-management)
- [Administrative Endpoints](#administrative-endpoints)

## Quick Start

### Development Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Start development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Access API documentation
open http://localhost:8000/docs
```

### Production Deployment
```bash
# Start with production settings
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## Authentication & Configuration

### Environment Variables

#### Required Configuration
```bash
# Redis Cache (Production)
RIS_REDIS_URL=rediss://user:password@host:port

# Database
RIS_DATABASE_URL=sqlite:///./supply_chain.db  # Development
RIS_DATABASE_URL=postgresql://user:pass@host:port/db  # Production
```

#### API Keys (Optional - Enable Real Data)
```bash
# ACLED (Geopolitical Events)
RIS_ACLED_EMAIL=your-email@domain.com
RIS_ACLED_PASSWORD=your-acled-key

# MarineTraffic (Port & Vessel Data)
RIS_MARINETRAFFIC_API_KEY=your-marinetraffic-key

# Additional providers automatically use free tiers
```

#### ML & Performance Settings
```bash
# Model Configuration
RIS_ML_MODEL_MAX_AGE_HOURS=72  # Model staleness tolerance

# Performance Tuning
RIS_CACHE_DEFAULT_TTL=3600     # Default cache TTL (seconds)
RIS_RATE_LIMIT_REQUESTS=100    # Requests per minute
```

## Core Analytics

### Global Economic Risk Index (GERI)
Real-time economic risk assessment based on multiple indicators.

#### `GET /api/v1/analytics/geri`
Compute current GERI score with breakdown by indicators.

**Response Example:**
```json
{
  "geri": {
    "score": 56.59,
    "risk_level": "moderate",
    "confidence": "low",
    "indicators": {
      "VIX": {"value": 20.0, "weight": 0.25, "contribution": 14.2},
      "YIELD_CURVE": {"value": 0.52, "weight": 0.20, "contribution": 17.0},
      "UNEMPLOYMENT": {"value": 4.3, "weight": 0.15, "contribution": 21.5}
    },
    "missing_data": ["FREIGHT_DIESEL", "BALTIC_DRY", "PMI"]
  }
}
```

#### `GET /api/v1/analytics/components`
Get individual risk component values and trends.

**Query Parameters:**
- `period` (optional): Time period for trend analysis (default: 30 days)

### Impact Analysis

#### `GET /api/v1/impact/ras`
Calculate Resilience Activation Score based on current conditions.

#### `GET /api/v1/impact/partners`
Analyze impact on key trade partners and relationships.

## Real-time Monitoring

### System Status

#### `GET /api/v1/realtime/status`
Comprehensive real-time system status with service health.

**Response Example:**
```json
{
  "system_status": {
    "overall_health": "healthy",
    "services_monitored": 9,
    "healthy_services": 7,
    "degraded_services": 2,
    "refresh_jobs": {
      "total": 9,
      "high_priority": 2,
      "medium_priority": 4,
      "low_priority": 3
    }
  }
}
```

#### `GET /api/v1/realtime/market-pulse`
Real-time market indicators and supply chain signals.

**Response Example:**
```json
{
  "market_pulse": {
    "vix_level": 20.0,
    "oil_price_usd": 60.94,
    "baltic_dry_index": null,
    "currency_volatility": "moderate",
    "supply_chain_alerts": [
      {
        "type": "port_congestion",
        "location": "Los Angeles",
        "severity": "medium"
      }
    ]
  }
}
```

### Anomaly Detection

#### `GET /api/v1/anomalies/latest`
Latest detected anomalies in supply chain metrics.

**Query Parameters:**
- `severity` (optional): Filter by severity (low, medium, high, critical)
- `limit` (optional): Maximum number of anomalies (default: 50)

## ML Intelligence

### Supply Chain Risk Prediction

#### `POST /api/v1/ml/supply-chain/predict`
Predict supply chain disruption risk for specific routes.

**Request Body:**
```json
{
  "route_data": {
    "origin_country": "China",
    "destination_country": "United States", 
    "transportation_mode": "sea",
    "cargo_type": "electronics"
  },
  "economic_data": {
    "gdp_growth": 2.1,
    "inflation_rate": 3.2,
    "currency_volatility": 0.15
  },
  "prediction_horizon": 30
}
```

### Market Intelligence

#### `POST /api/v1/ml/market-intelligence/trends`
Predict market trends and intelligence patterns.

#### `POST /api/v1/ml/anomaly-detection`
Detect anomalies in real-time data streams.

**Request Body:**
```json
{
  "market_data": {
    "commodity_prices": {"oil": 60.5, "gold": 1950.0},
    "currency_rates": {"EUR_USD": 1.08, "GBP_USD": 1.27},
    "trade_volumes": {"exports": 150000, "imports": 180000}
  },
  "sensitivity": 0.7
}
```

### Model Management

#### `GET /api/v1/ml/models/status`
Check status and performance of all ML models.

#### `POST /api/v1/ml/models/retrain`
Trigger model retraining with latest data.

## Supply Chain Network

### Network Analysis

#### `GET /api/v1/supply-chain/network/overview`
High-level supply chain network topology and health.

**Response Example:**
```json
{
  "network_overview": {
    "total_nodes": 156,
    "total_connections": 324,
    "critical_paths": 12,
    "vulnerability_score": 0.34,
    "resilience_index": 0.78,
    "major_hubs": ["Singapore", "Rotterdam", "Los Angeles"]
  }
}
```

#### `GET /api/v1/supply-chain/cascade/snapshot`
Current cascade risk snapshot across the network.

#### `POST /api/v1/supply-chain/cascade/simulate`
Simulate cascade effects from disruption scenarios.

**Request Body:**
```json
{
  "disruption_scenario": {
    "type": "port_closure",
    "location": "Shanghai",
    "duration_days": 7,
    "severity": 0.8
  },
  "simulation_horizon": 30
}
```

### Route Analysis

#### `GET /api/v1/supply-chain/routes/critical`
Identify critical supply chain routes and vulnerabilities.

#### `GET /api/v1/supply-chain/resilience/metrics`
Calculate resilience metrics for supply chain segments.

## Health & Error Monitoring

### System Health

#### `GET /api/v1/health/comprehensive`
Comprehensive system health check covering all components.

**Response Example:**
```json
{
  "health_check": {
    "overall_status": "healthy",
    "summary": {
      "total_services": 9,
      "healthy": 7,
      "degraded": 1,
      "unhealthy": 1,
      "health_percentage": 77.8
    },
    "services": {
      "database": {"status": "healthy", "response_time_ms": 2.1},
      "redis_cache": {"status": "healthy", "response_time_ms": 15.3},
      "api_comtrade": {"status": "healthy", "response_time_ms": 450.2}
    }
  }
}
```

#### `GET /api/v1/health/external-apis`
Detailed health status for external API integrations.

#### `GET /api/v1/health/production-readiness`
Production deployment readiness assessment.

#### `GET /api/v1/health/quick`
Fast health check for basic service availability.

### Error Monitoring

#### `GET /api/v1/errors/summary`
System-wide error summary with health scoring.

**Response Example:**
```json
{
  "error_summary": {
    "system_health": {
      "average_health_score": 85.5,
      "total_errors": 12,
      "error_rate_per_hour": 0.5,
      "problematic_services": ["acled"]
    },
    "alerts": [
      {
        "level": "warning",
        "message": "ACLED service health degraded",
        "action": "Check API credentials"
      }
    ]
  }
}
```

#### `GET /api/v1/errors/analytics`
Detailed error analytics with filtering options.

**Query Parameters:**
- `service` (optional): Filter by service name
- `hours` (optional): Analysis period (default: 24, max: 168)

#### `GET /api/v1/errors/services/{service}/health-score`
Health score (0-100) for specific service.

#### `GET /api/v1/errors/recent`
Recent error records with filtering.

**Query Parameters:**
- `service`, `category`, `severity` (optional): Filtering options
- `limit` (optional): Maximum records to return

## Cache Management

### Cache Operations

#### `GET /api/v1/cache/status`
Comprehensive cache system status and metrics.

#### `GET /api/v1/cache/stale-keys`
List cache keys that need refresh.

**Query Parameters:**
- `data_type` (optional): Filter by data type
- `limit` (optional): Maximum keys to return

#### `POST /api/v1/cache/invalidate/{data_type}`
Invalidate cache for specific data type or identifier.

**Path Parameters:**
- `data_type`: Type to invalidate or "all" for everything

**Query Parameters:**
- `identifier` (optional): Specific identifier or "*" for all (default: "*")

#### `POST /api/v1/cache/optimize`
Trigger cache optimization and cleanup.

### Cache Analytics

#### `GET /api/v1/cache/analytics`
Cache performance analytics and insights.

**Query Parameters:**
- `hours` (optional): Analysis period (default: 24, max: 168)

## Administrative Endpoints

### System Information

#### `GET /health`
Basic health check endpoint.

#### `GET /health/detailed`
Detailed system health with component breakdown.

#### `GET /docs`
Interactive API documentation (Swagger UI).

#### `GET /redoc`
Alternative API documentation (ReDoc).

### Monitoring Integration

#### `GET /metrics` (if enabled)
Prometheus metrics endpoint for monitoring integration.

## Error Responses

All endpoints follow consistent error response format:

```json
{
  "detail": "Error message describing what went wrong",
  "error_code": "SPECIFIC_ERROR_CODE",
  "timestamp": "2025-11-26T02:30:00Z"
}
```

### Common HTTP Status Codes
- `200` - Success
- `400` - Bad Request (invalid parameters)
- `401` - Unauthorized (missing/invalid auth)
- `404` - Not Found (endpoint or resource)
- `429` - Rate Limited
- `500` - Internal Server Error
- `503` - Service Unavailable

## Rate Limiting

### Default Limits
- **Analytics endpoints**: 100 requests/minute
- **ML endpoints**: 50 requests/minute  
- **System endpoints**: 200 requests/minute

### Headers
Rate limit information is provided in response headers:
- `X-RateLimit-Limit`: Requests allowed per window
- `X-RateLimit-Remaining`: Requests remaining in window
- `X-RateLimit-Reset`: Seconds until window resets

## Data Sources

### External API Integrations
- **UN Comtrade**: Trade statistics (free tier)
- **World Bank**: Economic indicators (free tier)
- **SEC Edgar**: Corporate data (free tier)
- **ACLED**: Geopolitical events (requires registration)
- **MarineTraffic**: Vessel/port data (requires API key)

### Real Data vs Mock Data
The system automatically falls back to mock data when:
- External APIs are unavailable
- API keys are not configured
- Rate limits are exceeded
- Network connectivity issues occur

## Performance Considerations

### Caching Strategy
- **L1 Cache (Redis)**: Real-time data, TTL 5-60 minutes
- **L2 Cache (Database)**: Historical data, persistent storage
- **Stale-while-revalidate**: Serve cached data while refreshing

### Background Processing
- Automated data refresh every 1-15 minutes
- Priority-based refresh scheduling
- Graceful degradation under load

### Scalability
- Horizontal scaling supported
- Redis cluster ready
- Database connection pooling
- Async request processing

## Development Tips

### Testing Endpoints
```bash
# Test basic health
curl http://localhost:8000/health

# Test GERI calculation
curl http://localhost:8000/api/v1/analytics/geri

# Test with parameters
curl "http://localhost:8000/api/v1/errors/analytics?service=acled&hours=24"
```

### Common Issues
1. **Redis Connection**: Check `RIS_REDIS_URL` configuration
2. **External APIs**: Monitor `/api/v1/health/external-apis` for status
3. **Rate Limits**: Check response headers for limit information
4. **Cache Performance**: Use `/api/v1/cache/status` for diagnostics

### Debugging
- Enable debug logging: Set `LOG_LEVEL=DEBUG`
- Monitor errors: Check `/api/v1/errors/recent`
- Health dashboard: Use `/api/v1/health/comprehensive`
- Cache diagnostics: Use `/api/v1/cache/analytics`

---

**Last Updated**: November 2025  
**API Version**: v1  
**Documentation Version**: 2.0