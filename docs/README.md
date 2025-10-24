# RiskX API Documentation

## Overview

The RiskX Risk Intelligence Platform provides a comprehensive REST API for real-time risk assessment across multiple domains including economic, geopolitical, environmental, and supply chain risks.

## Quick Start

### Base URL
- **Production**: `https://api.riskx.ai`

### Interactive Documentation
- **Swagger UI**: `/docs`
- **ReDoc**: `/redoc`
- **OpenAPI Schema**: `/openapi.json`

### Example Request
```bash
curl -X GET "https://api.riskx.ai/api/v1/risk/overview" \
     -H "accept: application/json"
```

## Core Endpoints

### Health & Status
- `GET /` - Root endpoint with platform information
- `GET /api/v1/health` - Comprehensive health check
- `GET /api/v1/status` - API status and features
- `GET /api/v1/test` - Test endpoint with sample data

### Risk Assessment
- `GET /api/v1/risk/overview` - Comprehensive risk overview
- `GET /api/v1/risk/factors` - Risk factors breakdown
- `GET /api/v1/risk/score/realtime` - Real-time risk score

### Machine Learning Predictions
- `GET /api/v1/risk/predictions/recession` - Recession probability
- `GET /api/v1/risk/predictions/supply-chain` - Supply chain risk
- `GET /api/v1/risk/predictions/market-volatility` - Market volatility
- `GET /api/v1/risk/predictions/geopolitical` - Geopolitical risk
- `GET /api/v1/risk/models/status` - ML model status
- `POST /api/v1/risk/models/train` - Trigger model training

### Economic Data
- `GET /api/v1/economic/indicators` - Economic indicators
- `GET /api/v1/economic/market` - Market data

### External Data Sources
- `GET /api/v1/external/fred/indicators` - FRED economic data
- `GET /api/v1/external/health` - External API health checks

### Network Analysis
- `GET /api/v1/network/overview` - Network overview
- `GET /api/v1/network/nodes` - Node analysis
- `GET /api/v1/network/centrality` - Centrality measures
- `GET /api/v1/network/vulnerabilities` - Vulnerability assessment
- `GET /api/v1/network/propagation` - Risk propagation analysis
- `GET /api/v1/network/critical-paths` - Critical path analysis
- `POST /api/v1/network/simulation` - Shock simulation

### Cache Management
- `GET /api/v1/cache/metrics` - Cache performance metrics
- `GET /api/v1/cache/status` - Cache system status
- `GET /api/v1/cache/demo` - Cache demonstration

### Database Operations
- `GET /api/v1/database/schema` - Database schema information
- `GET /api/v1/database/data/summary` - Data summary

### WebSocket Streaming
- `GET /api/v1/ws/status` - WebSocket status
- `POST /api/v1/ws/broadcast` - Broadcast message
- `WS /ws/risk-alerts` - Real-time risk alerts
- `WS /ws/market-data` - Real-time market data

## Response Format

All API responses follow a consistent format:

```json
{
  "status": "success|error|loading|service_initializing",
  "data": { ... },
  "source": "cache|real_time|computed",
  "timestamp": "2024-01-01T12:00:00Z",
  "message": "Optional status message",
  "error": "Error details if status is error"
}
```

## Data Sources

The API integrates with multiple authoritative data sources:

- **FRED**: Federal Reserve Economic Data
- **BEA**: Bureau of Economic Analysis  
- **BLS**: Bureau of Labor Statistics
- **Census**: U.S. Census Bureau
- **CISA**: Cybersecurity & Infrastructure Security Agency
- **NOAA**: National Oceanic and Atmospheric Administration
- **USGS**: U.S. Geological Survey

## Machine Learning Models

The platform uses four core ML models:

1. **Recession Predictor**: Analyzes economic indicators to predict recession probability
2. **Supply Chain Risk Model**: Assesses supply chain vulnerabilities and disruptions
3. **Market Volatility Model**: Predicts market volatility using financial indicators
4. **Geopolitical Risk Model**: Evaluates geopolitical tensions and their risk impact

## Caching Strategy

Three-tier intelligent caching system:

1. **Redis** (L1): Sub-second access for hot data (TTL: 5 minutes)
2. **PostgreSQL** (L2): Fast database access for warm data (TTL: 1 hour)
3. **File System** (L3): Long-term storage for cold data (TTL: 24 hours)

## Rate Limiting

Standard rate limiting is applied:
- **Default**: 100 requests per minute per IP
- **Burst**: Up to 200 requests per minute for short periods
- **Daily Limit**: 10,000 requests per day per IP

## Error Handling

The API uses standard HTTP status codes:

- `200 OK`: Successful request
- `400 Bad Request`: Invalid request parameters
- `404 Not Found`: Endpoint not found
- `422 Unprocessable Entity`: Validation error
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: Service temporarily unavailable

## Examples

### Get Risk Overview
```bash
curl -X GET "https://api.riskx.ai/api/v1/risk/overview" \
     -H "accept: application/json"
```

Response:
```json
{
  "status": "success",
  "data": {
    "overall_risk_score": 65.5,
    "risk_level": "Medium-High",
    "confidence": 0.85,
    "factors": {
      "economic": 70.0,
      "geopolitical": 60.0,
      "supply_chain": 65.0,
      "environmental": 45.0
    }
  },
  "source": "real_time",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### Run Risk Simulation
```bash
curl -X POST "https://api.riskx.ai/api/v1/network/simulation" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d '{
       "initial_nodes": ["BANK_A"],
       "shock_magnitude": 0.5,
       "simulation_steps": 20,
       "containment_threshold": 0.1
     }'
```

### Get Real-time Economic Indicators
```bash
curl -X GET "https://api.riskx.ai/api/v1/economic/indicators" \
     -H "accept: application/json"
```

## WebSocket Streaming

Connect to real-time data streams:

```javascript
// Risk alerts stream
const riskSocket = new WebSocket('wss://api.riskx.ai/ws/risk-alerts');
riskSocket.onmessage = (event) => {
  const riskAlert = JSON.parse(event.data);
  console.log('Risk Alert:', riskAlert);
};

// Market data stream  
const marketSocket = new WebSocket('wss://api.riskx.ai/ws/market-data');
marketSocket.onmessage = (event) => {
  const marketData = JSON.parse(event.data);
  console.log('Market Update:', marketData);
};
```

## SDKs and Client Libraries

Official SDKs available:
- **Python**: `pip install riskx-python`
- **JavaScript/Node.js**: `npm install riskx-js`
- **R**: `install.packages("riskx")`

Python SDK example:
```python
from riskx import RiskXClient

client = RiskXClient(base_url="https://api.riskx.ai")

# Get risk overview
risk_data = client.risk.get_overview()
print(f"Overall Risk Score: {risk_data.overall_risk_score}")

# Get recession probability
recession = client.ml.predict_recession()
print(f"Recession Probability: {recession.probability:.2%}")
```

## Support

- **Documentation**: https://docs.riskx.ai
- **GitHub**: https://github.com/riskx-platform
- **Issues**: https://github.com/riskx-platform/api/issues
- **Email**: support@riskx.ai
- **Discord**: https://discord.gg/riskx

## License

This API is provided under the MIT License for research and educational purposes.

## Changelog

### v1.0.0 (2024-01-01)
- Initial release with comprehensive risk assessment
- ML-powered predictions for 4 risk domains
- Real-time data integration from 7 sources
- Three-tier intelligent caching system
- WebSocket streaming for real-time updates
- Complete OpenAPI documentation