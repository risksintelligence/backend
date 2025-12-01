# RiskX Observatory - Backend

FastAPI service providing comprehensive supply chain risk intelligence, real-time monitoring, and predictive analytics for global trade disruptions.

## Quick Start

### Development
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Production
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## Key Features

- **Real-time Risk Analytics**: GERI scoring, anomaly detection, market pulse monitoring
- **ML-Powered Intelligence**: Supply chain risk prediction, market trend analysis
- **Network Analysis**: Cascade simulation, vulnerability assessment, resilience metrics
- **Comprehensive Monitoring**: Health checks, error logging, performance analytics
- **Production Ready**: Redis caching, background workers, rate limiting

## Core Endpoints

### Analytics
- `GET /api/v1/analytics/geri` - Global Economic Risk Index
- `GET /api/v1/analytics/components` - Risk component analysis
- `GET /api/v1/anomalies/latest` - Anomaly detection results

### Real-time Monitoring
- `GET /api/v1/realtime/status` - System status dashboard
- `GET /api/v1/realtime/market-pulse` - Live market indicators

### ML Intelligence
- `POST /api/v1/ml/supply-chain/predict` - Risk prediction
- `POST /api/v1/ml/anomaly-detection` - ML anomaly detection

### Health & Monitoring
- `GET /api/v1/health/comprehensive` - System health check
- `GET /api/v1/errors/summary` - Error monitoring dashboard

## Configuration

### Repository Information
- **Repository**: https://github.com/risksintelligence/backend
- **Production URL**: https://backend-1-s84g.onrender.com
- **Frontend Repository**: https://github.com/risksintelligence/frontend

### Required Environment Variables
```bash
# Core Configuration
RIS_ENV=production
RIS_POSTGRES_DSN=postgresql://...
RIS_REDIS_URL=rediss://...

# Security
RIS_JWT_SECRET=...
RIS_REVIEWER_API_KEY=...
RIS_ALLOWED_ORIGINS=https://frontend-1-tzlw.onrender.com,...

# API Keys (18 external data sources)
RIS_FRED_API_KEY=...
RIS_EIA_API_KEY=...
RIS_ALPHA_VANTAGE_API_KEY=...
RIS_ACLED_EMAIL=your-email@domain.com
RIS_ACLED_PASSWORD=your-acled-key
RIS_MARINETRAFFIC_API_KEY=your-marinetraffic-key
# See .env.example for complete list
```

## Documentation

- **📖 Full API Documentation**: [API_DOCUMENTATION.md](./API_DOCUMENTATION.md)
- **🚀 Deployment Guide**: [DEPLOYMENT.md](./DEPLOYMENT.md)
- **📊 Interactive Docs**: http://localhost:8000/docs (when running)

## Current Status

✅ **Production Ready** - All core systems operational  
✅ **Real Data Integration** - 5 external APIs connected  
✅ **ML Models** - Trained and operational  
✅ **Monitoring** - Comprehensive health & error tracking  
✅ **Caching** - Redis-based performance optimization

## Health Dashboard

Check system status: `GET /api/v1/health/comprehensive`

**Last Updated**: November 2025
