# RiskX Backend

FastAPI-based backend service for the RiskX Risk Intelligence Observatory.

## Features

- FastAPI web framework with async support
- PostgreSQL database with SQLAlchemy ORM
- Redis caching layer
- ML model serving (SHAP explainability)
- Real-time data processing from multiple sources
- WebSocket support for live updates
- Comprehensive monitoring and metrics

## Environment Variables

Required environment variables:

```bash
DATABASE_URL=postgresql://user:password@host:port/database
REDIS_URL=redis://host:port
FRED_API_KEY=your_fred_api_key
CENSUS_API_KEY=your_census_api_key
DEBUG=false
LOG_LEVEL=info
```

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set up database
python scripts/setup/init_db.py

# Start development server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

## Deployment

Deploy to Render using the provided render.yaml configuration:

```bash
# Connect your GitHub repository to Render
# Render will automatically deploy when you push to main branch
```

## API Documentation

Once deployed, API documentation is available at:
- Swagger UI: `https://your-backend-url.onrender.com/docs`
- ReDoc: `https://your-backend-url.onrender.com/redoc`

## Health Check

Health check endpoint: `https://your-backend-url.onrender.com/api/v1/health`