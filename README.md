# RRIO Backend

FastAPI service exposing GRII analytics and Resilience Activation Score endpoints.

## Development
```
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Endpoints
- `/health`
- `/api/v1/analytics/geri`
- `/api/v1/impact/ras`
