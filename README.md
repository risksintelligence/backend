# RIS Backend (FastAPI)

This directory hosts the RIS Engine API, background jobs, and ML training scripts. Follow the steps below to run the backend locally.

## Prerequisites

- Python 3.11+
- PostgreSQL (local or Docker)
- Redis

## Local Setup

1. **Start Postgres & Redis (Docker example)**
   ```bash
   docker run --name ris-postgres -p 5432:5432 -e POSTGRES_PASSWORD=risdev -e POSTGRES_DB=ris_dev -d postgres:15
   docker run --name ris-redis -p 6379:6379 -d redis:7
   export RIS_POSTGRES_DSN=postgresql://postgres:risdev@localhost:5432/ris_dev
   export RIS_REDIS_URL=redis://localhost:6379
   ```

2. **Install dependencies**
   ```bash
   cd backend
   export PYTHONPATH=.
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Bootstrap the database**
   ```bash
   psql "$RIS_POSTGRES_DSN" -f database/schema.sql
   ./scripts/backfill-data.sh
   ./scripts/train-models.sh
   ```

4. **Run the API server**
   ```bash
   uvicorn src.api.app:app --host 0.0.0.0 --port 8000
   ```

## Useful Scripts

- `scripts/backfill-data.sh` – fetch 10+ years of historical data via FRED.
- `scripts/train-models.sh` – train regime/forecast/anomaly models and register them.
- `scripts/ml/run-model-monitoring.sh` – run the drift monitoring job manually.
- `scripts/backup/run-backups.sh full` – snapshot Postgres + Redis to local files/S3.

## Tests

```bash
cd backend
export PYTHONPATH=.
pytest
```

Ensure `RIS_POSTGRES_DSN` and `RIS_REDIS_URL` point to reachable services before running tests or the API.
