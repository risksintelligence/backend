-- Add ML model registry tables and cron execution tracking

CREATE TABLE IF NOT EXISTS ml_models (
    model_id TEXT PRIMARY KEY,
    model_type TEXT NOT NULL,
    version TEXT NOT NULL,
    model_data JSONB NOT NULL,
    is_active BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ml_drift_metrics (
    id BIGSERIAL PRIMARY KEY,
    model_id TEXT REFERENCES ml_models(model_id) ON DELETE CASCADE,
    timestamp TIMESTAMPTZ NOT NULL,
    drift_data JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ml_drift_model_time
ON ml_drift_metrics (model_id, timestamp DESC);

CREATE TABLE IF NOT EXISTS cron_executions (
    execution_id TEXT PRIMARY KEY,
    job_name TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ,
    status TEXT DEFAULT 'running',
    success BOOLEAN,
    duration_seconds NUMERIC(10,2),
    exit_code INT,
    output TEXT,
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cron_executions_job_time
ON cron_executions (job_name, started_at DESC);
