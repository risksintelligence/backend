-- Initial schema setup for RIS database
-- This file creates all core tables and indexes for the RIS system

-- Core data tables
CREATE TABLE IF NOT EXISTS raw_observations (
    id BIGSERIAL PRIMARY KEY,
    series_id TEXT NOT NULL,
    source TEXT NOT NULL,
    observed_at TIMESTAMPTZ NOT NULL,
    value NUMERIC(18,6) NOT NULL,
    unit TEXT,
    source_url TEXT,
    fetched_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(series_id, observed_at)
);

CREATE INDEX IF NOT EXISTS idx_raw_observations_series_time 
ON raw_observations (series_id, observed_at DESC);

CREATE INDEX IF NOT EXISTS idx_raw_observations_fetched 
ON raw_observations (fetched_at DESC);

-- GERI computed values
CREATE TABLE IF NOT EXISTS computed_indices (
    id BIGSERIAL PRIMARY KEY,
    index_name TEXT NOT NULL DEFAULT 'geri_v1.0',
    ts_utc TIMESTAMPTZ NOT NULL,
    value NUMERIC(8,2) NOT NULL,
    band TEXT NOT NULL,
    drivers JSONB NOT NULL,
    components JSONB NOT NULL,
    inputs JSONB NOT NULL,
    sources JSONB NOT NULL,
    version TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(index_name, ts_utc)
);

CREATE INDEX IF NOT EXISTS idx_computed_indices_time 
ON computed_indices (index_name, ts_utc DESC);

-- Caching infrastructure
CREATE TABLE IF NOT EXISTS cache_metadata (
    cache_key TEXT PRIMARY KEY,
    soft_expiry TIMESTAMPTZ NOT NULL,
    hard_expiry TIMESTAMPTZ NOT NULL,
    last_refresh TIMESTAMPTZ,
    refresh_attempts INT DEFAULT 0,
    status TEXT DEFAULT 'active'
);

CREATE INDEX IF NOT EXISTS idx_cache_metadata_expiry 
ON cache_metadata (soft_expiry, hard_expiry);

-- ML infrastructure
CREATE TABLE IF NOT EXISTS feature_store_snapshots (
    id BIGSERIAL PRIMARY KEY,
    snapshot_time TIMESTAMPTZ NOT NULL,
    features JSONB NOT NULL,
    model_version TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_feature_snapshots_time 
ON feature_store_snapshots (snapshot_time DESC);

CREATE TABLE IF NOT EXISTS model_registry (
    id BIGSERIAL PRIMARY KEY,
    model_name TEXT NOT NULL,
    version TEXT NOT NULL,
    metadata JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(model_name, version)
);

-- Administration
CREATE TABLE IF NOT EXISTS admin_audit_log (
    id BIGSERIAL PRIMARY KEY,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    payload JSONB,
    occurred_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_admin_audit_time 
ON admin_audit_log (occurred_at DESC);

-- Research infrastructure
CREATE TABLE IF NOT EXISTS research_api_requests (
    id BIGSERIAL PRIMARY KEY,
    requested_at TIMESTAMPTZ DEFAULT NOW(),
    endpoint TEXT NOT NULL,
    params JSONB,
    requester TEXT DEFAULT 'anonymous'
);

CREATE INDEX IF NOT EXISTS idx_research_requests_time 
ON research_api_requests (requested_at DESC);

CREATE TABLE IF NOT EXISTS peer_reviews (
    id BIGSERIAL PRIMARY KEY,
    reviewer_name TEXT NOT NULL,
    reviewer_email TEXT,
    decision TEXT NOT NULL,
    comments TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Scenario Studio
CREATE TABLE IF NOT EXISTS scenario_runs (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    shocks JSONB NOT NULL,
    horizon_hours INT NOT NULL,
    baseline_value NUMERIC(6,2) NOT NULL,
    scenario_value NUMERIC(6,2) NOT NULL,
    requester TEXT DEFAULT 'internal'
);

CREATE INDEX IF NOT EXISTS idx_scenario_runs_time 
ON scenario_runs (created_at DESC);

-- Alert system
CREATE TABLE IF NOT EXISTS alert_subscriptions (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    channel TEXT NOT NULL,
    address TEXT NOT NULL,
    conditions JSONB NOT NULL,
    active BOOLEAN DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_alert_subscriptions_active 
ON alert_subscriptions (active, created_at DESC);

-- Create a view for latest GERI values
CREATE OR REPLACE VIEW latest_geri AS
SELECT DISTINCT ON (index_name)
    index_name,
    ts_utc,
    value,
    band,
    components,
    drivers,
    version,
    created_at
FROM computed_indices
ORDER BY index_name, ts_utc DESC;

-- Create a view for data freshness monitoring
CREATE OR REPLACE VIEW data_freshness_status AS
SELECT 
    series_id,
    MAX(observed_at) as last_observation,
    MAX(fetched_at) as last_fetch,
    COUNT(*) as total_observations,
    EXTRACT(EPOCH FROM (NOW() - MAX(observed_at))) / 3600 as hours_since_last_data
FROM raw_observations
GROUP BY series_id
ORDER BY last_observation DESC;
