-- Core database schema for RIS Engine

CREATE TABLE raw_observations (
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

CREATE TABLE computed_indices (
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

CREATE TABLE cache_metadata (
    cache_key TEXT PRIMARY KEY,
    soft_expiry TIMESTAMPTZ NOT NULL,
    hard_expiry TIMESTAMPTZ NOT NULL,
    last_refresh TIMESTAMPTZ,
    refresh_attempts INT DEFAULT 0,
    status TEXT DEFAULT 'active'
);

CREATE TABLE feature_store_snapshots (
    id BIGSERIAL PRIMARY KEY,
    snapshot_time TIMESTAMPTZ NOT NULL,
    features JSONB NOT NULL,
    model_version TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE model_registry (
    id BIGSERIAL PRIMARY KEY,
    model_name TEXT NOT NULL,
    version TEXT NOT NULL,
    metadata JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(model_name, version)
);

CREATE TABLE ml_models (
    model_id TEXT PRIMARY KEY,
    model_type TEXT NOT NULL,
    version TEXT NOT NULL,
    model_data JSONB NOT NULL,
    is_active BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE ml_drift_metrics (
    id BIGSERIAL PRIMARY KEY,
    model_id TEXT REFERENCES ml_models(model_id) ON DELETE CASCADE,
    timestamp TIMESTAMPTZ NOT NULL,
    drift_data JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_ml_drift_model_time
ON ml_drift_metrics (model_id, timestamp DESC);

CREATE TABLE cron_executions (
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

CREATE INDEX idx_cron_executions_job_time
ON cron_executions (job_name, started_at DESC);

CREATE TABLE admin_audit_log (
    id BIGSERIAL PRIMARY KEY,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    payload JSONB,
    occurred_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE research_api_requests (
    id BIGSERIAL PRIMARY KEY,
    requested_at TIMESTAMPTZ DEFAULT NOW(),
    endpoint TEXT NOT NULL,
    params JSONB,
    requester TEXT DEFAULT 'anonymous'
);

CREATE TABLE peer_reviews (
    id BIGSERIAL PRIMARY KEY,
    reviewer_name TEXT NOT NULL,
    reviewer_email TEXT,
    decision TEXT NOT NULL,
    comments TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE scenario_runs (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    shocks JSONB NOT NULL,
    horizon_hours INT NOT NULL,
    baseline_value NUMERIC(6,2) NOT NULL,
    scenario_value NUMERIC(6,2) NOT NULL,
    requester TEXT DEFAULT 'internal'
);

CREATE TABLE alert_subscriptions (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    channel TEXT NOT NULL,
    address TEXT NOT NULL,
    conditions JSONB NOT NULL,
    active BOOLEAN DEFAULT TRUE
);

-- Authentication and User Management Tables
CREATE TABLE users (
    id BIGSERIAL PRIMARY KEY,
    username TEXT NOT NULL UNIQUE,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'analyst',
    subscription_tier TEXT NOT NULL DEFAULT 'free',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_login TIMESTAMPTZ
);

CREATE TABLE api_keys (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT REFERENCES users(id) ON DELETE CASCADE,
    key_hash TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    permissions JSONB NOT NULL DEFAULT '[]',
    last_used TIMESTAMPTZ,
    expires_at TIMESTAMPTZ,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE user_sessions (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT REFERENCES users(id) ON DELETE CASCADE,
    session_token TEXT NOT NULL UNIQUE,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Feature Flag Management
CREATE TABLE feature_flags (
    id BIGSERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    is_enabled BOOLEAN DEFAULT FALSE,
    rollout_percentage INT DEFAULT 0 CHECK (rollout_percentage >= 0 AND rollout_percentage <= 100),
    target_roles JSONB DEFAULT '[]',
    target_subscription_tiers JSONB DEFAULT '[]',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Persistent Scenario Storage
CREATE TABLE saved_scenarios (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT REFERENCES users(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    description TEXT,
    shocks JSONB NOT NULL,
    horizon_hours INT NOT NULL,
    baseline_value NUMERIC(6,2),
    scenario_value NUMERIC(6,2),
    is_public BOOLEAN DEFAULT FALSE,
    shared_with JSONB DEFAULT '[]',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Alert Thresholds (Backend Storage)
CREATE TABLE alert_thresholds (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT REFERENCES users(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    geri_threshold NUMERIC(6,2),
    delta_threshold NUMERIC(6,2),
    conditions JSONB NOT NULL,
    notification_channels JSONB DEFAULT '[]',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Alert Events
CREATE TABLE alert_events (
    id BIGSERIAL PRIMARY KEY,
    threshold_id BIGINT REFERENCES alert_thresholds(id) ON DELETE CASCADE,
    user_id BIGINT REFERENCES users(id) ON DELETE CASCADE,
    trigger_type TEXT NOT NULL,
    trigger_value NUMERIC(10,2) NOT NULL,
    scenario_data JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Scenario Shares and Collaboration
CREATE TABLE scenario_shares (
    id BIGSERIAL PRIMARY KEY,
    scenario_id BIGINT REFERENCES saved_scenarios(id) ON DELETE CASCADE,
    shared_by BIGINT REFERENCES users(id) ON DELETE CASCADE,
    shared_with_user BIGINT REFERENCES users(id) ON DELETE SET NULL,
    shared_with_email TEXT,
    permission_level TEXT DEFAULT 'view' CHECK (permission_level IN ('view', 'edit', 'admin')),
    expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Premium Subscription Tracking
CREATE TABLE subscription_usage (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT REFERENCES users(id) ON DELETE CASCADE,
    feature TEXT NOT NULL,
    usage_count INT DEFAULT 0,
    period_start TIMESTAMPTZ NOT NULL,
    period_end TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Deployment Control Actions
CREATE TABLE deployment_actions (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT REFERENCES users(id) ON DELETE CASCADE,
    action_type TEXT NOT NULL,
    target_service TEXT,
    parameters JSONB DEFAULT '{}',
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    result JSONB,
    started_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

-- Indexes for performance
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_api_keys_hash ON api_keys(key_hash);
CREATE INDEX idx_api_keys_user_id ON api_keys(user_id);
CREATE INDEX idx_user_sessions_token ON user_sessions(session_token);
CREATE INDEX idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX idx_feature_flags_name ON feature_flags(name);
CREATE INDEX idx_saved_scenarios_user_id ON saved_scenarios(user_id);
CREATE INDEX idx_alert_thresholds_user_id ON alert_thresholds(user_id);
CREATE INDEX idx_alert_events_user_id ON alert_events(user_id);
CREATE INDEX idx_alert_events_threshold_id ON alert_events(threshold_id);
CREATE INDEX idx_scenario_shares_scenario_id ON scenario_shares(scenario_id);
CREATE INDEX idx_subscription_usage_user_id ON subscription_usage(user_id);
CREATE INDEX idx_deployment_actions_user_id ON deployment_actions(user_id);

-- Scenario Collaboration and Comments
CREATE TABLE scenario_comments (
    id BIGSERIAL PRIMARY KEY,
    scenario_id BIGINT REFERENCES saved_scenarios(id) ON DELETE CASCADE,
    user_id BIGINT REFERENCES users(id) ON DELETE CASCADE,
    comment_text TEXT NOT NULL,
    is_resolved BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Scenario Activity Log
CREATE TABLE scenario_activity (
    id BIGSERIAL PRIMARY KEY,
    scenario_id BIGINT REFERENCES saved_scenarios(id) ON DELETE CASCADE,
    user_id BIGINT REFERENCES users(id) ON DELETE CASCADE,
    activity_type TEXT NOT NULL, -- 'created', 'updated', 'shared', 'commented', 'forked'
    activity_data JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Alert Delivery Log (for production alert tracking)
CREATE TABLE alert_delivery_log (
    id BIGSERIAL PRIMARY KEY,
    subscription_id BIGINT,
    channel TEXT NOT NULL,
    address TEXT NOT NULL,
    payload JSONB NOT NULL,
    delivery_status TEXT NOT NULL,
    error_message TEXT,
    delivered_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Additional indexes for collaboration features
CREATE INDEX idx_scenario_comments_scenario_id ON scenario_comments(scenario_id);
CREATE INDEX idx_scenario_comments_user_id ON scenario_comments(user_id);
CREATE INDEX idx_scenario_activity_scenario_id ON scenario_activity(scenario_id);
CREATE INDEX idx_scenario_activity_user_id ON scenario_activity(user_id);
CREATE INDEX idx_alert_delivery_log_channel ON alert_delivery_log(channel);
CREATE INDEX idx_alert_delivery_log_status ON alert_delivery_log(delivery_status);

-- Advanced Export Records
CREATE TABLE export_records (
    id BIGSERIAL PRIMARY KEY,
    export_id TEXT NOT NULL UNIQUE,
    user_id BIGINT REFERENCES users(id) ON DELETE CASCADE,
    scope TEXT NOT NULL, -- 'scenario_runs', 'saved_scenarios', 'alert_history', 'collaboration_activity'
    format TEXT NOT NULL, -- 'csv', 'json', 'xlsx', 'pdf'
    file_path TEXT NOT NULL,
    file_size_bytes BIGINT NOT NULL,
    record_count INT NOT NULL,
    is_public BOOLEAN DEFAULT FALSE,
    expires_at TIMESTAMPTZ,
    filters JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_export_records_export_id ON export_records(export_id);
CREATE INDEX idx_export_records_user_id ON export_records(user_id);
CREATE INDEX idx_export_records_expires_at ON export_records(expires_at);
CREATE INDEX idx_export_records_scope ON export_records(scope);
CREATE INDEX idx_export_records_is_public ON export_records(is_public);
