-- RiskX Initial Database Schema
-- Risk Intelligence Observatory Database Structure

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "btree_gin";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS risk_intelligence;
CREATE SCHEMA IF NOT EXISTS economic_data;
CREATE SCHEMA IF NOT EXISTS supply_chain;
CREATE SCHEMA IF NOT EXISTS financial_data;
CREATE SCHEMA IF NOT EXISTS ml_models;
CREATE SCHEMA IF NOT EXISTS audit;

-- Set search path
SET search_path TO risk_intelligence, economic_data, supply_chain, financial_data, ml_models, audit, public;

-- ============================================================================
-- CORE RISK INTELLIGENCE TABLES
-- ============================================================================

-- Risk Score History
CREATE TABLE risk_intelligence.risk_scores (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    overall_score DECIMAL(5,2) NOT NULL CHECK (overall_score >= 0 AND overall_score <= 100),
    risk_level VARCHAR(20) NOT NULL CHECK (risk_level IN ('low', 'moderate', 'high', 'critical')),
    confidence DECIMAL(3,2) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    model_version VARCHAR(50) NOT NULL,
    data_sources TEXT[] NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB
);

-- Risk Factors
CREATE TABLE risk_intelligence.risk_factors (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    risk_score_id UUID NOT NULL REFERENCES risk_intelligence.risk_scores(id) ON DELETE CASCADE,
    factor_name VARCHAR(100) NOT NULL,
    factor_value DECIMAL(10,4) NOT NULL,
    normalized_value DECIMAL(5,4) NOT NULL CHECK (normalized_value >= 0 AND normalized_value <= 1),
    weight DECIMAL(5,4) NOT NULL CHECK (weight >= 0 AND weight <= 1),
    contribution DECIMAL(5,4) NOT NULL,
    category VARCHAR(50) NOT NULL,
    source VARCHAR(100) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Risk Alerts
CREATE TABLE risk_intelligence.risk_alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    alert_type VARCHAR(20) NOT NULL CHECK (alert_type IN ('critical', 'warning', 'info', 'success')),
    title VARCHAR(255) NOT NULL,
    message TEXT NOT NULL,
    source VARCHAR(100) NOT NULL,
    severity INTEGER NOT NULL CHECK (severity >= 1 AND severity <= 10),
    risk_score_id UUID REFERENCES risk_intelligence.risk_scores(id),
    affected_sectors TEXT[],
    action_required BOOLEAN NOT NULL DEFAULT FALSE,
    is_resolved BOOLEAN NOT NULL DEFAULT FALSE,
    resolved_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB
);

-- ============================================================================
-- ECONOMIC DATA TABLES
-- ============================================================================

-- FRED Economic Data
CREATE TABLE economic_data.fred_series (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    series_id VARCHAR(50) NOT NULL UNIQUE,
    title TEXT NOT NULL,
    frequency VARCHAR(20) NOT NULL,
    units VARCHAR(100),
    seasonal_adjustment VARCHAR(50),
    last_updated TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB
);

CREATE TABLE economic_data.fred_observations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    series_id VARCHAR(50) NOT NULL REFERENCES economic_data.fred_series(series_id),
    observation_date DATE NOT NULL,
    value DECIMAL(15,6),
    realtime_start DATE NOT NULL,
    realtime_end DATE NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(series_id, observation_date, realtime_start)
);

-- BEA Economic Data
CREATE TABLE economic_data.bea_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    dataset_name VARCHAR(100) NOT NULL,
    table_name VARCHAR(100) NOT NULL,
    line_code VARCHAR(20),
    line_description TEXT,
    time_period VARCHAR(20) NOT NULL,
    data_value DECIMAL(15,2),
    unit VARCHAR(50),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB
);

-- BLS Labor Data
CREATE TABLE economic_data.bls_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    series_id VARCHAR(50) NOT NULL,
    series_title TEXT,
    period VARCHAR(20) NOT NULL,
    year INTEGER NOT NULL,
    period_name VARCHAR(50),
    value DECIMAL(15,4),
    footnotes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(series_id, year, period)
);

-- ============================================================================
-- SUPPLY CHAIN TABLES
-- ============================================================================

-- Census Trade Data
CREATE TABLE supply_chain.trade_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    time_period VARCHAR(20) NOT NULL,
    trade_flow VARCHAR(20) NOT NULL CHECK (trade_flow IN ('imports', 'exports', 'balance')),
    partner_country VARCHAR(100),
    commodity_code VARCHAR(20),
    commodity_description TEXT,
    trade_value DECIMAL(15,2),
    quantity DECIMAL(15,4),
    unit VARCHAR(50),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB
);

-- Supply Chain Risk Indicators
CREATE TABLE supply_chain.risk_indicators (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    indicator_name VARCHAR(100) NOT NULL,
    indicator_value DECIMAL(10,4) NOT NULL,
    region VARCHAR(100),
    sector VARCHAR(100),
    risk_level VARCHAR(20) CHECK (risk_level IN ('low', 'moderate', 'high', 'critical')),
    data_source VARCHAR(100) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Port and Transportation Data
CREATE TABLE supply_chain.transportation_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMPTZ NOT NULL,
    location VARCHAR(100) NOT NULL,
    transportation_type VARCHAR(50) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(12,4) NOT NULL,
    unit VARCHAR(50),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================================
-- FINANCIAL DATA TABLES
-- ============================================================================

-- FDIC Banking Data
CREATE TABLE financial_data.banking_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    report_date DATE NOT NULL,
    institution_id VARCHAR(20) NOT NULL,
    institution_name TEXT,
    charter_type VARCHAR(50),
    total_assets DECIMAL(15,2),
    total_deposits DECIMAL(15,2),
    total_loans DECIMAL(15,2),
    capital_ratio DECIMAL(5,4),
    roi_assets DECIMAL(5,4),
    risk_based_capital_ratio DECIMAL(5,4),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB
);

-- Market Data
CREATE TABLE financial_data.market_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    data_type VARCHAR(50) NOT NULL, -- price, spread, volatility, etc.
    value DECIMAL(15,6) NOT NULL,
    currency VARCHAR(10),
    exchange VARCHAR(50),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Credit Risk Data
CREATE TABLE financial_data.credit_risk (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMPTZ NOT NULL,
    risk_category VARCHAR(100) NOT NULL,
    risk_metric VARCHAR(100) NOT NULL,
    value DECIMAL(12,6) NOT NULL,
    sector VARCHAR(100),
    rating VARCHAR(20),
    data_source VARCHAR(100) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================================
-- ML MODELS TABLES
-- ============================================================================

-- Model Registry
CREATE TABLE ml_models.model_registry (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    framework VARCHAR(50),
    file_path TEXT,
    training_date TIMESTAMPTZ,
    performance_metrics JSONB,
    feature_importance JSONB,
    model_status VARCHAR(20) NOT NULL DEFAULT 'training' CHECK (model_status IN ('training', 'active', 'retired')),
    created_by VARCHAR(100),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(model_name, model_version)
);

-- Predictions
CREATE TABLE ml_models.predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID NOT NULL REFERENCES ml_models.model_registry(id),
    prediction_type VARCHAR(50) NOT NULL,
    input_features JSONB NOT NULL,
    prediction_result JSONB NOT NULL,
    confidence_score DECIMAL(5,4),
    prediction_date TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    actual_outcome JSONB,
    outcome_date TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Feature Store
CREATE TABLE ml_models.feature_store (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    feature_name VARCHAR(100) NOT NULL,
    feature_value DECIMAL(15,6) NOT NULL,
    feature_type VARCHAR(50) NOT NULL,
    entity_id VARCHAR(100),
    timestamp TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB
);

-- ============================================================================
-- AUDIT TABLES
-- ============================================================================

-- Data Quality Metrics
CREATE TABLE audit.data_quality (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    table_name VARCHAR(100) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(10,6) NOT NULL,
    threshold DECIMAL(10,6),
    status VARCHAR(20) NOT NULL CHECK (status IN ('pass', 'warning', 'fail')),
    check_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- API Usage Logs
CREATE TABLE audit.api_usage (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    endpoint VARCHAR(255) NOT NULL,
    method VARCHAR(10) NOT NULL,
    user_id VARCHAR(100),
    ip_address INET,
    request_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    response_status INTEGER,
    response_time_ms INTEGER,
    request_size_bytes INTEGER,
    response_size_bytes INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- System Events
CREATE TABLE audit.system_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(50) NOT NULL,
    event_source VARCHAR(100) NOT NULL,
    event_description TEXT NOT NULL,
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('info', 'warning', 'error', 'critical')),
    event_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    user_id VARCHAR(100),
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

-- Risk Intelligence Indexes
CREATE INDEX idx_risk_scores_timestamp ON risk_intelligence.risk_scores(timestamp DESC);
CREATE INDEX idx_risk_scores_level ON risk_intelligence.risk_scores(risk_level);
CREATE INDEX idx_risk_factors_score_id ON risk_intelligence.risk_factors(risk_score_id);
CREATE INDEX idx_risk_factors_category ON risk_intelligence.risk_factors(category);
CREATE INDEX idx_risk_alerts_type ON risk_intelligence.risk_alerts(alert_type);
CREATE INDEX idx_risk_alerts_severity ON risk_intelligence.risk_alerts(severity DESC);
CREATE INDEX idx_risk_alerts_created ON risk_intelligence.risk_alerts(created_at DESC);

-- Economic Data Indexes
CREATE INDEX idx_fred_observations_series_date ON economic_data.fred_observations(series_id, observation_date DESC);
CREATE INDEX idx_bea_data_dataset_period ON economic_data.bea_data(dataset_name, time_period);
CREATE INDEX idx_bls_data_series_period ON economic_data.bls_data(series_id, year DESC, period);

-- Supply Chain Indexes
CREATE INDEX idx_trade_data_period ON supply_chain.trade_data(time_period DESC);
CREATE INDEX idx_trade_data_flow_country ON supply_chain.trade_data(trade_flow, partner_country);
CREATE INDEX idx_risk_indicators_timestamp ON supply_chain.risk_indicators(timestamp DESC);
CREATE INDEX idx_transportation_data_location ON supply_chain.transportation_data(location, timestamp DESC);

-- Financial Data Indexes
CREATE INDEX idx_banking_data_date ON financial_data.banking_data(report_date DESC);
CREATE INDEX idx_market_data_symbol_timestamp ON financial_data.market_data(symbol, timestamp DESC);
CREATE INDEX idx_credit_risk_timestamp ON financial_data.credit_risk(timestamp DESC);

-- ML Models Indexes
CREATE INDEX idx_model_registry_status ON ml_models.model_registry(model_status);
CREATE INDEX idx_predictions_model_date ON ml_models.predictions(model_id, prediction_date DESC);
CREATE INDEX idx_feature_store_name_timestamp ON ml_models.feature_store(feature_name, timestamp DESC);

-- Audit Indexes
CREATE INDEX idx_data_quality_table_timestamp ON audit.data_quality(table_name, check_timestamp DESC);
CREATE INDEX idx_api_usage_timestamp ON audit.api_usage(request_timestamp DESC);
CREATE INDEX idx_system_events_type_timestamp ON audit.system_events(event_type, event_timestamp DESC);

-- ============================================================================
-- PARTITIONING FOR LARGE TABLES
-- ============================================================================

-- Partition risk_scores by month for performance
-- ALTER TABLE risk_intelligence.risk_scores PARTITION BY RANGE (timestamp);

-- Partition observations by month
-- ALTER TABLE economic_data.fred_observations PARTITION BY RANGE (observation_date);

-- ============================================================================
-- VIEWS FOR COMMON QUERIES
-- ============================================================================

-- Latest Risk Score
CREATE VIEW risk_intelligence.latest_risk_score AS
SELECT DISTINCT ON (1) *
FROM risk_intelligence.risk_scores
ORDER BY 1, timestamp DESC;

-- Active Risk Alerts
CREATE VIEW risk_intelligence.active_alerts AS
SELECT *
FROM risk_intelligence.risk_alerts
WHERE NOT is_resolved
ORDER BY severity DESC, created_at DESC;

-- Current Economic Indicators
CREATE VIEW economic_data.current_indicators AS
SELECT 
    f.series_id,
    f.title,
    fo.value,
    fo.observation_date
FROM economic_data.fred_series f
JOIN LATERAL (
    SELECT value, observation_date
    FROM economic_data.fred_observations o
    WHERE o.series_id = f.series_id
    ORDER BY observation_date DESC
    LIMIT 1
) fo ON true;

-- ============================================================================
-- TRIGGERS AND FUNCTIONS
-- ============================================================================

-- Function to update risk level based on score
CREATE OR REPLACE FUNCTION update_risk_level()
RETURNS TRIGGER AS $$
BEGIN
    NEW.risk_level := CASE
        WHEN NEW.overall_score < 30 THEN 'low'
        WHEN NEW.overall_score < 60 THEN 'moderate'
        WHEN NEW.overall_score < 80 THEN 'high'
        ELSE 'critical'
    END;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to automatically set risk level
CREATE TRIGGER trigger_update_risk_level
    BEFORE INSERT OR UPDATE ON risk_intelligence.risk_scores
    FOR EACH ROW
    EXECUTE FUNCTION update_risk_level();

-- Function for audit logging
CREATE OR REPLACE FUNCTION log_data_changes()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO audit.system_events (
        event_type,
        event_source,
        event_description,
        severity,
        metadata
    ) VALUES (
        TG_OP,
        TG_TABLE_SCHEMA || '.' || TG_TABLE_NAME,
        'Data modification in ' || TG_TABLE_NAME,
        'info',
        jsonb_build_object('old', to_jsonb(OLD), 'new', to_jsonb(NEW))
    );
    
    IF TG_OP = 'DELETE' THEN
        RETURN OLD;
    ELSE
        RETURN NEW;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- GRANTS AND PERMISSIONS
-- ============================================================================

-- Create roles
CREATE ROLE riskx_read;
CREATE ROLE riskx_write;
CREATE ROLE riskx_admin;

-- Grant schema permissions
GRANT USAGE ON SCHEMA risk_intelligence, economic_data, supply_chain, financial_data, ml_models, audit TO riskx_read;
GRANT USAGE ON SCHEMA risk_intelligence, economic_data, supply_chain, financial_data, ml_models, audit TO riskx_write;
GRANT ALL ON SCHEMA risk_intelligence, economic_data, supply_chain, financial_data, ml_models, audit TO riskx_admin;

-- Grant table permissions
GRANT SELECT ON ALL TABLES IN SCHEMA risk_intelligence, economic_data, supply_chain, financial_data, ml_models, audit TO riskx_read;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA risk_intelligence, economic_data, supply_chain, financial_data, ml_models, audit TO riskx_write;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA risk_intelligence, economic_data, supply_chain, financial_data, ml_models, audit TO riskx_admin;

-- Grant sequence permissions
GRANT USAGE ON ALL SEQUENCES IN SCHEMA risk_intelligence, economic_data, supply_chain, financial_data, ml_models, audit TO riskx_write;
GRANT ALL ON ALL SEQUENCES IN SCHEMA risk_intelligence, economic_data, supply_chain, financial_data, ml_models, audit TO riskx_admin;

-- Grant function permissions
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA risk_intelligence, economic_data, supply_chain, financial_data, ml_models, audit TO riskx_write;