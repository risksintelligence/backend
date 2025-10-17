#!/usr/bin/env python3
"""
Table Creation Script

Creates custom tables required for the RiskX platform beyond the standard ORM models.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


async def create_custom_tables(engine):
    """Create custom tables not covered by ORM models"""
    logger = logging.getLogger("create_tables")
    
    tables_to_create = [
        create_cache_metadata_table,
        create_data_quality_log_table,
        create_pipeline_execution_log_table,
        create_model_performance_log_table,
        create_alert_log_table,
        create_user_session_table,
        create_api_usage_log_table,
        create_system_health_log_table
    ]
    
    async with engine.acquire() as conn:
        for table_creator in tables_to_create:
            try:
                await table_creator(conn)
                logger.info(f"Created table: {table_creator.__name__}")
            except Exception as e:
                logger.error(f"Error creating table {table_creator.__name__}: {str(e)}")
                raise


async def create_cache_metadata_table(conn):
    """Create cache metadata tracking table"""
    sql = """
    CREATE TABLE IF NOT EXISTS cache_metadata (
        id SERIAL PRIMARY KEY,
        cache_key VARCHAR(255) UNIQUE NOT NULL,
        data_source VARCHAR(100) NOT NULL,
        last_updated TIMESTAMP WITH TIME ZONE NOT NULL,
        expiry_time TIMESTAMP WITH TIME ZONE,
        size_bytes INTEGER,
        hit_count INTEGER DEFAULT 0,
        miss_count INTEGER DEFAULT 0,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE INDEX IF NOT EXISTS idx_cache_metadata_source ON cache_metadata(data_source);
    CREATE INDEX IF NOT EXISTS idx_cache_metadata_updated ON cache_metadata(last_updated);
    CREATE INDEX IF NOT EXISTS idx_cache_metadata_expiry ON cache_metadata(expiry_time);
    """
    await conn.execute(sql)


async def create_data_quality_log_table(conn):
    """Create data quality monitoring log table"""
    sql = """
    CREATE TABLE IF NOT EXISTS data_quality_log (
        id SERIAL PRIMARY KEY,
        source_name VARCHAR(100) NOT NULL,
        validation_type VARCHAR(50) NOT NULL,
        validation_result JSONB NOT NULL,
        quality_score DECIMAL(5,4),
        record_count INTEGER,
        error_count INTEGER DEFAULT 0,
        warning_count INTEGER DEFAULT 0,
        execution_time_ms INTEGER,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE INDEX IF NOT EXISTS idx_dq_log_source ON data_quality_log(source_name);
    CREATE INDEX IF NOT EXISTS idx_dq_log_type ON data_quality_log(validation_type);
    CREATE INDEX IF NOT EXISTS idx_dq_log_created ON data_quality_log(created_at);
    CREATE INDEX IF NOT EXISTS idx_dq_log_score ON data_quality_log(quality_score);
    """
    await conn.execute(sql)


async def create_pipeline_execution_log_table(conn):
    """Create ETL pipeline execution log table"""
    sql = """
    CREATE TABLE IF NOT EXISTS pipeline_execution_log (
        id SERIAL PRIMARY KEY,
        pipeline_name VARCHAR(100) NOT NULL,
        execution_id VARCHAR(100) UNIQUE NOT NULL,
        status VARCHAR(20) NOT NULL CHECK (status IN ('running', 'completed', 'failed', 'cancelled')),
        start_time TIMESTAMP WITH TIME ZONE NOT NULL,
        end_time TIMESTAMP WITH TIME ZONE,
        duration_seconds INTEGER,
        records_processed INTEGER DEFAULT 0,
        errors_encountered INTEGER DEFAULT 0,
        execution_details JSONB,
        error_message TEXT,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE INDEX IF NOT EXISTS idx_pipeline_log_name ON pipeline_execution_log(pipeline_name);
    CREATE INDEX IF NOT EXISTS idx_pipeline_log_status ON pipeline_execution_log(status);
    CREATE INDEX IF NOT EXISTS idx_pipeline_log_start ON pipeline_execution_log(start_time);
    CREATE INDEX IF NOT EXISTS idx_pipeline_log_execution_id ON pipeline_execution_log(execution_id);
    """
    await conn.execute(sql)


async def create_model_performance_log_table(conn):
    """Create ML model performance tracking table"""
    sql = """
    CREATE TABLE IF NOT EXISTS model_performance_log (
        id SERIAL PRIMARY KEY,
        model_name VARCHAR(100) NOT NULL,
        model_version VARCHAR(50) NOT NULL,
        evaluation_date TIMESTAMP WITH TIME ZONE NOT NULL,
        metric_name VARCHAR(50) NOT NULL,
        metric_value DECIMAL(10,6) NOT NULL,
        dataset_type VARCHAR(20) NOT NULL CHECK (dataset_type IN ('training', 'validation', 'test', 'production')),
        data_period_start DATE,
        data_period_end DATE,
        sample_size INTEGER,
        metadata JSONB,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE INDEX IF NOT EXISTS idx_model_perf_name ON model_performance_log(model_name);
    CREATE INDEX IF NOT EXISTS idx_model_perf_version ON model_performance_log(model_version);
    CREATE INDEX IF NOT EXISTS idx_model_perf_date ON model_performance_log(evaluation_date);
    CREATE INDEX IF NOT EXISTS idx_model_perf_metric ON model_performance_log(metric_name);
    """
    await conn.execute(sql)


async def create_alert_log_table(conn):
    """Create system alerts and notifications log table"""
    sql = """
    CREATE TABLE IF NOT EXISTS alert_log (
        id SERIAL PRIMARY KEY,
        alert_type VARCHAR(50) NOT NULL,
        severity VARCHAR(20) NOT NULL CHECK (severity IN ('low', 'medium', 'high', 'critical')),
        title VARCHAR(255) NOT NULL,
        message TEXT NOT NULL,
        source_system VARCHAR(100) NOT NULL,
        triggered_at TIMESTAMP WITH TIME ZONE NOT NULL,
        resolved_at TIMESTAMP WITH TIME ZONE,
        status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'acknowledged', 'resolved', 'false_positive')),
        assigned_to VARCHAR(100),
        resolution_notes TEXT,
        metadata JSONB,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE INDEX IF NOT EXISTS idx_alert_log_type ON alert_log(alert_type);
    CREATE INDEX IF NOT EXISTS idx_alert_log_severity ON alert_log(severity);
    CREATE INDEX IF NOT EXISTS idx_alert_log_status ON alert_log(status);
    CREATE INDEX IF NOT EXISTS idx_alert_log_triggered ON alert_log(triggered_at);
    CREATE INDEX IF NOT EXISTS idx_alert_log_source ON alert_log(source_system);
    """
    await conn.execute(sql)


async def create_user_session_table(conn):
    """Create user session tracking table"""
    sql = """
    CREATE TABLE IF NOT EXISTS user_sessions (
        id SERIAL PRIMARY KEY,
        session_id VARCHAR(255) UNIQUE NOT NULL,
        user_id VARCHAR(100),
        ip_address INET,
        user_agent TEXT,
        started_at TIMESTAMP WITH TIME ZONE NOT NULL,
        last_activity TIMESTAMP WITH TIME ZONE NOT NULL,
        ended_at TIMESTAMP WITH TIME ZONE,
        session_data JSONB,
        is_active BOOLEAN DEFAULT TRUE,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE INDEX IF NOT EXISTS idx_user_sessions_id ON user_sessions(session_id);
    CREATE INDEX IF NOT EXISTS idx_user_sessions_user ON user_sessions(user_id);
    CREATE INDEX IF NOT EXISTS idx_user_sessions_active ON user_sessions(is_active);
    CREATE INDEX IF NOT EXISTS idx_user_sessions_started ON user_sessions(started_at);
    """
    await conn.execute(sql)


async def create_api_usage_log_table(conn):
    """Create API usage and performance log table"""
    sql = """
    CREATE TABLE IF NOT EXISTS api_usage_log (
        id SERIAL PRIMARY KEY,
        request_id VARCHAR(100) UNIQUE NOT NULL,
        endpoint VARCHAR(255) NOT NULL,
        method VARCHAR(10) NOT NULL,
        status_code INTEGER NOT NULL,
        response_time_ms INTEGER NOT NULL,
        request_size_bytes INTEGER,
        response_size_bytes INTEGER,
        user_id VARCHAR(100),
        ip_address INET,
        user_agent TEXT,
        error_message TEXT,
        timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE INDEX IF NOT EXISTS idx_api_log_endpoint ON api_usage_log(endpoint);
    CREATE INDEX IF NOT EXISTS idx_api_log_status ON api_usage_log(status_code);
    CREATE INDEX IF NOT EXISTS idx_api_log_timestamp ON api_usage_log(timestamp);
    CREATE INDEX IF NOT EXISTS idx_api_log_user ON api_usage_log(user_id);
    CREATE INDEX IF NOT EXISTS idx_api_log_response_time ON api_usage_log(response_time_ms);
    """
    await conn.execute(sql)


async def create_system_health_log_table(conn):
    """Create system health monitoring log table"""
    sql = """
    CREATE TABLE IF NOT EXISTS system_health_log (
        id SERIAL PRIMARY KEY,
        component_name VARCHAR(100) NOT NULL,
        health_status VARCHAR(20) NOT NULL CHECK (health_status IN ('healthy', 'degraded', 'unhealthy', 'unknown')),
        metrics JSONB NOT NULL,
        response_time_ms INTEGER,
        error_message TEXT,
        checked_at TIMESTAMP WITH TIME ZONE NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE INDEX IF NOT EXISTS idx_health_log_component ON system_health_log(component_name);
    CREATE INDEX IF NOT EXISTS idx_health_log_status ON system_health_log(health_status);
    CREATE INDEX IF NOT EXISTS idx_health_log_checked ON system_health_log(checked_at);
    """
    await conn.execute(sql)


async def create_indexes_and_constraints():
    """Create additional indexes and constraints for performance"""
    sql_commands = [
        # Composite indexes for common query patterns
        """
        CREATE INDEX IF NOT EXISTS idx_cache_metadata_source_updated 
        ON cache_metadata(data_source, last_updated DESC);
        """,
        
        """
        CREATE INDEX IF NOT EXISTS idx_dq_log_source_created 
        ON data_quality_log(source_name, created_at DESC);
        """,
        
        """
        CREATE INDEX IF NOT EXISTS idx_pipeline_log_name_start 
        ON pipeline_execution_log(pipeline_name, start_time DESC);
        """,
        
        """
        CREATE INDEX IF NOT EXISTS idx_model_perf_name_date 
        ON model_performance_log(model_name, evaluation_date DESC);
        """,
        
        """
        CREATE INDEX IF NOT EXISTS idx_alert_log_status_severity 
        ON alert_log(status, severity, triggered_at DESC);
        """,
        
        """
        CREATE INDEX IF NOT EXISTS idx_api_log_endpoint_timestamp 
        ON api_usage_log(endpoint, timestamp DESC);
        """,
        
        # Partial indexes for active records
        """
        CREATE INDEX IF NOT EXISTS idx_user_sessions_active_last_activity 
        ON user_sessions(last_activity DESC) WHERE is_active = TRUE;
        """,
        
        """
        CREATE INDEX IF NOT EXISTS idx_alert_log_active 
        ON alert_log(triggered_at DESC) WHERE status = 'active';
        """
    ]
    
    return sql_commands


async def create_views():
    """Create useful database views"""
    views = [
        # Recent data quality summary view
        """
        CREATE OR REPLACE VIEW recent_data_quality AS
        SELECT 
            source_name,
            validation_type,
            AVG(quality_score) as avg_quality_score,
            COUNT(*) as validation_count,
            SUM(error_count) as total_errors,
            MAX(created_at) as last_validation
        FROM data_quality_log 
        WHERE created_at >= CURRENT_DATE - INTERVAL '7 days'
        GROUP BY source_name, validation_type
        ORDER BY source_name, validation_type;
        """,
        
        # Pipeline execution summary view
        """
        CREATE OR REPLACE VIEW pipeline_execution_summary AS
        SELECT 
            pipeline_name,
            status,
            COUNT(*) as execution_count,
            AVG(duration_seconds) as avg_duration_seconds,
            SUM(records_processed) as total_records_processed,
            MAX(start_time) as last_execution
        FROM pipeline_execution_log 
        WHERE start_time >= CURRENT_DATE - INTERVAL '30 days'
        GROUP BY pipeline_name, status
        ORDER BY pipeline_name, status;
        """,
        
        # API performance summary view
        """
        CREATE OR REPLACE VIEW api_performance_summary AS
        SELECT 
            endpoint,
            method,
            COUNT(*) as request_count,
            AVG(response_time_ms) as avg_response_time_ms,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY response_time_ms) as p95_response_time_ms,
            COUNT(*) FILTER (WHERE status_code >= 400) as error_count,
            COUNT(*) FILTER (WHERE status_code >= 200 AND status_code < 300) as success_count
        FROM api_usage_log 
        WHERE timestamp >= CURRENT_DATE - INTERVAL '24 hours'
        GROUP BY endpoint, method
        ORDER BY request_count DESC;
        """,
        
        # System health overview view
        """
        CREATE OR REPLACE VIEW system_health_overview AS
        SELECT DISTINCT ON (component_name) 
            component_name,
            health_status,
            metrics,
            response_time_ms,
            error_message,
            checked_at
        FROM system_health_log 
        ORDER BY component_name, checked_at DESC;
        """
    ]
    
    return views


async def apply_additional_optimizations(conn):
    """Apply additional database optimizations"""
    logger = logging.getLogger("create_tables")
    
    try:
        # Create indexes
        logger.info("Creating additional indexes...")
        index_commands = await create_indexes_and_constraints()
        for cmd in index_commands:
            await conn.execute(cmd)
        
        # Create views
        logger.info("Creating database views...")
        view_commands = await create_views()
        for cmd in view_commands:
            await conn.execute(cmd)
        
        logger.info("Additional optimizations applied successfully")
        
    except Exception as e:
        logger.error(f"Error applying optimizations: {str(e)}")
        raise


async def main():
    """Main execution function for standalone usage"""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    from src.core.database import get_database_engine
    
    logger = logging.getLogger("create_tables")
    
    try:
        logger.info("Creating custom tables...")
        engine = get_database_engine()
        
        await create_custom_tables(engine)
        
        async with engine.acquire() as conn:
            await apply_additional_optimizations(conn)
        
        logger.info("Custom tables created successfully!")
        
    except Exception as e:
        logger.error(f"Failed to create custom tables: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())