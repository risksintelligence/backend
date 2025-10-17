#!/usr/bin/env python3
"""
Data Seeding Script

Seeds the RiskX database with initial data, sample records, and default configurations.
"""

import asyncio
import logging
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


async def seed_initial_data(engine):
    """Seed the database with initial data"""
    logger = logging.getLogger("seed_data")
    
    async with engine.acquire() as conn:
        # Seed configuration data
        await seed_system_configuration(conn)
        
        # Seed sample cache metadata
        await seed_cache_metadata(conn)
        
        # Seed sample data quality records
        await seed_data_quality_samples(conn)
        
        # Seed sample pipeline execution records
        await seed_pipeline_execution_samples(conn)
        
        # Seed sample model performance records
        await seed_model_performance_samples(conn)
        
        # Seed sample system health records
        await seed_system_health_samples(conn)
        
        logger.info("Initial data seeding completed")


async def seed_system_configuration(conn):
    """Seed system configuration parameters"""
    logger = logging.getLogger("seed_data")
    
    # Sample configuration data
    config_data = [
        {
            'component': 'cache',
            'setting': 'default_ttl',
            'value': '3600',
            'description': 'Default cache TTL in seconds'
        },
        {
            'component': 'data_quality',
            'setting': 'min_quality_threshold',
            'value': '0.8',
            'description': 'Minimum acceptable data quality score'
        },
        {
            'component': 'alerts',
            'setting': 'critical_response_time',
            'value': '300',
            'description': 'Maximum acceptable response time for critical alerts in seconds'
        },
        {
            'component': 'pipeline',
            'setting': 'max_retry_attempts',
            'value': '3',
            'description': 'Maximum number of retry attempts for failed pipeline tasks'
        },
        {
            'component': 'api',
            'setting': 'rate_limit_per_minute',
            'value': '1000',
            'description': 'API rate limit per minute per client'
        }
    ]
    
    # Create configuration table if it doesn't exist
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS system_configuration (
            id SERIAL PRIMARY KEY,
            component VARCHAR(100) NOT NULL,
            setting VARCHAR(100) NOT NULL,
            value TEXT NOT NULL,
            description TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(component, setting)
        )
    """)
    
    # Insert configuration data
    for config in config_data:
        await conn.execute("""
            INSERT INTO system_configuration (component, setting, value, description)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (component, setting) DO UPDATE SET
                value = EXCLUDED.value,
                description = EXCLUDED.description,
                updated_at = CURRENT_TIMESTAMP
        """, config['component'], config['setting'], config['value'], config['description'])
    
    logger.info(f"Seeded {len(config_data)} system configuration records")


async def seed_cache_metadata(conn):
    """Seed sample cache metadata records"""
    logger = logging.getLogger("seed_data")
    
    cache_records = [
        {
            'cache_key': 'fred_gdp_quarterly',
            'data_source': 'fred',
            'last_updated': datetime.now() - timedelta(hours=2),
            'expiry_time': datetime.now() + timedelta(hours=22),
            'size_bytes': 15420,
            'hit_count': 45,
            'miss_count': 3
        },
        {
            'cache_key': 'bea_trade_balance_monthly',
            'data_source': 'bea',
            'last_updated': datetime.now() - timedelta(hours=1),
            'expiry_time': datetime.now() + timedelta(hours=23),
            'size_bytes': 8765,
            'hit_count': 23,
            'miss_count': 1
        },
        {
            'cache_key': 'bls_unemployment_rate',
            'data_source': 'bls',
            'last_updated': datetime.now() - timedelta(minutes=30),
            'expiry_time': datetime.now() + timedelta(hours=23, minutes=30),
            'size_bytes': 5432,
            'hit_count': 67,
            'miss_count': 2
        },
        {
            'cache_key': 'fdic_bank_health_indicators',
            'data_source': 'fdic',
            'last_updated': datetime.now() - timedelta(hours=6),
            'expiry_time': datetime.now() + timedelta(hours=18),
            'size_bytes': 34567,
            'hit_count': 12,
            'miss_count': 1
        },
        {
            'cache_key': 'noaa_weather_alerts',
            'data_source': 'noaa',
            'last_updated': datetime.now() - timedelta(minutes=15),
            'expiry_time': datetime.now() + timedelta(minutes=45),
            'size_bytes': 2345,
            'hit_count': 89,
            'miss_count': 5
        }
    ]
    
    for record in cache_records:
        await conn.execute("""
            INSERT INTO cache_metadata 
            (cache_key, data_source, last_updated, expiry_time, size_bytes, hit_count, miss_count)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (cache_key) DO UPDATE SET
                last_updated = EXCLUDED.last_updated,
                expiry_time = EXCLUDED.expiry_time,
                size_bytes = EXCLUDED.size_bytes,
                hit_count = EXCLUDED.hit_count,
                miss_count = EXCLUDED.miss_count,
                updated_at = CURRENT_TIMESTAMP
        """, record['cache_key'], record['data_source'], record['last_updated'],
        record['expiry_time'], record['size_bytes'], record['hit_count'], record['miss_count'])
    
    logger.info(f"Seeded {len(cache_records)} cache metadata records")


async def seed_data_quality_samples(conn):
    """Seed sample data quality records"""
    logger = logging.getLogger("seed_data")
    
    quality_records = []
    sources = ['fred', 'bea', 'bls', 'fdic', 'noaa', 'cisa', 'census']
    validation_types = ['schema', 'completeness', 'freshness', 'consistency']
    
    # Generate sample records for the last 7 days
    for days_ago in range(7):
        date = datetime.now() - timedelta(days=days_ago)
        
        for source in sources:
            for validation_type in validation_types:
                # Generate realistic quality scores
                base_score = 0.85 if source in ['fred', 'bea', 'bls'] else 0.75
                import random
                quality_score = min(1.0, max(0.0, base_score + random.uniform(-0.1, 0.1)))
                
                record = {
                    'source_name': source,
                    'validation_type': validation_type,
                    'validation_result': {
                        'passed_checks': random.randint(8, 12),
                        'failed_checks': random.randint(0, 2),
                        'warnings': random.randint(0, 3),
                        'details': f'Sample validation for {source} {validation_type}'
                    },
                    'quality_score': quality_score,
                    'record_count': random.randint(1000, 10000),
                    'error_count': random.randint(0, 5),
                    'warning_count': random.randint(0, 10),
                    'execution_time_ms': random.randint(100, 2000),
                    'created_at': date
                }
                quality_records.append(record)
    
    for record in quality_records:
        await conn.execute("""
            INSERT INTO data_quality_log 
            (source_name, validation_type, validation_result, quality_score, 
             record_count, error_count, warning_count, execution_time_ms, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        """, record['source_name'], record['validation_type'], 
        json.dumps(record['validation_result']), record['quality_score'],
        record['record_count'], record['error_count'], record['warning_count'],
        record['execution_time_ms'], record['created_at'])
    
    logger.info(f"Seeded {len(quality_records)} data quality records")


async def seed_pipeline_execution_samples(conn):
    """Seed sample pipeline execution records"""
    logger = logging.getLogger("seed_data")
    
    pipelines = [
        'economic_data_pipeline',
        'trade_data_pipeline', 
        'financial_risk_pipeline',
        'weather_events_pipeline',
        'cyber_threats_pipeline'
    ]
    
    execution_records = []
    
    # Generate records for the last 30 days
    for days_ago in range(30):
        date = datetime.now() - timedelta(days=days_ago)
        
        for pipeline in pipelines:
            # Most executions succeed
            import random
            status = 'completed' if random.random() > 0.1 else 'failed'
            
            start_time = date
            duration = random.randint(300, 1800)  # 5-30 minutes
            end_time = start_time + timedelta(seconds=duration)
            
            record = {
                'pipeline_name': pipeline,
                'execution_id': f"{pipeline}_{date.strftime('%Y%m%d_%H%M%S')}",
                'status': status,
                'start_time': start_time,
                'end_time': end_time if status == 'completed' else None,
                'duration_seconds': duration if status == 'completed' else None,
                'records_processed': random.randint(1000, 50000) if status == 'completed' else 0,
                'errors_encountered': 0 if status == 'completed' else random.randint(1, 5),
                'execution_details': {
                    'source_count': random.randint(3, 7),
                    'validation_passed': status == 'completed',
                    'cache_refreshed': True
                },
                'error_message': None if status == 'completed' else 'Sample error for demonstration'
            }
            execution_records.append(record)
    
    for record in execution_records:
        await conn.execute("""
            INSERT INTO pipeline_execution_log 
            (pipeline_name, execution_id, status, start_time, end_time, duration_seconds,
             records_processed, errors_encountered, execution_details, error_message)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        """, record['pipeline_name'], record['execution_id'], record['status'],
        record['start_time'], record['end_time'], record['duration_seconds'],
        record['records_processed'], record['errors_encountered'],
        json.dumps(record['execution_details']), record['error_message'])
    
    logger.info(f"Seeded {len(execution_records)} pipeline execution records")


async def seed_model_performance_samples(conn):
    """Seed sample model performance records"""
    logger = logging.getLogger("seed_data")
    
    models = [
        'risk_prediction_model',
        'supply_chain_risk_model',
        'financial_stress_model',
        'weather_impact_model'
    ]
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc', 'mse', 'mae']
    dataset_types = ['training', 'validation', 'test', 'production']
    
    performance_records = []
    
    # Generate monthly records for the last 6 months
    for months_ago in range(6):
        date = datetime.now() - timedelta(days=months_ago * 30)
        
        for model in models:
            for dataset_type in dataset_types:
                for metric in metrics:
                    # Generate realistic performance values
                    import random
                    if metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']:
                        value = random.uniform(0.75, 0.95)
                    else:  # MSE, MAE
                        value = random.uniform(0.01, 0.1)
                    
                    record = {
                        'model_name': model,
                        'model_version': f'v1.{6-months_ago}',
                        'evaluation_date': date,
                        'metric_name': metric,
                        'metric_value': value,
                        'dataset_type': dataset_type,
                        'data_period_start': date - timedelta(days=90),
                        'data_period_end': date,
                        'sample_size': random.randint(5000, 50000),
                        'metadata': {
                            'feature_count': random.randint(50, 200),
                            'training_time_minutes': random.randint(30, 180)
                        }
                    }
                    performance_records.append(record)
    
    for record in performance_records:
        await conn.execute("""
            INSERT INTO model_performance_log 
            (model_name, model_version, evaluation_date, metric_name, metric_value,
             dataset_type, data_period_start, data_period_end, sample_size, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        """, record['model_name'], record['model_version'], record['evaluation_date'],
        record['metric_name'], record['metric_value'], record['dataset_type'],
        record['data_period_start'], record['data_period_end'], record['sample_size'],
        json.dumps(record['metadata']))
    
    logger.info(f"Seeded {len(performance_records)} model performance records")


async def seed_system_health_samples(conn):
    """Seed sample system health records"""
    logger = logging.getLogger("seed_data")
    
    components = [
        'redis_cache',
        'postgresql_db',
        'fred_api',
        'bea_api',
        'bls_api',
        'fdic_api',
        'noaa_api',
        'cisa_api',
        'ml_model_service',
        'web_frontend'
    ]
    
    health_records = []
    
    # Generate hourly records for the last 24 hours
    for hours_ago in range(24):
        timestamp = datetime.now() - timedelta(hours=hours_ago)
        
        for component in components:
            # Most components are healthy
            import random
            status_rand = random.random()
            if status_rand > 0.95:
                status = 'unhealthy'
            elif status_rand > 0.85:
                status = 'degraded'
            else:
                status = 'healthy'
            
            response_time = random.randint(50, 500) if status == 'healthy' else random.randint(1000, 5000)
            
            record = {
                'component_name': component,
                'health_status': status,
                'metrics': {
                    'cpu_usage': random.uniform(10, 80),
                    'memory_usage': random.uniform(20, 70),
                    'response_time_ms': response_time,
                    'error_rate': random.uniform(0, 0.05) if status == 'healthy' else random.uniform(0.1, 0.3)
                },
                'response_time_ms': response_time,
                'error_message': None if status == 'healthy' else 'Sample error message',
                'checked_at': timestamp
            }
            health_records.append(record)
    
    for record in health_records:
        await conn.execute("""
            INSERT INTO system_health_log 
            (component_name, health_status, metrics, response_time_ms, error_message, checked_at)
            VALUES ($1, $2, $3, $4, $5, $6)
        """, record['component_name'], record['health_status'],
        json.dumps(record['metrics']), record['response_time_ms'],
        record['error_message'], record['checked_at'])
    
    logger.info(f"Seeded {len(health_records)} system health records")


async def main():
    """Main execution function for standalone usage"""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    from src.core.database import get_database_engine
    
    logger = logging.getLogger("seed_data")
    
    try:
        logger.info("Seeding initial data...")
        engine = get_database_engine()
        
        await seed_initial_data(engine)
        
        logger.info("Data seeding completed successfully!")
        
    except Exception as e:
        logger.error(f"Failed to seed data: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())