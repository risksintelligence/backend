"""
Risk Intelligence Data ETL Pipeline
Automated pipeline for risk intelligence data from CISA, NOAA, USGS, and Supply Chain sources
"""
from datetime import datetime, timedelta
import logging
import asyncio
from typing import Dict, Any

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.redis.hooks.redis import RedisHook
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.utils.dates import days_ago

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default DAG arguments
default_args = {
    'owner': 'riskx-platform',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

# DAG definition
dag = DAG(
    'risk_intelligence_pipeline',
    default_args=default_args,
    description='Extract, transform, and load risk intelligence data from multiple sources',
    schedule_interval=timedelta(hours=4),  # Run every 4 hours
    max_active_runs=1,
    catchup=False,
    tags=['risk', 'intelligence', 'security', 'etl'],
)


def extract_cisa_data(**context):
    """Extract cybersecurity threat data from CISA"""
    import sys
    import os
    sys.path.append('/Users/omoshola/Documents/riskxx/backend')
    
    from src.data.sources.cisa import get_cybersecurity_threats
    
    logger.info("Starting CISA data extraction...")
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        cisa_data = loop.run_until_complete(get_cybersecurity_threats())
        loop.close()
        
        if cisa_data and 'indicators' in cisa_data:
            logger.info(f"Successfully extracted {cisa_data.get('count', 0)} CISA indicators")
            context['task_instance'].xcom_push(key='cisa_data', value=cisa_data)
            return cisa_data
        else:
            raise ValueError("No CISA data received")
            
    except Exception as e:
        logger.error(f"CISA data extraction failed: {e}")
        raise


def extract_noaa_data(**context):
    """Extract environmental risk data from NOAA"""
    import sys
    import os
    sys.path.append('/Users/omoshola/Documents/riskxx/backend')
    
    from src.data.sources.noaa import get_environmental_risks
    
    logger.info("Starting NOAA data extraction...")
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        noaa_data = loop.run_until_complete(get_environmental_risks())
        loop.close()
        
        if noaa_data and 'indicators' in noaa_data:
            logger.info(f"Successfully extracted {noaa_data.get('count', 0)} NOAA indicators")
            context['task_instance'].xcom_push(key='noaa_data', value=noaa_data)
            return noaa_data
        else:
            raise ValueError("No NOAA data received")
            
    except Exception as e:
        logger.error(f"NOAA data extraction failed: {e}")
        raise


def extract_usgs_data(**context):
    """Extract geological hazard data from USGS"""
    import sys
    import os
    sys.path.append('/Users/omoshola/Documents/riskxx/backend')
    
    from src.data.sources.usgs import get_geological_hazards
    
    logger.info("Starting USGS data extraction...")
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        usgs_data = loop.run_until_complete(get_geological_hazards())
        loop.close()
        
        if usgs_data and 'indicators' in usgs_data:
            logger.info(f"Successfully extracted {usgs_data.get('count', 0)} USGS indicators")
            context['task_instance'].xcom_push(key='usgs_data', value=usgs_data)
            return usgs_data
        else:
            raise ValueError("No USGS data received")
            
    except Exception as e:
        logger.error(f"USGS data extraction failed: {e}")
        raise


def extract_supply_chain_data(**context):
    """Extract supply chain risk data"""
    import sys
    import os
    sys.path.append('/Users/omoshola/Documents/riskxx/backend')
    
    from src.data.sources.supply_chain import get_supply_chain_risks
    
    logger.info("Starting Supply Chain data extraction...")
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        supply_data = loop.run_until_complete(get_supply_chain_risks())
        loop.close()
        
        if supply_data and 'indicators' in supply_data:
            logger.info(f"Successfully extracted {supply_data.get('count', 0)} Supply Chain indicators")
            context['task_instance'].xcom_push(key='supply_chain_data', value=supply_data)
            return supply_data
        else:
            raise ValueError("No Supply Chain data received")
            
    except Exception as e:
        logger.error(f"Supply Chain data extraction failed: {e}")
        raise


def transform_risk_data(**context):
    """Transform and normalize risk intelligence data from all sources"""
    logger.info("Starting risk intelligence data transformation...")
    
    try:
        # Get data from previous tasks
        ti = context['task_instance']
        cisa_data = ti.xcom_pull(key='cisa_data', task_ids='extract_cisa_data')
        noaa_data = ti.xcom_pull(key='noaa_data', task_ids='extract_noaa_data')
        usgs_data = ti.xcom_pull(key='usgs_data', task_ids='extract_usgs_data')
        supply_data = ti.xcom_pull(key='supply_chain_data', task_ids='extract_supply_chain_data')
        
        # Transform and standardize data structure
        transformed_data = {
            'risk_indicators': {},
            'risk_assessments': {},
            'metadata': {
                'extraction_timestamp': datetime.utcnow().isoformat(),
                'sources': ['cisa', 'noaa', 'usgs', 'supply_chain'],
                'total_indicators': 0,
                'overall_risk_scores': {}
            }
        }
        
        # Process CISA data
        if cisa_data and 'indicators' in cisa_data:
            for indicator_name, indicator_data in cisa_data['indicators'].items():
                standardized_indicator = {
                    'value': indicator_data.get('risk_score', 0),
                    'risk_level': indicator_data.get('risk_level', 'Unknown'),
                    'source': 'cisa',
                    'category': 'cybersecurity',
                    'last_updated': cisa_data.get('last_updated'),
                    'metadata': {
                        'vulnerabilities': indicator_data.get('total_vulnerabilities', 0),
                        'ransomware_threats': indicator_data.get('ransomware_associated', 0),
                        'details': indicator_data
                    }
                }
                transformed_data['risk_indicators'][f'cisa_{indicator_name}'] = standardized_indicator
            
            # Store overall cybersecurity risk assessment
            transformed_data['risk_assessments']['cybersecurity'] = {
                'overall_risk_score': cisa_data.get('overall_cybersecurity_risk', 0),
                'risk_level': cisa_data.get('risk_level', 'Unknown'),
                'source': 'cisa',
                'assessment_date': cisa_data.get('last_updated')
            }
            transformed_data['metadata']['overall_risk_scores']['cybersecurity'] = cisa_data.get('overall_cybersecurity_risk', 0)
        
        # Process NOAA data
        if noaa_data and 'indicators' in noaa_data:
            for indicator_name, indicator_data in noaa_data['indicators'].items():
                standardized_indicator = {
                    'value': indicator_data.get('weather_risk_score', indicator_data.get('climate_risk_score', 0)),
                    'risk_level': indicator_data.get('risk_level', 'Unknown'),
                    'source': 'noaa',
                    'category': 'environmental',
                    'last_updated': noaa_data.get('last_updated'),
                    'metadata': {
                        'active_alerts': indicator_data.get('total_alerts', 0),
                        'weather_events': indicator_data.get('active_weather_events', 0),
                        'details': indicator_data
                    }
                }
                transformed_data['risk_indicators'][f'noaa_{indicator_name}'] = standardized_indicator
            
            # Store overall environmental risk assessment
            transformed_data['risk_assessments']['environmental'] = {
                'overall_risk_score': noaa_data.get('overall_environmental_risk', 0),
                'risk_level': noaa_data.get('risk_level', 'Unknown'),
                'source': 'noaa',
                'assessment_date': noaa_data.get('last_updated')
            }
            transformed_data['metadata']['overall_risk_scores']['environmental'] = noaa_data.get('overall_environmental_risk', 0)
        
        # Process USGS data
        if usgs_data and 'indicators' in usgs_data:
            for indicator_name, indicator_data in usgs_data['indicators'].items():
                standardized_indicator = {
                    'value': indicator_data.get('seismic_risk_score', indicator_data.get('composite_hazard_score', 0)),
                    'risk_level': indicator_data.get('risk_level', 'Unknown'),
                    'source': 'usgs',
                    'category': 'geological',
                    'last_updated': usgs_data.get('last_updated'),
                    'metadata': {
                        'earthquakes_count': indicator_data.get('total_earthquakes', 0),
                        'significant_events': indicator_data.get('significant_earthquakes', 0),
                        'details': indicator_data
                    }
                }
                transformed_data['risk_indicators'][f'usgs_{indicator_name}'] = standardized_indicator
            
            # Store overall geological risk assessment
            transformed_data['risk_assessments']['geological'] = {
                'overall_risk_score': usgs_data.get('overall_geological_risk', 0),
                'risk_level': usgs_data.get('risk_level', 'Unknown'),
                'source': 'usgs',
                'assessment_date': usgs_data.get('last_updated')
            }
            transformed_data['metadata']['overall_risk_scores']['geological'] = usgs_data.get('overall_geological_risk', 0)
        
        # Process Supply Chain data
        if supply_data and 'indicators' in supply_data:
            for indicator_name, indicator_data in supply_data['indicators'].items():
                standardized_indicator = {
                    'value': indicator_data.get('overall_supply_chain_risk', indicator_data.get('overall_infrastructure_risk', 0)),
                    'risk_level': indicator_data.get('risk_level', 'Unknown'),
                    'source': 'supply_chain',
                    'category': 'supply_chain',
                    'last_updated': supply_data.get('last_updated'),
                    'metadata': {
                        'critical_nodes': indicator_data.get('total_critical_nodes', 0),
                        'disruption_score': indicator_data.get('composite_disruption_score', 0),
                        'details': indicator_data
                    }
                }
                transformed_data['risk_indicators'][f'supply_{indicator_name}'] = standardized_indicator
            
            # Store overall supply chain risk assessment
            transformed_data['risk_assessments']['supply_chain'] = {
                'overall_risk_score': supply_data.get('overall_supply_chain_risk', 0),
                'risk_level': supply_data.get('risk_level', 'Unknown'),
                'source': 'supply_chain',
                'assessment_date': supply_data.get('last_updated')
            }
            transformed_data['metadata']['overall_risk_scores']['supply_chain'] = supply_data.get('overall_supply_chain_risk', 0)
        
        # Calculate composite risk score
        risk_scores = list(transformed_data['metadata']['overall_risk_scores'].values())
        if risk_scores:
            composite_risk = sum(risk_scores) / len(risk_scores)
            transformed_data['metadata']['composite_risk_score'] = composite_risk
            
            # Determine overall risk level
            if composite_risk >= 75:
                overall_risk_level = "Critical"
            elif composite_risk >= 50:
                overall_risk_level = "High"
            elif composite_risk >= 25:
                overall_risk_level = "Medium"
            else:
                overall_risk_level = "Low"
                
            transformed_data['metadata']['overall_risk_level'] = overall_risk_level
        
        # Update metadata
        transformed_data['metadata']['total_indicators'] = len(transformed_data['risk_indicators'])
        
        logger.info(f"Successfully transformed {transformed_data['metadata']['total_indicators']} risk indicators")
        logger.info(f"Composite risk score: {transformed_data['metadata'].get('composite_risk_score', 0):.1f}")
        
        # Store transformed data in XCom
        ti.xcom_push(key='transformed_risk_data', value=transformed_data)
        return transformed_data
        
    except Exception as e:
        logger.error(f"Risk data transformation failed: {e}")
        raise


def load_risk_data_to_database(**context):
    """Load transformed risk data to PostgreSQL database"""
    logger.info("Starting risk data database load operation...")
    
    try:
        # Get transformed data
        ti = context['task_instance']
        transformed_data = ti.xcom_pull(key='transformed_risk_data', task_ids='transform_risk_data')
        
        if not transformed_data:
            raise ValueError("No transformed risk data available")
        
        # Connect to PostgreSQL
        postgres_hook = PostgresHook(postgres_conn_id='postgres_default')
        
        # Insert risk indicators
        indicators = transformed_data.get('risk_indicators', {})
        inserted_count = 0
        
        for indicator_name, indicator_data in indicators.items():
            # Prepare SQL statement for risk indicators
            insert_sql = """
                INSERT INTO risk_indicators 
                (name, risk_value, risk_level, source, category, last_updated, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (name, source) 
                DO UPDATE SET 
                    risk_value = EXCLUDED.risk_value,
                    risk_level = EXCLUDED.risk_level,
                    last_updated = EXCLUDED.last_updated,
                    metadata = EXCLUDED.metadata;
            """
            
            # Execute insert
            postgres_hook.run(
                insert_sql,
                parameters=[
                    indicator_name,
                    indicator_data.get('value'),
                    indicator_data.get('risk_level'),
                    indicator_data.get('source'),
                    indicator_data.get('category'),
                    indicator_data.get('last_updated'),
                    indicator_data.get('metadata', {})
                ]
            )
            inserted_count += 1
        
        # Insert risk assessments
        assessments = transformed_data.get('risk_assessments', {})
        for assessment_name, assessment_data in assessments.items():
            assessment_sql = """
                INSERT INTO risk_assessments 
                (assessment_type, overall_risk_score, risk_level, source, assessment_date, details)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (assessment_type, source, assessment_date) 
                DO UPDATE SET 
                    overall_risk_score = EXCLUDED.overall_risk_score,
                    risk_level = EXCLUDED.risk_level,
                    details = EXCLUDED.details;
            """
            
            postgres_hook.run(
                assessment_sql,
                parameters=[
                    assessment_name,
                    assessment_data.get('overall_risk_score'),
                    assessment_data.get('risk_level'),
                    assessment_data.get('source'),
                    assessment_data.get('assessment_date'),
                    assessment_data
                ]
            )
        
        logger.info(f"Successfully loaded {inserted_count} risk indicators and {len(assessments)} assessments to database")
        
        # Log pipeline execution
        log_sql = """
            INSERT INTO etl_pipeline_logs 
            (pipeline_name, execution_timestamp, status, records_processed, execution_details)
            VALUES (%s, %s, %s, %s, %s);
        """
        
        postgres_hook.run(
            log_sql,
            parameters=[
                'risk_intelligence_pipeline',
                datetime.utcnow(),
                'success',
                inserted_count,
                {
                    'sources': transformed_data['metadata']['sources'],
                    'composite_risk_score': transformed_data['metadata'].get('composite_risk_score', 0),
                    'risk_level': transformed_data['metadata'].get('overall_risk_level', 'Unknown')
                }
            ]
        )
        
        return {'status': 'success', 'records_loaded': inserted_count, 'assessments_loaded': len(assessments)}
        
    except Exception as e:
        logger.error(f"Risk data database load failed: {e}")
        raise


def load_risk_data_to_cache(**context):
    """Load transformed risk data to Redis cache"""
    logger.info("Starting risk data cache load operation...")
    
    try:
        # Get transformed data
        ti = context['task_instance']
        transformed_data = ti.xcom_pull(key='transformed_risk_data', task_ids='transform_risk_data')
        
        if not transformed_data:
            raise ValueError("No transformed risk data available")
        
        # Connect to Redis
        redis_hook = RedisHook(redis_conn_id='redis_default')
        redis_client = redis_hook.get_conn()
        
        # Cache individual risk indicators
        indicators = transformed_data.get('risk_indicators', {})
        cached_count = 0
        
        for indicator_name, indicator_data in indicators.items():
            # Create cache key
            cache_key = f"risk_indicator:{indicator_name}"
            
            # Cache with 6-hour TTL (since pipeline runs every 4 hours)
            redis_client.setex(
                cache_key,
                21600,  # 6 hours
                str(indicator_data)
            )
            cached_count += 1
        
        # Cache risk assessments
        assessments = transformed_data.get('risk_assessments', {})
        for assessment_name, assessment_data in assessments.items():
            cache_key = f"risk_assessment:{assessment_name}"
            redis_client.setex(
                cache_key,
                21600,  # 6 hours
                str(assessment_data)
            )
        
        # Cache aggregated risk data
        redis_client.setex(
            "risk_intelligence:all",
            21600,
            str(transformed_data)
        )
        
        # Cache composite risk score
        if 'composite_risk_score' in transformed_data['metadata']:
            redis_client.setex(
                "risk_intelligence:composite_score",
                21600,
                str(transformed_data['metadata']['composite_risk_score'])
            )
        
        # Cache metadata
        redis_client.setex(
            "risk_intelligence:metadata",
            21600,
            str(transformed_data.get('metadata', {}))
        )
        
        logger.info(f"Successfully cached {cached_count} risk indicators and {len(assessments)} assessments to Redis")
        return {'status': 'success', 'records_cached': cached_count, 'assessments_cached': len(assessments)}
        
    except Exception as e:
        logger.error(f"Risk data cache load failed: {e}")
        raise


def risk_data_quality_check(**context):
    """Perform data quality checks on risk intelligence data"""
    logger.info("Starting risk data quality checks...")
    
    try:
        # Get transformed data
        ti = context['task_instance']
        transformed_data = ti.xcom_pull(key='transformed_risk_data', task_ids='transform_risk_data')
        
        if not transformed_data:
            raise ValueError("No transformed risk data available")
        
        quality_issues = []
        indicators = transformed_data.get('risk_indicators', {})
        assessments = transformed_data.get('risk_assessments', {})
        
        # Check for missing risk values
        for indicator_name, indicator_data in indicators.items():
            if indicator_data.get('value') is None:
                quality_issues.append(f"Missing risk value for {indicator_name}")
            
            if not indicator_data.get('risk_level'):
                quality_issues.append(f"Missing risk level for {indicator_name}")
            
            if not indicator_data.get('source'):
                quality_issues.append(f"Missing source for {indicator_name}")
        
        # Check risk value ranges (0-100)
        for indicator_name, indicator_data in indicators.items():
            value = indicator_data.get('value')
            if value is not None:
                if value < 0 or value > 100:
                    quality_issues.append(f"Risk value out of range (0-100): {value} for {indicator_name}")
        
        # Check data freshness (within last 24 hours)
        current_time = datetime.utcnow()
        stale_threshold = current_time - timedelta(hours=24)
        
        for indicator_name, indicator_data in indicators.items():
            last_updated = indicator_data.get('last_updated')
            if last_updated:
                try:
                    update_time = datetime.fromisoformat(last_updated.replace('Z', '+00:00'))
                    if update_time < stale_threshold:
                        quality_issues.append(f"Stale risk data for {indicator_name} (last updated: {last_updated})")
                except:
                    quality_issues.append(f"Invalid timestamp format for {indicator_name}")
        
        # Check assessment completeness
        expected_categories = ['cybersecurity', 'environmental', 'geological', 'supply_chain']
        missing_assessments = [cat for cat in expected_categories if cat not in assessments]
        if missing_assessments:
            quality_issues.append(f"Missing risk assessments for categories: {missing_assessments}")
        
        # Check composite risk score calculation
        metadata = transformed_data.get('metadata', {})
        if 'composite_risk_score' not in metadata:
            quality_issues.append("Missing composite risk score calculation")
        
        # Log quality check results
        if quality_issues:
            logger.warning(f"Risk data quality issues found: {quality_issues}")
            return {'status': 'warning', 'issues': quality_issues}
        else:
            logger.info("All risk data quality checks passed")
            return {'status': 'success', 'issues': []}
            
    except Exception as e:
        logger.error(f"Risk data quality check failed: {e}")
        raise


# Define task dependencies
extract_cisa_task = PythonOperator(
    task_id='extract_cisa_data',
    python_callable=extract_cisa_data,
    dag=dag,
)

extract_noaa_task = PythonOperator(
    task_id='extract_noaa_data',
    python_callable=extract_noaa_data,
    dag=dag,
)

extract_usgs_task = PythonOperator(
    task_id='extract_usgs_data',
    python_callable=extract_usgs_data,
    dag=dag,
)

extract_supply_chain_task = PythonOperator(
    task_id='extract_supply_chain_data',
    python_callable=extract_supply_chain_data,
    dag=dag,
)

transform_risk_task = PythonOperator(
    task_id='transform_risk_data',
    python_callable=transform_risk_data,
    dag=dag,
)

load_risk_db_task = PythonOperator(
    task_id='load_risk_data_to_database',
    python_callable=load_risk_data_to_database,
    dag=dag,
)

load_risk_cache_task = PythonOperator(
    task_id='load_risk_data_to_cache',
    python_callable=load_risk_data_to_cache,
    dag=dag,
)

risk_quality_check_task = PythonOperator(
    task_id='risk_data_quality_check',
    python_callable=risk_data_quality_check,
    dag=dag,
)

# Set task dependencies
# Parallel extraction
[extract_cisa_task, extract_noaa_task, extract_usgs_task, extract_supply_chain_task] >> transform_risk_task

# Parallel loading after transformation
transform_risk_task >> [load_risk_db_task, load_risk_cache_task]

# Quality check after loading
[load_risk_db_task, load_risk_cache_task] >> risk_quality_check_task