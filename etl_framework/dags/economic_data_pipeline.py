"""
Economic Data ETL Pipeline
Automated data pipeline for extracting, transforming, and loading economic data
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
    'economic_data_pipeline',
    default_args=default_args,
    description='Extract, transform, and load economic data from multiple sources',
    schedule_interval=timedelta(hours=6),  # Run every 6 hours
    max_active_runs=1,
    catchup=False,
    tags=['economic', 'data', 'etl'],
)


def extract_fred_data(**context):
    """Extract data from FRED API"""
    import sys
    import os
    sys.path.append('/Users/omoshola/Documents/riskxx/backend')
    
    from src.data.sources.fred import get_key_indicators
    
    logger.info("Starting FRED data extraction...")
    
    try:
        # Run async function in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        fred_data = loop.run_until_complete(get_key_indicators())
        loop.close()
        
        if fred_data and 'indicators' in fred_data:
            logger.info(f"Successfully extracted {fred_data.get('count', 0)} FRED indicators")
            
            # Store data in XCom for downstream tasks
            context['task_instance'].xcom_push(key='fred_data', value=fred_data)
            return fred_data
        else:
            raise ValueError("No FRED data received")
            
    except Exception as e:
        logger.error(f"FRED data extraction failed: {e}")
        raise


def extract_bea_data(**context):
    """Extract data from BEA API"""
    import sys
    import os
    sys.path.append('/Users/omoshola/Documents/riskxx/backend')
    
    from src.data.sources.bea import get_economic_accounts
    
    logger.info("Starting BEA data extraction...")
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        bea_data = loop.run_until_complete(get_economic_accounts())
        loop.close()
        
        if bea_data and 'indicators' in bea_data:
            logger.info(f"Successfully extracted {bea_data.get('count', 0)} BEA indicators")
            context['task_instance'].xcom_push(key='bea_data', value=bea_data)
            return bea_data
        else:
            raise ValueError("No BEA data received")
            
    except Exception as e:
        logger.error(f"BEA data extraction failed: {e}")
        raise


def extract_bls_data(**context):
    """Extract data from BLS API"""
    import sys
    import os
    sys.path.append('/Users/omoshola/Documents/riskxx/backend')
    
    from src.data.sources.bls import get_labor_statistics
    
    logger.info("Starting BLS data extraction...")
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        bls_data = loop.run_until_complete(get_labor_statistics())
        loop.close()
        
        if bls_data and 'indicators' in bls_data:
            logger.info(f"Successfully extracted {bls_data.get('count', 0)} BLS indicators")
            context['task_instance'].xcom_push(key='bls_data', value=bls_data)
            return bls_data
        else:
            raise ValueError("No BLS data received")
            
    except Exception as e:
        logger.error(f"BLS data extraction failed: {e}")
        raise


def extract_census_data(**context):
    """Extract data from Census Bureau API"""
    import sys
    import os
    sys.path.append('/Users/omoshola/Documents/riskxx/backend')
    
    from src.data.sources.census import get_population_data, get_household_income
    
    logger.info("Starting Census data extraction...")
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Get both population and income data
        pop_data = loop.run_until_complete(get_population_data())
        income_data = loop.run_until_complete(get_household_income())
        
        loop.close()
        
        census_data = {
            'population': pop_data,
            'income': income_data,
            'source': 'census',
            'last_updated': datetime.utcnow().isoformat()
        }
        
        logger.info("Successfully extracted Census data")
        context['task_instance'].xcom_push(key='census_data', value=census_data)
        return census_data
        
    except Exception as e:
        logger.error(f"Census data extraction failed: {e}")
        raise


def transform_economic_data(**context):
    """Transform and normalize economic data from all sources"""
    logger.info("Starting economic data transformation...")
    
    try:
        # Get data from previous tasks
        ti = context['task_instance']
        fred_data = ti.xcom_pull(key='fred_data', task_ids='extract_fred_data')
        bea_data = ti.xcom_pull(key='bea_data', task_ids='extract_bea_data')
        bls_data = ti.xcom_pull(key='bls_data', task_ids='extract_bls_data')
        census_data = ti.xcom_pull(key='census_data', task_ids='extract_census_data')
        
        # Transform and standardize data structure
        transformed_data = {
            'economic_indicators': {},
            'metadata': {
                'extraction_timestamp': datetime.utcnow().isoformat(),
                'sources': ['fred', 'bea', 'bls', 'census'],
                'total_indicators': 0
            }
        }
        
        # Process FRED data
        if fred_data and 'indicators' in fred_data:
            for indicator_name, indicator_data in fred_data['indicators'].items():
                standardized_indicator = {
                    'value': indicator_data.get('value'),
                    'units': indicator_data.get('units', ''),
                    'date': indicator_data.get('date', ''),
                    'source': 'fred',
                    'category': 'economic',
                    'last_updated': fred_data.get('last_updated'),
                    'metadata': {
                        'title': indicator_data.get('title', ''),
                        'frequency': indicator_data.get('frequency', '')
                    }
                }
                transformed_data['economic_indicators'][f'fred_{indicator_name}'] = standardized_indicator
        
        # Process BEA data
        if bea_data and 'indicators' in bea_data:
            for indicator_name, indicator_data in bea_data['indicators'].items():
                standardized_indicator = {
                    'value': indicator_data.get('value'),
                    'units': indicator_data.get('units', ''),
                    'date': indicator_data.get('time_period', ''),
                    'source': 'bea',
                    'category': 'economic',
                    'last_updated': bea_data.get('last_updated'),
                    'metadata': {
                        'description': indicator_data.get('line_description', ''),
                        'frequency': indicator_data.get('frequency', '')
                    }
                }
                transformed_data['economic_indicators'][f'bea_{indicator_name}'] = standardized_indicator
        
        # Process BLS data
        if bls_data and 'indicators' in bls_data:
            for indicator_name, indicator_data in bls_data['indicators'].items():
                standardized_indicator = {
                    'value': indicator_data.get('value'),
                    'units': indicator_data.get('units', ''),
                    'date': indicator_data.get('period', ''),
                    'source': 'bls',
                    'category': 'labor',
                    'last_updated': bls_data.get('last_updated'),
                    'metadata': {
                        'series_id': indicator_data.get('series_id', ''),
                        'yoy_change': indicator_data.get('yoy_change_percent')
                    }
                }
                transformed_data['economic_indicators'][f'bls_{indicator_name}'] = standardized_indicator
        
        # Process Census data
        if census_data:
            if census_data.get('population'):
                pop_data = census_data['population']
                standardized_indicator = {
                    'value': pop_data.get('population'),
                    'units': 'people',
                    'date': pop_data.get('year', ''),
                    'source': 'census',
                    'category': 'demographic',
                    'last_updated': census_data.get('last_updated'),
                    'metadata': {
                        'name': pop_data.get('name', '')
                    }
                }
                transformed_data['economic_indicators']['census_population'] = standardized_indicator
            
            if census_data.get('income'):
                income_data = census_data['income']
                standardized_indicator = {
                    'value': income_data.get('median_household_income'),
                    'units': income_data.get('currency', 'USD'),
                    'date': income_data.get('year', ''),
                    'source': 'census',
                    'category': 'economic',
                    'last_updated': census_data.get('last_updated'),
                    'metadata': {}
                }
                transformed_data['economic_indicators']['census_median_income'] = standardized_indicator
        
        # Update metadata
        transformed_data['metadata']['total_indicators'] = len(transformed_data['economic_indicators'])
        
        logger.info(f"Successfully transformed {transformed_data['metadata']['total_indicators']} economic indicators")
        
        # Store transformed data in XCom
        ti.xcom_push(key='transformed_data', value=transformed_data)
        return transformed_data
        
    except Exception as e:
        logger.error(f"Data transformation failed: {e}")
        raise


def load_to_database(**context):
    """Load transformed data to PostgreSQL database"""
    logger.info("Starting database load operation...")
    
    try:
        # Get transformed data with risk assessment
        ti = context['task_instance']
        transformed_data = ti.xcom_pull(key='transformed_data_with_risk', task_ids='calculate_risk_scores')
        
        if not transformed_data:
            raise ValueError("No transformed data available")
        
        # Connect to PostgreSQL
        postgres_hook = PostgresHook(postgres_conn_id='postgres_default')
        
        # Insert economic indicators
        indicators = transformed_data.get('economic_indicators', {})
        inserted_count = 0
        
        for indicator_name, indicator_data in indicators.items():
            # Prepare SQL statement
            insert_sql = """
                INSERT INTO economic_indicators 
                (name, value, units, date_recorded, source, category, last_updated, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (name, source, date_recorded) 
                DO UPDATE SET 
                    value = EXCLUDED.value,
                    last_updated = EXCLUDED.last_updated,
                    metadata = EXCLUDED.metadata;
            """
            
            # Execute insert
            postgres_hook.run(
                insert_sql,
                parameters=[
                    indicator_name,
                    indicator_data.get('value'),
                    indicator_data.get('units'),
                    indicator_data.get('date'),
                    indicator_data.get('source'),
                    indicator_data.get('category'),
                    indicator_data.get('last_updated'),
                    indicator_data.get('metadata', {})
                ]
            )
            inserted_count += 1
        
        logger.info(f"Successfully loaded {inserted_count} indicators to database")
        
        # Log pipeline execution
        log_sql = """
            INSERT INTO etl_pipeline_logs 
            (pipeline_name, execution_timestamp, status, records_processed, execution_details)
            VALUES (%s, %s, %s, %s, %s);
        """
        
        postgres_hook.run(
            log_sql,
            parameters=[
                'economic_data_pipeline',
                datetime.utcnow(),
                'success',
                inserted_count,
                {'sources': transformed_data['metadata']['sources']}
            ]
        )
        
        return {'status': 'success', 'records_loaded': inserted_count}
        
    except Exception as e:
        logger.error(f"Database load failed: {e}")
        
        # Log failure
        try:
            postgres_hook = PostgresHook(postgres_conn_id='postgres_default')
            log_sql = """
                INSERT INTO etl_pipeline_logs 
                (pipeline_name, execution_timestamp, status, records_processed, execution_details)
                VALUES (%s, %s, %s, %s, %s);
            """
            postgres_hook.run(
                log_sql,
                parameters=[
                    'economic_data_pipeline',
                    datetime.utcnow(),
                    'failed',
                    0,
                    {'error': str(e)}
                ]
            )
        except:
            pass
        
        raise


def load_to_cache(**context):
    """Load transformed data to Redis cache"""
    logger.info("Starting cache load operation...")
    
    try:
        # Get transformed data with risk assessment
        ti = context['task_instance']
        transformed_data = ti.xcom_pull(key='transformed_data_with_risk', task_ids='calculate_risk_scores')
        
        if not transformed_data:
            raise ValueError("No transformed data available")
        
        # Connect to Redis
        redis_hook = RedisHook(redis_conn_id='redis_default')
        redis_client = redis_hook.get_conn()
        
        # Cache individual indicators
        indicators = transformed_data.get('economic_indicators', {})
        cached_count = 0
        
        for indicator_name, indicator_data in indicators.items():
            # Create cache key
            cache_key = f"economic_indicator:{indicator_name}"
            
            # Cache with 24-hour TTL
            redis_client.setex(
                cache_key,
                86400,  # 24 hours
                str(indicator_data)
            )
            cached_count += 1
        
        # Cache aggregated data
        redis_client.setex(
            "economic_indicators:all",
            86400,
            str(transformed_data)
        )
        
        # Cache metadata
        redis_client.setex(
            "economic_indicators:metadata",
            86400,
            str(transformed_data.get('metadata', {}))
        )
        
        logger.info(f"Successfully cached {cached_count} indicators to Redis")
        return {'status': 'success', 'records_cached': cached_count}
        
    except Exception as e:
        logger.error(f"Cache load failed: {e}")
        raise


def calculate_risk_scores(**context):
    """Calculate risk scores using ML models from transformed economic data"""
    logger.info("Starting risk score calculation...")
    
    try:
        # Get transformed data
        ti = context['task_instance']
        transformed_data = ti.xcom_pull(key='transformed_data', task_ids='transform_data')
        
        if not transformed_data:
            raise ValueError("No transformed data available for risk calculation")
        
        # Initialize model server for risk calculations
        import asyncio
        import sys
        import os
        sys.path.append('/Users/omoshola/Documents/riskxx/backend')
        
        from src.ml.serving.model_server import ModelServer
        
        # Create event loop for async operations
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Initialize model server
            model_server = ModelServer()
            
            # Get comprehensive risk assessment
            comprehensive_assessment = loop.run_until_complete(
                model_server.get_comprehensive_risk_assessment()
            )
            
            # Get individual model predictions
            individual_predictions = {}
            
            # Recession probability
            recession_pred = loop.run_until_complete(
                model_server.predict_recession_probability()
            )
            individual_predictions['recession'] = recession_pred
            
            # Supply chain risk
            supply_chain_pred = loop.run_until_complete(
                model_server.predict_supply_chain_risk()
            )
            individual_predictions['supply_chain'] = supply_chain_pred
            
            # Market volatility
            market_vol_pred = loop.run_until_complete(
                model_server.predict_market_volatility()
            )
            individual_predictions['market_volatility'] = market_vol_pred
            
            # Geopolitical risk
            geopolitical_pred = loop.run_until_complete(
                model_server.predict_geopolitical_risk()
            )
            individual_predictions['geopolitical'] = geopolitical_pred
            
        finally:
            loop.close()
        
        # Extract key risk metrics
        risk_summary = {
            'overall_risk_score': comprehensive_assessment.get('overall_risk_score', 0.0),
            'recession_probability': individual_predictions.get('recession', {}).get('prediction', {}).get('probability', 0.0),
            'supply_chain_risk': individual_predictions.get('supply_chain', {}).get('prediction', {}).get('risk_score', 0.0),
            'market_volatility_score': individual_predictions.get('market_volatility', {}).get('prediction', {}).get('volatility_score', 0.0),
            'geopolitical_risk_score': individual_predictions.get('geopolitical', {}).get('prediction', {}).get('risk_score', 0.0),
            'last_updated': datetime.utcnow().isoformat(),
            'model_confidence': comprehensive_assessment.get('confidence', 0.0),
            'risk_factors': comprehensive_assessment.get('factors', {}),
            'risk_trend': comprehensive_assessment.get('trend', 'stable')
        }
        
        # Add risk scores to transformed data
        transformed_data['risk_assessment'] = {
            'summary': risk_summary,
            'detailed_predictions': individual_predictions,
            'comprehensive_assessment': comprehensive_assessment,
            'calculation_timestamp': datetime.utcnow().isoformat()
        }
        
        # Update metadata to include risk calculation
        transformed_data['metadata']['risk_calculation_status'] = 'completed'
        transformed_data['metadata']['models_used'] = [
            'recession_predictor',
            'supply_chain_risk_model', 
            'market_volatility_model',
            'geopolitical_risk_model'
        ]
        
        logger.info(f"Risk calculation completed. Overall risk score: {risk_summary['overall_risk_score']:.3f}")
        
        # Push enhanced data to XCom
        ti.xcom_push(key='transformed_data_with_risk', value=transformed_data)
        
        return {
            'status': 'success',
            'overall_risk_score': risk_summary['overall_risk_score'],
            'risk_factors_calculated': len(risk_summary['risk_factors'])
        }
        
    except Exception as e:
        logger.error(f"Risk calculation failed: {e}")
        
        # Risk calculation failed - pipeline must halt
        logger.error("Risk calculation failed - terminating pipeline to prevent incomplete data")
        raise RuntimeError("Risk calculation failed - cannot proceed with incomplete assessment")
                    'risk_factors': {},
                    'risk_trend': 'unknown'
                },
                'calculation_timestamp': datetime.utcnow().isoformat(),
                'error': str(e)
            }
            transformed_data['metadata']['risk_calculation_status'] = 'failed'
            ti.xcom_push(key='transformed_data_with_risk', value=transformed_data)
        
        raise


# Define the DAG
dag = DAG(
    'economic_data_pipeline',
    default_args={
        'owner': 'riskx-team',
        'depends_on_past': False,
        'start_date': datetime(2024, 1, 1),
        'email_on_failure': False,
        'email_on_retry': False,
        'retries': 2,
        'retry_delay': timedelta(minutes=5),
        'max_active_runs': 1
    },
    description='Economic data extraction, transformation, risk calculation, and loading pipeline',
    schedule_interval=timedelta(minutes=30),  # Run every 30 minutes
    catchup=False,
    tags=['economic-data', 'etl', 'risk-assessment', 'ml-predictions']
)

# Define tasks
extract_fred_task = PythonOperator(
    task_id='extract_fred_data',
    python_callable=extract_fred_data,
    dag=dag,
    pool='api_pool',
    priority_weight=10
)

extract_bea_task = PythonOperator(
    task_id='extract_bea_data',
    python_callable=extract_bea_data,
    dag=dag,
    pool='api_pool',
    priority_weight=8
)

extract_bls_task = PythonOperator(
    task_id='extract_bls_data',
    python_callable=extract_bls_data,
    dag=dag,
    pool='api_pool',
    priority_weight=8
)

extract_census_task = PythonOperator(
    task_id='extract_census_data',
    python_callable=extract_census_data,
    dag=dag,
    pool='api_pool',
    priority_weight=6
)

transform_task = PythonOperator(
    task_id='transform_data',
    python_callable=transform_data,
    dag=dag,
    priority_weight=15
)

calculate_risk_task = PythonOperator(
    task_id='calculate_risk_scores',
    python_callable=calculate_risk_scores,
    dag=dag,
    priority_weight=20
)

load_db_task = PythonOperator(
    task_id='load_to_database',
    python_callable=load_to_database,
    dag=dag,
    priority_weight=25
)

load_cache_task = PythonOperator(
    task_id='load_to_cache',
    python_callable=load_to_cache,
    dag=dag,
    priority_weight=25
)

# Define task dependencies
# Extract tasks run in parallel
extract_tasks = [extract_fred_task, extract_bea_task, extract_bls_task, extract_census_task]

# Transform after all extractions complete
for extract_task in extract_tasks:
    extract_task >> transform_task

# Calculate risk scores after transformation
transform_task >> calculate_risk_task

# Load to database and cache in parallel after risk calculation
calculate_risk_task >> [load_db_task, load_cache_task]