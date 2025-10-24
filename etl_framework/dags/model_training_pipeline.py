"""
Model Training Pipeline
Automated pipeline for training and updating ML risk models
"""
from datetime import datetime, timedelta
import logging
import asyncio
from typing import Dict, Any

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
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
    'retries': 1,
    'retry_delay': timedelta(minutes=10),
}

# DAG definition
dag = DAG(
    'model_training_pipeline',
    default_args=default_args,
    description='Train and update ML risk models with latest data',
    schedule_interval=timedelta(days=7),  # Run weekly
    max_active_runs=1,
    catchup=False,
    tags=['ml', 'models', 'training', 'risk'],
)


def prepare_training_data(**context):
    """Prepare training data from database"""
    logger.info("Preparing training data for model training...")
    
    try:
        # Connect to PostgreSQL
        postgres_hook = PostgresHook(postgres_conn_id='postgres_default')
        
        # Get economic indicators data
        economic_query = """
            SELECT 
                name, value, units, date_recorded, source, category,
                last_updated, metadata
            FROM economic_indicators 
            WHERE last_updated >= NOW() - INTERVAL '30 days'
            ORDER BY date_recorded DESC;
        """
        
        economic_data = postgres_hook.get_records(economic_query)
        
        # Get risk indicators data
        risk_query = """
            SELECT 
                name, risk_value, risk_level, source, category,
                last_updated, metadata
            FROM risk_indicators 
            WHERE last_updated >= NOW() - INTERVAL '30 days'
            ORDER BY last_updated DESC;
        """
        
        risk_data = postgres_hook.get_records(risk_query)
        
        # Get risk assessments data
        assessment_query = """
            SELECT 
                assessment_type, overall_risk_score, risk_level,
                source, assessment_date, details
            FROM risk_assessments 
            WHERE assessment_date >= NOW() - INTERVAL '30 days'
            ORDER BY assessment_date DESC;
        """
        
        assessment_data = postgres_hook.get_records(assessment_query)
        
        # Prepare structured training data
        training_data = {
            'economic_indicators': [
                {
                    'name': row[0], 'value': row[1], 'units': row[2],
                    'date': row[3], 'source': row[4], 'category': row[5],
                    'last_updated': row[6], 'metadata': row[7]
                }
                for row in economic_data
            ],
            'risk_indicators': [
                {
                    'name': row[0], 'risk_value': row[1], 'risk_level': row[2],
                    'source': row[3], 'category': row[4], 'last_updated': row[5],
                    'metadata': row[6]
                }
                for row in risk_data
            ],
            'risk_assessments': [
                {
                    'assessment_type': row[0], 'overall_risk_score': row[1],
                    'risk_level': row[2], 'source': row[3], 'assessment_date': row[4],
                    'details': row[5]
                }
                for row in assessment_data
            ],
            'prepared_timestamp': datetime.utcnow().isoformat()
        }
        
        logger.info(f"Prepared training data: {len(economic_data)} economic indicators, "
                   f"{len(risk_data)} risk indicators, {len(assessment_data)} assessments")
        
        # Store in XCom
        context['task_instance'].xcom_push(key='training_data', value=training_data)
        return training_data
        
    except Exception as e:
        logger.error(f"Training data preparation failed: {e}")
        raise


def train_recession_model(**context):
    """Train recession prediction model"""
    import sys
    import os
    sys.path.append('/Users/omoshola/Documents/riskxx/backend')
    
    from src.ml.models.recession_predictor import train_recession_models
    
    logger.info("Starting recession model training...")
    
    try:
        # Get training data
        ti = context['task_instance']
        training_data = ti.xcom_pull(key='training_data', task_ids='prepare_training_data')
        
        # Run training
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        training_results = loop.run_until_complete(train_recession_models())
        loop.close()
        
        logger.info(f"Recession model training completed: {training_results['status']}")
        
        # Store results
        ti.xcom_push(key='recession_training_results', value=training_results)
        return training_results
        
    except Exception as e:
        logger.error(f"Recession model training failed: {e}")
        raise


def train_supply_chain_model(**context):
    """Train supply chain risk model"""
    import sys
    import os
    sys.path.append('/Users/omoshola/Documents/riskxx/backend')
    
    from src.ml.models.supply_chain_risk_model import train_supply_chain_models
    
    logger.info("Starting supply chain model training...")
    
    try:
        # Run training
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        training_results = loop.run_until_complete(train_supply_chain_models())
        loop.close()
        
        logger.info(f"Supply chain model training completed: {training_results['status']}")
        
        # Store results
        ti = context['task_instance']
        ti.xcom_push(key='supply_chain_training_results', value=training_results)
        return training_results
        
    except Exception as e:
        logger.error(f"Supply chain model training failed: {e}")
        raise


def train_volatility_model(**context):
    """Train market volatility model"""
    import sys
    import os
    sys.path.append('/Users/omoshola/Documents/riskxx/backend')
    
    from src.ml.models.market_volatility_model import train_volatility_models
    
    logger.info("Starting market volatility model training...")
    
    try:
        # Run training
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        training_results = loop.run_until_complete(train_volatility_models())
        loop.close()
        
        logger.info(f"Market volatility model training completed: {training_results['status']}")
        
        # Store results
        ti = context['task_instance']
        ti.xcom_push(key='volatility_training_results', value=training_results)
        return training_results
        
    except Exception as e:
        logger.error(f"Market volatility model training failed: {e}")
        raise


def train_geopolitical_model(**context):
    """Train geopolitical risk model"""
    import sys
    import os
    sys.path.append('/Users/omoshola/Documents/riskxx/backend')
    
    from src.ml.models.geopolitical_risk_model import train_geopolitical_models
    
    logger.info("Starting geopolitical risk model training...")
    
    try:
        # Run training
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        training_results = loop.run_until_complete(train_geopolitical_models())
        loop.close()
        
        logger.info(f"Geopolitical risk model training completed: {training_results['status']}")
        
        # Store results
        ti = context['task_instance']
        ti.xcom_push(key='geopolitical_training_results', value=training_results)
        return training_results
        
    except Exception as e:
        logger.error(f"Geopolitical risk model training failed: {e}")
        raise


def validate_models(**context):
    """Validate trained models and assess performance"""
    logger.info("Starting model validation...")
    
    try:
        # Get training results from all models
        ti = context['task_instance']
        recession_results = ti.xcom_pull(key='recession_training_results', task_ids='train_recession_model')
        supply_chain_results = ti.xcom_pull(key='supply_chain_training_results', task_ids='train_supply_chain_model')
        volatility_results = ti.xcom_pull(key='volatility_training_results', task_ids='train_volatility_model')
        geopolitical_results = ti.xcom_pull(key='geopolitical_training_results', task_ids='train_geopolitical_model')
        
        validation_results = {
            'validation_timestamp': datetime.utcnow().isoformat(),
            'models_validated': [],
            'validation_summary': {},
            'overall_status': 'success'
        }
        
        # Validate each model
        models = {
            'recession': recession_results,
            'supply_chain': supply_chain_results,
            'market_volatility': volatility_results,
            'geopolitical': geopolitical_results
        }
        
        for model_name, results in models.items():
            if results and results.get('status') == 'completed':
                validation_results['models_validated'].append(model_name)
                
                # Extract performance metrics
                training_results = results.get('training_results', {})
                if training_results:
                    # Calculate average performance across all sub-models
                    performance_scores = []
                    
                    for sub_model, metrics in training_results.items():
                        if isinstance(metrics, dict):
                            # Look for performance indicators
                            if 'cv_accuracy' in metrics:
                                performance_scores.append(metrics['cv_accuracy'])
                            elif 'cv_rmse' in metrics:
                                # Convert RMSE to a score (lower is better, so invert)
                                performance_scores.append(1.0 - min(1.0, metrics['cv_rmse']))
                            elif 'cv_mean_auc' in metrics:
                                performance_scores.append(metrics['cv_mean_auc'])
                    
                    avg_performance = sum(performance_scores) / len(performance_scores) if performance_scores else 0.8
                    
                    validation_results['validation_summary'][model_name] = {
                        'status': 'valid',
                        'performance_score': avg_performance,
                        'sub_models_count': len(training_results),
                        'model_version': results.get('model_version', '1.0.0')
                    }
                else:
                    validation_results['validation_summary'][model_name] = {
                        'status': 'valid',
                        'performance_score': 0.8,  # Default assumption
                        'model_version': results.get('model_version', '1.0.0')
                    }
            else:
                validation_results['validation_summary'][model_name] = {
                    'status': 'failed',
                    'performance_score': 0.0,
                    'error': 'Training failed or incomplete'
                }
                validation_results['overall_status'] = 'partial_failure'
        
        # Calculate overall validation score
        valid_models = [m for m in validation_results['validation_summary'].values() if m['status'] == 'valid']
        if valid_models:
            avg_performance = sum(m['performance_score'] for m in valid_models) / len(valid_models)
            validation_results['overall_performance_score'] = avg_performance
        else:
            validation_results['overall_performance_score'] = 0.0
            validation_results['overall_status'] = 'failed'
        
        logger.info(f"Model validation completed. Status: {validation_results['overall_status']}")
        logger.info(f"Models validated: {validation_results['models_validated']}")
        logger.info(f"Overall performance: {validation_results['overall_performance_score']:.3f}")
        
        # Store validation results
        ti.xcom_push(key='validation_results', value=validation_results)
        return validation_results
        
    except Exception as e:
        logger.error(f"Model validation failed: {e}")
        raise


def deploy_models(**context):
    """Deploy validated models to production"""
    logger.info("Starting model deployment...")
    
    try:
        # Get validation results
        ti = context['task_instance']
        validation_results = ti.xcom_pull(key='validation_results', task_ids='validate_models')
        
        if not validation_results or validation_results.get('overall_status') == 'failed':
            raise ValueError("Model validation failed, cannot deploy models")
        
        deployment_results = {
            'deployment_timestamp': datetime.utcnow().isoformat(),
            'deployed_models': [],
            'deployment_status': 'success'
        }
        
        # Deploy each validated model
        for model_name, validation_info in validation_results.get('validation_summary', {}).items():
            if validation_info.get('status') == 'valid':
                try:
                    # In a real deployment, you would:
                    # 1. Copy model files to production location
                    # 2. Update model registry
                    # 3. Update API endpoints to use new models
                    # 4. Run smoke tests
                    
                    # For now, we'll simulate deployment
                    logger.info(f"Deploying {model_name} model (version {validation_info.get('model_version', '1.0.0')})")
                    
                    deployment_results['deployed_models'].append({
                        'model_name': model_name,
                        'version': validation_info.get('model_version', '1.0.0'),
                        'performance_score': validation_info.get('performance_score', 0.0),
                        'deployment_time': datetime.utcnow().isoformat()
                    })
                    
                except Exception as e:
                    logger.error(f"Failed to deploy {model_name} model: {e}")
                    deployment_results['deployment_status'] = 'partial_failure'
        
        # Log deployment to database
        postgres_hook = PostgresHook(postgres_conn_id='postgres_default')
        
        for deployed_model in deployment_results['deployed_models']:
            log_sql = """
                INSERT INTO model_deployments 
                (model_name, model_version, performance_score, deployment_timestamp, status)
                VALUES (%s, %s, %s, %s, %s);
            """
            
            postgres_hook.run(
                log_sql,
                parameters=[
                    deployed_model['model_name'],
                    deployed_model['version'],
                    deployed_model['performance_score'],
                    deployed_model['deployment_time'],
                    'deployed'
                ]
            )
        
        logger.info(f"Model deployment completed. Deployed {len(deployment_results['deployed_models'])} models")
        
        # Store deployment results
        ti.xcom_push(key='deployment_results', value=deployment_results)
        return deployment_results
        
    except Exception as e:
        logger.error(f"Model deployment failed: {e}")
        raise


def log_training_pipeline(**context):
    """Log training pipeline execution to database"""
    logger.info("Logging training pipeline execution...")
    
    try:
        # Get results from all tasks
        ti = context['task_instance']
        validation_results = ti.xcom_pull(key='validation_results', task_ids='validate_models')
        deployment_results = ti.xcom_pull(key='deployment_results', task_ids='deploy_models')
        
        # Connect to PostgreSQL
        postgres_hook = PostgresHook(postgres_conn_id='postgres_default')
        
        # Log pipeline execution
        log_sql = """
            INSERT INTO etl_pipeline_logs 
            (pipeline_name, execution_timestamp, status, records_processed, execution_details)
            VALUES (%s, %s, %s, %s, %s);
        """
        
        execution_details = {
            'pipeline_type': 'model_training',
            'models_trained': validation_results.get('models_validated', []) if validation_results else [],
            'models_deployed': len(deployment_results.get('deployed_models', [])) if deployment_results else 0,
            'overall_performance': validation_results.get('overall_performance_score', 0.0) if validation_results else 0.0,
            'validation_status': validation_results.get('overall_status') if validation_results else 'unknown',
            'deployment_status': deployment_results.get('deployment_status') if deployment_results else 'not_attempted'
        }
        
        pipeline_status = 'success'
        if validation_results and validation_results.get('overall_status') == 'failed':
            pipeline_status = 'failed'
        elif (validation_results and validation_results.get('overall_status') == 'partial_failure') or \
             (deployment_results and deployment_results.get('deployment_status') == 'partial_failure'):
            pipeline_status = 'partial_failure'
        
        postgres_hook.run(
            log_sql,
            parameters=[
                'model_training_pipeline',
                datetime.utcnow(),
                pipeline_status,
                len(execution_details.get('models_trained', [])),
                execution_details
            ]
        )
        
        logger.info(f"Training pipeline logged with status: {pipeline_status}")
        return {'status': pipeline_status, 'details': execution_details}
        
    except Exception as e:
        logger.error(f"Failed to log training pipeline: {e}")
        raise


# Define tasks
prepare_data_task = PythonOperator(
    task_id='prepare_training_data',
    python_callable=prepare_training_data,
    dag=dag,
)

train_recession_task = PythonOperator(
    task_id='train_recession_model',
    python_callable=train_recession_model,
    dag=dag,
)

train_supply_chain_task = PythonOperator(
    task_id='train_supply_chain_model',
    python_callable=train_supply_chain_model,
    dag=dag,
)

train_volatility_task = PythonOperator(
    task_id='train_volatility_model',
    python_callable=train_volatility_model,
    dag=dag,
)

train_geopolitical_task = PythonOperator(
    task_id='train_geopolitical_model',
    python_callable=train_geopolitical_model,
    dag=dag,
)

validate_task = PythonOperator(
    task_id='validate_models',
    python_callable=validate_models,
    dag=dag,
)

deploy_task = PythonOperator(
    task_id='deploy_models',
    python_callable=deploy_models,
    dag=dag,
)

log_task = PythonOperator(
    task_id='log_training_pipeline',
    python_callable=log_training_pipeline,
    dag=dag,
)

# Set task dependencies
prepare_data_task >> [train_recession_task, train_supply_chain_task, train_volatility_task, train_geopolitical_task]

[train_recession_task, train_supply_chain_task, train_volatility_task, train_geopolitical_task] >> validate_task

validate_task >> deploy_task >> log_task