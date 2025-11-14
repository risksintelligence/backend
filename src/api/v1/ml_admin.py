"""
ML Model Administration API
Endpoints for managing ML models, drift detection, and retraining
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from backend.src.services.model_registry_service import get_model_registry_service, ModelRegistryService
from backend.src.services.auth_service import User
from backend.src.api.middleware.auth import require_admin, require_deployment_control
from backend.src.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/admin/ml", tags=["ml_admin"])

class ModelPromotionRequest(BaseModel):
    model_id: str
    deployment_notes: Optional[str] = ""

class RetrainingRequest(BaseModel):
    model_type: str
    force_retrain: bool = False
    hyperparameters: Optional[Dict[str, Any]] = {}

@router.get("/models")
async def list_models(
    model_type: Optional[str] = None,
    include_inactive: bool = False,
    model_registry: ModelRegistryService = Depends(get_model_registry_service),
    current_user: User = Depends(require_admin)
):
    """List all registered ML models."""
    try:
        from backend.src.core.database import get_database_pool
        
        pool = await get_database_pool()
        async with pool.acquire() as conn:
            query = """
                SELECT model_id, model_type, version, is_active, created_at, model_data
                FROM ml_models
            """
            params = []
            
            conditions = []
            if model_type:
                conditions.append(f"model_type = ${len(params) + 1}")
                params.append(model_type)
                
            if not include_inactive:
                conditions.append("is_active = TRUE")
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY model_type, created_at DESC"
            
            rows = await conn.fetch(query, *params)
            
            models = []
            for row in rows:
                model_data = row['model_data'] if isinstance(row['model_data'], dict) else {}
                
                models.append({
                    'model_id': row['model_id'],
                    'model_type': row['model_type'],
                    'version': row['version'],
                    'is_active': row['is_active'],
                    'created_at': row['created_at'].isoformat(),
                    'performance_metrics': model_data.get('performance_metrics', {}),
                    'features_used': model_data.get('features_used', []),
                    'file_size_bytes': model_data.get('file_size_bytes', 0),
                    'deployment_notes': model_data.get('deployment_notes', '')
                })
            
            return {
                'models': models,
                'total_count': len(models),
                'active_count': len([m for m in models if m['is_active']]),
                'model_types': list(set(m['model_type'] for m in models))
            }
            
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {e}")

@router.get("/models/{model_id}")
async def get_model_details(
    model_id: str,
    model_registry: ModelRegistryService = Depends(get_model_registry_service),
    current_user: User = Depends(require_admin)
):
    """Get detailed information about a specific model."""
    try:
        from backend.src.core.database import get_database_pool
        
        pool = await get_database_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT model_id, model_type, version, is_active, created_at, model_data
                FROM ml_models
                WHERE model_id = $1
            """, model_id)
            
            if not row:
                raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
            
            model_data = row['model_data'] if isinstance(row['model_data'], dict) else {}
            
            # Get recent predictions for this model
            recent_predictions = await conn.fetch("""
                SELECT prediction_data, accuracy_score, created_at
                FROM ml_model_predictions
                WHERE model_id = $1
                ORDER BY created_at DESC
                LIMIT 20
            """, model_id)
            
            # Get drift metrics
            drift_metrics = await conn.fetch("""
                SELECT drift_data, timestamp
                FROM ml_drift_metrics
                WHERE model_id = $1
                ORDER BY timestamp DESC
                LIMIT 10
            """, model_id)
            
            return {
                'model_id': row['model_id'],
                'model_type': row['model_type'],
                'version': row['version'],
                'is_active': row['is_active'],
                'created_at': row['created_at'].isoformat(),
                'metadata': model_data,
                'recent_predictions': [
                    {
                        'prediction': pred['prediction_data'],
                        'accuracy': pred['accuracy_score'],
                        'timestamp': pred['created_at'].isoformat()
                    }
                    for pred in recent_predictions
                ],
                'drift_history': [
                    {
                        'metrics': drift['drift_data'],
                        'timestamp': drift['timestamp'].isoformat()
                    }
                    for drift in drift_metrics
                ]
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model details: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model details: {e}")

@router.post("/models/{model_id}/promote")
async def promote_model(
    model_id: str,
    request: ModelPromotionRequest,
    model_registry: ModelRegistryService = Depends(get_model_registry_service),
    current_user: User = Depends(require_deployment_control)
):
    """Promote a model to active status."""
    try:
        logger.info(f"Model promotion requested by {current_user.username}: {model_id}")
        
        success = await model_registry.promote_model(model_id)
        
        if success:
            # Log the promotion action
            from backend.src.services.admin_service import get_admin_service
            admin_service = get_admin_service()
            
            await admin_service.log_admin_action(
                current_user.username,
                "model_promotion",
                {
                    'model_id': model_id,
                    'deployment_notes': request.deployment_notes
                }
            )
            
            return {
                'status': 'success',
                'message': f'Model {model_id} promoted to active status',
                'promoted_by': current_user.username,
                'promoted_at': datetime.now(timezone.utc).isoformat()
            }
        else:
            raise HTTPException(status_code=400, detail=f"Failed to promote model {model_id}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error promoting model: {e}")
        raise HTTPException(status_code=500, detail=f"Model promotion failed: {e}")

@router.get("/drift/summary")
async def get_drift_summary(
    days: int = 7,
    model_registry: ModelRegistryService = Depends(get_model_registry_service),
    current_user: User = Depends(require_admin)
):
    """Get summary of model drift metrics."""
    try:
        from backend.src.core.database import get_database_pool
        from datetime import timedelta
        
        since_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        pool = await get_database_pool()
        async with pool.acquire() as conn:
            # Get drift metrics summary
            drift_summary = await conn.fetch("""
                SELECT dm.model_id, m.model_type, dm.drift_data, dm.timestamp
                FROM ml_drift_metrics dm
                JOIN ml_models m ON dm.model_id = m.model_id
                WHERE dm.timestamp >= $1
                ORDER BY dm.timestamp DESC
            """, since_date)
            
            # Aggregate drift statistics
            drift_stats = {}
            alert_count = 0
            
            for row in drift_summary:
                model_type = row['model_type']
                drift_data = row['drift_data'] if isinstance(row['drift_data'], dict) else {}
                
                if model_type not in drift_stats:
                    drift_stats[model_type] = {
                        'model_type': model_type,
                        'sample_count': 0,
                        'avg_feature_drift': 0.0,
                        'avg_prediction_drift': 0.0,
                        'avg_performance_degradation': 0.0,
                        'alert_count': 0
                    }
                
                stats = drift_stats[model_type]
                stats['sample_count'] += 1
                stats['avg_feature_drift'] += drift_data.get('feature_drift_score', 0.0)
                stats['avg_prediction_drift'] += drift_data.get('prediction_drift_score', 0.0)
                stats['avg_performance_degradation'] += drift_data.get('performance_degradation', 0.0)
                
                if drift_data.get('alert_threshold_exceeded', False):
                    stats['alert_count'] += 1
                    alert_count += 1
            
            # Calculate averages
            for stats in drift_stats.values():
                if stats['sample_count'] > 0:
                    stats['avg_feature_drift'] /= stats['sample_count']
                    stats['avg_prediction_drift'] /= stats['sample_count']
                    stats['avg_performance_degradation'] /= stats['sample_count']
            
            return {
                'period_days': days,
                'total_drift_measurements': len(drift_summary),
                'total_alerts': alert_count,
                'drift_by_model_type': list(drift_stats.values()),
                'generated_at': datetime.now(timezone.utc).isoformat()
            }
            
    except Exception as e:
        logger.error(f"Error getting drift summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get drift summary: {e}")

@router.post("/drift/check")
async def trigger_drift_check(
    model_type: Optional[str] = None,
    background_tasks: BackgroundTasks,
    model_registry: ModelRegistryService = Depends(get_model_registry_service),
    current_user: User = Depends(require_deployment_control)
):
    """Manually trigger drift detection for models."""
    try:
        logger.info(f"Manual drift check triggered by {current_user.username}")
        
        async def run_drift_check():
            from backend.src.jobs.model_monitoring import ModelMonitoringJob
            
            monitoring_job = ModelMonitoringJob()
            results = await monitoring_job.run_monitoring_cycle()
            
            logger.info(f"Manual drift check completed: {results}")
        
        # Run drift check in background
        background_tasks.add_task(run_drift_check)
        
        return {
            'status': 'initiated',
            'message': 'Drift detection initiated in background',
            'model_type': model_type or 'all',
            'initiated_by': current_user.username,
            'initiated_at': datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error triggering drift check: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to trigger drift check: {e}")

@router.post("/retrain")
async def trigger_retraining(
    request: RetrainingRequest,
    background_tasks: BackgroundTasks,
    model_registry: ModelRegistryService = Depends(get_model_registry_service),
    current_user: User = Depends(require_deployment_control)
):
    """Trigger model retraining."""
    try:
        logger.info(f"Model retraining requested by {current_user.username}: {request.model_type}")
        
        async def run_retraining():
            # This would integrate with the actual training pipeline
            # For now, just log the request
            logger.info(f"Retraining {request.model_type} with params: {request.hyperparameters}")
            
            # This is where you'd call your actual training code
            # await train_model(request.model_type, request.hyperparameters)
        
        # Run retraining in background
        background_tasks.add_task(run_retraining)
        
        # Log the retraining request
        from backend.src.services.admin_service import get_admin_service
        admin_service = get_admin_service()
        
        await admin_service.log_admin_action(
            current_user.username,
            "model_retrain_triggered",
            {
                'model_type': request.model_type,
                'force_retrain': request.force_retrain,
                'hyperparameters': request.hyperparameters
            }
        )
        
        return {
            'status': 'initiated',
            'message': f'Retraining initiated for {request.model_type}',
            'model_type': request.model_type,
            'initiated_by': current_user.username,
            'initiated_at': datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error triggering retraining: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to trigger retraining: {e}")

@router.get("/performance")
async def get_model_performance(
    days: int = 7,
    model_type: Optional[str] = None,
    model_registry: ModelRegistryService = Depends(get_model_registry_service),
    current_user: User = Depends(require_admin)
):
    """Get model performance metrics over time."""
    try:
        from backend.src.core.database import get_database_pool
        from datetime import timedelta
        
        since_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        pool = await get_database_pool()
        async with pool.acquire() as conn:
            query = """
                SELECT model_type, model_id, accuracy_score, 
                       DATE_TRUNC('day', created_at) as day,
                       COUNT(*) as prediction_count,
                       AVG(accuracy_score) as avg_accuracy
                FROM ml_model_predictions
                WHERE created_at >= $1
            """
            params = [since_date]
            
            if model_type:
                query += " AND model_type = $2"
                params.append(model_type)
            
            query += """
                GROUP BY model_type, model_id, DATE_TRUNC('day', created_at)
                ORDER BY day DESC, model_type
            """
            
            performance_data = await conn.fetch(query, *params)
            
            # Organize by model type
            performance_by_type = {}
            for row in performance_data:
                mtype = row['model_type']
                if mtype not in performance_by_type:
                    performance_by_type[mtype] = []
                
                performance_by_type[mtype].append({
                    'date': row['day'].isoformat(),
                    'model_id': row['model_id'],
                    'prediction_count': row['prediction_count'],
                    'avg_accuracy': float(row['avg_accuracy']) if row['avg_accuracy'] else 0.0
                })
            
            return {
                'period_days': days,
                'performance_by_model_type': performance_by_type,
                'generated_at': datetime.now(timezone.utc).isoformat()
            }
            
    except Exception as e:
        logger.error(f"Error getting model performance: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model performance: {e}")

@router.post("/cleanup")
async def cleanup_old_models(
    keep_versions: int = 5,
    model_registry: ModelRegistryService = Depends(get_model_registry_service),
    current_user: User = Depends(require_deployment_control)
):
    """Clean up old model versions."""
    try:
        logger.info(f"Model cleanup requested by {current_user.username}: keep {keep_versions} versions")
        
        deleted_count = await model_registry.cleanup_old_models(keep_versions)
        
        # Log the cleanup action
        from backend.src.services.admin_service import get_admin_service
        admin_service = get_admin_service()
        
        await admin_service.log_admin_action(
            current_user.username,
            "model_cleanup",
            {
                'deleted_count': deleted_count,
                'keep_versions': keep_versions
            }
        )
        
        return {
            'status': 'completed',
            'message': f'Cleaned up {deleted_count} old model versions',
            'deleted_count': deleted_count,
            'keep_versions': keep_versions,
            'cleaned_by': current_user.username,
            'cleaned_at': datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error cleaning up models: {e}")
        raise HTTPException(status_code=500, detail=f"Model cleanup failed: {e}")