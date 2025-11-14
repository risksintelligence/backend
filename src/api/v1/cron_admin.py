"""
Cron Job Administration API
Endpoints for monitoring and managing cron job execution
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone, timedelta

from backend.src.services.cron_monitor_service import get_cron_monitor_service, CronMonitorService
from backend.src.services.auth_service import User
from backend.src.api.middleware.auth import require_admin, require_deployment_control
from backend.src.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/admin/cron", tags=["cron_admin"])

class CronJobExecutionResponse(BaseModel):
    execution_id: str
    job_name: str
    started_at: datetime
    completed_at: Optional[datetime]
    success: Optional[bool]
    duration_seconds: Optional[float]
    status: str

@router.get("/jobs")
async def list_cron_jobs(
    cron_monitor: CronMonitorService = Depends(get_cron_monitor_service),
    current_user: User = Depends(require_admin)
):
    """List all monitored cron jobs and their definitions."""
    try:
        # Get job definitions from the service
        monitored_jobs = cron_monitor.monitored_jobs
        
        jobs_info = []
        for job_name, job_def in monitored_jobs.items():
            jobs_info.append({
                'job_name': job_def.job_name,
                'schedule': job_def.schedule,
                'description': job_def.description,
                'expected_duration_seconds': job_def.expected_duration_seconds,
                'max_duration_seconds': job_def.max_duration_seconds,
                'alert_on_failure': job_def.alert_on_failure,
                'alert_on_late': job_def.alert_on_late,
                'required_for_system': job_def.required_for_system
            })
        
        return {
            'total_jobs': len(jobs_info),
            'jobs': jobs_info
        }
        
    except Exception as e:
        logger.error(f"Error listing cron jobs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list cron jobs: {e}")

@router.get("/jobs/{job_name}/health")
async def get_job_health(
    job_name: str,
    cron_monitor: CronMonitorService = Depends(get_cron_monitor_service),
    current_user: User = Depends(require_admin)
):
    """Get health status for a specific cron job."""
    try:
        health = await cron_monitor.get_job_health(job_name)
        
        return {
            'job_name': health.job_name,
            'status': health.status,
            'last_execution': health.last_execution.isoformat() if health.last_execution else None,
            'last_success': health.last_success.isoformat() if health.last_success else None,
            'success_rate_24h': health.success_rate_24h,
            'avg_duration_seconds': health.avg_duration_seconds,
            'next_expected': health.next_expected.isoformat() if health.next_expected else None,
            'issues': health.issues
        }
        
    except Exception as e:
        logger.error(f"Error getting job health for {job_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get job health: {e}")

@router.get("/jobs/health/summary")
async def get_all_jobs_health(
    cron_monitor: CronMonitorService = Depends(get_cron_monitor_service),
    current_user: User = Depends(require_admin)
):
    """Get health summary for all cron jobs."""
    try:
        all_health = await cron_monitor.get_all_jobs_health()
        
        # Convert to API response format
        health_data = []
        for health in all_health:
            health_data.append({
                'job_name': health.job_name,
                'status': health.status,
                'success_rate_24h': health.success_rate_24h,
                'avg_duration_seconds': health.avg_duration_seconds,
                'last_execution': health.last_execution.isoformat() if health.last_execution else None,
                'issues_count': len(health.issues),
                'issues': health.issues
            })
        
        # Calculate summary statistics
        total_jobs = len(health_data)
        healthy_jobs = len([h for h in health_data if h['status'] == 'healthy'])
        warning_jobs = len([h for h in health_data if h['status'] == 'warning'])
        critical_jobs = len([h for h in health_data if h['status'] == 'critical'])
        
        return {
            'summary': {
                'total_jobs': total_jobs,
                'healthy_jobs': healthy_jobs,
                'warning_jobs': warning_jobs,
                'critical_jobs': critical_jobs,
                'health_percentage': (healthy_jobs / total_jobs * 100) if total_jobs > 0 else 0
            },
            'jobs': health_data,
            'generated_at': datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting jobs health summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get jobs health summary: {e}")

@router.get("/jobs/{job_name}/executions")
async def get_job_executions(
    job_name: str,
    limit: int = 50,
    hours: int = 24,
    cron_monitor: CronMonitorService = Depends(get_cron_monitor_service),
    current_user: User = Depends(require_admin)
):
    """Get recent execution history for a specific cron job."""
    try:
        from backend.src.core.database import get_database_pool
        
        since_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        pool = await get_database_pool()
        async with pool.acquire() as conn:
            executions = await conn.fetch("""
                SELECT execution_id, job_name, started_at, completed_at, 
                       success, duration_seconds, exit_code, status, 
                       error_message, created_at
                FROM cron_executions 
                WHERE job_name = $1 AND started_at >= $2
                ORDER BY started_at DESC
                LIMIT $3
            """, job_name, since_time, limit)
            
            execution_data = []
            for exec_row in executions:
                execution_data.append({
                    'execution_id': exec_row['execution_id'],
                    'job_name': exec_row['job_name'],
                    'started_at': exec_row['started_at'].isoformat(),
                    'completed_at': exec_row['completed_at'].isoformat() if exec_row['completed_at'] else None,
                    'success': exec_row['success'],
                    'duration_seconds': float(exec_row['duration_seconds']) if exec_row['duration_seconds'] else None,
                    'exit_code': exec_row['exit_code'],
                    'status': exec_row['status'],
                    'error_message': exec_row['error_message'],
                    'created_at': exec_row['created_at'].isoformat()
                })
            
            return {
                'job_name': job_name,
                'period_hours': hours,
                'total_executions': len(execution_data),
                'successful_executions': len([e for e in execution_data if e['success']]),
                'failed_executions': len([e for e in execution_data if e['success'] is False]),
                'executions': execution_data
            }
            
    except Exception as e:
        logger.error(f"Error getting job executions for {job_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get job executions: {e}")

@router.get("/report")
async def get_comprehensive_report(
    cron_monitor: CronMonitorService = Depends(get_cron_monitor_service),
    current_user: User = Depends(require_admin)
):
    """Generate comprehensive cron job health report."""
    try:
        report = await cron_monitor.generate_health_report()
        return report
        
    except Exception as e:
        logger.error(f"Error generating comprehensive report: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {e}")

@router.get("/executions/recent")
async def get_recent_executions(
    limit: int = 100,
    hours: int = 24,
    status: Optional[str] = None,
    cron_monitor: CronMonitorService = Depends(get_cron_monitor_service),
    current_user: User = Depends(require_admin)
):
    """Get recent cron job executions across all jobs."""
    try:
        from backend.src.core.database import get_database_pool
        
        since_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        pool = await get_database_pool()
        async with pool.acquire() as conn:
            query = """
                SELECT execution_id, job_name, started_at, completed_at, 
                       success, duration_seconds, status, error_message
                FROM cron_executions 
                WHERE started_at >= $1
            """
            params = [since_time]
            
            if status:
                query += " AND status = $2"
                params.append(status)
            
            query += " ORDER BY started_at DESC LIMIT $" + str(len(params) + 1)
            params.append(limit)
            
            executions = await conn.fetch(query, *params)
            
            execution_data = []
            for exec_row in executions:
                execution_data.append({
                    'execution_id': exec_row['execution_id'],
                    'job_name': exec_row['job_name'],
                    'started_at': exec_row['started_at'].isoformat(),
                    'completed_at': exec_row['completed_at'].isoformat() if exec_row['completed_at'] else None,
                    'success': exec_row['success'],
                    'duration_seconds': float(exec_row['duration_seconds']) if exec_row['duration_seconds'] else None,
                    'status': exec_row['status'],
                    'has_error': bool(exec_row['error_message'])
                })
            
            return {
                'period_hours': hours,
                'filter_status': status,
                'total_executions': len(execution_data),
                'executions': execution_data,
                'generated_at': datetime.now(timezone.utc).isoformat()
            }
            
    except Exception as e:
        logger.error(f"Error getting recent executions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get recent executions: {e}")

@router.get("/executions/{execution_id}")
async def get_execution_details(
    execution_id: str,
    current_user: User = Depends(require_admin)
):
    """Get detailed information about a specific execution."""
    try:
        from backend.src.core.database import get_database_pool
        
        pool = await get_database_pool()
        async with pool.acquire() as conn:
            execution = await conn.fetchrow("""
                SELECT execution_id, job_name, started_at, completed_at, 
                       success, duration_seconds, exit_code, status, 
                       output, error_message, created_at
                FROM cron_executions 
                WHERE execution_id = $1
            """, execution_id)
            
            if not execution:
                raise HTTPException(status_code=404, detail=f"Execution not found: {execution_id}")
            
            return {
                'execution_id': execution['execution_id'],
                'job_name': execution['job_name'],
                'started_at': execution['started_at'].isoformat(),
                'completed_at': execution['completed_at'].isoformat() if execution['completed_at'] else None,
                'success': execution['success'],
                'duration_seconds': float(execution['duration_seconds']) if execution['duration_seconds'] else None,
                'exit_code': execution['exit_code'],
                'status': execution['status'],
                'output': execution['output'] if execution['output'] else '',
                'error_message': execution['error_message'],
                'created_at': execution['created_at'].isoformat()
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting execution details: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get execution details: {e}")

@router.delete("/executions/cleanup")
async def cleanup_old_executions(
    days: int = 30,
    current_user: User = Depends(require_deployment_control)
):
    """Clean up old cron execution records."""
    try:
        from backend.src.core.database import get_database_pool
        
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        pool = await get_database_pool()
        async with pool.acquire() as conn:
            result = await conn.execute("""
                DELETE FROM cron_executions 
                WHERE started_at < $1
            """, cutoff_date)
            
            # Extract the number of deleted rows
            deleted_count = int(result.split()[-1]) if result else 0
            
            logger.info(f"Cleaned up {deleted_count} old cron execution records older than {days} days")
            
            return {
                'status': 'completed',
                'message': f'Cleaned up {deleted_count} execution records older than {days} days',
                'deleted_count': deleted_count,
                'cutoff_date': cutoff_date.isoformat(),
                'cleaned_by': current_user.username,
                'cleaned_at': datetime.now(timezone.utc).isoformat()
            }
            
    except Exception as e:
        logger.error(f"Error cleaning up execution records: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {e}")

@router.get("/status/overview")
async def get_status_overview(
    cron_monitor: CronMonitorService = Depends(get_cron_monitor_service),
    current_user: User = Depends(require_admin)
):
    """Get high-level overview of cron job system status."""
    try:
        # Get comprehensive health report
        report = await cron_monitor.generate_health_report()
        
        if 'error' in report:
            raise Exception(report['error'])
        
        # Extract key metrics for dashboard
        summary = report.get('summary', {})
        critical_issues = report.get('critical_issues', [])
        
        # Determine overall system status
        overall_status = report.get('overall_status', 'unknown')
        status_emoji = {
            'healthy': '✅',
            'warning': '⚠️',
            'critical': '🚨',
            'unknown': '❓'
        }.get(overall_status, '❓')
        
        return {
            'overall_status': overall_status,
            'status_emoji': status_emoji,
            'health_percentage': summary.get('health_percentage', 0),
            'total_jobs': summary.get('total_jobs', 0),
            'healthy_jobs': summary.get('healthy_jobs', 0),
            'warning_jobs': summary.get('warning_jobs', 0),
            'critical_jobs': summary.get('critical_jobs', 0),
            'has_critical_issues': len(critical_issues) > 0,
            'critical_issues_count': len(critical_issues),
            'last_updated': report.get('generated_at'),
            'critical_job_names': [issue['job_name'] for issue in critical_issues]
        }
        
    except Exception as e:
        logger.error(f"Error getting status overview: {e}")
        return {
            'overall_status': 'error',
            'status_emoji': '❌',
            'error_message': str(e),
            'last_updated': datetime.now(timezone.utc).isoformat()
        }