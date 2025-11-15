"""
Cron Job Monitoring and Alerting Service
Tracks cron job execution, success rates, and sends alerts for failures
"""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import json

from src.core.database import get_database_pool
from src.core.logging import get_logger
from src.monitoring.observability import get_observability_service
from src.monitoring.metrics import (
    CRON_JOB_LAST_RUN_TIMESTAMP,
    CRON_JOB_LAST_SUCCESS_TIMESTAMP,
    CRON_JOB_LAST_FAILURE_TIMESTAMP,
    CRON_JOB_LAST_DURATION_SECONDS,
    CRON_JOB_STATUS,
    CRON_JOB_FAILURES_TOTAL,
    CRON_JOB_EXPECTED_INTERVAL_SECONDS,
)
from src.services.alerts_delivery import get_alert_delivery_service

logger = get_logger(__name__)

@dataclass
class CronJobDefinition:
    """Definition of a monitored cron job."""
    job_name: str
    schedule: str  # Cron expression
    expected_duration_seconds: int
    max_duration_seconds: int
    description: str
    alert_on_failure: bool = True
    alert_on_late: bool = True
    required_for_system: bool = True
    expected_interval_seconds: int = 3600

@dataclass
class CronJobExecution:
    """Record of a cron job execution."""
    job_name: str
    started_at: datetime
    completed_at: Optional[datetime]
    success: bool
    duration_seconds: Optional[float]
    exit_code: Optional[int]
    output: str
    error_message: Optional[str]

@dataclass
class CronJobHealth:
    """Health status of a cron job."""
    job_name: str
    last_execution: Optional[datetime]
    last_success: Optional[datetime]
    success_rate_24h: float
    avg_duration_seconds: float
    status: str  # 'healthy', 'warning', 'critical', 'unknown'
    next_expected: Optional[datetime]
    issues: List[str]

class CronMonitorService:
    """Service for monitoring cron job execution and health."""
    
    def __init__(self):
        self.observability = get_observability_service()
        self.alert_service = get_alert_delivery_service()
        
        # Define monitored cron jobs
        self.monitored_jobs = {
            'data_snapshots': CronJobDefinition(
                job_name='data_snapshots',
                schedule='0 * * * *',  # Hourly
                expected_duration_seconds=300,  # 5 minutes
                max_duration_seconds=900,  # 15 minutes
                description='Hourly economic data snapshot collection',
                required_for_system=True,
                expected_interval_seconds=3600,
            ),
            'backup_daily': CronJobDefinition(
                job_name='backup_daily',
                schedule='0 2 * * *',  # Daily at 2 AM
                expected_duration_seconds=1800,  # 30 minutes
                max_duration_seconds=3600,  # 1 hour
                description='Daily full system backup',
                required_for_system=False,
                expected_interval_seconds=86400,
            ),
            'backup_hourly': CronJobDefinition(
                job_name='backup_hourly',
                schedule='0 9-21 * * *',  # Business hours
                expected_duration_seconds=600,  # 10 minutes
                max_duration_seconds=1200,  # 20 minutes
                description='Hourly PostgreSQL snapshots',
                required_for_system=False,
                expected_interval_seconds=3600,
            ),
            'model_monitoring': CronJobDefinition(
                job_name='model_monitoring',
                schedule='0 */4 * * *',  # Every 4 hours
                expected_duration_seconds=300,  # 5 minutes
                max_duration_seconds=900,  # 15 minutes
                description='ML model drift detection and monitoring',
                required_for_system=True,
                expected_interval_seconds=14400,
            ),
            'model_training': CronJobDefinition(
                job_name='model_training',
                schedule='0 3 * * *',  # Daily at 3 AM
                expected_duration_seconds=3600,  # 1 hour
                max_duration_seconds=7200,  # 2 hours
                description='Daily ML model retraining',
                required_for_system=True,
                expected_interval_seconds=86400,
            ),
            'cache_cleanup': CronJobDefinition(
                job_name='cache_cleanup',
                schedule='0 4 * * *',  # Daily at 4 AM
                expected_duration_seconds=180,  # 3 minutes
                max_duration_seconds=600,  # 10 minutes
                description='Daily cache cleanup and maintenance',
                required_for_system=False,
                expected_interval_seconds=86400,
            ),
            'database_maintenance': CronJobDefinition(
                job_name='database_maintenance',
                schedule='0 5 * * 0',  # Weekly on Sunday at 5 AM
                expected_duration_seconds=1800,  # 30 minutes
                max_duration_seconds=3600,  # 1 hour
                description='Weekly database maintenance (VACUUM, REINDEX)',
                required_for_system=False,
                expected_interval_seconds=604800,
            ),
            'alert_processing': CronJobDefinition(
                job_name='alert_processing',
                schedule='*/15 * * * *',  # Every 15 minutes
                expected_duration_seconds=60,  # 1 minute
                max_duration_seconds=300,  # 5 minutes
                description='Process pending alert deliveries',
                required_for_system=True,
                expected_interval_seconds=900,
            )
        }
        # Initialize expected interval gauges
        for definition in self.monitored_jobs.values():
            CRON_JOB_EXPECTED_INTERVAL_SECONDS.labels(job_name=definition.job_name).set(
                definition.expected_interval_seconds
            )
    
    def _expected_interval(self, job_name: str) -> int:
        definition = self.monitored_jobs.get(job_name)
        return definition.expected_interval_seconds if definition else 3600
    
    async def record_job_start(self, job_name: str) -> str:
        """Record the start of a cron job execution."""
        execution_id = f"{job_name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        now_ts = datetime.now(timezone.utc).timestamp()
        
        try:
            pool = await get_database_pool()
            async with pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO cron_executions (
                        execution_id, job_name, started_at, status
                    ) VALUES ($1, $2, $3, 'running')
                """, execution_id, job_name, datetime.now(timezone.utc))
            
            logger.info(f"Cron job started: {job_name} (ID: {execution_id})")
            
            # Update Prometheus metrics
            CRON_JOB_LAST_RUN_TIMESTAMP.labels(job_name=job_name).set(now_ts)
            CRON_JOB_EXPECTED_INTERVAL_SECONDS.labels(job_name=job_name).set(self._expected_interval(job_name))
            
            # Update Prometheus metrics
            self.observability.record_job_execution(job_name, 0, True)  # Start marker
            
            return execution_id
            
        except Exception as e:
            logger.error(f"Error recording job start for {job_name}: {e}")
            return execution_id
    
    async def record_job_completion(
        self, 
        execution_id: str, 
        job_name: str, 
        success: bool, 
        exit_code: int = 0,
        output: str = "",
        error_message: str = ""
    ):
        """Record the completion of a cron job execution."""
        completed_at = datetime.now(timezone.utc)
        
        try:
            pool = await get_database_pool()
            async with pool.acquire() as conn:
                # Get start time to calculate duration
                row = await conn.fetchrow("""
                    SELECT started_at FROM cron_executions 
                    WHERE execution_id = $1
                """, execution_id)
                
                if row:
                    started_at = row['started_at']
                    duration = (completed_at - started_at).total_seconds()
                    
                    # Update execution record
                    await conn.execute("""
                        UPDATE cron_executions 
                        SET completed_at = $1, status = $2, success = $3,
                            duration_seconds = $4, exit_code = $5, 
                            output = $6, error_message = $7
                        WHERE execution_id = $8
                    """, 
                    completed_at, 
                    'completed' if success else 'failed',
                    success,
                    duration,
                    exit_code,
                    output[:10000],  # Limit output size
                    error_message[:1000] if error_message else None,
                    execution_id
                    )
                    
                    logger.info(f"Cron job completed: {job_name} (success: {success}, duration: {duration:.1f}s)")
                    
                    # Update Prometheus metrics
                    self.observability.record_job_execution(job_name, duration, success)
                    CRON_JOB_LAST_DURATION_SECONDS.labels(job_name=job_name).set(duration)
                    if success:
                        CRON_JOB_LAST_SUCCESS_TIMESTAMP.labels(job_name=job_name).set(completed_at.timestamp())
                        CRON_JOB_STATUS.labels(job_name=job_name).set(1)
                    else:
                        CRON_JOB_LAST_FAILURE_TIMESTAMP.labels(job_name=job_name).set(completed_at.timestamp())
                        CRON_JOB_STATUS.labels(job_name=job_name).set(0)
                        CRON_JOB_FAILURES_TOTAL.labels(job_name=job_name).inc()
                    
                    # Check for issues and send alerts
                    await self._check_job_health_and_alert(job_name, duration, success, error_message)
                    
                else:
                    logger.warning(f"No start record found for execution ID: {execution_id}")
                    
        except Exception as e:
            logger.error(f"Error recording job completion for {job_name}: {e}")
    
    async def get_job_health(self, job_name: str) -> CronJobHealth:
        """Get health status for a specific cron job."""
        try:
            pool = await get_database_pool()
            async with pool.acquire() as conn:
                # Get recent execution statistics
                since_24h = datetime.now(timezone.utc) - timedelta(hours=24)
                
                recent_stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_runs,
                        COUNT(*) FILTER (WHERE success = true) as successful_runs,
                        MAX(started_at) as last_execution,
                        MAX(started_at) FILTER (WHERE success = true) as last_success,
                        AVG(duration_seconds) FILTER (WHERE duration_seconds IS NOT NULL) as avg_duration
                    FROM cron_executions 
                    WHERE job_name = $1 AND started_at >= $2
                """, job_name, since_24h)
                
                if recent_stats and recent_stats['total_runs'] > 0:
                    success_rate = (recent_stats['successful_runs'] / recent_stats['total_runs']) * 100
                    last_execution = recent_stats['last_execution']
                    last_success = recent_stats['last_success']
                    avg_duration = float(recent_stats['avg_duration']) if recent_stats['avg_duration'] else 0.0
                else:
                    success_rate = 0.0
                    last_execution = None
                    last_success = None
                    avg_duration = 0.0
                
                # Determine health status and issues
                status, issues = self._assess_job_health(
                    job_name, 
                    last_execution, 
                    last_success, 
                    success_rate, 
                    avg_duration
                )
                
                # Calculate next expected execution (simplified)
                next_expected = self._calculate_next_execution(job_name)
                
                return CronJobHealth(
                    job_name=job_name,
                    last_execution=last_execution,
                    last_success=last_success,
                    success_rate_24h=success_rate,
                    avg_duration_seconds=avg_duration,
                    status=status,
                    next_expected=next_expected,
                    issues=issues
                )
                
        except Exception as e:
            logger.error(f"Error getting job health for {job_name}: {e}")
            return CronJobHealth(
                job_name=job_name,
                last_execution=None,
                last_success=None,
                success_rate_24h=0.0,
                avg_duration_seconds=0.0,
                status='unknown',
                next_expected=None,
                issues=[f"Error retrieving health data: {e}"]
            )
    
    async def get_all_jobs_health(self) -> List[CronJobHealth]:
        """Get health status for all monitored cron jobs."""
        health_reports = []
        
        for job_name in self.monitored_jobs.keys():
            health = await self.get_job_health(job_name)
            health_reports.append(health)
        
        return health_reports
    
    async def _check_job_health_and_alert(
        self, 
        job_name: str, 
        duration: float, 
        success: bool, 
        error_message: str = ""
    ):
        """Check job health and send alerts if needed."""
        if job_name not in self.monitored_jobs:
            return
        
        job_def = self.monitored_jobs[job_name]
        alerts_needed = []
        
        # Check for failure
        if not success and job_def.alert_on_failure:
            alerts_needed.append({
                'type': 'job_failure',
                'severity': 'critical' if job_def.required_for_system else 'high',
                'message': f"Cron job '{job_name}' failed",
                'details': {
                    'job_name': job_name,
                    'error_message': error_message,
                    'expected_duration': job_def.expected_duration_seconds,
                    'actual_duration': duration
                }
            })
        
        # Check for excessive duration
        if success and duration > job_def.max_duration_seconds and job_def.alert_on_late:
            alerts_needed.append({
                'type': 'job_slow',
                'severity': 'medium',
                'message': f"Cron job '{job_name}' took longer than expected",
                'details': {
                    'job_name': job_name,
                    'expected_max_duration': job_def.max_duration_seconds,
                    'actual_duration': duration,
                    'slowdown_factor': duration / job_def.max_duration_seconds
                }
            })
        
        # Send alerts
        for alert in alerts_needed:
            await self._send_cron_alert(alert)
    
    def _assess_job_health(
        self, 
        job_name: str, 
        last_execution: Optional[datetime], 
        last_success: Optional[datetime],
        success_rate: float, 
        avg_duration: float
    ) -> tuple[str, List[str]]:
        """Assess the health status of a cron job."""
        if job_name not in self.monitored_jobs:
            return 'unknown', ['Job not in monitoring configuration']
        
        job_def = self.monitored_jobs[job_name]
        issues = []
        
        now = datetime.now(timezone.utc)
        
        # Check if job has run recently
        if last_execution is None:
            return 'critical', ['No executions recorded']
        
        # Check time since last execution
        time_since_last = (now - last_execution).total_seconds()
        expected_interval = self._get_expected_interval(job_def.schedule)
        
        if time_since_last > expected_interval * 2:  # Missing for 2 intervals
            issues.append(f"No execution for {time_since_last/3600:.1f} hours")
        
        # Check success rate
        if success_rate < 50:
            issues.append(f"Low success rate: {success_rate:.1f}%")
        elif success_rate < 90:
            issues.append(f"Moderate success rate: {success_rate:.1f}%")
        
        # Check average duration
        if avg_duration > job_def.max_duration_seconds:
            issues.append(f"Average duration ({avg_duration:.1f}s) exceeds maximum ({job_def.max_duration_seconds}s)")
        
        # Check time since last successful execution
        if last_success and (now - last_success).total_seconds() > expected_interval * 3:
            issues.append(f"No successful execution for {(now - last_success).total_seconds()/3600:.1f} hours")
        
        # Determine overall status
        if not issues:
            status = 'healthy'
        elif len(issues) == 1 and 'Moderate success rate' in issues[0]:
            status = 'warning'
        elif any('No execution' in issue or 'Low success rate' in issue for issue in issues):
            status = 'critical'
        else:
            status = 'warning'
        
        return status, issues
    
    def _get_expected_interval(self, cron_schedule: str) -> int:
        """Get expected interval in seconds for a cron schedule (simplified)."""
        # This is a simplified implementation
        # In production, use a proper cron parser like croniter
        
        schedule_intervals = {
            '*/15 * * * *': 15 * 60,  # Every 15 minutes
            '0 * * * *': 3600,  # Hourly
            '0 */4 * * *': 4 * 3600,  # Every 4 hours
            '0 2 * * *': 24 * 3600,  # Daily at 2 AM
            '0 3 * * *': 24 * 3600,  # Daily at 3 AM
            '0 4 * * *': 24 * 3600,  # Daily at 4 AM
            '0 9-21 * * *': 3600,  # Business hours (treat as hourly)
            '0 5 * * 0': 7 * 24 * 3600,  # Weekly on Sunday
        }
        
        return schedule_intervals.get(cron_schedule, 3600)  # Default to 1 hour
    
    def _calculate_next_execution(self, job_name: str) -> Optional[datetime]:
        """Calculate next expected execution time (simplified)."""
        if job_name not in self.monitored_jobs:
            return None
        
        # This is simplified - use croniter in production for accurate calculations
        job_def = self.monitored_jobs[job_name]
        interval = self._get_expected_interval(job_def.schedule)
        
        return datetime.now(timezone.utc) + timedelta(seconds=interval)
    
    async def _send_cron_alert(self, alert: Dict[str, Any]):
        """Send cron job alert notification."""
        try:
            alert_message = f"🔧 CRON JOB ALERT: {alert['message']}\n\n"
            alert_message += f"**Details:**\n"
            
            for key, value in alert['details'].items():
                alert_message += f"- {key.replace('_', ' ').title()}: {value}\n"
            
            # This would integrate with the alert delivery service
            # For now, just log the alert
            logger.warning(f"Cron alert [{alert['severity']}]: {alert['message']}")
            
            # Record alert in observability system
            self.observability.record_alert_trigger("cron_job", alert['severity'])
            
        except Exception as e:
            logger.error(f"Error sending cron alert: {e}")
    
    async def generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive cron job health report."""
        try:
            all_health = await self.get_all_jobs_health()
            
            # Aggregate statistics
            total_jobs = len(all_health)
            healthy_jobs = len([h for h in all_health if h.status == 'healthy'])
            warning_jobs = len([h for h in all_health if h.status == 'warning'])
            critical_jobs = len([h for h in all_health if h.status == 'critical'])
            
            # Find jobs with issues
            jobs_with_issues = [h for h in all_health if h.issues]
            
            # Calculate overall system health
            if critical_jobs > 0:
                overall_status = 'critical'
            elif warning_jobs > 0:
                overall_status = 'warning'
            else:
                overall_status = 'healthy'
            
            return {
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'overall_status': overall_status,
                'summary': {
                    'total_jobs': total_jobs,
                    'healthy_jobs': healthy_jobs,
                    'warning_jobs': warning_jobs,
                    'critical_jobs': critical_jobs,
                    'health_percentage': (healthy_jobs / total_jobs * 100) if total_jobs > 0 else 0
                },
                'job_details': [
                    {
                        'job_name': h.job_name,
                        'status': h.status,
                        'success_rate_24h': h.success_rate_24h,
                        'last_execution': h.last_execution.isoformat() if h.last_execution else None,
                        'avg_duration_seconds': h.avg_duration_seconds,
                        'issues': h.issues,
                        'next_expected': h.next_expected.isoformat() if h.next_expected else None
                    }
                    for h in all_health
                ],
                'critical_issues': [
                    {
                        'job_name': h.job_name,
                        'issues': h.issues
                    }
                    for h in jobs_with_issues if h.status == 'critical'
                ]
            }
            
        except Exception as e:
            logger.error(f"Error generating health report: {e}")
            return {
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'error': str(e),
                'overall_status': 'unknown'
            }

# Global service instance
_cron_monitor_service = None

def get_cron_monitor_service() -> CronMonitorService:
    global _cron_monitor_service
    if _cron_monitor_service is None:
        _cron_monitor_service = CronMonitorService()
    return _cron_monitor_service
