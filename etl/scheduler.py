"""
Simple ETL scheduler for RiskX data pipeline.

This provides a lightweight alternative to Airflow for basic scheduling needs.
"""
import asyncio
import logging
import schedule
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Callable
import threading
import json
from pathlib import Path

from etl.tasks.fred_fetch import run_fred_etl_task

logger = logging.getLogger(__name__)


class ETLScheduler:
    """
    Simple ETL scheduler for RiskX data pipeline.
    """
    
    def __init__(self):
        self.is_running = False
        self.task_history = []
        self.max_history = 100
        self.log_file = Path("logs/etl_scheduler.log")
        self.log_file.parent.mkdir(exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Register tasks
        self._register_tasks()
    
    def _setup_logging(self):
        """Setup logging for the scheduler."""
        handler = logging.FileHandler(self.log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    def _register_tasks(self):
        """Register ETL tasks with the scheduler."""
        logger.info("Registering ETL tasks")
        
        # FRED data updates - every 4 hours during market hours
        schedule.every(4).hours.do(
            self._run_task_with_logging,
            task_name="fred_update",
            task_func=run_fred_etl_task,
            force_refresh=False
        )
        
        # Daily comprehensive refresh at 6 AM
        schedule.every().day.at("06:00").do(
            self._run_task_with_logging,
            task_name="daily_comprehensive_refresh",
            task_func=run_fred_etl_task,
            force_refresh=True
        )
        
        # Weekly data validation on Sundays at 2 AM
        schedule.every().sunday.at("02:00").do(
            self._run_task_with_logging,
            task_name="weekly_validation",
            task_func=self._run_weekly_validation
        )
        
        logger.info("ETL tasks registered successfully")
    
    def _run_task_with_logging(self, task_name: str, task_func: Callable, **kwargs):
        """
        Run a task with comprehensive logging and error handling.
        
        Args:
            task_name: Name of the task
            task_func: Function to execute
            **kwargs: Arguments to pass to the task function
        """
        start_time = datetime.utcnow()
        logger.info(f"Starting task: {task_name}")
        
        task_record = {
            "task_name": task_name,
            "start_time": start_time.isoformat(),
            "status": "running",
            "result": None,
            "error": None,
            "duration_seconds": None
        }
        
        try:
            # Execute the task
            result = task_func(**kwargs)
            
            # Record success
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            task_record.update({
                "status": "completed",
                "result": result,
                "end_time": end_time.isoformat(),
                "duration_seconds": duration
            })
            
            logger.info(f"Task {task_name} completed successfully in {duration:.2f} seconds")
            
        except Exception as e:
            # Record failure
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            error_msg = str(e)
            
            task_record.update({
                "status": "failed",
                "error": error_msg,
                "end_time": end_time.isoformat(),
                "duration_seconds": duration
            })
            
            logger.error(f"Task {task_name} failed after {duration:.2f} seconds: {error_msg}")
        
        # Add to history
        self.task_history.append(task_record)
        
        # Maintain history size
        if len(self.task_history) > self.max_history:
            self.task_history.pop(0)
        
        # Save task history to file
        self._save_task_history()
    
    def _run_weekly_validation(self) -> Dict[str, Any]:
        """Run weekly data validation checks."""
        logger.info("Running weekly data validation")
        
        from etl.tasks.fred_fetch import FREDDataFetcher
        
        fetcher = FREDDataFetcher()
        
        validation_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "fred_validation": fetcher.validate_data_freshness(),
            "etl_metrics": fetcher.get_etl_metrics(),
            "overall_status": "healthy"
        }
        
        # Determine overall status
        if validation_results["fred_validation"]["overall_status"] != "healthy":
            validation_results["overall_status"] = "degraded"
        
        logger.info(f"Weekly validation completed with status: {validation_results['overall_status']}")
        return validation_results
    
    def _save_task_history(self):
        """Save task history to file."""
        try:
            history_file = Path("logs/etl_task_history.json")
            history_file.parent.mkdir(exist_ok=True)
            
            with open(history_file, 'w') as f:
                json.dump(self.task_history, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error saving task history: {str(e)}")
    
    def start(self):
        """Start the ETL scheduler."""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
        
        self.is_running = True
        logger.info("Starting ETL scheduler")
        
        # Run initial data fetch
        logger.info("Running initial data fetch")
        self._run_task_with_logging(
            task_name="initial_fetch",
            task_func=run_fred_etl_task,
            force_refresh=True
        )
        
        # Start the scheduler loop
        try:
            while self.is_running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")
        except Exception as e:
            logger.error(f"Scheduler error: {str(e)}")
        finally:
            self.is_running = False
            logger.info("ETL scheduler stopped")
    
    def stop(self):
        """Stop the ETL scheduler."""
        logger.info("Stopping ETL scheduler")
        self.is_running = False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current scheduler status."""
        return {
            "is_running": self.is_running,
            "scheduled_jobs": len(schedule.jobs),
            "recent_tasks": self.task_history[-10:],  # Last 10 tasks
            "next_run": str(schedule.next_run()) if schedule.jobs else None
        }
    
    def run_task_now(self, task_name: str, **kwargs) -> Dict[str, Any]:
        """
        Manually trigger a task to run immediately.
        
        Args:
            task_name: Name of the task to run
            **kwargs: Arguments to pass to the task
            
        Returns:
            Task execution result
        """
        logger.info(f"Manually triggering task: {task_name}")
        
        if task_name == "fred_update":
            self._run_task_with_logging(
                task_name="manual_fred_update",
                task_func=run_fred_etl_task,
                force_refresh=kwargs.get("force_refresh", False)
            )
        elif task_name == "weekly_validation":
            self._run_task_with_logging(
                task_name="manual_validation",
                task_func=self._run_weekly_validation
            )
        else:
            raise ValueError(f"Unknown task: {task_name}")
        
        # Return the latest task result
        return self.task_history[-1] if self.task_history else {}


def run_scheduler():
    """Main function to run the ETL scheduler."""
    scheduler = ETLScheduler()
    
    try:
        scheduler.start()
    except KeyboardInterrupt:
        scheduler.stop()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the scheduler
    run_scheduler()