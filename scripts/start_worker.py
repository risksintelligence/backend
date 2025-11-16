#!/usr/bin/env python3
"""
Production Background Worker Startup Script

This script starts the RRIO background worker with proper production configuration.
It handles data ingestion, model training scheduling, and system maintenance tasks.

Usage:
    python scripts/start_worker.py

Environment Variables Required:
    - RIS_POSTGRES_DSN: Database connection string
    - RIS_REDIS_URL: Redis connection string (optional)
    - WORKER_ROLE: Type of worker (ingestion, training, maintenance)
"""

import os
import sys
import time
import logging
import signal
from pathlib import Path
from typing import Dict, Any

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import get_settings
from app.services.ingestion import ingest_local_series
from app.services.training import train_all_models
from app.services.transparency import add_transparency_log
from app.db import SessionLocal, Base, engine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RRIOWorker:
    """Production background worker for RRIO system."""
    
    def __init__(self):
        self.settings = get_settings()
        self.running = True
        self.worker_role = os.getenv('WORKER_ROLE', 'ingestion')
        self.setup_signal_handlers()
        
    def setup_signal_handlers(self):
        """Setup graceful shutdown signal handlers."""
        def shutdown_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down gracefully...")
            self.running = False
        
        signal.signal(signal.SIGINT, shutdown_handler)
        signal.signal(signal.SIGTERM, shutdown_handler)
    
    def check_database_connection(self) -> bool:
        """Check if database connection is working."""
        try:
            from sqlalchemy import text
            db = SessionLocal()
            db.execute(text("SELECT 1"))
            db.close()
            return True
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False
    
    def ensure_database(self) -> None:
        """Ensure database tables exist before starting worker."""
        try:
            logger.info("Ensuring database tables exist...")
            Base.metadata.create_all(bind=engine)
            logger.info("Database tables verified/created successfully")
        except Exception as exc:
            logger.error(f"Database initialization failed: {exc}")
            raise
    
    def run_ingestion_worker(self):
        """Run data ingestion worker."""
        logger.info("Starting data ingestion worker...")
        
        while self.running:
            try:
                logger.info("Running data ingestion cycle...")
                observations = ingest_local_series()
                
                total_obs = sum(len(series_data) for series_data in observations.values())
                logger.info(f"Ingested {total_obs} observations across {len(observations)} series")
                
                # Log transparency event
                add_transparency_log(
                    event_type="data_update",
                    description=f"Automated data ingestion completed: {total_obs} observations",
                    metadata={"series_count": len(observations), "observation_count": total_obs}
                )
                
                # Sleep for 1 hour between ingestion cycles
                self.sleep_with_interrupt(3600)
                
            except Exception as e:
                logger.error(f"Ingestion cycle failed: {e}")
                self.sleep_with_interrupt(600)  # Retry after 10 minutes
    
    def run_training_worker(self):
        """Run model training worker with deployment-friendly initialization."""
        logger.info("Starting model training worker...")
        
        # Check if we're in a deployment context
        is_deployment = os.getenv('RENDER_SERVICE_TYPE') == 'background_worker'
        
        try:
            # Always do initial training
            logger.info("Running initial model training...")
            train_all_models()
            logger.info("Model training completed successfully")
            
            # Log transparency event
            add_transparency_log(
                event_type="model_retrain", 
                description="Initial model training completed",
                metadata={"models": ["regime_classifier", "forecast_model", "anomaly_detector"]}
            )
            
            if is_deployment:
                # In deployment mode, exit after initial training
                logger.info("Deployment mode: initial training complete, exiting gracefully")
                return
            
        except Exception as e:
            logger.error(f"Initial training failed: {e}")
            if is_deployment:
                # Don't fail deployment for training issues
                logger.warning("Training failed in deployment mode, continuing...")
                return
            else:
                raise
        
        # Continue with periodic training only in production mode
        while self.running:
            try:
                # Sleep for 24 hours between training cycles
                self.sleep_with_interrupt(86400)
                
                if not self.running:
                    break
                    
                logger.info("Running scheduled model retraining...")
                train_all_models()
                logger.info("Scheduled model training completed successfully")
                
                # Log transparency event
                add_transparency_log(
                    event_type="model_retrain",
                    description="Scheduled model retraining completed",
                    metadata={"models": ["regime_classifier", "forecast_model", "anomaly_detector"]}
                )
                
            except Exception as e:
                logger.error(f"Training cycle failed: {e}")
                self.sleep_with_interrupt(3600)  # Retry after 1 hour
    
    def run_maintenance_worker(self):
        """Run system maintenance worker."""
        logger.info("Starting maintenance worker...")
        
        while self.running:
            try:
                logger.info("Running maintenance cycle...")
                
                # Cleanup old logs, optimize database, etc.
                self.cleanup_old_logs()
                
                # Log transparency event
                add_transparency_log(
                    event_type="system_maintenance",
                    description="Automated system maintenance completed",
                    metadata={"tasks": ["log_cleanup", "database_optimization"]}
                )
                
                # Sleep for 6 hours between maintenance cycles
                self.sleep_with_interrupt(21600)
                
            except Exception as e:
                logger.error(f"Maintenance cycle failed: {e}")
                self.sleep_with_interrupt(3600)  # Retry after 1 hour
    
    def cleanup_old_logs(self):
        """Clean up old logs and temporary data."""
        try:
            db = SessionLocal()
            
            # Delete transparency logs older than 90 days
            from app.models import TransparencyLogModel
            from datetime import datetime, timedelta
            
            cutoff_date = datetime.utcnow() - timedelta(days=90)
            old_logs = db.query(TransparencyLogModel).filter(
                TransparencyLogModel.timestamp < cutoff_date
            )
            deleted_count = old_logs.count()
            old_logs.delete()
            db.commit()
            db.close()
            
            logger.info(f"Cleaned up {deleted_count} old transparency logs")
            
        except Exception as e:
            logger.error(f"Log cleanup failed: {e}")
    
    def sleep_with_interrupt(self, seconds: int):
        """Sleep for given seconds, but allow interruption."""
        start_time = time.time()
        while time.time() - start_time < seconds and self.running:
            time.sleep(min(1, seconds - (time.time() - start_time)))
    
    def run(self):
        """Main worker run loop."""
        logger.info(f"Starting RRIO worker in {self.worker_role} mode")
        logger.info(f"Database URL: {getattr(self.settings, 'database_url', 'Not configured')[:20]}...")
        
        # Ensure database tables exist
        self.ensure_database()
        
        # Check database connection
        if not self.check_database_connection():
            logger.error("Cannot connect to database. Exiting.")
            sys.exit(1)
        
        logger.info("Database connection successful")
        
        # Route to appropriate worker based on role
        try:
            if self.worker_role == 'ingestion':
                self.run_ingestion_worker()
            elif self.worker_role == 'training':
                self.run_training_worker()
            elif self.worker_role == 'maintenance':
                self.run_maintenance_worker()
            else:
                logger.error(f"Unknown worker role: {self.worker_role}")
                sys.exit(1)
                
        except KeyboardInterrupt:
            logger.info("Worker interrupted by user")
        except Exception as e:
            logger.error(f"Worker failed with error: {e}")
            sys.exit(1)
        finally:
            logger.info("Worker shutdown complete")


def main():
    """Main entry point."""
    worker = RRIOWorker()
    worker.run()


if __name__ == "__main__":
    main()
