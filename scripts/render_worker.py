#!/usr/bin/env python3
"""
Render-optimized Worker Startup Script

This script is specifically designed to run workers on Render.com with
proper environment handling, error recovery, and resource management.

Usage:
    For ingestion worker: WORKER_ROLE=ingestion python scripts/render_worker.py  
    For training worker: WORKER_ROLE=training python scripts/render_worker.py
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

# Configure logging for Render console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

def verify_environment():
    """Verify required environment variables for Render deployment."""
    logger.info("🔍 Verifying Render environment...")
    
    required_vars = {
        'RIS_POSTGRES_DSN': 'Database connection string',
        'RIS_JWT_SECRET': 'JWT authentication secret',
        'WORKER_ROLE': 'Worker role (ingestion/training/maintenance)'
    }
    
    missing_vars = []
    for var, description in required_vars.items():
        if not os.getenv(var):
            missing_vars.append(f"{var} ({description})")
        else:
            # Log first/last few chars for security
            value = os.getenv(var)
            masked_value = f"{value[:10]}...{value[-5:]}" if len(value) > 15 else "***"
            logger.info(f"  ✅ {var}: {masked_value}")
    
    if missing_vars:
        logger.error("❌ Missing required environment variables:")
        for var in missing_vars:
            logger.error(f"  - {var}")
        return False
    
    # Optional environment variables
    optional_vars = ['RIS_REDIS_URL', 'RIS_ENV']
    for var in optional_vars:
        if os.getenv(var):
            logger.info(f"  ✅ {var}: configured")
        else:
            logger.warning(f"  ⚠️ {var}: not set (optional)")
    
    logger.info("✅ Environment verification complete")
    return True

def setup_database():
    """Initialize database connection and create tables if needed."""
    logger.info("🗄️ Setting up database connection...")
    
    try:
        from app.db import SessionLocal, Base, engine
        from app.core.config import get_settings
        from sqlalchemy import text
        
        settings = get_settings()
        logger.info(f"📊 Database URL: {settings.database_url[:30]}...")
        
        # Test connection
        logger.info("🔗 Testing database connection...")
        db = SessionLocal()
        result = db.execute(text("SELECT 1")).fetchone()
        db.close()
        logger.info("✅ Database connection successful")
        
        # Create tables
        logger.info("🛠️ Creating/verifying database tables...")
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables ready")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Database setup failed: {e}")
        return False

def setup_cache():
    """Initialize cache system (Redis or file fallback)."""
    logger.info("🔄 Setting up cache system...")
    
    try:
        from app.core.cache import RedisCache
        
        redis_url = os.getenv('RIS_REDIS_URL')
        if redis_url:
            cache = RedisCache("worker_test")
            if cache.available:
                logger.info("✅ Redis cache available")
                
                # Test cache operations
                test_key = "render_worker_test"
                cache.set(test_key, {"status": "ok", "timestamp": time.time()}, ttl=60)
                test_value = cache.get(test_key)
                
                if test_value:
                    logger.info("✅ Cache operations working")
                    cache.delete(test_key)
                else:
                    logger.warning("⚠️ Cache operations failed, but continuing")
            else:
                logger.warning("⚠️ Redis unavailable, using file cache fallback")
        else:
            logger.info("📁 Using file-based cache (no Redis URL provided)")
        
        return True
        
    except Exception as e:
        logger.warning(f"⚠️ Cache setup issue (non-fatal): {e}")
        return True  # Cache is not critical

def start_ingestion_worker():
    """Start the data ingestion worker with Render optimizations."""
    logger.info("📊 Starting ingestion worker...")
    
    from app.services.ingestion import ingest_local_series
    from app.services.transparency import add_transparency_log
    
    cycle_count = 0
    while True:
        try:
            cycle_count += 1
            logger.info(f"🔄 Starting ingestion cycle #{cycle_count}")
            
            # Run ingestion
            observations = ingest_local_series()
            total_obs = sum(len(series_data) for series_data in observations.values())
            
            logger.info(f"✅ Cycle #{cycle_count} complete: {total_obs} observations across {len(observations)} series")
            
            # Log transparency event
            add_transparency_log(
                event_type="data_update",
                description=f"Render ingestion cycle #{cycle_count}: {total_obs} observations",
                metadata={
                    "cycle": cycle_count,
                    "series_count": len(observations),
                    "observation_count": total_obs,
                    "environment": "render"
                }
            )
            
            # Sleep for 1 hour, but check for shutdown every minute
            logger.info("😴 Sleeping for 1 hour...")
            for i in range(60):
                time.sleep(60)  # Sleep in 1-minute chunks
                if i % 15 == 0:  # Log every 15 minutes
                    logger.info(f"⏰ {60-i} minutes until next ingestion cycle")
            
        except KeyboardInterrupt:
            logger.info("🛑 Ingestion worker stopped by signal")
            break
        except Exception as e:
            logger.error(f"❌ Ingestion cycle #{cycle_count} failed: {e}")
            logger.info("🔄 Retrying in 10 minutes...")
            time.sleep(600)

def start_training_worker():
    """Start the model training worker with Render optimizations."""
    logger.info("🧠 Starting training worker...")
    
    from app.services.training import train_all_models
    from app.services.transparency import add_transparency_log
    
    cycle_count = 0
    while True:
        try:
            cycle_count += 1
            logger.info(f"🔄 Starting training cycle #{cycle_count}")
            
            # Run model training
            train_all_models()
            
            logger.info(f"✅ Training cycle #{cycle_count} complete")
            
            # Log transparency event
            add_transparency_log(
                event_type="model_retrain",
                description=f"Render training cycle #{cycle_count} completed",
                metadata={
                    "cycle": cycle_count,
                    "models": ["regime_classifier", "forecast_model", "anomaly_detector"],
                    "environment": "render"
                }
            )
            
            # Sleep for 24 hours, but log progress every hour
            logger.info("😴 Sleeping for 24 hours...")
            for i in range(24):
                time.sleep(3600)  # Sleep in 1-hour chunks
                logger.info(f"⏰ {24-i} hours until next training cycle")
            
        except KeyboardInterrupt:
            logger.info("🛑 Training worker stopped by signal")
            break
        except Exception as e:
            logger.error(f"❌ Training cycle #{cycle_count} failed: {e}")
            logger.info("🔄 Retrying in 1 hour...")
            time.sleep(3600)

def main():
    """Main entry point for Render worker."""
    logger.info("🚀 Starting RRIO Worker on Render.com")
    logger.info("=" * 50)
    
    # Setup signal handlers
    def shutdown_handler(signum, frame):
        logger.info(f"📞 Received signal {signum}, shutting down gracefully...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    
    # Verification steps
    if not verify_environment():
        logger.error("💥 Environment verification failed")
        sys.exit(1)
    
    if not setup_database():
        logger.error("💥 Database setup failed")
        sys.exit(1)
    
    if not setup_cache():
        logger.warning("⚠️ Cache setup had issues (continuing anyway)")
    
    # Get worker role
    worker_role = os.getenv('WORKER_ROLE', 'ingestion').lower()
    
    logger.info(f"👷 Starting {worker_role} worker")
    logger.info("=" * 50)
    
    # Start appropriate worker
    try:
        if worker_role == 'ingestion':
            start_ingestion_worker()
        elif worker_role == 'training':
            start_training_worker()
        else:
            logger.error(f"❌ Unknown worker role: {worker_role}")
            logger.error("Valid roles: ingestion, training")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"💥 Worker failed: {e}")
        sys.exit(1)
    
    logger.info("✅ Worker shutdown complete")

if __name__ == "__main__":
    main()