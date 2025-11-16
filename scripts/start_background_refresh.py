#!/usr/bin/env python3
"""
Background Refresh Worker

Starts the TTL-based background refresh service that implements
stale-while-revalidate pattern per architecture requirements.

Usage:
    python scripts/start_background_refresh.py
"""

import asyncio
import sys
import signal
import logging
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.background_refresh import refresh_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/background_refresh.log') if Path('logs').exists() else logging.NullHandler()
    ]
)

logger = logging.getLogger(__name__)

class BackgroundRefreshWorker:
    """Worker process for running background refresh service."""
    
    def __init__(self):
        self.running = False
        self.refresh_task = None
        
    async def start(self) -> None:
        """Start the background refresh worker."""
        logger.info("🚀 Starting Background Refresh Worker...")
        logger.info("📋 Worker will monitor cache TTLs and refresh stale data")
        logger.info("🔄 Implementing stale-while-revalidate pattern")
        
        self.running = True
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        try:
            # Start the background refresh service
            self.refresh_task = asyncio.create_task(refresh_service.start_refresh_loop())
            
            # Keep the worker running
            while self.running:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("🛑 Received keyboard interrupt")
        except Exception as e:
            logger.error(f"❌ Worker error: {e}")
        finally:
            await self.shutdown()
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the worker."""
        logger.info("🔄 Shutting down Background Refresh Worker...")
        
        self.running = False
        
        # Stop the refresh service
        refresh_service.stop_refresh_loop()
        
        # Cancel the refresh task
        if self.refresh_task and not self.refresh_task.done():
            self.refresh_task.cancel()
            try:
                await self.refresh_task
            except asyncio.CancelledError:
                pass
        
        logger.info("✅ Background Refresh Worker shutdown complete")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"📨 Received signal {signum}, initiating shutdown...")
        self.running = False

async def main():
    """Main entry point for the background refresh worker."""
    worker = BackgroundRefreshWorker()
    
    try:
        await worker.start()
    except Exception as e:
        logger.error(f"❌ Failed to start background refresh worker: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)
    
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except Exception as e:
        logger.error(f"❌ Critical error: {e}")
        sys.exit(1)