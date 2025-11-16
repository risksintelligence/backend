#!/usr/bin/env python3
"""
Simple Background Worker Runner for Render

This module provides a simplified entry point for background workers that 
handles Render deployment timing issues gracefully.
"""

import sys
import time
import subprocess
from pathlib import Path

print("🚀 Starting RRIO background worker...")

# Check if critical dependencies are available
def check_dependencies():
    try:
        import pydantic
        import fastapi
        import sqlalchemy
        return True
    except ImportError as e:
        print(f"⚠️ Missing dependencies: {e}")
        return False

# Wait for dependencies to be available (Render build completion)
max_wait = 300  # 5 minutes
wait_interval = 15
elapsed = 0

while elapsed < max_wait:
    if check_dependencies():
        print("✅ Dependencies available")
        break
    print(f"⏳ Waiting for dependencies... ({elapsed}s/{max_wait}s)")
    time.sleep(wait_interval)
    elapsed += wait_interval
else:
    print("❌ Dependencies not available after 5 minutes")
    print("🔄 Exiting for Render to retry...")
    sys.exit(0)

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

try:
    from start_worker import main as run_production_worker
    print("✅ Worker module imported successfully")
except ImportError as e:
    print(f"❌ Worker startup failed: {e}")
    # Don't exit with error - let Render handle the retry
    print("🔄 Exiting for Render to retry...")
    sys.exit(0)

def main() -> None:
    """Main entry point for background worker jobs."""
    try:
        print("▶️ Starting worker...")
        run_production_worker()
    except Exception as e:
        print(f"❌ Worker error: {e}")
        # Exit gracefully to allow Render restart
        sys.exit(0)

if __name__ == "__main__":
    main()