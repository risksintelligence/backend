#!/usr/bin/env python3
"""
Simple Background Worker Runner for Render

This module provides a simplified entry point for background workers that 
handles Render deployment timing issues gracefully.
"""

import sys
import time
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

print("🚀 Starting RRIO background worker...")

# Simple approach: brief delay then try import
time.sleep(3)

try:
    from start_worker import main as run_production_worker
    print("✅ Worker module imported successfully")
except ImportError as e:
    print(f"⚠️ Import failed, retrying: {e}")
    time.sleep(10)
    try:
        from start_worker import main as run_production_worker
        print("✅ Worker module imported on retry")
    except ImportError as e2:
        print(f"❌ Worker startup failed: {e2}")
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