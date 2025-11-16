#!/usr/bin/env python3
"""
Jobs Runner Module

This module provides the entry point for background workers as expected
by Render deployment configuration. It delegates to the production worker
implementation in scripts/start_worker.py.

Usage:
    python -m jobs.runner

Environment Variables:
    - WORKER_ROLE: Type of worker (ingestion, training, maintenance)
"""

import sys
import os
from pathlib import Path

def check_dependencies():
    """Check required dependencies and log status. Don't fail hard in production."""
    required_packages = ['pydantic', 'fastapi', 'sqlalchemy', 'redis']
    missing = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        missing_list = ', '.join(missing)
        print("ℹ️  This may be normal during Render's build phase")
        print(f"⚠️  Some packages may not be importable: {missing_list}")
        print("   Worker will attempt to continue - imports will fail if truly missing")
        return False

    print("✅ All required packages available")
    return True

# Log dependency status but continue (let actual imports fail if needed)
check_dependencies()

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

# Wait for build to complete on Render
import time
import subprocess

def wait_for_dependencies():
    """Wait for dependencies to be available, with exponential backoff."""
    max_wait_time = 300  # 5 minutes max
    wait_time = 10  # Start with 10 seconds
    total_waited = 0
    
    while total_waited < max_wait_time:
        try:
            # Try to import a key dependency
            import pydantic
            print("✅ Dependencies are ready")
            return True
        except ImportError:
            print(f"⏳ Waiting {wait_time}s for build to complete... ({total_waited}s/{max_wait_time}s)")
            time.sleep(wait_time)
            total_waited += wait_time
            wait_time = min(wait_time * 1.5, 60)  # Exponential backoff, max 60s
    
    print(f"❌ Timeout waiting for dependencies after {max_wait_time}s")
    return False

# Wait for dependencies to be ready
if not wait_for_dependencies():
    print("Exiting - dependencies not available after waiting")
    sys.exit(1)

# Now try to import the worker
try:
    from start_worker import main as run_production_worker
    print("✅ Worker module imported successfully")
except ImportError as e:
    print(f"❌ Failed to import worker: {e}")
    sys.exit(1)

def main() -> None:
    """Main entry point for background worker jobs."""
    run_production_worker()

if __name__ == "__main__":
    main()
