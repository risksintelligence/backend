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

# Try to import worker with retry logic for Render build delays
import time
max_retries = 3
retry_delay = 5

for attempt in range(max_retries):
    try:
        from start_worker import main as run_production_worker
        break
    except ImportError as e:
        if attempt < max_retries - 1:
            print(f"Import failed (attempt {attempt + 1}): {e}")
            print(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
        else:
            print(f"Final import attempt failed: {e}")
            print("Worker startup failed - dependencies not available")
            sys.exit(1)

def main() -> None:
    """Main entry point for background worker jobs."""
    run_production_worker()

if __name__ == "__main__":
    main()
