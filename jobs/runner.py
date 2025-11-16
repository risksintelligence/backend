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
    """Check required dependencies and exit gracefully if they're not yet installed."""
    required_packages = ['pydantic', 'fastapi', 'sqlalchemy', 'redis']
    missing = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        missing_list = ', '.join(missing)
        print("ℹ️  This may be normal during Render's import phase")
        print(f"⚠️  Some packages may not be importable: {missing_list}")
        print("   Exiting worker startup until dependencies are installed (pip install -r requirements.txt).")
        return False

    print("✅ All required packages available")
    return True

# Abort early if dependencies aren't ready (Render will retry after build)
if not check_dependencies():
    sys.exit(0)

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from start_worker import main as run_production_worker

def main() -> None:
    """Main entry point for background worker jobs."""
    run_production_worker()

if __name__ == "__main__":
    main()
