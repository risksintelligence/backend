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

def ensure_dependencies():
    """Verify required dependencies exist. Do not auto-install in managed environments."""
    required_packages = ['pydantic', 'fastapi', 'sqlalchemy', 'redis']
    missing = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        missing_list = ', '.join(missing)
        print(f"❌ Missing required packages: {missing_list}")
        print("⚠️  Install dependencies via your deployment build step (e.g., pip install -r requirements.txt)")
        print("ℹ️  Render managed environments block runtime pip installs; use a virtualenv or image build instead.")
        sys.exit(1)

# Ensure dependencies are present before importing the worker
ensure_dependencies()

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from start_worker import main as run_production_worker

def main() -> None:
    """Main entry point for background worker jobs."""
    run_production_worker()

if __name__ == "__main__":
    main()
