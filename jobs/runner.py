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

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from start_worker import main as run_production_worker

def main() -> None:
    """Main entry point for background worker jobs."""
    run_production_worker()

if __name__ == "__main__":
    main()
