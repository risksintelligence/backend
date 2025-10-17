"""
RiskX Test Suite

Comprehensive testing framework for the RiskX AI Risk Intelligence Observatory.
"""

import os
import sys
from pathlib import Path

# Add project root to Python path for testing
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Test configuration
TEST_CONFIG = {
    "database_url": os.getenv("TEST_DATABASE_URL", "sqlite:///test_riskx.db"),
    "cache_backend": "memory",
    "log_level": "DEBUG",
    "api_test_timeout": 30,
    "integration_test_timeout": 60
}

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "test_data"
TEST_DATA_DIR.mkdir(exist_ok=True)