#!/usr/bin/env python3
"""
Database Setup Script for Phase A

Creates database tables and runs initial migrations.
Run this before training models.

Usage:
    python scripts/setup_database.py
"""

import sys
import os
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db import Base, engine
from app.models import ObservationModel


def main():
    """Set up database tables."""
    print("🗄️  Setting up database for Phase A...")
    print("📊 Creating tables for observations")
    
    try:
        # Create all tables
        Base.metadata.create_all(bind=engine)
        print("✅ Database tables created successfully!")
        print("\n📋 Tables created:")
        print("   • observations - Time series data storage")
        print("\n🚀 Ready for data ingestion and model training!")
        
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()