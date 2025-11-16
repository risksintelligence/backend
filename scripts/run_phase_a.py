#!/usr/bin/env python3
"""
Phase A Complete Setup Script

Runs the complete Phase A setup pipeline:
1. Database setup
2. Data ingestion  
3. ML model training

Usage:
    python scripts/run_phase_a.py
"""

import sys
import os
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db import Base, engine
from app.services.ingestion import ingest_local_series
from app.services.training import train_all_models


def setup_database():
    """Set up database tables."""
    print("🗄️  Setting up database...")
    Base.metadata.create_all(bind=engine)
    print("✅ Database ready!")


def ingest_data():
    """Ingest historical data."""
    print("\n📊 Ingesting historical data...")
    observations = ingest_local_series()
    total_obs = sum(len(series_data) for series_data in observations.values())
    print(f"✅ Ingested {total_obs} observations across {len(observations)} series")
    return observations


def train_models():
    """Train ML models."""
    print("\n🧠 Training ML models...")
    train_all_models()
    print("✅ All models trained successfully!")


def main():
    """Run complete Phase A setup."""
    print("🚀 Starting Phase A Complete Setup")
    print("=" * 50)
    
    try:
        # Step 1: Database setup
        setup_database()
        
        # Step 2: Data ingestion
        observations = ingest_data()
        
        # Step 3: Model training
        train_models()
        
        print("\n" + "=" * 50)
        print("🎉 Phase A Setup Complete!")
        print("\n📋 What's ready:")
        print("   ✅ Database with historical observations")
        print("   ✅ Trained ML models for regime, forecast, anomaly detection")
        print("   ✅ API endpoints ready for production")
        print("\n🌐 Available endpoints:")
        print("   • /api/v1/analytics/geri - GERII score")
        print("   • /api/v1/ai/regime/current - Market regime")
        print("   • /api/v1/ai/forecast/next-24h - 24h forecasts")
        print("   • /api/v1/anomalies/latest - Anomaly detection")
        print("   • /api/v1/impact/ras - RAS snapshot")
        print("\n🚀 Start the server: uvicorn app.main:app --reload")
        
    except Exception as e:
        print(f"\n❌ Phase A setup failed: {e}")
        print("\n🔍 Troubleshooting:")
        print("   • Ensure virtual environment is activated")
        print("   • Check all dependencies are installed")
        print("   • Verify data provider access")
        sys.exit(1)


if __name__ == "__main__":
    main()