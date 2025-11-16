#!/usr/bin/env python3
"""
Data Ingestion Script for Phase A

Ingests historical data for ML training using the 5-year window requirement.
Run this after database setup and before model training.

Usage:
    python scripts/ingest_data.py
"""

import sys
import os
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.ingestion import ingest_local_series


def main():
    """Ingest data for training."""
    print("📊 Starting data ingestion for Phase A...")
    print("⏰ Processing 5-year historical window")
    print("🔄 Ingesting from data providers...")
    
    try:
        observations = ingest_local_series()
        
        total_obs = sum(len(series_data) for series_data in observations.values())
        print(f"\n✅ Data ingestion completed!")
        print(f"📈 Processed {len(observations)} series")
        print(f"📊 Total observations: {total_obs}")
        print("\n🧮 Series breakdown:")
        for series_id, series_data in observations.items():
            print(f"   • {series_id}: {len(series_data)} observations")
        
        print("\n🚀 Ready for ML model training!")
        
    except Exception as e:
        print(f"❌ Data ingestion failed: {e}")
        print("\n🔍 Troubleshooting:")
        print("   • Check data provider API keys")
        print("   • Verify network connectivity")
        print("   • Ensure database is set up")
        sys.exit(1)


if __name__ == "__main__":
    main()