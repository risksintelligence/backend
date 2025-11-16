#!/usr/bin/env python3
"""
ML Model Training Script for Phase A

This script trains all ML models using the 5-year data window and saves them
to the models/ directory. Run this after setting up the Python environment
and database with historical data.

Usage:
    python scripts/train_models.py

Requirements:
    - Virtual environment activated
    - Database populated with historical observations
    - Required ML packages installed (see requirements.txt)
"""

import sys
import os
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.training import train_all_models


def main():
    """Main training execution."""
    print("🤖 Starting Phase A ML model training...")
    print("📊 Using 5-year historical data window")
    print("💾 Models will be saved to models/ directory")
    print("-" * 50)
    
    try:
        train_all_models()
        print("\n✅ Training completed successfully!")
        print("\n📂 Trained models:")
        print("   • regime_classifier.pkl - Market regime classification")
        print("   • regime_scaler.pkl - Feature scaler for regime model")
        print("   • forecast_model.pkl - 24-hour delta forecasting")
        print("   • forecast_scaler.pkl - Feature scaler for forecast model")
        print("   • anomaly_detector.pkl - Anomaly detection")
        print("\n🚀 Models are ready for API endpoints!")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        print("\n🔍 Troubleshooting:")
        print("   • Ensure database is populated with observations")
        print("   • Check that all dependencies are installed")
        print("   • Verify 5+ years of historical data exists")
        sys.exit(1)


if __name__ == "__main__":
    main()