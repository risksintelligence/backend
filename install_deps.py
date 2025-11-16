#!/usr/bin/env python3
"""
Dependency installer for Render workers.
Ensures all required packages are installed before starting workers.
"""

import sys
import subprocess
import os

def install_requirements():
    """Install packages from requirements.txt if not already installed."""
    try:
        # Try to install from requirements.txt
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
        ], capture_output=True, text=True, check=True)
        
        print("✅ Dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False

def verify_critical_packages():
    """Verify that critical packages are available."""
    critical_packages = {
        'pydantic': 'pydantic>=2.5.0',
        'fastapi': 'fastapi>=0.100.0',
        'sqlalchemy': 'sqlalchemy>=2.0.0',
        'redis': 'redis>=5.0.0'
    }
    
    missing = []
    for package, version_spec in critical_packages.items():
        try:
            __import__(package)
            print(f"✅ {package} is available")
        except ImportError:
            missing.append(version_spec)
            print(f"❌ {package} is missing")
    
    if missing:
        print(f"\n🔧 Installing missing packages: {missing}")
        try:
            subprocess.run([
                sys.executable, '-m', 'pip', 'install'
            ] + missing, check=True)
            print("✅ Missing packages installed")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install missing packages: {e}")
            return False
    
    return True

if __name__ == "__main__":
    print("🔧 Installing dependencies for Render worker...")
    
    if not install_requirements():
        sys.exit(1)
    
    if not verify_critical_packages():
        sys.exit(1)
    
    print("🎉 All dependencies are ready!")