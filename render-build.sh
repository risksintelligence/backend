#!/bin/bash
set -e

echo "=== RIS Backend Build Script ==="
echo "Python version check:"
python --version

# Ensure we're using Python 3.11
if ! python --version | grep -q "3.11"; then
    echo "ERROR: Python 3.11 required but found $(python --version)"
    exit 1
fi

# Upgrade pip and setuptools
pip install --upgrade pip setuptools wheel

# Install dependencies with explicit Python 3.11 compatibility
pip install -r requirements.txt

echo "Build completed successfully"