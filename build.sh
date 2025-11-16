#!/bin/bash
# Render build script for both web service and background workers

set -e

echo "🔧 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "📦 Installing additional production dependencies..."
pip install psycopg2-binary

echo "🗂️ Creating required directories..."
mkdir -p models data cache logs

echo "✅ Build completed successfully"
echo "📋 Installed packages:"
pip list | grep -E "(pydantic|fastapi|sqlalchemy|redis)"