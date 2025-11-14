#!/bin/bash
set -e

echo "=== Setup Health Check ==="

# Check environment variables
echo "Environment Check:"
echo "- RIS_POSTGRES_DSN: ${RIS_POSTGRES_DSN:+SET}" 
echo "- RIS_REDIS_URL: ${RIS_REDIS_URL:+SET}"
echo "- ENVIRONMENT: ${ENVIRONMENT:-not set}"

# Check database connection
if [[ -n "${RIS_POSTGRES_DSN:-}" ]]; then
    echo "Testing database connection..."
    if psql "$RIS_POSTGRES_DSN" -c "SELECT COUNT(*) FROM users;" 2>/dev/null; then
        echo "✓ Database connection successful"
    else
        echo "✗ Database connection failed"
    fi
    
    # Check key tables exist
    tables=("users" "feature_flags" "ml_models" "alert_subscriptions")
    for table in "${tables[@]}"; do
        if psql "$RIS_POSTGRES_DSN" -c "SELECT 1 FROM $table LIMIT 1;" &>/dev/null; then
            echo "✓ Table $table exists and accessible"
        else
            echo "✗ Table $table missing or inaccessible"
        fi
    done
else
    echo "✗ RIS_POSTGRES_DSN not set"
fi

# Test Python imports
echo "Testing Python dependencies..."
python3 -c "
import sys
modules = ['fastapi', 'uvicorn', 'asyncpg', 'psycopg2', 'redis', 'sklearn', 'pandas']
for module in modules:
    try:
        __import__(module)
        print(f'✓ {module} imported successfully')
    except ImportError as e:
        print(f'✗ {module} import failed: {e}')
"

echo "Health check completed"