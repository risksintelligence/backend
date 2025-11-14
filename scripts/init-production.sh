#!/bin/bash
set -e

echo "=== Production Initialization ==="

# Check if database is available
if [[ -z "${RIS_POSTGRES_DSN:-}" ]]; then
    echo "WARNING: RIS_POSTGRES_DSN not set - cannot initialize database"
    exit 0
fi

echo "Testing database connection..."
if ! psql "$RIS_POSTGRES_DSN" -c "SELECT 1;" &>/dev/null; then
    echo "WARNING: Database not available - cannot initialize"
    exit 0
fi

echo "Database connection successful"

# Create minimal ML models if they don't exist
echo "Initializing ML models..."
python3 -c "
import os
import json
from datetime import datetime

# Create minimal ML model records
models = [
    {
        'model_id': 'regime_classifier_v1',
        'model_type': 'regime_classification',
        'version': '1.0',
        'model_data': {
            'type': 'mock_classifier',
            'classes': ['stable', 'volatile', 'crisis'],
            'default_prediction': 'stable',
            'confidence': 0.75
        },
        'is_active': True
    },
    {
        'model_id': 'forecast_model_v1',
        'model_type': 'geri_forecast',
        'version': '1.0',
        'model_data': {
            'type': 'mock_forecaster',
            'default_delta': 0.0,
            'range': [-5.0, 5.0]
        },
        'is_active': True
    },
    {
        'model_id': 'anomaly_detector_v1',
        'model_type': 'anomaly_detection',
        'version': '1.0',
        'model_data': {
            'type': 'mock_detector',
            'threshold': -2.0,
            'default_score': -0.5
        },
        'is_active': True
    }
]

# Insert models if they don't exist
import psycopg2
conn = psycopg2.connect(os.environ['RIS_POSTGRES_DSN'])
cursor = conn.cursor()

for model in models:
    try:
        cursor.execute('''
            INSERT INTO ml_models (model_id, model_type, version, model_data, is_active)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (model_id) DO NOTHING
        ''', (
            model['model_id'],
            model['model_type'], 
            model['version'],
            json.dumps(model['model_data']),
            model['is_active']
        ))
        print(f'Initialized model: {model[\"model_id\"]}')
    except Exception as e:
        print(f'Error initializing model {model[\"model_id\"]}: {e}')

conn.commit()
conn.close()
print('ML model initialization completed')
"

# Add some basic feature flags
echo "Initializing feature flags..."
python3 -c "
import os
import psycopg2

flags = [
    ('ai_predictions', 'AI prediction endpoints', True, 100),
    ('scenario_studio', 'Scenario simulation features', True, 100),
    ('advanced_alerts', 'Advanced alert configurations', True, 50),
    ('data_exports', 'Data export functionality', True, 100)
]

conn = psycopg2.connect(os.environ['RIS_POSTGRES_DSN'])
cursor = conn.cursor()

for name, desc, enabled, rollout in flags:
    try:
        cursor.execute('''
            INSERT INTO feature_flags (name, description, is_enabled, rollout_percentage)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (name) DO NOTHING
        ''', (name, desc, enabled, rollout))
        print(f'Initialized feature flag: {name}')
    except Exception as e:
        print(f'Error initializing flag {name}: {e}')

conn.commit()
conn.close()
print('Feature flags initialization completed')
"

echo "Production initialization completed successfully"