#!/usr/bin/env python3
"""
Test script to verify worker functionality in Render environment.
"""
import os
import sys
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import get_settings
from app.db import SessionLocal, Base, engine
from sqlalchemy import text

def test_environment():
    """Test environment variables and configuration."""
    print("=== Environment Test ===")
    
    required_vars = [
        'RIS_POSTGRES_DSN',
        'RIS_JWT_SECRET'
    ]
    
    optional_vars = [
        'RIS_REDIS_URL',
        'RIS_ENV',
        'WORKER_ROLE'
    ]
    
    print("Required environment variables:")
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"  ✅ {var}: {'*' * 10}...{value[-10:] if len(value) > 10 else value}")
        else:
            print(f"  ❌ {var}: Not set")
    
    print("\nOptional environment variables:")
    for var in optional_vars:
        value = os.getenv(var)
        if value:
            print(f"  ✅ {var}: {value}")
        else:
            print(f"  ⚠️ {var}: Not set")

def test_configuration():
    """Test configuration loading."""
    print("\n=== Configuration Test ===")
    
    try:
        settings = get_settings()
        print(f"  ✅ Configuration loaded successfully")
        print(f"  📊 Environment: {settings.environment}")
        print(f"  🗄️ Database URL: {settings.database_url[:30]}...")
        if hasattr(settings, 'redis_url') and settings.redis_url:
            print(f"  🔄 Redis URL: {settings.redis_url[:30]}...")
        else:
            print(f"  ⚠️ Redis URL: Not configured")
        print(f"  📁 Models directory: {settings.models_dir}")
    except Exception as e:
        print(f"  ❌ Configuration failed: {e}")
        return False
    
    return True

def test_database():
    """Test database connection and table creation."""
    print("\n=== Database Test ===")
    
    try:
        # Test database connection
        print("  🔗 Testing database connection...")
        db = SessionLocal()
        result = db.execute(text("SELECT 1")).fetchone()
        db.close()
        print(f"  ✅ Database connection successful")
        
        # Test table creation
        print("  🛠️ Creating database tables...")
        Base.metadata.create_all(bind=engine)
        print(f"  ✅ Database tables created/verified")
        
        # Test model imports
        print("  📦 Testing model imports...")
        from app.models import ObservationModel, ModelMetadataModel, TransparencyLogModel
        print(f"  ✅ Model imports successful")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Database test failed: {e}")
        return False

def test_redis():
    """Test Redis connection (optional)."""
    print("\n=== Redis Test ===")
    
    redis_url = os.getenv('RIS_REDIS_URL')
    if not redis_url:
        print("  ⚠️ Redis URL not configured, skipping Redis test")
        return True
    
    try:
        from app.core.cache import RedisCache
        cache = RedisCache("test")
        
        if cache.available:
            print("  ✅ Redis connection successful")
            
            # Test basic operations
            cache.set("test_key", "test_value", ttl=60)
            value = cache.get("test_key")
            
            if value == "test_value":
                print("  ✅ Redis operations working")
                cache.delete("test_key")
                return True
            else:
                print("  ❌ Redis operations failed")
                return False
        else:
            print("  ⚠️ Redis not available (will use file cache fallback)")
            return True
    
    except Exception as e:
        print(f"  ❌ Redis test failed: {e}")
        return True  # Redis is optional

def test_ingestion():
    """Test data ingestion functionality."""
    print("\n=== Ingestion Test ===")
    
    try:
        # Test import of key modules
        print("  📦 Testing ingestion imports...")
        from app.services.ingestion import ingest_local_series
        from app.data.registry import SERIES_REGISTRY
        print(f"  ✅ Ingestion imports successful")
        print(f"  📊 Series registry has {len(SERIES_REGISTRY)} series")
        
        # Test provider failover
        print("  🔄 Testing provider failover...")
        from app.core.provider_failover import failover_manager
        print(f"  ✅ Provider failover manager available")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Ingestion test failed: {e}")
        return False

def test_training():
    """Test model training functionality."""
    print("\n=== Training Test ===")
    
    try:
        # Test import of key modules
        print("  📦 Testing training imports...")
        from app.services.training import train_all_models, fetch_training_window
        print(f"  ✅ Training imports successful")
        
        # Test scikit-learn availability
        print("  🧠 Testing ML library availability...")
        import sklearn
        import joblib
        print(f"  ✅ ML libraries available (sklearn {sklearn.__version__})")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Training test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 RRIO Worker Environment Test")
    print("=" * 50)
    
    tests = [
        test_environment,
        test_configuration,
        test_database,
        test_redis,
        test_ingestion,
        test_training
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  💥 Test crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("✅ All tests passed! Worker should be ready for deployment.")
        return 0
    else:
        print("❌ Some tests failed. Please fix issues before deploying.")
        return 1

if __name__ == "__main__":
    sys.exit(main())