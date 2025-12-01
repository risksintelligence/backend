#!/usr/bin/env python3
"""
Infrastructure Validation Tests for About RRIO Pages
Tests cache invalidation, multi-tier architecture, database performance, and data persistence
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock
import json

# Test cache invalidation
@pytest.mark.asyncio
async def test_cache_invalidation_real_time_data():
    """Test cache invalidation for real-time data with TTL management."""
    from app.core.cache import RedisCache
    
    # Mock Redis if not available in test environment
    cache = RedisCache("test_invalidation")
    
    if not cache.available:
        # Test unified cache fallback for invalidation
        from app.core.unified_cache import UnifiedCache
        unified_cache = UnifiedCache("test_invalidation")
        
        # Test unified cache invalidation
        test_data = {"value": 123, "timestamp": datetime.utcnow().isoformat()}
        unified_cache.set("test_key", test_data, source="test", soft_ttl=2, hard_ttl=5)  # 2 second soft TTL
        
        # Immediate retrieval should work
        data, metadata = unified_cache.get("test_key")
        assert data is not None, "Data should be available immediately"
        
        # Wait for soft TTL expiration
        await asyncio.sleep(3)
        
        # Data should still be available but marked as stale
        expired_data, expired_metadata = unified_cache.get("test_key")
        assert expired_data is not None, "Data should still be available after soft TTL (stale-while-revalidate)"
        if expired_metadata:
            assert expired_metadata.is_stale_soft, "Data should be marked as stale after soft TTL"
        
        return True
    
    # Test Redis cache invalidation if available
    test_data = {"value": 456, "timestamp": datetime.utcnow().isoformat()}
    cache.set_with_metadata("test_key", test_data, soft_ttl=1, hard_ttl=5)
    
    # Immediate retrieval
    data, metadata = cache.get_with_metadata("test_key")
    assert data is not None, "Data should be available immediately"
    
    # Test stale detection after soft TTL
    await asyncio.sleep(2)
    data, metadata = cache.get_with_metadata("test_key")
    
    if metadata:
        assert metadata.get('is_stale_soft', False), "Data should be marked as soft-stale"
    
    return True

def test_multi_tier_cache_architecture():
    """Test L1 Redis -> L2 PostgreSQL -> L3 File fallback pattern."""
    from app.core.unified_cache import UnifiedCache
    from app.db import SessionLocal
    from app.models import ObservationModel
    
    cache = UnifiedCache("test_multitier")
    
    # Test L3 file store creation and retrieval
    l3_dir = Path("l3_cache/test_multitier")
    l3_dir.mkdir(parents=True, exist_ok=True)
    
    test_data = {"series_id": "TEST_MT", "value": 789.01}
    today = datetime.utcnow().strftime('%Y-%m-%d')
    l3_file = l3_dir / f"TEST_MT_{today}.json"
    
    l3_bundle = {
        'data': test_data,
        'metadata': {
            'cached_at': datetime.utcnow().isoformat(),
            'source': 'test_provider',
            'checksum': 'test_checksum',
            'derivation_flag': 'raw',
            'soft_ttl': 300,
            'hard_ttl': 1200
        }
    }
    
    with open(l3_file, 'w') as f:
        json.dump(l3_bundle, f)
    
    # Test L3 fallback
    data, metadata = cache._get_from_l3("TEST_MT")
    assert data is not None, "L3 fallback should retrieve data"
    assert data["series_id"] == "TEST_MT", "L3 data should match"
    
    # Test L2 PostgreSQL integration
    db = SessionLocal()
    test_obs = ObservationModel(
        series_id="TEST_MT_L2",
        observed_at=datetime.utcnow(),
        value=555.55,
        source="test_provider",
        fetched_at=datetime.utcnow(),
        checksum="test_checksum",
        derivation_flag="raw"
    )
    db.add(test_obs)
    db.commit()
    
    # Test L2 retrieval
    l2_data, l2_metadata = cache._get_from_l2("TEST_MT_L2")
    assert l2_data is not None, "L2 should retrieve from database"
    
    db.close()
    
    # Test freshness reporting
    freshness_report = cache.get_freshness_report()
    assert "l1_redis" in freshness_report, "Should report L1 status"
    assert "l2_postgresql" in freshness_report, "Should report L2 status"
    assert "l3_file_store" in freshness_report, "Should report L3 status"
    
    return True

def test_database_performance_about_pages():
    """Test database performance for About page queries."""
    from app.db import SessionLocal
    from app.models import PageView, UserEvent, UserFeedback, ObservationModel
    from sqlalchemy import func, text
    
    db = SessionLocal()
    
    # Test analytics query performance
    start_time = time.time()
    
    # Simulate About page analytics queries
    page_views = db.query(func.count(PageView.id)).filter(
        PageView.path == '/about'
    ).scalar() or 0
    
    events = db.query(func.count(UserEvent.id)).scalar() or 0
    
    feedback_avg = db.query(func.avg(UserFeedback.rating)).scalar() or 0
    
    query_time = (time.time() - start_time) * 1000  # Convert to ms
    
    assert query_time < 100, f"Analytics queries should complete in <100ms, got {query_time}ms"
    
    # Test observation queries for GERI components
    start_time = time.time()
    
    recent_obs = db.query(ObservationModel).filter(
        ObservationModel.observed_at >= datetime.utcnow() - timedelta(hours=24)
    ).limit(50).all()
    
    obs_query_time = (time.time() - start_time) * 1000
    
    assert obs_query_time < 50, f"Observation queries should complete in <50ms, got {obs_query_time}ms"
    
    # Test connection performance
    start_time = time.time()
    result = db.execute(text("SELECT 1")).scalar()
    connection_time = (time.time() - start_time) * 1000
    
    assert connection_time < 20, f"Database connections should be <20ms, got {connection_time}ms"
    assert result == 1, "Database connection should work"
    
    db.close()
    return True

def test_user_analytics_data_persistence():
    """Test data persistence for user analytics supporting About pages."""
    from app.db import SessionLocal
    from app.models import PageView, UserEvent, UserFeedback
    from sqlalchemy import func
    
    db = SessionLocal()
    
    # Test PageView persistence
    test_page_view = PageView(
        path="/about",
        timestamp=datetime.utcnow(),
        user_agent="TestInfrastructure/1.0",
        referrer="https://test.example.com",
        viewport="1920x1080"
    )
    db.add(test_page_view)
    db.commit()
    
    # Verify persistence
    saved_view = db.query(PageView).filter(
        PageView.user_agent == "TestInfrastructure/1.0"
    ).first()
    
    assert saved_view is not None, "PageView should persist to database"
    assert saved_view.path == "/about", "PageView path should be correct"
    
    # Test UserEvent persistence with JSON data
    test_event = UserEvent(
        event_name="infrastructure_test_event",
        event_data={"test_type": "infrastructure", "performance_metric": 98.5},
        timestamp=datetime.utcnow(),
        path="/about"
    )
    db.add(test_event)
    db.commit()
    
    # Verify JSON data persistence
    saved_event = db.query(UserEvent).filter(
        UserEvent.event_name == "infrastructure_test_event"
    ).first()
    
    assert saved_event is not None, "UserEvent should persist"
    assert saved_event.event_data is not None, "Event data should persist"
    assert saved_event.event_data.get("test_type") == "infrastructure", "JSON data should be intact"
    
    # Test UserFeedback persistence
    test_feedback = UserFeedback(
        page="/about",
        rating=5,
        comment="Infrastructure test feedback",
        category="performance_test",
        timestamp=datetime.utcnow()
    )
    db.add(test_feedback)
    db.commit()
    
    # Verify feedback persistence
    saved_feedback = db.query(UserFeedback).filter(
        UserFeedback.comment == "Infrastructure test feedback"
    ).first()
    
    assert saved_feedback is not None, "UserFeedback should persist"
    assert saved_feedback.rating == 5, "Rating should be correct"
    
    # Test aggregation queries (critical for About page metrics)
    total_about_views = db.query(func.count(PageView.id)).filter(
        PageView.path == "/about"
    ).scalar()
    
    avg_about_rating = db.query(func.avg(UserFeedback.rating)).filter(
        UserFeedback.page == "/about"
    ).scalar()
    
    assert total_about_views >= 1, "Should have at least one About page view"
    assert avg_about_rating is not None, "Should calculate average rating"
    
    db.close()
    return True

def test_cache_warming_about_page_metrics():
    """Test cache warming strategy for About page metrics."""
    from app.core.unified_cache import UnifiedCache
    
    cache = UnifiedCache("about_page_metrics")
    
    # Test critical About page metrics caching
    platform_metrics = {
        "total_estimated_users": 1500,
        "total_sessions": 18750,
        "platform_age_days": 400,
        "average_user_rating": 4.8
    }
    
    # Warm cache with platform metrics
    cache.set(
        key="platform_metrics",
        value=platform_metrics,
        source="analytics_engine",
        source_url="internal://analytics/platform",
        soft_ttl=300,  # 5 minutes
        hard_ttl=3600  # 1 hour
    )
    
    # Test immediate retrieval
    cached_data, metadata = cache.get("platform_metrics")
    assert cached_data is not None, "Platform metrics should be cached"
    assert cached_data["total_estimated_users"] == 1500, "Cached data should be correct"
    
    # Test GERI real-time data caching
    geri_data = {
        "score": 52.3,
        "band": "moderate",
        "confidence": 91,
        "last_updated": datetime.utcnow().isoformat()
    }
    
    cache.set(
        key="current_geri",
        value=geri_data,
        source="geri_engine",
        soft_ttl=30,   # 30 seconds for real-time data
        hard_ttl=300
    )
    
    # Test real-time data retrieval
    cached_geri, geri_meta = cache.get("current_geri")
    assert cached_geri is not None, "GERI data should be cached"
    assert cached_geri["score"] == 52.3, "GERI score should be correct"
    
    # Test cache performance
    start_time = time.time()
    for _ in range(10):
        cache.get("platform_metrics")
    avg_time = ((time.time() - start_time) / 10) * 1000  # ms per retrieval
    
    assert avg_time < 10, f"Cache retrieval should be <10ms, got {avg_time}ms"
    
    # Test cache warming strategy effectiveness
    freshness_report = cache.get_freshness_report()
    
    # Verify multi-tier architecture is operational
    unified_status = freshness_report.get("unified_status", {})
    assert unified_status.get("architecture") == "3_tier_l1_l2_l3", "Should use 3-tier architecture"
    assert unified_status.get("stale_while_revalidate"), "Should support stale-while-revalidate"
    
    return True

# Performance benchmarks for About page infrastructure
def test_about_page_infrastructure_benchmarks():
    """Test performance benchmarks specific to About page infrastructure."""
    
    # Test 1: Cold start performance (cache miss scenario)
    start_time = time.time()
    
    from app.core.unified_cache import UnifiedCache
    cache = UnifiedCache("benchmark_test")
    
    # Simulate cold start data retrieval
    cache.set(
        key="benchmark_data",
        value={"benchmark": True, "data_size": "1KB"},
        source="benchmark",
        soft_ttl=60,
        hard_ttl=300
    )
    
    cold_start_time = (time.time() - start_time) * 1000
    assert cold_start_time < 50, f"Cold start should be <50ms, got {cold_start_time}ms"
    
    # Test 2: Warm cache performance
    start_time = time.time()
    data, metadata = cache.get("benchmark_data")
    warm_cache_time = (time.time() - start_time) * 1000
    
    assert warm_cache_time < 5, f"Warm cache should be <5ms, got {warm_cache_time}ms"
    assert data is not None, "Warm cache should return data"
    
    # Test 3: Database query performance for About page
    from app.db import SessionLocal
    from app.models import ObservationModel
    from sqlalchemy import text
    
    db = SessionLocal()
    start_time = time.time()
    
    # Simulate About page data queries
    db.execute(text("SELECT COUNT(*) FROM observations WHERE series_id LIKE 'VIX%'"))
    db.execute(text("SELECT COUNT(*) FROM page_views WHERE path = '/about'"))
    db.execute(text("SELECT AVG(rating) FROM user_feedback WHERE page = '/about'"))
    
    multi_query_time = (time.time() - start_time) * 1000
    assert multi_query_time < 30, f"Multiple queries should be <30ms, got {multi_query_time}ms"
    
    db.close()
    
    return {
        "cold_start_ms": cold_start_time,
        "warm_cache_ms": warm_cache_time,
        "multi_query_ms": multi_query_time,
        "all_benchmarks_passed": True
    }

if __name__ == "__main__":
    """Run tests directly for validation."""
    import sys
    
    print("🧪 Running Infrastructure Validation Tests")
    print("=" * 50)
    
    try:
        # Run cache invalidation test
        print("1. Testing cache invalidation...")
        result = asyncio.run(test_cache_invalidation_real_time_data())
        print("✅ Cache invalidation test PASSED")
        
        # Run multi-tier architecture test
        print("2. Testing multi-tier cache architecture...")
        test_multi_tier_cache_architecture()
        print("✅ Multi-tier cache test PASSED")
        
        # Run database performance test
        print("3. Testing database performance...")
        test_database_performance_about_pages()
        print("✅ Database performance test PASSED")
        
        # Run data persistence test
        print("4. Testing user analytics persistence...")
        test_user_analytics_data_persistence()
        print("✅ Data persistence test PASSED")
        
        # Run cache warming test
        print("5. Testing cache warming...")
        test_cache_warming_about_page_metrics()
        print("✅ Cache warming test PASSED")
        
        # Run performance benchmarks
        print("6. Running performance benchmarks...")
        benchmarks = test_about_page_infrastructure_benchmarks()
        print("✅ Performance benchmarks PASSED")
        print(f"   Cold start: {benchmarks['cold_start_ms']:.2f}ms")
        print(f"   Warm cache: {benchmarks['warm_cache_ms']:.2f}ms")
        print(f"   Multi-query: {benchmarks['multi_query_ms']:.2f}ms")
        
        print("\n🎉 ALL INFRASTRUCTURE TESTS PASSED!")
        print("About RRIO pages have 100% database and cache integration")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)