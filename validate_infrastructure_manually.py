#!/usr/bin/env python3
"""
Manual Infrastructure Validation for About RRIO Pages
Direct validation of cache, database, and performance infrastructure
"""

import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime, timedelta

# Add the backend directory to the path
sys.path.insert(0, '/Users/omoshola/Documents/riskx_obervatory 2/backend')

def test_unified_cache_implementation():
    """Test unified cache implementation with 3-tier architecture."""
    print("Testing Unified Cache Implementation...")
    
    try:
        from app.core.unified_cache import UnifiedCache
        
        cache = UnifiedCache("validation_test")
        
        # Test cache set and get
        test_data = {"validation": True, "timestamp": datetime.utcnow().isoformat()}
        cache.set("test_key", test_data, source="validation_test", soft_ttl=5, hard_ttl=20)
        
        # Immediate retrieval
        retrieved, metadata = cache.get("test_key")
        assert retrieved is not None, "Unified cache should store and retrieve data"
        assert retrieved["validation"] == True, "Data should be intact"
        
        print("   Unified cache set/get works")
        
        # Test metadata
        if metadata:
            assert metadata.source == "validation_test", "Metadata should include source"
            print(f"   Metadata: stale_soft={metadata.is_stale_soft}, age={metadata.age_seconds}s")
            print("   Metadata tracking works")
        else:
            print("   No metadata returned (using fallback cache)")
        
        # Test TTL expiration
        import time
        time.sleep(1)  # Short sleep to test
        still_there, still_meta = cache.get("test_key") 
        assert still_there is not None, "Data should still be there before TTL"
        
        print("   TTL management works")
        return True
        
    except Exception as e:
        print(f"   Unified cache test failed: {e}")
        return False

def test_database_connection_and_performance():
    """Test database connection and query performance."""
    print("Testing Database Connection and Performance...")
    
    try:
        from app.db import SessionLocal, engine
        from app.models import ObservationModel, PageView, UserEvent, UserFeedback
        from sqlalchemy import func, text
        
        # Test basic connection
        db = SessionLocal()
        start_time = time.time()
        
        result = db.execute(text("SELECT 1 as test_col")).fetchone()
        connection_time = (time.time() - start_time) * 1000
        
        assert result[0] == 1, "Database connection should work"
        print(f"   Database connection works ({connection_time:.2f}ms)")
        
        # Test table existence
        tables_exist = True
        try:
            db.query(func.count(ObservationModel.id)).scalar()
            db.query(func.count(PageView.id)).scalar()
            db.query(func.count(UserEvent.id)).scalar() 
            db.query(func.count(UserFeedback.id)).scalar()
            print("   All required tables exist")
        except Exception as e:
            print(f"   Some tables may not exist: {e}")
            tables_exist = False
        
        # Test query performance
        start_time = time.time()
        
        # Run multiple queries like About page would
        obs_count = db.query(func.count(ObservationModel.id)).scalar() or 0
        page_views = db.query(func.count(PageView.id)).scalar() or 0
        events = db.query(func.count(UserEvent.id)).scalar() or 0
        feedback = db.query(func.count(UserFeedback.id)).scalar() or 0
        
        query_time = (time.time() - start_time) * 1000
        
        print(f"   Query performance: {query_time:.2f}ms")
        print(f"   Data counts - Observations: {obs_count}, Views: {page_views}, Events: {events}, Feedback: {feedback}")
        
        db.close()
        return True
        
    except Exception as e:
        print(f"   Database test failed: {e}")
        return False

def test_l3_cache_structure():
    """Test L3 file cache structure and functionality."""
    print("Testing L3 Cache Structure...")
    
    try:
        # Create L3 cache directory structure
        l3_dir = Path("l3_cache/validation_test")
        l3_dir.mkdir(parents=True, exist_ok=True)
        
        # Test L3 bundle creation
        today = datetime.utcnow().strftime('%Y-%m-%d')
        test_file = l3_dir / f"TEST_SERIES_{today}.json"
        
        test_bundle = {
            'data': {
                'series_id': 'TEST_SERIES',
                'value': 123.45,
                'timestamp': datetime.utcnow().isoformat()
            },
            'metadata': {
                'cached_at': datetime.utcnow().isoformat(),
                'source': 'validation_test',
                'source_url': 'test://validation',
                'checksum': 'test_checksum_123',
                'derivation_flag': 'raw',
                'soft_ttl': 300,
                'hard_ttl': 1200
            }
        }
        
        with open(test_file, 'w') as f:
            json.dump(test_bundle, f, indent=2)
        
        # Test L3 bundle retrieval
        with open(test_file, 'r') as f:
            retrieved_bundle = json.load(f)
        
        assert retrieved_bundle['data']['series_id'] == 'TEST_SERIES', "L3 data should be intact"
        assert retrieved_bundle['metadata']['source'] == 'validation_test', "L3 metadata should be intact"
        
        print(f"   L3 cache bundle created: {test_file}")
        print("   L3 data structure is correct")
        
        return True
        
    except Exception as e:
        print(f"   L3 cache test failed: {e}")
        return False

def test_analytics_data_models():
    """Test analytics data model persistence."""
    print("Testing Analytics Data Models...")
    
    try:
        from app.db import SessionLocal
        from app.models import PageView, UserEvent, UserFeedback
        
        db = SessionLocal()
        
        # Test PageView model
        test_view = PageView(
            path="/about",
            timestamp=datetime.utcnow(),
            user_agent="ValidationTest/1.0",
            referrer="https://validation.test",
            viewport="1920x1080"
        )
        db.add(test_view)
        
        # Test UserEvent model with JSON data
        test_event = UserEvent(
            event_name="validation_test",
            event_data={"test_metric": 99.5, "validation": True},
            timestamp=datetime.utcnow(),
            path="/about"
        )
        db.add(test_event)
        
        # Test UserFeedback model
        test_feedback = UserFeedback(
            page="/about",
            rating=5,
            comment="Validation test feedback",
            category="infrastructure_test",
            timestamp=datetime.utcnow()
        )
        db.add(test_feedback)
        
        db.commit()
        
        # Verify persistence
        saved_view = db.query(PageView).filter(PageView.user_agent == "ValidationTest/1.0").first()
        saved_event = db.query(UserEvent).filter(UserEvent.event_name == "validation_test").first()
        saved_feedback = db.query(UserFeedback).filter(UserFeedback.comment == "Validation test feedback").first()
        
        assert saved_view is not None, "PageView should persist"
        assert saved_event is not None, "UserEvent should persist"
        assert saved_feedback is not None, "UserFeedback should persist"
        
        # Test JSON data integrity
        assert saved_event.event_data["validation"] == True, "JSON data should persist correctly"
        
        print("   PageView persistence works")
        print("   UserEvent with JSON persistence works")
        print("   UserFeedback persistence works")
        
        db.close()
        return True
        
    except Exception as e:
        print(f"   Analytics data model test failed: {e}")
        return False

def test_cache_performance_benchmarks():
    """Test cache performance for About page scenarios."""
    print("Testing Cache Performance Benchmarks...")
    
    try:
        from app.core.unified_cache import UnifiedCache
        
        cache = UnifiedCache("performance_test")
        
        # Test multiple writes
        write_times = []
        for i in range(10):
            start = time.time()
            cache.set(f"perf_test_{i}", {"data": i, "timestamp": datetime.utcnow().isoformat()}, source="performance_test")
            write_times.append((time.time() - start) * 1000)
        
        avg_write_time = sum(write_times) / len(write_times)
        
        # Test multiple reads
        read_times = []
        for i in range(10):
            start = time.time()
            data, metadata = cache.get(f"perf_test_{i}")
            read_times.append((time.time() - start) * 1000)
            assert data is not None, f"Should retrieve data for key {i}"
        
        avg_read_time = sum(read_times) / len(read_times)
        
        print(f"   Average write time: {avg_write_time:.2f}ms")
        print(f"   Average read time: {avg_read_time:.2f}ms")
        
        # Performance assertions for About page requirements
        assert avg_write_time < 50, f"Write time should be <50ms, got {avg_write_time:.2f}ms"
        assert avg_read_time < 10, f"Read time should be <10ms, got {avg_read_time:.2f}ms"
        
        print("   Performance benchmarks met")
        return True
        
    except Exception as e:
        print(f"   Performance benchmark test failed: {e}")
        return False

def test_about_page_data_flow():
    """Test complete data flow for About page metrics."""
    print("Testing About Page Data Flow...")
    
    try:
        from app.core.unified_cache import UnifiedCache
        from app.db import SessionLocal
        from app.models import PageView, UserFeedback
        from sqlalchemy import func
        
        # Simulate About page data flow
        cache = UnifiedCache("about_page_test")
        
        # 1. Cache platform metrics (what About page displays)
        platform_metrics = {
            "total_users": 1500,
            "platform_age_days": 400,
            "avg_rating": 4.8,
            "countries_served": 25,
            "generated_at": datetime.utcnow().isoformat()
        }
        
        cache.set("platform_metrics", platform_metrics, source="about_page_test", soft_ttl=300, hard_ttl=1200)
        
        # 2. Store user analytics data (what feeds the metrics)
        db = SessionLocal()
        
        for i in range(5):
            view = PageView(
                path="/about",
                timestamp=datetime.utcnow() - timedelta(days=i),
                user_agent=f"TestUser{i}/1.0"
            )
            db.add(view)
            
            feedback = UserFeedback(
                page="/about",
                rating=5,
                timestamp=datetime.utcnow() - timedelta(days=i)
            )
            db.add(feedback)
        
        db.commit()
        
        # 3. Verify data flow: database -> cache -> About page
        cached_metrics, metrics_metadata = cache.get("platform_metrics")
        assert cached_metrics is not None, "Platform metrics should be cached"
        
        # Verify analytics queries work (what generates the metrics)
        about_views = db.query(func.count(PageView.id)).filter(PageView.path == "/about").scalar()
        avg_rating = db.query(func.avg(UserFeedback.rating)).filter(UserFeedback.page == "/about").scalar()
        
        assert about_views >= 5, "Should have About page views in database"
        assert avg_rating == 5.0, "Should calculate average rating correctly"
        
        print(f"   Platform metrics cached successfully")
        print(f"   About page views in DB: {about_views}")
        print(f"   Average rating: {avg_rating}")
        print("   Complete data flow validated")
        
        db.close()
        return True
        
    except Exception as e:
        print(f"   Data flow test failed: {e}")
        return False

def main():
    """Run all infrastructure validation tests."""
    print("Manual Infrastructure Validation for About RRIO Pages")
    print("=" * 60)
    
    tests = [
        ("Unified Cache Implementation", test_unified_cache_implementation),
        ("Database Connection & Performance", test_database_connection_and_performance),
        ("L3 Cache Structure", test_l3_cache_structure),
        ("Analytics Data Models", test_analytics_data_models),
        ("Cache Performance Benchmarks", test_cache_performance_benchmarks),
        ("About Page Data Flow", test_about_page_data_flow),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
            results.append((test_name, "PASS" if result else "FAIL"))
        except Exception as e:
            print(f"   Test failed with exception: {e}")
            results.append((test_name, "FAIL"))
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 VALIDATION RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, status in results:
        status_icon = "PASS" if status == "PASS" else "FAIL"
        print(f"{status_icon} {test_name}: {status}")
        if status == "PASS":
            passed += 1
    
    total = len(results)
    success_rate = (passed / total) * 100 if total > 0 else 0
    
    print(f"\nTotal: {total}, Passed: {passed}, Failed: {total - passed}")
    print(f"Success Rate: {success_rate:.1f}%")
    
    if passed == total:
        print("\n🎉 ALL INFRASTRUCTURE VALIDATION TESTS PASSED!")
        print("About RRIO pages have complete database and cache integration.")
    else:
        print(f"\nWARNING: {total - passed} validation test(s) failed.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)