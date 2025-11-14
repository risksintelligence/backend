"""Test core functionality without importing full FastAPI routers."""
import pytest
from unittest.mock import AsyncMock, MagicMock

def test_scenario_sharing_core():
    """Test that scenario sharing core functionality works."""
    from src.services.scenario_sharing_service import (
        ScenarioSharingService, 
        PermissionLevel,
        SavedScenario,
        SharingError
    )
    
    # Test service can be initialized
    service = ScenarioSharingService(postgres_dsn="postgresql://test:test@localhost/test")
    assert service.postgres_dsn == "postgresql://test:test@localhost/test"
    
    # Test permission levels
    assert PermissionLevel.VIEW.value == "view"
    assert PermissionLevel.EDIT.value == "edit"
    assert PermissionLevel.ADMIN.value == "admin"

def test_collaboration_core():
    """Test that collaboration core functionality works."""
    from src.services.scenario_collaboration_service import (
        ScenarioCollaborationService,
        ActivityType,
        ScenarioComment
    )
    
    # Test service can be initialized 
    service = ScenarioCollaborationService(postgres_dsn="postgresql://test:test@localhost/test")
    assert service.postgres_dsn == "postgresql://test:test@localhost/test"
    
    # Test activity types
    assert ActivityType.CREATED.value == "created"
    assert ActivityType.COMMENTED.value == "commented"
    assert ActivityType.FORKED.value == "forked"

def test_advanced_exports_core():
    """Test that advanced exports core functionality works."""
    from src.services.advanced_export_service import (
        AdvancedExportService,
        ExportFormat,
        ExportScope,
        ExportRequest
    )
    
    # Test service can be initialized
    service = AdvancedExportService(
        postgres_dsn="postgresql://test:test@localhost/test",
        export_base_path="/tmp/test_exports"
    )
    assert service.postgres_dsn == "postgresql://test:test@localhost/test"
    assert service.export_base_path == "/tmp/test_exports"
    
    # Test export formats and scopes
    assert ExportFormat.CSV.value == "csv"
    assert ExportFormat.JSON.value == "json"
    assert ExportFormat.EXCEL.value == "xlsx"
    
    assert ExportScope.SCENARIO_RUNS.value == "scenario_runs"
    assert ExportScope.SAVED_SCENARIOS.value == "saved_scenarios"
    
    # Test export request creation
    request = ExportRequest(
        user_id=1,
        scope=ExportScope.SCENARIO_RUNS,
        format=ExportFormat.CSV,
        filters={"date_from": "2023-01-01"},
        limit=100,
        include_metadata=True,
        share_publicly=False,
        expires_in_hours=24
    )
    assert request.user_id == 1
    assert request.limit == 100

def test_auth_service_core():
    """Test that auth service core functionality works."""
    from src.services.auth_service import (
        AuthService,
        User,
        AuthenticationError
    )
    
    # Test service can be initialized
    service = AuthService(postgres_dsn="postgresql://test:test@localhost/test")
    assert service.postgres_dsn == "postgresql://test:test@localhost/test"

def test_subscription_service_core():
    """Test that subscription service core functionality works."""
    from src.services.subscription_service import (
        SubscriptionService,
        FeatureCategory,
        SubscriptionTier
    )
    
    # Test service can be initialized
    service = SubscriptionService(postgres_dsn="postgresql://test:test@localhost/test")
    assert service.postgres_dsn == "postgresql://test:test@localhost/test"
    
    # Test enums
    assert FeatureCategory.COLLABORATION.value == "collaboration"
    assert FeatureCategory.ADVANCED_ANALYTICS.value == "advanced_analytics"
    assert SubscriptionTier.FREE.value == "free"
    assert SubscriptionTier.PREMIUM.value == "premium"

def test_alerts_delivery_enhanced():
    """Test that enhanced alerts delivery works.""" 
    from src.services.alerts_delivery import (
        AlertDeliveryService,
        DeliveredAlert
    )
    
    # Test service can be initialized
    service = AlertDeliveryService(postgres_dsn="postgresql://test:test@localhost/test")
    assert hasattr(service, '_send_email_with_retry')
    assert hasattr(service, '_send_webhook_with_retry')

def test_middleware_components():
    """Test that auth middleware components can be imported."""
    from src.api.middleware.auth import (
        AuthenticationRequired,
        OptionalAuth,
        check_subscription_tier,
        check_usage_limit
    )
    
    # Test middleware classes can be instantiated
    auth_required = AuthenticationRequired()
    assert auth_required.required_permission is None
    
    auth_with_permission = AuthenticationRequired("admin:*")  
    assert auth_with_permission.required_permission == "admin:*"
    
    optional_auth = OptionalAuth()
    assert optional_auth is not None

def test_database_schema_completeness():
    """Test that database schema includes all required tables."""
    from pathlib import Path
    
    schema_path = Path(__file__).parent.parent / "database" / "schema.sql"
    schema_content = schema_path.read_text()
    
    # Core tables
    core_tables = [
        "raw_observations", "computed_indices", "cache_metadata",
        "feature_store_snapshots", "model_registry", "scenario_runs"
    ]
    
    # New feature tables
    feature_tables = [
        "users", "api_keys", "user_sessions", "feature_flags",
        "saved_scenarios", "alert_thresholds", "scenario_shares", 
        "scenario_comments", "scenario_activity", "export_records",
        "alert_delivery_log", "subscription_usage", "deployment_actions"
    ]
    
    # ML and monitoring tables  
    ml_tables = ["ml_models", "ml_drift_metrics", "cron_executions"]
    
    all_tables = core_tables + feature_tables + ml_tables
    
    for table in all_tables:
        assert f"CREATE TABLE {table}" in schema_content, f"Missing table: {table}"
    
    # Check for indexes
    key_indexes = [
        "idx_users_email", "idx_saved_scenarios_user_id",
        "idx_scenario_shares_scenario_id", "idx_export_records_export_id"
    ]
    
    for index in key_indexes:
        assert f"CREATE INDEX {index}" in schema_content, f"Missing index: {index}"

def test_file_structure_completeness():
    """Test that all expected files exist."""
    from pathlib import Path
    
    base_path = Path(__file__).parent.parent / "src"
    
    # Core service files
    expected_services = [
        "services/scenario_sharing_service.py",
        "services/scenario_collaboration_service.py", 
        "services/advanced_export_service.py",
        "services/auth_service.py",
        "services/subscription_service.py"
    ]
    
    # API endpoint files
    expected_apis = [
        "api/v1/scenario_sharing.py",
        "api/v1/advanced_exports.py",
        "api/middleware/auth.py"
    ]
    
    all_expected = expected_services + expected_apis
    
    for file_path in all_expected:
        full_path = base_path / file_path
        assert full_path.exists(), f"Missing file: {file_path}"