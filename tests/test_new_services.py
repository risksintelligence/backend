"""Tests for new services implemented: scenario sharing, collaboration, and advanced exports."""
import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

# Test the new services can be imported
def test_scenario_sharing_service_imports():
    """Test that scenario sharing service can be imported."""
    try:
        from src.services.scenario_sharing_service import (
            ScenarioSharingService, 
            PermissionLevel,
            SavedScenario
        )
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import scenario sharing service: {e}")

def test_scenario_collaboration_service_imports():
    """Test that scenario collaboration service can be imported."""
    try:
        from src.services.scenario_collaboration_service import (
            ScenarioCollaborationService,
            ActivityType,
            ScenarioComment
        )
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import scenario collaboration service: {e}")

def test_advanced_export_service_imports():
    """Test that advanced export service can be imported."""
    try:
        from src.services.advanced_export_service import (
            AdvancedExportService,
            ExportFormat,
            ExportScope,
            ExportRequest
        )
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import advanced export service: {e}")

def test_scenario_sharing_api_imports():
    """Test that scenario sharing API can be imported."""
    try:
        from src.api.v1.scenario_sharing import router
        assert router is not None
    except ImportError as e:
        pytest.fail(f"Failed to import scenario sharing API: {e}")

def test_advanced_exports_api_imports():
    """Test that advanced exports API can be imported."""
    try:
        from src.api.v1.advanced_exports import router
        assert router is not None
    except ImportError as e:
        pytest.fail(f"Failed to import advanced exports API: {e}")

def test_permission_levels():
    """Test permission level enum values."""
    from src.services.scenario_sharing_service import PermissionLevel
    
    assert PermissionLevel.VIEW.value == "view"
    assert PermissionLevel.EDIT.value == "edit"  
    assert PermissionLevel.ADMIN.value == "admin"

def test_export_formats():
    """Test export format enum values."""
    from src.services.advanced_export_service import ExportFormat, ExportScope
    
    assert ExportFormat.CSV.value == "csv"
    assert ExportFormat.JSON.value == "json"
    assert ExportFormat.EXCEL.value == "xlsx"
    
    assert ExportScope.SCENARIO_RUNS.value == "scenario_runs"
    assert ExportScope.SAVED_SCENARIOS.value == "saved_scenarios"
    assert ExportScope.ALERT_HISTORY.value == "alert_history"
    assert ExportScope.COLLABORATION_ACTIVITY.value == "collaboration_activity"

def test_activity_types():
    """Test activity type enum values."""
    from src.services.scenario_collaboration_service import ActivityType
    
    assert ActivityType.CREATED.value == "created"
    assert ActivityType.UPDATED.value == "updated"
    assert ActivityType.SHARED.value == "shared"
    assert ActivityType.COMMENTED.value == "commented"
    assert ActivityType.FORKED.value == "forked"

@pytest.mark.asyncio
async def test_scenario_sharing_service_initialization():
    """Test that ScenarioSharingService can be initialized with mock DSN."""
    from src.services.scenario_sharing_service import ScenarioSharingService
    
    # Test initialization with explicit DSN
    service = ScenarioSharingService(postgres_dsn="postgresql://test:test@localhost/test")
    assert service.postgres_dsn == "postgresql://test:test@localhost/test"

@pytest.mark.asyncio  
async def test_export_request_creation():
    """Test that ExportRequest can be created with valid parameters."""
    from src.services.advanced_export_service import ExportRequest, ExportFormat, ExportScope
    
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
    assert request.scope == ExportScope.SCENARIO_RUNS
    assert request.format == ExportFormat.CSV
    assert request.filters == {"date_from": "2023-01-01"}
    assert request.limit == 100
    assert request.include_metadata is True
    assert request.share_publicly is False
    assert request.expires_in_hours == 24

def test_schema_includes_new_tables():
    """Test that the database schema includes our new tables."""
    from pathlib import Path
    
    schema_path = Path(__file__).parent.parent / "database" / "schema.sql"
    schema_content = schema_path.read_text()
    
    # Check for new tables we added
    required_tables = [
        "scenario_comments",
        "scenario_activity", 
        "alert_delivery_log",
        "export_records",
        "users",
        "api_keys",
        "user_sessions",
        "feature_flags",
        "saved_scenarios",
        "alert_thresholds",
        "scenario_shares"
    ]
    
    for table in required_tables:
        assert f"CREATE TABLE {table}" in schema_content, f"Missing table: {table}"

def test_auth_middleware_imports():
    """Test that auth middleware can be imported."""
    try:
        from src.api.middleware.auth import (
            require_auth,
            require_admin,
            limit_scenarios,
            limit_exports,
            check_subscription_tier
        )
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import auth middleware: {e}")

def test_auth_service_imports():
    """Test that auth service can be imported.""" 
    try:
        from src.services.auth_service import (
            AuthService,
            User,
            AuthenticationError,
            get_auth_service
        )
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import auth service: {e}")

def test_subscription_service_imports():
    """Test that subscription service can be imported."""
    try:
        from src.services.subscription_service import (
            SubscriptionService,
            FeatureCategory,
            SubscriptionTier,
            get_subscription_service
        )
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import subscription service: {e}")

def test_alerts_delivery_service_imports():
    """Test that alerts delivery service can be imported."""
    try:
        from src.services.alerts_delivery import (
            AlertDeliveryService,
            DeliveredAlert
        )
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import alerts delivery service: {e}")