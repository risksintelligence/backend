"""
Tests for ETL utility functions
"""

import pytest
import asyncio
import pandas as pd
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

from etl.utils.connectors import APIConnector, DatabaseConnector, ConnectorManager
from etl.utils.validators import DataValidator, SchemaValidator, QualityValidator, ValidationResult
from etl.utils.notifications import NotificationManager, EmailNotifier, WebhookNotifier, NotificationEvent, NotificationLevel


class TestAPIConnector:
    """Test API connector functionality"""
    
    @pytest.fixture
    def api_connector(self):
        config = {
            'base_url': 'https://api.example.com',
            'headers': {'User-Agent': 'Test/1.0'},
            'timeout': 30,
            'rate_limit': 100,
            'cache_ttl': 3600
        }
        return APIConnector("test_api", config)
    
    @pytest.mark.asyncio
    async def test_connector_initialization(self, api_connector):
        """Test API connector initializes correctly"""
        assert api_connector.name == "test_api"
        assert api_connector.base_url == 'https://api.example.com'
        assert api_connector.rate_limit == 100
        assert api_connector.cache_ttl == 3600
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, api_connector):
        """Test rate limiting functionality"""
        # Fill up the rate limit
        for _ in range(api_connector.rate_limit):
            assert api_connector._check_rate_limit() is True
        
        # Next request should be rate limited
        assert api_connector._check_rate_limit() is False
    
    @pytest.mark.asyncio
    async def test_cache_key_generation(self, api_connector):
        """Test cache key generation"""
        endpoint = "/test"
        params = {"param1": "value1", "param2": "value2"}
        
        key1 = api_connector._generate_cache_key(endpoint, params)
        key2 = api_connector._generate_cache_key(endpoint, params)
        
        # Same input should generate same key
        assert key1 == key2
        
        # Different params should generate different key
        different_params = {"param1": "different", "param2": "value2"}
        key3 = api_connector._generate_cache_key(endpoint, different_params)
        assert key1 != key3
    
    @pytest.mark.asyncio
    async def test_fetch_data_with_cache(self, api_connector, mock_http_session):
        """Test data fetching with caching"""
        with patch.object(api_connector, 'session', mock_http_session):
            with patch.object(api_connector.cache, 'get', return_value=None) as mock_cache_get:
                with patch.object(api_connector.cache, 'set') as mock_cache_set:
                    
                    # First call should hit API
                    result = await api_connector.fetch_data("/test", {"param": "value"})
                    
                    mock_cache_get.assert_called_once()
                    mock_cache_set.assert_called_once()
                    assert result == {"status": "success", "data": []}
    
    @pytest.mark.asyncio
    async def test_error_handling_with_fallback(self, api_connector):
        """Test error handling with cache fallback"""
        cached_data = {"cached": True, "timestamp": datetime.now().isoformat()}
        
        with patch.object(api_connector, 'session') as mock_session:
            # Mock session to raise an exception
            mock_session.get.side_effect = Exception("Network error")
            
            with patch.object(api_connector.cache, 'get', return_value=cached_data):
                # Should return cached data on error
                result = await api_connector.fetch_data("/test")
                assert result == cached_data
    
    @pytest.mark.asyncio
    async def test_health_check(self, api_connector):
        """Test health check functionality"""
        api_connector.config['health_endpoint'] = '/health'
        
        with patch.object(api_connector, 'fetch_data', return_value={"status": "ok"}):
            health_result = await api_connector.health_check()
            
            assert health_result["status"] == "healthy"
            assert "response_time_ms" in health_result
            assert "timestamp" in health_result


class TestDatabaseConnector:
    """Test database connector functionality"""
    
    @pytest.fixture
    def db_connector(self):
        config = {
            'connection_string': 'postgresql://test:test@localhost:5432/testdb'
        }
        return DatabaseConnector("test_db", config)
    
    @pytest.mark.asyncio
    async def test_connector_initialization(self, db_connector):
        """Test database connector initializes correctly"""
        assert db_connector.name == "test_db"
        assert db_connector.connection_string == 'postgresql://test:test@localhost:5432/testdb'
    
    @pytest.mark.asyncio
    async def test_fetch_data(self, db_connector):
        """Test database data fetching"""
        mock_pool = AsyncMock()
        mock_connection = AsyncMock()
        mock_rows = [{"id": 1, "name": "test"}, {"id": 2, "name": "test2"}]
        
        mock_connection.fetch.return_value = mock_rows
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        
        db_connector.pool = mock_pool
        
        result = await db_connector.fetch_data("SELECT * FROM test_table")
        
        assert len(result) == 2
        assert result[0]["id"] == 1
        assert result[1]["name"] == "test2"
    
    @pytest.mark.asyncio
    async def test_execute_query(self, db_connector):
        """Test database query execution"""
        mock_pool = AsyncMock()
        mock_connection = AsyncMock()
        mock_connection.execute.return_value = "UPDATE 5"
        
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        db_connector.pool = mock_pool
        
        result = await db_connector.execute("UPDATE test_table SET name = $1", ("new_name",))
        
        assert result == 5
    
    @pytest.mark.asyncio
    async def test_health_check(self, db_connector):
        """Test database health check"""
        mock_pool = AsyncMock()
        mock_connection = AsyncMock()
        mock_connection.fetch.return_value = [{"health_check": 1}]
        
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        db_connector.pool = mock_pool
        
        health_result = await db_connector.health_check()
        
        assert health_result["status"] == "healthy"
        assert "response_time_ms" in health_result


class TestConnectorManager:
    """Test connector manager functionality"""
    
    @pytest.fixture
    def connector_manager(self):
        return ConnectorManager()
    
    def test_add_connector(self, connector_manager):
        """Test adding connectors to manager"""
        mock_connector = Mock()
        mock_connector.name = "test_connector"
        
        connector_manager.add_connector("test", mock_connector)
        
        assert "test" in connector_manager.connectors
        assert connector_manager.connectors["test"] == mock_connector
    
    @pytest.mark.asyncio
    async def test_connect_all(self, connector_manager):
        """Test connecting all connectors"""
        mock_connector1 = AsyncMock()
        mock_connector1.connect.return_value = True
        mock_connector2 = AsyncMock()
        mock_connector2.connect.return_value = False
        
        connector_manager.add_connector("conn1", mock_connector1)
        connector_manager.add_connector("conn2", mock_connector2)
        
        results = await connector_manager.connect_all()
        
        assert results["conn1"] is True
        assert results["conn2"] is False
    
    @pytest.mark.asyncio
    async def test_health_check_all(self, connector_manager):
        """Test health checking all connectors"""
        mock_connector = AsyncMock()
        mock_connector.health_check.return_value = {"status": "healthy"}
        
        connector_manager.add_connector("test", mock_connector)
        
        results = await connector_manager.health_check_all()
        
        assert "test" in results
        assert results["test"]["status"] == "healthy"


class TestDataValidators:
    """Test data validation utilities"""
    
    @pytest.fixture
    def schema_validator(self):
        return SchemaValidator()
    
    @pytest.fixture
    def quality_validator(self):
        return QualityValidator()
    
    @pytest.fixture
    def sample_dataframe(self):
        return pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=10, freq="D"),
            "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "series_id": ["TEST"] * 10
        })
    
    @pytest.mark.asyncio
    async def test_schema_validation(self, schema_validator, sample_dataframe):
        """Test schema validation"""
        config = {
            "source_name": "test_source",
            "schema_name": "fred_data"
        }
        
        report = await schema_validator.validate(sample_dataframe, config)
        
        assert report.source_name == "test_source"
        assert report.total_checks > 0
    
    @pytest.mark.asyncio
    async def test_quality_validation(self, quality_validator, sample_dataframe):
        """Test data quality validation"""
        config = {
            "source_name": "test_source",
            "completeness_threshold": 80,
            "max_age_days": 7,
            "date_column": "date"
        }
        
        report = await quality_validator.validate(sample_dataframe, config)
        
        assert report.source_name == "test_source"
        assert report.total_checks > 0
    
    @pytest.mark.asyncio
    async def test_data_completeness_check(self, quality_validator):
        """Test data completeness validation"""
        # Create DataFrame with missing values
        df_with_nulls = pd.DataFrame({
            "col1": [1, 2, None, 4, 5],
            "col2": [1, None, None, 4, 5],
            "col3": [1, 2, 3, 4, 5]
        })
        
        config = {"source_name": "test", "completeness_threshold": 80}
        
        report = await quality_validator.validate(df_with_nulls, config)
        
        # Should detect completeness issues
        assert any("completeness" in result.rule_name for result in report.errors + report.warnings_list)
    
    @pytest.mark.asyncio
    async def test_data_freshness_check(self, quality_validator):
        """Test data freshness validation"""
        # Create DataFrame with old data
        old_dates = pd.date_range("2020-01-01", periods=5, freq="D")
        df_old = pd.DataFrame({
            "date": old_dates,
            "value": [1, 2, 3, 4, 5]
        })
        
        config = {
            "source_name": "test",
            "date_column": "date",
            "max_age_days": 7
        }
        
        report = await quality_validator.validate(df_old, config)
        
        # Should detect freshness issues
        freshness_results = [r for r in report.errors + report.warnings_list if "freshness" in r.rule_name]
        assert len(freshness_results) > 0
    
    def test_validation_result_creation(self):
        """Test ValidationResult creation"""
        result = ValidationResult(
            is_valid=True,
            rule_name="test_rule",
            message="Test passed",
            severity="info"
        )
        
        assert result.is_valid is True
        assert result.rule_name == "test_rule"
        assert result.message == "Test passed"
        assert result.severity == "info"
        assert isinstance(result.timestamp, datetime)


class TestNotificationSystem:
    """Test notification system"""
    
    @pytest.fixture
    def notification_manager(self):
        return NotificationManager()
    
    @pytest.fixture
    def email_notifier(self):
        config = {
            "smtp_server": "localhost",
            "smtp_port": 587,
            "from_email": "test@example.com",
            "to_emails": ["recipient@example.com"],
            "dry_run": True  # Don't actually send emails in tests
        }
        return EmailNotifier(config)
    
    @pytest.fixture
    def webhook_notifier(self):
        config = {
            "webhook_url": "https://hooks.example.com/webhook",
            "timeout": 30,
            "retry_count": 2
        }
        return WebhookNotifier(config)
    
    def test_notification_event_creation(self):
        """Test notification event creation"""
        event = NotificationEvent(
            level=NotificationLevel.WARNING,
            title="Test Alert",
            message="This is a test alert",
            source="test_system"
        )
        
        assert event.level == NotificationLevel.WARNING
        assert event.title == "Test Alert"
        assert event.message == "This is a test alert"
        assert event.source == "test_system"
        assert isinstance(event.timestamp, datetime)
    
    @pytest.mark.asyncio
    async def test_email_notification(self, email_notifier):
        """Test email notification sending"""
        event = NotificationEvent(
            level=NotificationLevel.ERROR,
            title="Test Error",
            message="Test error message",
            source="test_system"
        )
        
        # Should succeed in dry run mode
        result = await email_notifier.send_notification(event)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_webhook_notification(self, webhook_notifier, mock_http_session):
        """Test webhook notification sending"""
        event = NotificationEvent(
            level=NotificationLevel.INFO,
            title="Test Info",
            message="Test info message",
            source="test_system"
        )
        
        with patch('aiohttp.ClientSession', return_value=mock_http_session):
            result = await webhook_notifier.send_notification(event)
            assert result is True
    
    def test_notification_level_filtering(self, email_notifier):
        """Test notification level filtering"""
        email_notifier.min_level = NotificationLevel.WARNING
        
        # Should notify for warning level
        warning_event = NotificationEvent(
            level=NotificationLevel.WARNING,
            title="Warning",
            message="Warning message",
            source="test"
        )
        assert email_notifier.should_notify(warning_event) is True
        
        # Should not notify for info level (below threshold)
        info_event = NotificationEvent(
            level=NotificationLevel.INFO,
            title="Info",
            message="Info message", 
            source="test"
        )
        assert email_notifier.should_notify(info_event) is False
    
    @pytest.mark.asyncio
    async def test_notification_manager_workflow(self, notification_manager):
        """Test notification manager workflow"""
        # Add mock notifiers
        mock_email = AsyncMock()
        mock_email.notify.return_value = True
        mock_webhook = AsyncMock()
        mock_webhook.notify.return_value = True
        
        notification_manager.add_notifier("email", mock_email)
        notification_manager.add_notifier("webhook", mock_webhook)
        
        event = NotificationEvent(
            level=NotificationLevel.CRITICAL,
            title="Critical Alert",
            message="Critical system alert",
            source="system"
        )
        
        results = await notification_manager.notify(event)
        
        assert "email" in results
        assert "webhook" in results
        assert results["email"] is True
        assert results["webhook"] is True
    
    @pytest.mark.asyncio
    async def test_pipeline_notification_helpers(self, notification_manager):
        """Test pipeline-specific notification helpers"""
        with patch.object(notification_manager, 'notify', return_value={"email": True}) as mock_notify:
            
            # Test pipeline start notification
            await notification_manager.notify_pipeline_start("test_pipeline", {"param": "value"})
            mock_notify.assert_called()
            
            # Verify notification event was created correctly
            call_args = mock_notify.call_args[0][0]
            assert call_args.level == NotificationLevel.INFO
            assert "Started" in call_args.title
            assert "test_pipeline" in call_args.title


class TestETLUtilsIntegration:
    """Integration tests for ETL utilities"""
    
    @pytest.mark.asyncio
    async def test_connector_validator_integration(self):
        """Test integration between connectors and validators"""
        # Create mock connector
        api_connector = APIConnector("test", {"base_url": "https://api.test.com"})
        
        # Mock data fetch
        mock_data = [
            {"date": "2023-01-01", "value": 100.0, "series_id": "TEST"},
            {"date": "2023-01-02", "value": 101.0, "series_id": "TEST"}
        ]
        
        with patch.object(api_connector, 'fetch_data', return_value=mock_data):
            # Fetch data
            data = await api_connector.fetch_data("/test")
            
            # Validate the fetched data
            validator = DataValidator()
            df = pd.DataFrame(data)
            
            validation_config = {
                "source_name": "test_api",
                "schema_name": "fred_data",
                "validate_schema": True,
                "validate_quality": True
            }
            
            results = await validator.validate_data_source(df, validation_config)
            
            assert "schema" in results or "quality" in results
    
    @pytest.mark.asyncio
    async def test_notification_validator_integration(self):
        """Test integration between validators and notifications"""
        notification_manager = NotificationManager()
        
        # Mock notifier
        mock_notifier = AsyncMock()
        mock_notifier.notify.return_value = True
        notification_manager.add_notifier("test", mock_notifier)
        
        # Create validation failure
        await notification_manager.notify_data_quality_issue(
            "test_source",
            "Data completeness below threshold",
            {"completeness": 0.5, "threshold": 0.8}
        )
        
        # Verify notification was sent
        mock_notifier.notify.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_error_propagation_through_utils(self):
        """Test error propagation through utility chain"""
        # Test connector error propagation
        api_connector = APIConnector("test", {"base_url": "https://api.test.com"})
        
        with patch.object(api_connector, 'session') as mock_session:
            mock_session.get.side_effect = Exception("Network timeout")
            
            # Should handle error gracefully
            with pytest.raises(Exception, match="Network timeout"):
                await api_connector.fetch_data("/test")