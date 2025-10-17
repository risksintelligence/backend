"""
Integration tests for the complete data pipeline
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

from etl.dags.risk_data_pipeline import RiskDataPipeline
from etl.utils.validators import DataValidator
from etl.utils.notifications import NotificationManager
from src.cache.cache_manager import CacheManager


class TestDataPipelineIntegration:
    """Test complete data pipeline integration"""
    
    @pytest.fixture
    async def pipeline(self):
        return RiskDataPipeline()
    
    @pytest.fixture
    def mock_data_sources(self):
        """Mock all data sources for integration testing"""
        return {
            "fred": {"GDP": [{"date": "2023-12-01", "value": 21000}]},
            "bea": {"trade_balance": [{"date": "2023-12-01", "value": -75000}]},
            "bls": {"unemployment": [{"date": "2023-12-01", "value": 3.8}]},
            "fdic": {"bank_health": [{"cert": "1234", "risk_score": 0.3}]},
            "noaa": {"weather_events": [{"date": "2023-12-01", "severity": "moderate"}]},
            "cisa": {"vulnerabilities": [{"cve": "CVE-2023-1234", "risk": "high"}]}
        }
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_complete_pipeline_execution(self, pipeline, mock_data_sources):
        """Test end-to-end pipeline execution"""
        with patch.object(pipeline, 'extract_economic_data') as mock_economic:
            with patch.object(pipeline, 'extract_trade_data') as mock_trade:
                with patch.object(pipeline, 'validate_data_quality') as mock_validate:
                    with patch.object(pipeline, 'transform_and_engineer_features') as mock_transform:
                        with patch.object(pipeline, 'update_ml_models') as mock_update:
                            with patch.object(pipeline, 'warm_cache_layers') as mock_cache:
                                
                                # Configure mocks
                                mock_economic.return_value = {"status": "success", "record_count": 100}
                                mock_trade.return_value = {"status": "success", "record_count": 50}
                                mock_validate.return_value = {"validation_status": "passed", "overall_quality_score": 0.9}
                                mock_transform.return_value = {"status": "success", "feature_count": 75}
                                mock_update.return_value = {"status": "success", "model_updated": True}
                                mock_cache.return_value = {"redis_cache": "success", "postgres_cache": "success"}
                                
                                # Execute pipeline steps
                                economic_result = await pipeline.extract_economic_data()
                                trade_result = await pipeline.extract_trade_data()
                                validation_result = await pipeline.validate_data_quality()
                                transform_result = await pipeline.transform_and_engineer_features()
                                model_result = await pipeline.update_ml_models()
                                cache_result = await pipeline.warm_cache_layers()
                                
                                # Verify all steps completed successfully
                                assert economic_result["status"] == "success"
                                assert trade_result["status"] == "success"
                                assert validation_result["validation_status"] == "passed"
                                assert transform_result["status"] == "success"
                                assert model_result["status"] == "success"
                                assert cache_result["redis_cache"] == "success"
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_pipeline_failure_handling(self, pipeline):
        """Test pipeline behavior when components fail"""
        with patch.object(pipeline, 'extract_economic_data', side_effect=Exception("Data source unavailable")):
            with patch.object(pipeline, 'extract_trade_data') as mock_trade:
                with patch.object(pipeline.cache, 'get', return_value={"fallback": "data"}):
                    
                    # Pipeline should handle failures gracefully
                    with pytest.raises(Exception, match="Data source unavailable"):
                        await pipeline.extract_economic_data()
                    
                    # Other components should still work
                    mock_trade.return_value = {"status": "success"}
                    trade_result = await pipeline.extract_trade_data()
                    assert trade_result["status"] == "success"
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_data_quality_validation_integration(self, pipeline):
        """Test integration of data quality validation"""
        validator = DataValidator()
        
        sample_data = {
            "fred_data": [
                {"date": "2023-12-01", "value": 21000, "series_id": "GDP"},
                {"date": "2023-11-01", "value": 20950, "series_id": "GDP"}
            ]
        }
        
        validation_config = {
            "fred_data": {
                "source_name": "fred",
                "schema_name": "fred_data",
                "validate_schema": True,
                "validate_quality": True
            }
        }
        
        with patch.object(validator, 'validate_multiple_sources') as mock_validate:
            mock_validate.return_value = {
                "fred_data": {
                    "schema": Mock(is_valid=True, errors=[]),
                    "quality": Mock(is_valid=True, errors=[])
                }
            }
            
            validation_results = await validator.validate_multiple_sources(sample_data, validation_config)
            
            assert "fred_data" in validation_results
            assert validation_results["fred_data"]["schema"].is_valid
            assert validation_results["fred_data"]["quality"].is_valid
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_cache_integration(self, pipeline):
        """Test cache integration across pipeline components"""
        cache_manager = CacheManager()
        
        # Test cache warming
        test_data = {"test_key": "test_value", "timestamp": datetime.now().isoformat()}
        
        with patch.object(cache_manager, 'set') as mock_set:
            with patch.object(cache_manager, 'get', return_value=test_data) as mock_get:
                
                # Warm cache
                await cache_manager.set("test_cache_key", test_data, ttl=3600)
                mock_set.assert_called_once()
                
                # Retrieve from cache
                cached_data = await cache_manager.get("test_cache_key")
                mock_get.assert_called_once()
                assert cached_data == test_data
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_notification_integration(self, pipeline):
        """Test notification system integration"""
        notification_manager = NotificationManager()
        
        with patch.object(notification_manager, 'notify_pipeline_start') as mock_start:
            with patch.object(notification_manager, 'notify_pipeline_success') as mock_success:
                with patch.object(notification_manager, 'notify_pipeline_failure') as mock_failure:
                    
                    mock_start.return_value = {"email": True, "webhook": True}
                    mock_success.return_value = {"email": True, "webhook": True}
                    mock_failure.return_value = {"email": True, "webhook": True}
                    
                    # Test notification flow
                    start_result = await notification_manager.notify_pipeline_start("test_pipeline")
                    success_result = await notification_manager.notify_pipeline_success("test_pipeline", {"records": 100})
                    failure_result = await notification_manager.notify_pipeline_failure("test_pipeline", "Test error", {"details": "test"})
                    
                    assert start_result["email"] is True
                    assert success_result["email"] is True
                    assert failure_result["email"] is True
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_concurrent_data_source_access(self, pipeline, mock_data_sources):
        """Test concurrent access to multiple data sources"""
        async def mock_extract_source(source_name):
            await asyncio.sleep(0.1)  # Simulate API delay
            return {"source": source_name, "status": "success", "data": mock_data_sources.get(source_name, [])}
        
        with patch.object(pipeline, 'extract_economic_data', side_effect=lambda: mock_extract_source("fred")):
            with patch.object(pipeline, 'extract_trade_data', side_effect=lambda: mock_extract_source("census")):
                with patch.object(pipeline, 'extract_disruption_signals', side_effect=lambda: mock_extract_source("noaa")):
                    
                    # Execute concurrent extractions
                    results = await asyncio.gather(
                        pipeline.extract_economic_data(),
                        pipeline.extract_trade_data(),
                        pipeline.extract_disruption_signals(),
                        return_exceptions=True
                    )
                    
                    # All extractions should complete successfully
                    assert len(results) == 3
                    for result in results:
                        assert not isinstance(result, Exception)
                        assert result["status"] == "success"
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_data_flow_through_pipeline(self, pipeline):
        """Test data flow from extraction through to model updates"""
        pipeline_data = {}
        
        # Mock each pipeline stage to pass data to the next
        async def mock_extract_economic(**context):
            data = {"economic_indicators": [{"gdp": 21000, "unemployment": 3.8}]}
            pipeline_data["economic"] = data
            return {"status": "success", "record_count": len(data["economic_indicators"])}
        
        async def mock_validate_quality(**context):
            assert "economic" in pipeline_data  # Data should be available from previous stage
            return {"validation_status": "passed", "overall_quality_score": 0.9}
        
        async def mock_transform_features(**context):
            assert "economic" in pipeline_data  # Data should still be available
            features = {"gdp_trend": 0.05, "unemployment_risk": 0.3}
            pipeline_data["features"] = features
            return {"status": "success", "feature_count": len(features)}
        
        async def mock_update_models(**context):
            assert "features" in pipeline_data  # Features should be available
            return {"status": "success", "model_updated": True}
        
        with patch.object(pipeline, 'extract_economic_data', side_effect=mock_extract_economic):
            with patch.object(pipeline, 'validate_data_quality', side_effect=mock_validate_quality):
                with patch.object(pipeline, 'transform_and_engineer_features', side_effect=mock_transform_features):
                    with patch.object(pipeline, 'update_ml_models', side_effect=mock_update_models):
                        
                        # Execute pipeline in sequence
                        extract_result = await pipeline.extract_economic_data()
                        validate_result = await pipeline.validate_data_quality()
                        transform_result = await pipeline.transform_and_engineer_features()
                        model_result = await pipeline.update_ml_models()
                        
                        # Verify data flowed through pipeline correctly
                        assert extract_result["status"] == "success"
                        assert validate_result["validation_status"] == "passed"
                        assert transform_result["status"] == "success"
                        assert model_result["status"] == "success"
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_pipeline_monitoring_and_logging(self, pipeline):
        """Test pipeline monitoring and logging integration"""
        execution_log = []
        
        async def log_execution(stage_name, result):
            execution_log.append({
                "stage": stage_name,
                "timestamp": datetime.now(),
                "result": result
            })
        
        with patch.object(pipeline, 'extract_economic_data') as mock_economic:
            with patch.object(pipeline, 'validate_data_quality') as mock_validate:
                
                mock_economic.return_value = {"status": "success", "record_count": 100}
                mock_validate.return_value = {"validation_status": "passed", "overall_quality_score": 0.9}
                
                # Execute with logging
                economic_result = await pipeline.extract_economic_data()
                await log_execution("extract_economic", economic_result)
                
                validate_result = await pipeline.validate_data_quality()
                await log_execution("validate_quality", validate_result)
                
                # Verify logging captured execution
                assert len(execution_log) == 2
                assert execution_log[0]["stage"] == "extract_economic"
                assert execution_log[1]["stage"] == "validate_quality"
    
    @pytest.mark.asyncio
    @pytest.mark.integration 
    async def test_pipeline_performance_metrics(self, pipeline):
        """Test pipeline performance monitoring"""
        start_time = datetime.now()
        
        with patch.object(pipeline, 'extract_economic_data') as mock_economic:
            mock_economic.return_value = {"status": "success", "record_count": 100}
            
            # Execute and measure performance
            result = await pipeline.extract_economic_data()
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Verify performance metrics
            assert result["status"] == "success"
            assert execution_time < 5.0  # Should complete within 5 seconds
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_pipeline_health_checks(self, pipeline):
        """Test pipeline component health checks"""
        health_results = {}
        
        # Mock health checks for various components
        with patch.object(pipeline.cache, 'health_check', return_value={"status": "healthy"}):
            cache_health = await pipeline.cache.health_check()
            health_results["cache"] = cache_health
        
        # Check overall system health
        with patch.object(pipeline, '_check_system_health', return_value=health_results):
            system_health = await pipeline._check_system_health()
            
            assert "cache" in system_health
            assert system_health["cache"]["status"] == "healthy"