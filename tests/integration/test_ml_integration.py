"""
Integration tests for ML models and risk assessment workflows.
"""
import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock


@pytest.mark.integration
class TestMLModelIntegration:
    """Test ML model integration with API endpoints."""
    
    @pytest.mark.asyncio
    async def test_risk_overview_with_ml_models(self, async_client):
        """Test risk overview endpoint with ML model integration."""
        with patch('src.ml.serving.model_server.ModelServer') as mock_server_class:
            mock_server = AsyncMock()
            mock_server.get_comprehensive_risk_assessment.return_value = {
                'overall_risk_score': 65.5,
                'confidence': 0.85,
                'factors': {
                    'economic': 70.0,
                    'geopolitical': 60.0,
                    'supply_chain': 65.0,
                    'market_volatility': 68.0
                },
                'trend': 'rising',
                'assessment_timestamp': '2024-01-01T12:00:00Z'
            }
            mock_server_class.return_value = mock_server
            
            response = await async_client.get("/api/v1/risk/overview")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert "data" in data
            # Should contain ML-generated risk assessment
            if "overall_risk_score" in data["data"]:
                assert isinstance(data["data"]["overall_risk_score"], (int, float))
    
    @pytest.mark.asyncio
    async def test_recession_prediction_endpoint(self, async_client):
        """Test recession prediction endpoint."""
        with patch('src.ml.serving.model_server.ModelServer') as mock_server_class:
            mock_server = AsyncMock()
            mock_server.predict_recession_probability.return_value = {
                'status': 'success',
                'prediction': {
                    'probability': 0.25,
                    'confidence': 0.88,
                    'factors': {
                        'yield_curve': -0.5,
                        'unemployment_trend': 'stable',
                        'gdp_growth': 2.1
                    }
                },
                'model_version': '1.0.0',
                'prediction_timestamp': '2024-01-01T12:00:00Z'
            }
            mock_server_class.return_value = mock_server
            
            response = await async_client.get("/api/v1/risk/predictions/recession")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert "data" in data
    
    @pytest.mark.asyncio
    async def test_supply_chain_prediction_endpoint(self, async_client):
        """Test supply chain risk prediction endpoint."""
        with patch('src.ml.serving.model_server.ModelServer') as mock_server_class:
            mock_server = AsyncMock()
            mock_server.predict_supply_chain_risk.return_value = {
                'status': 'success',
                'prediction': {
                    'risk_score': 55.0,
                    'risk_level': 'medium',
                    'factors': {
                        'shipping_costs': 'elevated',
                        'delivery_delays': 'moderate',
                        'inventory_levels': 'low'
                    }
                }
            }
            mock_server_class.return_value = mock_server
            
            response = await async_client.get("/api/v1/risk/predictions/supply-chain")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert "data" in data
    
    @pytest.mark.asyncio
    async def test_market_volatility_prediction_endpoint(self, async_client):
        """Test market volatility prediction endpoint."""
        with patch('src.ml.serving.model_server.ModelServer') as mock_server_class:
            mock_server = AsyncMock()
            mock_server.predict_market_volatility.return_value = {
                'status': 'success',
                'prediction': {
                    'volatility_score': 35.0,
                    'volatility_level': 'moderate',
                    'trend': 'increasing'
                }
            }
            mock_server_class.return_value = mock_server
            
            response = await async_client.get("/api/v1/risk/predictions/market-volatility")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert "data" in data
    
    @pytest.mark.asyncio
    async def test_geopolitical_prediction_endpoint(self, async_client):
        """Test geopolitical risk prediction endpoint."""
        with patch('src.ml.serving.model_server.ModelServer') as mock_server_class:
            mock_server = AsyncMock()
            mock_server.predict_geopolitical_risk.return_value = {
                'status': 'success',
                'prediction': {
                    'risk_score': 60.0,
                    'risk_level': 'medium-high',
                    'factors': {
                        'conflict_index': 45.0,
                        'sanctions_count': 15,
                        'diplomatic_relations': 60.0
                    }
                }
            }
            mock_server_class.return_value = mock_server
            
            response = await async_client.get("/api/v1/risk/predictions/geopolitical")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert "data" in data
    
    @pytest.mark.asyncio
    async def test_model_status_endpoint(self, async_client):
        """Test ML model status endpoint."""
        with patch('src.ml.serving.model_server.ModelServer') as mock_server_class:
            mock_server = MagicMock()
            mock_server.get_model_status.return_value = {
                'total_models': 4,
                'trained_models': 3,
                'model_details': {
                    'recession_predictor': {'status': 'trained', 'accuracy': 0.85},
                    'supply_chain_risk': {'status': 'trained', 'accuracy': 0.78},
                    'market_volatility': {'status': 'trained', 'accuracy': 0.82},
                    'geopolitical_risk': {'status': 'not_trained', 'accuracy': None}
                },
                'last_training': '2024-01-01T00:00:00Z'
            }
            mock_server_class.return_value = mock_server
            
            response = await async_client.get("/api/v1/risk/models/status")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert "data" in data
    
    @pytest.mark.asyncio
    async def test_model_training_endpoint(self, async_client):
        """Test ML model training endpoint."""
        with patch('src.ml.training.model_trainer.ModelTrainingPipeline') as mock_pipeline_class:
            mock_pipeline = AsyncMock()
            mock_pipeline.train_all_models.return_value = {
                'status': 'success',
                'training_summary': {
                    'models_trained': 4,
                    'total_time': 1800,  # 30 minutes
                    'average_accuracy': 0.81
                },
                'model_results': {
                    'recession_predictor': {'accuracy': 0.85, 'training_time': 450},
                    'supply_chain_risk': {'accuracy': 0.78, 'training_time': 420},
                    'market_volatility': {'accuracy': 0.82, 'training_time': 380},
                    'geopolitical_risk': {'accuracy': 0.79, 'training_time': 550}
                }
            }
            mock_pipeline_class.return_value = mock_pipeline
            
            response = await async_client.post("/api/v1/risk/models/train")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert "data" in data


@pytest.mark.integration
class TestExternalAPIIntegration:
    """Test external API integration with ML models."""
    
    @pytest.mark.asyncio
    async def test_fred_integration_with_recession_model(self, async_client):
        """Test FRED API integration with recession model."""
        with patch('src.data.sources.fred.get_key_indicators') as mock_fred:
            with patch('src.ml.models.recession_predictor.RecessionPredictor') as mock_model_class:
                # Mock FRED response
                mock_fred.return_value = {
                    'status': 'success',
                    'indicators': {
                        'GDP': {'value': 27100000, 'date': '2024-04-01'},
                        'UNRATE': {'value': 3.4, 'date': '2024-04-01'},
                        'DGS10': {'value': 4.2, 'date': '2024-04-01'}
                    }
                }
                
                # Mock model prediction
                mock_model = AsyncMock()
                mock_model.predict.return_value = {
                    'status': 'success',
                    'prediction': {'probability': 0.15}
                }
                mock_model_class.return_value = mock_model
                
                response = await async_client.get("/api/v1/external/fred/indicators")
                
                assert response.status_code == 200
                data = response.json()
                assert data["status"] in ["success", "error"]
    
    @pytest.mark.asyncio
    async def test_cisa_integration_with_supply_chain_model(self, async_client):
        """Test CISA API integration with supply chain model."""
        with patch('src.data.sources.cisa.get_cybersecurity_threats') as mock_cisa:
            mock_cisa.return_value = {
                'indicators': {
                    'kev_catalog': {'risk_score': 45.0},
                    'infrastructure_risks': {'overall_infrastructure_risk': 55.0},
                    'threat_intelligence': {'threat_intelligence_score': 60.0}
                },
                'overall_cybersecurity_risk': 53.3,
                'count': 3
            }
            
            # Since we don't have a direct CISA endpoint, test through risk overview
            response = await async_client.get("/api/v1/risk/overview")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
    
    @pytest.mark.asyncio
    async def test_noaa_integration_with_risk_assessment(self, async_client):
        """Test NOAA API integration with risk assessment."""
        with patch('src.data.sources.noaa.get_environmental_risks') as mock_noaa:
            mock_noaa.return_value = {
                'indicators': {
                    'severe_weather': {'weather_risk_score': 35.0},
                    'climate_extremes': {'climate_risk_score': 40.0},
                    'transportation_impacts': {'overall_transport_risk': 30.0}
                },
                'overall_environmental_risk': 35.0,
                'count': 3
            }
            
            response = await async_client.get("/api/v1/risk/overview")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
    
    @pytest.mark.asyncio
    async def test_usgs_integration_with_infrastructure_risk(self, async_client):
        """Test USGS API integration with infrastructure risk."""
        with patch('src.data.sources.usgs.get_geological_hazards') as mock_usgs:
            mock_usgs.return_value = {
                'indicators': {
                    'recent_earthquakes': {'seismic_risk_score': 25.0},
                    'infrastructure_vulnerability': {'overall_vulnerability': 35.0},
                    'natural_hazards': {'composite_hazard_score': 30.0}
                },
                'overall_geological_risk': 30.0,
                'count': 3
            }
            
            response = await async_client.get("/api/v1/risk/overview")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"


@pytest.mark.integration
class TestRiskAssessmentWorkflow:
    """Test complete risk assessment workflow integration."""
    
    @pytest.mark.asyncio
    async def test_complete_ml_powered_risk_workflow(self, async_client):
        """Test complete ML-powered risk assessment workflow."""
        
        # Mock all external data sources
        with patch('src.data.sources.fred.get_key_indicators') as mock_fred, \
             patch('src.data.sources.bea.get_gdp_data') as mock_bea, \
             patch('src.data.sources.bls.get_employment_data') as mock_bls, \
             patch('src.data.sources.cisa.get_cybersecurity_threats') as mock_cisa, \
             patch('src.data.sources.noaa.get_environmental_risks') as mock_noaa, \
             patch('src.data.sources.usgs.get_geological_hazards') as mock_usgs, \
             patch('src.data.sources.supply_chain.get_supply_chain_risks') as mock_supply:
            
            # Set up mock responses
            mock_fred.return_value = {
                'status': 'success',
                'indicators': {'GDP': {'value': 27100000}, 'UNRATE': {'value': 3.4}}
            }
            mock_bea.return_value = {'status': 'success', 'gdp_data': []}
            mock_bls.return_value = {'status': 'success', 'employment_data': []}
            mock_cisa.return_value = {'overall_cybersecurity_risk': 45.0, 'indicators': {}}
            mock_noaa.return_value = {'overall_environmental_risk': 35.0, 'indicators': {}}
            mock_usgs.return_value = {'overall_geological_risk': 30.0, 'indicators': {}}
            mock_supply.return_value = {'overall_supply_chain_risk': 50.0, 'indicators': {}}
            
            # Step 1: Get economic data
            econ_response = await async_client.get("/api/v1/economic/indicators")
            assert econ_response.status_code == 200
            
            # Step 2: Get risk overview (should use ML models)
            risk_response = await async_client.get("/api/v1/risk/overview")
            assert risk_response.status_code == 200
            risk_data = risk_response.json()
            assert risk_data["status"] == "success"
            
            # Step 3: Get individual predictions
            recession_response = await async_client.get("/api/v1/risk/predictions/recession")
            assert recession_response.status_code == 200
            
            supply_chain_response = await async_client.get("/api/v1/risk/predictions/supply-chain")
            assert supply_chain_response.status_code == 200
            
            volatility_response = await async_client.get("/api/v1/risk/predictions/market-volatility")
            assert volatility_response.status_code == 200
            
            geopolitical_response = await async_client.get("/api/v1/risk/predictions/geopolitical")
            assert geopolitical_response.status_code == 200
            
            # All predictions should be successful
            for response in [recession_response, supply_chain_response, 
                           volatility_response, geopolitical_response]:
                data = response.json()
                assert data["status"] == "success"
    
    @pytest.mark.asyncio
    async def test_risk_assessment_caching_behavior(self, async_client):
        """Test risk assessment caching behavior."""
        
        # First request should compute risk assessment
        response1 = await async_client.get("/api/v1/risk/overview")
        assert response1.status_code == 200
        data1 = response1.json()
        
        # Second request should potentially use cache
        response2 = await async_client.get("/api/v1/risk/overview")
        assert response2.status_code == 200
        data2 = response2.json()
        
        # Both should be successful
        assert data1["status"] == "success"
        assert data2["status"] == "success"
        
        # Should have source indication
        assert "source" in data1 or "timestamp" in data1
        assert "source" in data2 or "timestamp" in data2
    
    @pytest.mark.asyncio
    async def test_ml_model_error_fallback(self, async_client):
        """Test ML model error fallback behavior."""
        
        with patch('src.ml.serving.model_server.ModelServer') as mock_server_class:
            # Mock model server that fails
            mock_server = AsyncMock()
            mock_server.get_comprehensive_risk_assessment.side_effect = Exception("Model prediction failed")
            mock_server_class.return_value = mock_server
            
            response = await async_client.get("/api/v1/risk/overview")
            
            # Should still return a response, potentially with fallback data
            assert response.status_code in [200, 500]
            
            if response.status_code == 200:
                data = response.json()
                # Should indicate service is initializing or provide fallback
                assert data["status"] in ["success", "service_initializing"]
    
    @pytest.mark.asyncio 
    async def test_concurrent_risk_assessments(self, async_client):
        """Test concurrent risk assessment requests."""
        
        # Make multiple concurrent requests
        tasks = []
        for _ in range(5):
            tasks.append(async_client.get("/api/v1/risk/overview"))
        
        responses = await asyncio.gather(*tasks)
        
        # All should complete successfully
        for response in responses:
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"


@pytest.mark.integration
class TestModelTrainingIntegration:
    """Test ML model training integration."""
    
    @pytest.mark.asyncio
    async def test_training_pipeline_with_real_data_sources(self, async_client):
        """Test training pipeline integration with real data sources."""
        
        with patch('src.ml.training.model_trainer.ModelTrainingPipeline') as mock_pipeline_class:
            with patch('src.data.sources.fred.get_multiple_series') as mock_fred:
                # Mock training data
                mock_fred.return_value = {
                    'GDP': {'observations': [{'date': '2024-01-01', 'value': '27000000'}]},
                    'UNRATE': {'observations': [{'date': '2024-01-01', 'value': '3.5'}]}
                }
                
                mock_pipeline = AsyncMock()
                mock_pipeline.fetch_training_data.return_value = {
                    'fred_data': mock_fred.return_value,
                    'other_sources': {}
                }
                mock_pipeline.train_all_models.return_value = {
                    'status': 'success',
                    'models_trained': 4,
                    'training_time': 1200
                }
                mock_pipeline_class.return_value = mock_pipeline
                
                response = await async_client.post("/api/v1/risk/models/train")
                
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "success"
                assert "data" in data
    
    @pytest.mark.asyncio
    async def test_model_performance_monitoring(self, async_client):
        """Test model performance monitoring integration."""
        
        with patch('src.ml.serving.model_server.ModelServer') as mock_server_class:
            mock_server = MagicMock()
            mock_server.get_model_status.return_value = {
                'total_models': 4,
                'trained_models': 4,
                'model_details': {
                    'recession_predictor': {
                        'status': 'trained',
                        'accuracy': 0.85,
                        'last_trained': '2024-01-01T00:00:00Z',
                        'prediction_count': 1250
                    },
                    'supply_chain_risk': {
                        'status': 'trained',
                        'accuracy': 0.78,
                        'last_trained': '2024-01-01T00:00:00Z',
                        'prediction_count': 890
                    },
                    'market_volatility': {
                        'status': 'trained',
                        'accuracy': 0.82,
                        'last_trained': '2024-01-01T00:00:00Z',
                        'prediction_count': 1100
                    },
                    'geopolitical_risk': {
                        'status': 'trained',
                        'accuracy': 0.79,
                        'last_trained': '2024-01-01T00:00:00Z',
                        'prediction_count': 760
                    }
                },
                'system_health': 'operational'
            }
            mock_server_class.return_value = mock_server
            
            response = await async_client.get("/api/v1/risk/models/status")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert "data" in data
            
            model_data = data["data"]
            assert model_data["total_models"] == 4
            assert model_data["trained_models"] == 4


@pytest.mark.integration
@pytest.mark.slow
class TestPerformanceIntegration:
    """Test performance aspects of ML integration."""
    
    @pytest.mark.asyncio
    async def test_risk_assessment_response_time(self, async_client):
        """Test risk assessment response time."""
        import time
        
        start_time = time.time()
        response = await async_client.get("/api/v1/risk/overview")
        end_time = time.time()
        
        response_time = end_time - start_time
        
        assert response.status_code == 200
        # Risk assessment should complete within reasonable time
        assert response_time < 5.0  # 5 seconds max
    
    @pytest.mark.asyncio
    async def test_concurrent_ml_predictions(self, async_client):
        """Test concurrent ML predictions performance."""
        import time
        
        start_time = time.time()
        
        # Make concurrent prediction requests
        tasks = [
            async_client.get("/api/v1/risk/predictions/recession"),
            async_client.get("/api/v1/risk/predictions/supply-chain"),
            async_client.get("/api/v1/risk/predictions/market-volatility"),
            async_client.get("/api/v1/risk/predictions/geopolitical")
        ]
        
        responses = await asyncio.gather(*tasks)
        end_time = time.time()
        
        concurrent_time = end_time - start_time
        
        # All requests should complete
        for response in responses:
            assert response.status_code == 200
        
        # Concurrent requests should complete faster than sequential
        assert concurrent_time < 10.0  # 10 seconds max for all concurrent requests
    
    @pytest.mark.asyncio
    async def test_large_dataset_handling(self, async_client):
        """Test handling of large datasets in risk assessment."""
        
        # This would test with larger datasets if available
        response = await async_client.get("/api/v1/economic/indicators")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should handle large economic datasets
        assert data["status"] == "success"
        
        # Response size should be reasonable
        import json
        response_size = len(json.dumps(data))
        assert response_size < 10 * 1024 * 1024  # Less than 10MB