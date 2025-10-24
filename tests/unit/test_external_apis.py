"""
Unit tests for external API integrations.
"""
import pytest
import aiohttp
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

from src.data.sources.fred import FREDClient, get_key_indicators
from src.data.sources.bea import BEAClient, get_gdp_data
from src.data.sources.bls import BLSClient, get_employment_data
from src.data.sources.census import CensusClient, get_population_data
from src.data.sources.cisa import CISAClient, get_cybersecurity_threats
from src.data.sources.noaa import NOAAClient, get_environmental_risks
from src.data.sources.usgs import USGSClient, get_geological_hazards
from src.data.sources.supply_chain import SupplyChainClient, get_supply_chain_risks


@pytest.mark.unit
class TestFREDClient:
    """Test FRED API client."""
    
    @pytest.fixture
    def fred_client(self):
        """Create FRED client instance."""
        return FREDClient()
    
    @pytest.mark.asyncio
    async def test_fred_client_initialization(self, fred_client):
        """Test FRED client initialization."""
        assert fred_client.session is None
        assert fred_client.rate_limit_delay == 1.0
        assert fred_client.last_request_time == 0
    
    @pytest.mark.asyncio
    async def test_fred_client_context_manager(self, fred_client):
        """Test FRED client as context manager."""
        async with fred_client as client:
            assert client.session is not None
            assert isinstance(client.session, aiohttp.ClientSession)
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, fred_client):
        """Test rate limiting functionality."""
        fred_client.last_request_time = fred_client._get_current_time()
        
        start_time = fred_client._get_current_time()
        await fred_client._rate_limit()
        end_time = fred_client._get_current_time()
        
        # Should have delayed for rate limit
        assert end_time - start_time >= 0.9  # Close to 1 second delay
    
    @pytest.mark.asyncio
    async def test_successful_api_request(self, fred_client):
        """Test successful API request."""
        mock_response_data = {
            "observations": [
                {"date": "2024-01-01", "value": "27000000.0"},
                {"date": "2024-04-01", "value": "27100000.0"}
            ]
        }
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = mock_response_data
            mock_get.return_value.__aenter__.return_value = mock_response
            
            fred_client.session = AsyncMock()
            result = await fred_client._make_request("test_url", {"param": "value"})
            
            assert result == mock_response_data
    
    @pytest.mark.asyncio
    async def test_api_error_handling(self, fred_client):
        """Test API error handling."""
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 404
            mock_response.text.return_value = "Not Found"
            mock_get.return_value.__aenter__.return_value = mock_response
            
            fred_client.session = AsyncMock()
            result = await fred_client._make_request("test_url")
            
            assert result is None
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, fred_client):
        """Test timeout handling."""
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.side_effect = aiohttp.ServerTimeoutError()
            
            fred_client.session = AsyncMock()
            result = await fred_client._make_request("test_url")
            
            assert result is None
    
    @pytest.mark.asyncio
    async def test_get_series_data(self, fred_client):
        """Test getting series data."""
        mock_data = {
            "observations": [
                {"date": "2024-01-01", "value": "2.5"},
                {"date": "2024-02-01", "value": "2.6"}
            ]
        }
        
        with patch.object(fred_client, '_make_request', return_value=mock_data):
            fred_client.session = AsyncMock()
            result = await fred_client.get_series_data("GDP")
            
            assert result is not None
            assert "observations" in result
            assert len(result["observations"]) == 2
    
    @pytest.mark.asyncio
    async def test_get_multiple_series(self, fred_client):
        """Test getting multiple series."""
        mock_data = {
            "observations": [{"date": "2024-01-01", "value": "2.5"}]
        }
        
        with patch.object(fred_client, 'get_series_data', return_value=mock_data):
            fred_client.session = AsyncMock()
            result = await fred_client.get_multiple_series(
                ["GDP", "UNRATE"], "2024-01-01", "2024-12-31"
            )
            
            assert isinstance(result, dict)
            assert "GDP" in result
            assert "UNRATE" in result


@pytest.mark.unit
class TestCISAClient:
    """Test CISA API client."""
    
    @pytest.fixture
    def cisa_client(self):
        """Create CISA client instance."""
        return CISAClient()
    
    @pytest.mark.asyncio
    async def test_get_kev_catalog(self, cisa_client):
        """Test getting KEV catalog."""
        mock_kev_data = {
            "catalogVersion": "2024.01.01",
            "dateReleased": "2024-01-01T00:00:00.000Z",
            "count": 2,
            "vulnerabilities": [
                {
                    "cveID": "CVE-2024-0001",
                    "vendorProject": "Test Vendor",
                    "product": "Test Product",
                    "vulnerabilityName": "Test Vulnerability",
                    "dateAdded": "2024-01-01",
                    "shortDescription": "Test description",
                    "requiredAction": "Apply patch",
                    "dueDate": "2024-02-01",
                    "knownRansomwareCampaignUse": "Known"
                },
                {
                    "cveID": "CVE-2024-0002",
                    "vendorProject": "Another Vendor",
                    "product": "Another Product",
                    "vulnerabilityName": "Another Vulnerability",
                    "dateAdded": "2024-01-15",
                    "shortDescription": "Another description",
                    "requiredAction": "Update software",
                    "dueDate": "2024-03-01",
                    "knownRansomwareCampaignUse": "Unknown"
                }
            ]
        }
        
        with patch.object(cisa_client, '_make_request', return_value=mock_kev_data):
            cisa_client.session = AsyncMock()
            result = await cisa_client.get_kev_catalog()
            
            assert result is not None
            assert "catalog_version" in result
            assert "total_vulnerabilities" in result
            assert "recent_vulnerabilities" in result
            assert "risk_score" in result
            assert isinstance(result["risk_score"], (int, float))
    
    @pytest.mark.asyncio
    async def test_get_infrastructure_sectors(self, cisa_client):
        """Test getting infrastructure sector risks."""
        mock_kev_data = {
            "recent_high_priority": [
                {
                    "short_description": "Remote code execution in power grid software",
                    "product": "SCADA System",
                    "vendor_project": "Energy Corp"
                }
            ]
        }
        
        with patch.object(cisa_client, 'get_kev_catalog', return_value=mock_kev_data):
            cisa_client.session = AsyncMock()
            result = await cisa_client.get_infrastructure_sectors()
            
            assert result is not None
            assert "sectors" in result
            assert "overall_infrastructure_risk" in result
            assert "Energy" in result["sectors"]
    
    @pytest.mark.asyncio
    async def test_get_threat_intelligence(self, cisa_client):
        """Test getting threat intelligence."""
        mock_kev_data = {
            "recent_high_priority": [
                {
                    "short_description": "Remote code execution vulnerability",
                    "vendor_project": "Microsoft",
                    "date_added": "2024-01-01"
                }
            ],
            "ransomware_associated": 5
        }
        
        with patch.object(cisa_client, 'get_kev_catalog', return_value=mock_kev_data):
            cisa_client.session = AsyncMock()
            result = await cisa_client.get_threat_intelligence()
            
            assert result is not None
            assert "active_exploited_vulnerabilities" in result
            assert "threat_types" in result
            assert "threat_intelligence_score" in result


@pytest.mark.unit
class TestNOAAClient:
    """Test NOAA API client."""
    
    @pytest.fixture
    def noaa_client(self):
        """Create NOAA client instance."""
        return NOAAClient()
    
    @pytest.mark.asyncio
    async def test_get_severe_weather_alerts(self, noaa_client):
        """Test getting severe weather alerts."""
        mock_alerts_data = {
            "features": [
                {
                    "properties": {
                        "severity": "Severe",
                        "event": "Thunderstorm Warning",
                        "urgency": "Immediate",
                        "headline": "Severe Thunderstorm Warning",
                        "areaDesc": "Test County",
                        "effective": "2024-01-01T12:00:00Z",
                        "expires": "2024-01-01T18:00:00Z"
                    }
                },
                {
                    "properties": {
                        "severity": "Extreme",
                        "event": "Hurricane Warning",
                        "urgency": "Expected",
                        "headline": "Hurricane Warning",
                        "areaDesc": "Coastal Area",
                        "effective": "2024-01-01T06:00:00Z",
                        "expires": "2024-01-02T06:00:00Z"
                    }
                }
            ]
        }
        
        with patch.object(noaa_client, '_make_weather_request', return_value=mock_alerts_data):
            noaa_client.session = AsyncMock()
            result = await noaa_client.get_severe_weather_alerts()
            
            assert result is not None
            assert "total_alerts" in result
            assert "by_severity" in result
            assert "weather_risk_score" in result
            assert result["total_alerts"] == 2
            assert result["by_severity"]["Severe"] == 1
            assert result["by_severity"]["Extreme"] == 1
    
    @pytest.mark.asyncio
    async def test_get_climate_extremes(self, noaa_client):
        """Test getting climate extremes."""
        with patch.dict('os.environ', {'NOAA_API_KEY': 'test_key'}):
            noaa_client.session = AsyncMock()
            result = await noaa_client.get_climate_extremes()
            
            assert result is not None
            assert "climate_risk_score" in result
            assert "source" in result
    
    @pytest.mark.asyncio
    async def test_get_transportation_impacts(self, noaa_client):
        """Test getting transportation impacts."""
        mock_weather_data = {
            "high_impact_alerts": [
                {
                    "event": "Blizzard Warning",
                    "severity": "Extreme",
                    "area_desc": "Interstate 95 Corridor"
                }
            ]
        }
        
        with patch.object(noaa_client, 'get_severe_weather_alerts', return_value=mock_weather_data):
            noaa_client.session = AsyncMock()
            result = await noaa_client.get_transportation_impacts()
            
            assert result is not None
            assert "transportation_modes" in result
            assert "overall_transport_risk" in result
            assert "aviation" in result["transportation_modes"]
            assert "highway" in result["transportation_modes"]


@pytest.mark.unit
class TestUSGSClient:
    """Test USGS API client."""
    
    @pytest.fixture
    def usgs_client(self):
        """Create USGS client instance."""
        return USGSClient()
    
    @pytest.mark.asyncio
    async def test_get_recent_earthquakes(self, usgs_client):
        """Test getting recent earthquakes."""
        mock_earthquake_data = {
            "features": [
                {
                    "properties": {
                        "mag": 5.2,
                        "place": "10km NE of Los Angeles, CA",
                        "time": 1704067200000  # 2024-01-01 timestamp
                    },
                    "geometry": {
                        "coordinates": [-118.2437, 34.0522, 10.0]
                    }
                },
                {
                    "properties": {
                        "mag": 6.1,
                        "place": "50km W of San Francisco, CA",
                        "time": 1704153600000  # 2024-01-02 timestamp
                    },
                    "geometry": {
                        "coordinates": [-122.4194, 37.7749, 15.0]
                    }
                }
            ]
        }
        
        with patch.object(usgs_client, '_make_request', return_value=mock_earthquake_data):
            usgs_client.session = AsyncMock()
            result = await usgs_client.get_recent_earthquakes(7)
            
            assert result is not None
            assert "total_earthquakes" in result
            assert "significant_earthquakes" in result
            assert "magnitude_distribution" in result
            assert "seismic_risk_score" in result
            assert result["total_earthquakes"] == 2
            assert result["magnitude_distribution"]["6+"] == 1
            assert result["magnitude_distribution"]["5-6"] == 1
    
    @pytest.mark.asyncio
    async def test_get_infrastructure_vulnerability(self, usgs_client):
        """Test getting infrastructure vulnerability."""
        mock_earthquake_data = {
            "significant_events": [
                {"magnitude": 5.5, "place": "Los Angeles, CA"},
                {"magnitude": 6.0, "place": "San Francisco, CA"}
            ],
            "regional_activity": {
                "California": 15,
                "Alaska": 8,
                "Nevada": 3
            }
        }
        
        with patch.object(usgs_client, 'get_recent_earthquakes', return_value=mock_earthquake_data):
            usgs_client.session = AsyncMock()
            result = await usgs_client.get_infrastructure_vulnerability()
            
            assert result is not None
            assert "infrastructure_systems" in result
            assert "overall_vulnerability" in result
            assert "power_grid" in result["infrastructure_systems"]
            assert "transportation" in result["infrastructure_systems"]
    
    @pytest.mark.asyncio
    async def test_get_realtime_feed(self, usgs_client):
        """Test getting real-time earthquake feed."""
        mock_feed_data = {
            "features": [
                {
                    "properties": {"mag": 4.5}
                },
                {
                    "properties": {"mag": 5.2}
                }
            ]
        }
        
        with patch.object(usgs_client, '_make_request', return_value=mock_feed_data):
            usgs_client.session = AsyncMock()
            result = await usgs_client.get_realtime_feed()
            
            assert result is not None
            assert "realtime_feeds" in result
            assert "realtime_risk_score" in result


@pytest.mark.unit
class TestSupplyChainClient:
    """Test supply chain data client."""
    
    @pytest.fixture
    def supply_chain_client(self):
        """Create supply chain client instance."""
        return SupplyChainClient()
    
    @pytest.mark.asyncio
    async def test_get_critical_infrastructure_nodes(self, supply_chain_client):
        """Test getting critical infrastructure nodes."""
        supply_chain_client.session = AsyncMock()
        result = await supply_chain_client.get_critical_infrastructure_nodes()
        
        assert result is not None
        assert "critical_nodes" in result
        assert "total_critical_nodes" in result
        assert "overall_infrastructure_risk" in result
        assert "ports" in result["critical_nodes"]
        assert "rail_hubs" in result["critical_nodes"]
        assert "distribution_centers" in result["critical_nodes"]
    
    @pytest.mark.asyncio
    async def test_get_transportation_vulnerabilities(self, supply_chain_client):
        """Test getting transportation vulnerabilities."""
        supply_chain_client.session = AsyncMock()
        result = await supply_chain_client.get_transportation_vulnerabilities()
        
        assert result is not None
        assert "transportation_modes" in result
        assert "weighted_transportation_risk" in result
        assert "maritime" in result["transportation_modes"]
        assert "rail" in result["transportation_modes"]
        assert "trucking" in result["transportation_modes"]
        assert "air_cargo" in result["transportation_modes"]
    
    @pytest.mark.asyncio
    async def test_get_supply_chain_disruption_indicators(self, supply_chain_client):
        """Test getting supply chain disruption indicators."""
        supply_chain_client.session = AsyncMock()
        result = await supply_chain_client.get_supply_chain_disruption_indicators()
        
        assert result is not None
        assert "disruption_indicators" in result
        assert "composite_disruption_score" in result
        assert "critical_risks" in result
        assert "global_events" in result["disruption_indicators"]
        assert "operational_indicators" in result["disruption_indicators"]
        assert "resource_constraints" in result["disruption_indicators"]
    
    @pytest.mark.asyncio
    async def test_get_logistics_performance_metrics(self, supply_chain_client):
        """Test getting logistics performance metrics."""
        supply_chain_client.session = AsyncMock()
        result = await supply_chain_client.get_logistics_performance_metrics()
        
        assert result is not None
        assert "performance_metrics" in result
        assert "overall_performance_score" in result
        assert "performance_gaps" in result
        assert "capacity_utilization" in result["performance_metrics"]
        assert "performance_indicators" in result["performance_metrics"]
        assert "network_resilience" in result["performance_metrics"]


@pytest.mark.unit
class TestAPIHealthChecks:
    """Test API health check functions."""
    
    @pytest.mark.asyncio
    async def test_fred_health_check(self):
        """Test FRED API health check."""
        with patch('src.data.sources.fred.FREDClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get_series_data.return_value = {"observations": []}
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            from src.data.sources.fred import health_check
            result = await health_check()
            
            assert isinstance(result, bool)
    
    @pytest.mark.asyncio
    async def test_cisa_health_check(self):
        """Test CISA API health check."""
        with patch('src.data.sources.cisa.CISAClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get_kev_catalog.return_value = {"vulnerabilities": []}
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            from src.data.sources.cisa import health_check
            result = await health_check()
            
            assert isinstance(result, bool)
    
    @pytest.mark.asyncio
    async def test_noaa_health_check(self):
        """Test NOAA API health check."""
        with patch('src.data.sources.noaa.NOAAClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get_severe_weather_alerts.return_value = {"features": []}
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            from src.data.sources.noaa import health_check
            result = await health_check()
            
            assert isinstance(result, bool)
    
    @pytest.mark.asyncio
    async def test_usgs_health_check(self):
        """Test USGS API health check."""
        with patch('src.data.sources.usgs.USGSClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get_recent_earthquakes.return_value = {"features": []}
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            from src.data.sources.usgs import health_check
            result = await health_check()
            
            assert isinstance(result, bool)
    
    @pytest.mark.asyncio
    async def test_supply_chain_health_check(self):
        """Test supply chain health check."""
        with patch('src.data.sources.supply_chain.SupplyChainClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get_critical_infrastructure_nodes.return_value = {"critical_nodes": {}}
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            from src.data.sources.supply_chain import health_check
            result = await health_check()
            
            assert isinstance(result, bool)


@pytest.mark.unit
class TestAPIIntegrationFunctions:
    """Test high-level API integration functions."""
    
    @pytest.mark.asyncio
    async def test_get_key_indicators(self):
        """Test getting key economic indicators."""
        with patch('src.data.sources.fred.FREDClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get_multiple_series.return_value = {
                "GDP": {"observations": [{"date": "2024-01-01", "value": "27000000"}]},
                "UNRATE": {"observations": [{"date": "2024-01-01", "value": "3.5"}]}
            }
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            result = await get_key_indicators()
            
            assert result["status"] == "success"
            assert "indicators" in result
            assert len(result["indicators"]) > 0
    
    @pytest.mark.asyncio
    async def test_get_cybersecurity_threats(self):
        """Test getting cybersecurity threats."""
        with patch('src.data.sources.cisa.CISAClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get_kev_catalog.return_value = {"risk_score": 45.0}
            mock_client.get_infrastructure_sectors.return_value = {"overall_infrastructure_risk": 55.0}
            mock_client.get_threat_intelligence.return_value = {"threat_intelligence_score": 60.0}
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            result = await get_cybersecurity_threats()
            
            assert "indicators" in result
            assert "overall_cybersecurity_risk" in result
            assert "count" in result
    
    @pytest.mark.asyncio
    async def test_get_environmental_risks(self):
        """Test getting environmental risks."""
        with patch('src.data.sources.noaa.NOAAClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get_severe_weather_alerts.return_value = {"weather_risk_score": 35.0}
            mock_client.get_climate_extremes.return_value = {"climate_risk_score": 40.0}
            mock_client.get_transportation_impacts.return_value = {"overall_transport_risk": 30.0}
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            result = await get_environmental_risks()
            
            assert "indicators" in result
            assert "overall_environmental_risk" in result
            assert "count" in result
    
    @pytest.mark.asyncio
    async def test_get_geological_hazards(self):
        """Test getting geological hazards."""
        with patch('src.data.sources.usgs.USGSClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get_recent_earthquakes.return_value = {"seismic_risk_score": 25.0}
            mock_client.get_infrastructure_vulnerability.return_value = {"overall_vulnerability": 35.0}
            mock_client.get_natural_hazard_assessment.return_value = {"composite_hazard_score": 30.0}
            mock_client.get_realtime_feed.return_value = {"realtime_risk_score": 20.0}
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            result = await get_geological_hazards()
            
            assert "indicators" in result
            assert "overall_geological_risk" in result
            assert "count" in result
    
    @pytest.mark.asyncio
    async def test_get_supply_chain_risks(self):
        """Test getting supply chain risks."""
        with patch('src.data.sources.supply_chain.SupplyChainClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get_critical_infrastructure_nodes.return_value = {"overall_infrastructure_risk": 45.0}
            mock_client.get_transportation_vulnerabilities.return_value = {"weighted_transportation_risk": 55.0}
            mock_client.get_supply_chain_disruption_indicators.return_value = {"composite_disruption_score": 50.0}
            mock_client.get_logistics_performance_metrics.return_value = {"overall_performance_score": 75.0}
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            result = await get_supply_chain_risks()
            
            assert "indicators" in result
            assert "overall_supply_chain_risk" in result
            assert "count" in result


@pytest.mark.unit
class TestAPIErrorHandling:
    """Test external API error handling."""
    
    @pytest.mark.asyncio
    async def test_network_error_handling(self):
        """Test network error handling."""
        with patch('src.data.sources.fred.FREDClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get_multiple_series.side_effect = aiohttp.ClientError("Network error")
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            result = await get_key_indicators()
            
            assert result["status"] == "error"
            assert "error" in result
    
    @pytest.mark.asyncio
    async def test_timeout_error_handling(self):
        """Test timeout error handling."""
        with patch('src.data.sources.cisa.CISAClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get_kev_catalog.side_effect = asyncio.TimeoutError("Request timeout")
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            result = await get_cybersecurity_threats()
            
            # Should handle timeout gracefully
            assert "indicators" in result
            assert result["count"] >= 0
    
    @pytest.mark.asyncio
    async def test_invalid_response_handling(self):
        """Test invalid response handling."""
        with patch('src.data.sources.noaa.NOAAClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get_severe_weather_alerts.return_value = None  # Invalid response
            mock_client.get_climate_extremes.return_value = {"invalid": "data"}
            mock_client.get_transportation_impacts.return_value = None
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            result = await get_environmental_risks()
            
            # Should handle invalid responses gracefully
            assert "indicators" in result
            assert isinstance(result["overall_environmental_risk"], (int, float))
    
    @pytest.mark.asyncio
    async def test_partial_failure_handling(self):
        """Test partial API failure handling."""
        with patch('src.data.sources.usgs.USGSClient') as mock_client_class:
            mock_client = AsyncMock()
            # First call succeeds, others fail
            mock_client.get_recent_earthquakes.return_value = {"seismic_risk_score": 25.0}
            mock_client.get_infrastructure_vulnerability.side_effect = Exception("API Error")
            mock_client.get_natural_hazard_assessment.side_effect = Exception("API Error")
            mock_client.get_realtime_feed.return_value = {"realtime_risk_score": 20.0}
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            result = await get_geological_hazards()
            
            # Should return partial results
            assert "indicators" in result
            assert len(result["indicators"]) >= 1  # At least one successful call
            assert isinstance(result["overall_geological_risk"], (int, float))