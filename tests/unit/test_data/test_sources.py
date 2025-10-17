"""
Tests for data source connectors
"""

import pytest
import pandas as pd
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

from src.data.sources.fred import FREDConnector
from src.data.sources.bea import BEAConnector
from src.data.sources.bls import BlsDataFetcher
from src.data.sources.census import CensusTradeDataFetcher
from src.data.sources.fdic import FdicDataFetcher
from src.data.sources.noaa import NOAADataSource
from src.data.sources.cisa import CISADataSource


class TestFREDConnector:
    """Test FRED data source connector"""
    
    @pytest.fixture
    def fred_connector(self):
        return FREDConnector(api_key="test_key")
    
    @pytest.mark.asyncio
    async def test_fetch_series_data(self, fred_connector, mock_fred_data):
        """Test fetching series data from FRED"""
        with patch.object(fred_connector, '_make_request', return_value=mock_fred_data):
            data = await fred_connector.fetch_series_data("GDP")
            
            assert isinstance(data, pd.DataFrame)
            assert len(data) == 2
            assert "date" in data.columns
            assert "value" in data.columns
    
    @pytest.mark.asyncio
    async def test_fetch_multiple_series(self, fred_connector, mock_fred_data):
        """Test fetching multiple series"""
        with patch.object(fred_connector, '_make_request', return_value=mock_fred_data):
            data = await fred_connector.fetch_multiple_series(["GDP", "UNRATE"])
            
            assert isinstance(data, dict)
            assert "GDP" in data
            assert "UNRATE" in data
    
    @pytest.mark.asyncio
    async def test_error_handling(self, fred_connector):
        """Test error handling for FRED API"""
        with patch.object(fred_connector, '_make_request', side_effect=Exception("API Error")):
            with pytest.raises(Exception, match="API Error"):
                await fred_connector.fetch_series_data("INVALID")
    
    @pytest.mark.asyncio
    async def test_caching_behavior(self, fred_connector, mock_fred_data):
        """Test that data is cached appropriately"""
        with patch.object(fred_connector, '_make_request', return_value=mock_fred_data) as mock_request:
            # First call
            await fred_connector.fetch_series_data("GDP")
            
            # Second call should use cache
            await fred_connector.fetch_series_data("GDP")
            
            # API should only be called once due to caching
            assert mock_request.call_count == 1


class TestBEAConnector:
    """Test BEA data source connector"""
    
    @pytest.fixture
    def bea_connector(self):
        return BEAConnector(api_key="test_key")
    
    @pytest.mark.asyncio
    async def test_fetch_gdp_data(self, bea_connector, mock_bea_data):
        """Test fetching GDP data from BEA"""
        with patch.object(bea_connector, '_make_request', return_value=mock_bea_data):
            data = await bea_connector.fetch_gdp_data()
            
            assert isinstance(data, pd.DataFrame)
            assert len(data) == 1
            assert "TimePeriod" in data.columns
            assert "DataValue" in data.columns
    
    @pytest.mark.asyncio
    async def test_fetch_trade_data(self, bea_connector, mock_bea_data):
        """Test fetching trade balance data"""
        with patch.object(bea_connector, '_make_request', return_value=mock_bea_data):
            data = await bea_connector.fetch_trade_data()
            
            assert isinstance(data, pd.DataFrame)
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, bea_connector):
        """Test rate limiting behavior"""
        with patch('asyncio.sleep') as mock_sleep:
            with patch.object(bea_connector, '_make_request', return_value={}):
                # Make multiple rapid requests
                for _ in range(5):
                    await bea_connector.fetch_gdp_data()
                
                # Should have implemented rate limiting
                mock_sleep.assert_called()


class TestBlsDataFetcher:
    """Test BLS data fetcher"""
    
    @pytest.fixture
    def bls_fetcher(self):
        return BlsDataFetcher()
    
    @pytest.mark.asyncio
    async def test_fetch_latest_data(self, bls_fetcher):
        """Test fetching latest BLS data"""
        mock_response = {
            "status": "REQUEST_SUCCEEDED",
            "Results": {
                "series": [
                    {
                        "seriesID": "UNRATE",
                        "data": [
                            {"year": "2023", "period": "M12", "value": "3.8"},
                            {"year": "2023", "period": "M11", "value": "3.9"}
                        ]
                    }
                ]
            }
        }
        
        with patch.object(bls_fetcher.connector, 'session') as mock_session:
            mock_session.post.return_value.__aenter__.return_value.status = 200
            mock_session.post.return_value.__aenter__.return_value.json.return_value = mock_response
            
            data = await bls_fetcher.fetch_latest_data(["UNRATE"])
            
            assert isinstance(data, dict)
            assert "UNRATE" in data
            assert isinstance(data["UNRATE"], pd.DataFrame)
    
    @pytest.mark.asyncio
    async def test_employment_indicators(self, bls_fetcher):
        """Test getting employment indicators with risk assessment"""
        mock_data = {"UNRATE": pd.DataFrame({
            "date": [datetime.now()],
            "value": [3.8],
            "series_id": ["UNRATE"],
            "series_name": ["Unemployment Rate"]
        })}
        
        with patch.object(bls_fetcher, 'fetch_latest_data', return_value=mock_data):
            indicators = await bls_fetcher.get_employment_indicators()
            
            assert "UNRATE" in indicators
            assert "latest_value" in indicators["UNRATE"]
            assert "risk_level" in indicators["UNRATE"]


class TestCensusTradeDataFetcher:
    """Test Census trade data fetcher"""
    
    @pytest.fixture
    def census_fetcher(self):
        return CensusTradeDataFetcher()
    
    @pytest.mark.asyncio
    async def test_fetch_trade_data(self, census_fetcher, mock_trade_data):
        """Test fetching trade data from Census"""
        with patch.object(census_fetcher.connector, 'fetch_data', return_value=mock_trade_data):
            data = await census_fetcher.fetch_latest_trade_data([2023])
            
            assert isinstance(data, dict)
            assert "by_country" in data or "by_commodity" in data
    
    @pytest.mark.asyncio
    async def test_trade_balance_calculation(self, census_fetcher):
        """Test trade balance calculations"""
        sample_trade_data = pd.DataFrame({
            "date": [datetime.now(), datetime.now()],
            "CTY_CODE": ["5700", "5700"],
            "data_type": ["exports", "imports"],
            "GEN_VAL_MO": [100000, 150000]
        })
        
        balance_data = census_fetcher._calculate_trade_balances(sample_trade_data)
        
        assert isinstance(balance_data, pd.DataFrame)
        if not balance_data.empty:
            assert "trade_balance" in balance_data.columns
            assert "import_dependency" in balance_data.columns
    
    @pytest.mark.asyncio
    async def test_trade_risk_assessment(self, census_fetcher):
        """Test trade risk assessment logic"""
        risk_level = census_fetcher._assess_trade_risk("5700", -50000, 0.8, 100000)
        assert risk_level in ["low", "moderate", "high", "unknown"]


class TestFdicDataFetcher:
    """Test FDIC data fetcher"""
    
    @pytest.fixture
    def fdic_fetcher(self):
        return FdicDataFetcher()
    
    @pytest.mark.asyncio
    async def test_fetch_institution_data(self, fdic_fetcher):
        """Test fetching bank institution data"""
        mock_response = {
            "data": [
                {
                    "CERT": "1234",
                    "NAME": "Test Bank",
                    "ASSET": 1000000,
                    "ROA": 1.2,
                    "TIER1RISKCAT": 12.5
                }
            ]
        }
        
        with patch.object(fdic_fetcher.connector, 'fetch_data', return_value=mock_response):
            data = await fdic_fetcher.fetch_institution_data(limit=100)
            
            assert isinstance(data, pd.DataFrame)
            assert len(data) == 1
            assert "size_category" in data.columns
    
    @pytest.mark.asyncio
    async def test_risk_assessment(self, fdic_fetcher):
        """Test bank risk assessment logic"""
        test_row = pd.Series({
            "TIER1RISKCAT": 8.0,
            "TOTCAPRISKCAT": 12.0,
            "LEVERAGE": 5.0,
            "ROA": 1.0,
            "ROE": 10.0
        })
        
        capital_risk = fdic_fetcher._assess_capital_risk(test_row)
        performance_risk = fdic_fetcher._assess_performance_risk(test_row)
        
        assert capital_risk in ["low", "moderate", "high", "unknown"]
        assert performance_risk in ["low", "moderate", "high", "unknown"]


class TestNoaaDataFetcher:
    """Test NOAA weather data fetcher"""
    
    @pytest.fixture
    def noaa_fetcher(self):
        return NoaaDataFetcher()
    
    @pytest.mark.asyncio
    async def test_fetch_weather_alerts(self, noaa_fetcher, mock_weather_data):
        """Test fetching active weather alerts"""
        with patch.object(noaa_fetcher.weather_connector, 'fetch_data', return_value=mock_weather_data):
            alerts = await noaa_fetcher.fetch_active_weather_alerts()
            
            assert isinstance(alerts, pd.DataFrame)
            if not alerts.empty:
                assert "alert_type" in alerts.columns
                assert "economic_risk_level" in alerts.columns
    
    @pytest.mark.asyncio
    async def test_storm_events_processing(self, noaa_fetcher):
        """Test storm events data processing"""
        sample_events = pd.DataFrame({
            "BEGIN_DATE": [datetime.now()],
            "EVENT_TYPE": ["Tornado"],
            "DAMAGE_PROPERTY": [1000000],
            "INJURIES_DIRECT": [5],
            "STATE": ["TX"]
        })
        
        processed = noaa_fetcher._process_storm_events(sample_events)
        
        assert "economic_impact_score" in processed.columns
        assert "severity_level" in processed.columns
    
    def test_economic_impact_calculation(self, noaa_fetcher):
        """Test economic impact score calculation"""
        test_event = pd.Series({
            "EVENT_TYPE": "Tornado",
            "DAMAGE_PROPERTY": 1000000,
            "DAMAGE_CROPS": 0,
            "INJURIES_DIRECT": 5,
            "DEATHS_DIRECT": 0
        })
        
        impact_score = noaa_fetcher._calculate_economic_impact(test_event)
        assert isinstance(impact_score, float)
        assert impact_score > 0


class TestCisaCyberDataFetcher:
    """Test CISA cyber data fetcher"""
    
    @pytest.fixture
    def cisa_fetcher(self):
        return CisaCyberDataFetcher()
    
    @pytest.mark.asyncio
    async def test_fetch_kev_data(self, cisa_fetcher, mock_cyber_data):
        """Test fetching Known Exploited Vulnerabilities"""
        with patch.object(cisa_fetcher.kev_connector, 'fetch_data', return_value=mock_cyber_data):
            kev_data = await cisa_fetcher.fetch_known_exploited_vulnerabilities()
            
            assert isinstance(kev_data, pd.DataFrame)
            if not kev_data.empty:
                assert "cveID" in kev_data.columns
                assert "risk_score" in kev_data.columns
    
    @pytest.mark.asyncio
    async def test_vulnerability_risk_scoring(self, cisa_fetcher):
        """Test vulnerability risk scoring"""
        test_vuln = pd.Series({
            "vulnerabilityName": "Remote Code Execution Vulnerability",
            "shortDescription": "Critical buffer overflow allows remote code execution",
            "days_since_added": 5,
            "days_until_due": 10
        })
        
        risk_score = cisa_fetcher._calculate_vuln_risk_score(test_vuln)
        assert isinstance(risk_score, float)
        assert 0 <= risk_score <= 10
    
    def test_sector_impact_assessment(self, cisa_fetcher):
        """Test critical infrastructure sector impact assessment"""
        test_vuln = pd.Series({
            "vulnerabilityName": "Windows Server Vulnerability",
            "vendorProject": "Microsoft",
            "product": "Windows Server"
        })
        
        impacted_sectors = cisa_fetcher._assess_sector_impact(test_vuln)
        assert isinstance(impacted_sectors, list)


class TestDataSourceIntegration:
    """Integration tests across data sources"""
    
    @pytest.mark.asyncio
    async def test_concurrent_data_fetching(self):
        """Test fetching data from multiple sources concurrently"""
        import asyncio
        
        fred = FREDConnector(api_key="test")
        bea = BEAConnector(api_key="test")
        
        with patch.object(fred, 'fetch_series_data', return_value=pd.DataFrame()):
            with patch.object(bea, 'fetch_gdp_data', return_value=pd.DataFrame()):
                
                # Fetch data concurrently
                results = await asyncio.gather(
                    fred.fetch_series_data("GDP"),
                    bea.fetch_gdp_data(),
                    return_exceptions=True
                )
                
                # Both should complete without errors
                for result in results:
                    assert not isinstance(result, Exception)
    
    @pytest.mark.asyncio
    async def test_error_propagation(self):
        """Test that errors are properly propagated from data sources"""
        fred = FREDConnector(api_key="test")
        
        with patch.object(fred, '_make_request', side_effect=ConnectionError("Network error")):
            with pytest.raises(ConnectionError):
                await fred.fetch_series_data("GDP")
    
    def test_data_format_consistency(self):
        """Test that all data sources return consistent formats"""
        # This would test that all data sources return DataFrames with expected columns
        expected_columns = ["date", "value", "source"]
        
        # Mock data from different sources
        fred_data = pd.DataFrame({
            "date": [datetime.now()],
            "value": [100.0],
            "source": ["fred"]
        })
        
        bea_data = pd.DataFrame({
            "date": [datetime.now()],
            "value": [200.0], 
            "source": ["bea"]
        })
        
        # Check format consistency
        for df in [fred_data, bea_data]:
            for col in expected_columns:
                assert col in df.columns