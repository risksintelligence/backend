import asyncio
from datetime import datetime

import httpx
import pytest

from src.data.fetchers.fred_client import FredClient
from src.data.fetchers.yahoo_client import YahooFinanceClient


@pytest.mark.asyncio
async def test_fred_client_parses_observations_correctly():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["series_id"] == "VIXCLS"
        payload = {
            "observations": [
                {"date": "2024-01-02", "value": "12.34"},
                {"date": "2024-01-01", "value": "."},
            ]
        }
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = FredClient(api_key="demo", http_client=http_client)
        observations = await client.get_series_observations("VIXCLS", limit=2)
        parsed = [FredClient.parse_observation(obs) for obs in observations]

    assert parsed[0]["value"] == 12.34
    assert parsed[1]["value"] is None
    assert parsed[0]["observed_at"] == datetime.fromisoformat("2024-01-02")


@pytest.mark.asyncio
async def test_yahoo_finance_client_returns_quote():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["symbols"] == "^VIX"
        payload = {
            "quoteResponse": {
                "result": [
                    {
                        "symbol": "^VIX",
                        "regularMarketPrice": 19.5,
                        "regularMarketChangePercent": -1.2,
                    }
                ]
            }
        }
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = YahooFinanceClient(http_client=http_client)
        quote = await client.get_quote("^VIX")

    assert quote["symbol"] == "^VIX"
    assert quote["regularMarketPrice"] == 19.5
    assert quote["regularMarketChangePercent"] == -1.2
