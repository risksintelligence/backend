"""Yahoo Finance quote client."""
from __future__ import annotations

from typing import Dict

from .base_client import BaseAPIClient, ExternalAPIError


class YahooFinanceClient(BaseAPIClient):
    def __init__(self, *, http_client=None) -> None:
        super().__init__("https://query1.finance.yahoo.com", http_client=http_client)

    async def get_quote(self, symbol: str) -> Dict[str, float]:
        params = {"symbols": symbol}
        data = await self._request("GET", "/v7/finance/quote", params=params)
        try:
            quote = data["quoteResponse"]["result"][0]
        except (KeyError, IndexError) as exc:
            raise ExternalAPIError("Malformed Yahoo Finance response") from exc
        return {
            "symbol": quote["symbol"],
            "regularMarketPrice": float(quote["regularMarketPrice"]),
            "regularMarketChangePercent": float(quote.get("regularMarketChangePercent", 0.0)),
        }
