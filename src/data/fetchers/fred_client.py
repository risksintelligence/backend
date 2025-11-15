"""FRED API client handling cadence and TTL metadata."""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from .base_client import BaseAPIClient, ExternalAPIError


class FredClient(BaseAPIClient):
    def __init__(self, api_key: str, *, http_client=None) -> None:
        super().__init__("https://api.stlouisfed.org", http_client=http_client)
        self._api_key = api_key

    async def get_series_observations(
        self,
        series_id: str,
        limit: int = 100,
        sort_order: str = "desc",
    ) -> List[Dict[str, str]]:
        params = {
            "series_id": series_id,
            "api_key": self._api_key,
            "file_type": "json",
            "sort_order": sort_order,
            "limit": limit,
        }
        data = await self._request("GET", "/fred/series/observations", params=params)
        if "observations" not in data:
            raise ExternalAPIError("FRED response missing observations")
        return data["observations"]

    @staticmethod
    def parse_observation(value: Dict[str, str]) -> Dict[str, Optional[float]]:
        observ_time = datetime.fromisoformat(value["date"])
        val = value.get("value")
        return {
            "observed_at": observ_time,
            "value": None if val in (".", None) else float(val),
        }
