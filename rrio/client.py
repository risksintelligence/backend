import os
from dataclasses import dataclass
from typing import Any, Dict

import httpx

BASE_URL = os.getenv("RRIO_API_BASE", "https://backend-9t5o.onrender.com/api/v1")
API_KEY = os.getenv("RRIO_API_KEY")


@dataclass
class RRIOClient:
    base_url: str = BASE_URL
    api_key: str = API_KEY

    def _headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _get(self, path: str) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        response = httpx.get(url, headers=self._headers(), timeout=10)
        response.raise_for_status()
        return response.json()

    def current_geri(self) -> Dict[str, Any]:
        return self._get("/analytics/geri")

    def current_regime(self) -> Dict[str, Any]:
        return self._get("/ai/regime/current")

    def forecast(self) -> Dict[str, Any]:
        return self._get("/ai/forecast/next-24h")

    def anomalies(self) -> Dict[str, Any]:
        return self._get("/anomalies/latest")

    def ras(self) -> Dict[str, Any]:
        return self._get("/impact/ras")
