"""Snapshot service orchestrating deterministic GERI payloads."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Tuple

from backend.src.analytics.geri import IndicatorSnapshot, compute_geri, normalize_snapshot
from backend.src.analytics.normalization import NormalizationParams
from backend.src.data.series_registry import SERIES_REGISTRY


@dataclass
class IndicatorRecord:
    value: float
    params: NormalizationParams
    provider: str
    series_id: str
    observed_at: datetime
    soft_ttl: timedelta
    hard_ttl: timedelta


class GERISnapshotService:
    def __init__(self) -> None:
        self._records: Dict[str, IndicatorRecord] = {}

    def ingest(self, name: str, record: IndicatorRecord) -> None:
        self._records[name] = record

    def has_inputs(self) -> bool:
        return bool(self._records)

    def build_payload(self) -> Tuple[dict, timedelta, timedelta]:
        if not self._records:
            raise ValueError("No indicator inputs available")
        indicators = {
            name: normalize_snapshot(
                IndicatorSnapshot(name=name, value=record.value, params=record.params)
            )
            for name, record in self._records.items()
        }
        result = compute_geri(indicators)
        now = datetime.now(timezone.utc)
        data_freshness = {}
        provenance = {}
        for name, record in self._records.items():
            age = now - record.observed_at
            data_freshness[name] = {
                "observed_at": record.observed_at.isoformat(),
                "stale": age > record.soft_ttl,
                "soft_ttl_seconds": record.soft_ttl.total_seconds(),
                "hard_ttl_seconds": record.hard_ttl.total_seconds(),
            }
            provenance[name] = {
                "provider": record.provider,
                "series_id": record.series_id,
            }
        payload = {
            "index_name": "geri_v1.0",
            **result,
            "data_freshness": data_freshness,
            "provenance": provenance,
        }
        soft_ttl = min(record.soft_ttl for record in self._records.values())
        hard_ttl = min(record.hard_ttl for record in self._records.values())
        return payload, soft_ttl, hard_ttl


DEFAULT_PARAMS = {
    "VIX": NormalizationParams(20, 5, "positive"),
    "YC_10Y_2Y": NormalizationParams(0.5, 0.2, "negative"),
    "CREDIT_SPREAD": NormalizationParams(1.0, 0.5, "positive"),
    "FREIGHT_SHIPPING": NormalizationParams(100, 10, "positive"),
    "PMI_MANUFACTURING": NormalizationParams(52, 3, "negative"),
    "OIL_WTI": NormalizationParams(70, 5, "positive"),
    "CPI_INDEX": NormalizationParams(2, 0.5, "positive"),
    "UNEMPLOYMENT": NormalizationParams(4, 0.5, "positive"),
}


def seed_demo_data(service: GERISnapshotService) -> None:
    now = datetime.now(timezone.utc)
    for name, config in SERIES_REGISTRY.items():
        params = DEFAULT_PARAMS[name]
        value = params.mean
        service.ingest(
            name,
            IndicatorRecord(
                value=value,
                params=params,
                provider=config.provider,
                series_id=config.series_id,
                observed_at=now,
                soft_ttl=timedelta(seconds=config.soft_ttl_seconds),
                hard_ttl=timedelta(seconds=config.hard_ttl_seconds),
            ),
        )
