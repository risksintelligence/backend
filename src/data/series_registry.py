"""Authoritative cadence/TTL registry derived from documentation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class SeriesConfig:
    provider: str
    series_id: str
    cadence: str
    soft_ttl_seconds: int
    hard_ttl_seconds: int
    description: str


SERIES_REGISTRY: Dict[str, SeriesConfig] = {
    "VIX": SeriesConfig(
        provider="Yahoo Finance",
        series_id="^VIX",
        cadence="*/5 * * * *",
        soft_ttl_seconds=900,
        hard_ttl_seconds=86400,
        description="CBOE Volatility Index",
    ),
    "YC_10Y_2Y": SeriesConfig(
        provider="FRED",
        series_id="DGS10,DGS2",
        cadence="0 9 * * 1-5",
        soft_ttl_seconds=3600,
        hard_ttl_seconds=10368000,
        description="10Y-2Y Treasury Yield Curve",
    ),
    "CREDIT_SPREAD": SeriesConfig(
        provider="FRED",
        series_id="BAA10YM",
        cadence="0 10 * * 1-5",
        soft_ttl_seconds=3600,
        hard_ttl_seconds=10368000,
        description="Credit Spread Proxy",
    ),
    "PMI_MANUFACTURING": SeriesConfig(
        provider="FRED",
        series_id="NAPM",
        cadence="0 14 1 * *",
        soft_ttl_seconds=86400,
        hard_ttl_seconds=10368000,
        description="ISM PMI (derived only)",
    ),
    "FREIGHT_SHIPPING": SeriesConfig(
        provider="FRED",
        series_id="PET.EMD_EPD2DXL0_PTE_NUS_DPG.W",
        cadence="0 12 * * 3",
        soft_ttl_seconds=3600,
        hard_ttl_seconds=172800,
        description="Freight proxy",
    ),
    "OIL_WTI": SeriesConfig(
        provider="FRED",
        series_id="DCOILWTICO",
        cadence="0 11 * * 1-5",
        soft_ttl_seconds=1800,
        hard_ttl_seconds=10368000,
        description="WTI crude spot",
    ),
    "CPI_INDEX": SeriesConfig(
        provider="FRED",
        series_id="CPIAUCSL",
        cadence="0 13 15 * *",
        soft_ttl_seconds=86400,
        hard_ttl_seconds=10368000,
        description="CPI All Items YoY",
    ),
    "UNEMPLOYMENT": SeriesConfig(
        provider="FRED",
        series_id="UNRATE",
        cadence="0 8 1 * *",
        soft_ttl_seconds=86400,
        hard_ttl_seconds=10368000,
        description="Unemployment rate",
    ),
}
