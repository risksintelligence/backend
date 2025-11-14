"""Deterministic GERI computation engine."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .normalization import NormalizationParams, normalize_indicator

FINANCIAL_WEIGHTS = {"VIX": 0.5, "YC_10Y_2Y": 0.3, "CREDIT_SPREAD": 0.2}
SUPPLY_WEIGHTS = {"FREIGHT_SHIPPING": 0.4, "PMI_MANUFACTURING": 0.4, "OIL_WTI": 0.2}
MACRO_WEIGHTS = {"CPI_INDEX": 0.6, "UNEMPLOYMENT": 0.4}
COMPONENTS = {
    "financial": FINANCIAL_WEIGHTS,
    "supply_chain": SUPPLY_WEIGHTS,
    "macro": MACRO_WEIGHTS,
}
COMPONENT_WEIGHTS = {"financial": 0.45, "supply_chain": 0.35, "macro": 0.20}
RISK_BANDS = [
    (19, "Minimal"),
    (39, "Low"),
    (59, "Moderate"),
    (79, "High"),
    (100, "Critical"),
]


@dataclass
class IndicatorSnapshot:
    name: str
    value: float
    params: NormalizationParams


@dataclass
class IndicatorResult:
    score: float
    z_score: float
    value: float


def normalize_snapshot(snapshot: IndicatorSnapshot) -> IndicatorResult:
    score, z = normalize_indicator(snapshot.value, snapshot.params)
    return IndicatorResult(score=score, z_score=z, value=snapshot.value)


def compute_component_score(component: str, indicators: Dict[str, IndicatorResult]) -> float:
    weights = COMPONENTS[component]
    total = 0.0
    for indicator, weight in weights.items():
        total += indicators[indicator].score * weight
    return total


def compute_geri(indicators: Dict[str, IndicatorResult]) -> Dict[str, object]:
    required = set().union(*COMPONENTS.values())
    missing = required - indicators.keys()
    if missing:
        raise ValueError(f"Missing indicators: {', '.join(sorted(missing))}")
    components = {
        name: compute_component_score(name, indicators) for name in COMPONENTS
    }
    value = sum(components[name] * COMPONENT_WEIGHTS[name] for name in components)
    band = determine_risk_band(value)
    drivers = compute_top_drivers(indicators)
    return {
        "value": round(value, 2),
        "band": band,
        "components": {k: round(v, 2) for k, v in components.items()},
        "drivers": drivers,
    }


def determine_risk_band(value: float) -> str:
    for threshold, label in RISK_BANDS:
        if value <= threshold:
            return label
    return "Critical"


def compute_top_drivers(indicators: Dict[str, IndicatorResult]) -> List[Dict[str, float]]:
    impacts: List[tuple[str, float]] = []
    for component_name, weights in COMPONENTS.items():
        comp_weight = COMPONENT_WEIGHTS[component_name]
        for indicator, weight in weights.items():
            z = indicators[indicator].z_score
            impact = abs(z) * comp_weight * weight
            impacts.append((indicator, impact))
    impacts.sort(key=lambda item: item[1], reverse=True)
    top = impacts[:3]
    return [
        {"series": series, "weighted_impact": round(impact, 4)} for series, impact in top
    ]
