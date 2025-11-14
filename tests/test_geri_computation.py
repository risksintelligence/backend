from backend.src.analytics.geri import (
    IndicatorSnapshot,
    compute_geri,
    determine_risk_band,
    normalize_snapshot,
)
from backend.src.analytics.normalization import NormalizationParams
import pytest


def test_normalize_snapshot_applies_direction():
    params_pos = NormalizationParams(mean=10, stddev=2, direction="positive")
    params_neg = NormalizationParams(mean=50, stddev=5, direction="negative")
    result_pos = normalize_snapshot(IndicatorSnapshot(name="VIX", value=14, params=params_pos))
    result_neg = normalize_snapshot(IndicatorSnapshot(name="PMI", value=45, params=params_neg))
    assert pytest.approx(result_pos.score, rel=1e-5) == 90  # z=2
    assert result_pos.z_score == 2
    assert result_neg.z_score > 0  # direction flipped


def _default_params():
    return {
        "VIX": NormalizationParams(20, 5, "positive"),
        "YC_10Y_2Y": NormalizationParams(0.5, 0.2, "negative"),
        "CREDIT_SPREAD": NormalizationParams(1.0, 0.5, "positive"),
        "FREIGHT_SHIPPING": NormalizationParams(100, 10, "positive"),
        "PMI_MANUFACTURING": NormalizationParams(52, 3, "negative"),
        "OIL_WTI": NormalizationParams(70, 5, "positive"),
        "CPI_INDEX": NormalizationParams(2, 0.5, "positive"),
        "UNEMPLOYMENT": NormalizationParams(4, 0.5, "positive"),
    }


def _normalize_inputs(values):
    params = _default_params()
    return {
        name: normalize_snapshot(IndicatorSnapshot(name=name, value=value, params=params[name]))
        for name, value in values.items()
    }


def test_compute_geri_matches_expected_weights():
    indicators = _normalize_inputs(
        {
            "VIX": 25,
            "YC_10Y_2Y": 0.2,
            "CREDIT_SPREAD": 1.5,
            "FREIGHT_SHIPPING": 110,
            "PMI_MANUFACTURING": 50,
            "OIL_WTI": 80,
            "CPI_INDEX": 3,
            "UNEMPLOYMENT": 5,
        }
    )
    result = compute_geri(indicators)
    assert pytest.approx(result["value"], rel=1e-3) == 75.82
    assert result["band"] == "High"
    assert result["components"]["financial"] > result["components"]["supply_chain"]
    assert len(result["drivers"]) == 3


def test_missing_indicator_raises_error():
    indicators = _normalize_inputs({"VIX": 25})
    with pytest.raises(ValueError):
        compute_geri(indicators)


def test_determine_risk_band_edges():
    assert determine_risk_band(10) == "Minimal"
    assert determine_risk_band(25) == "Low"
    assert determine_risk_band(45) == "Moderate"
    assert determine_risk_band(65) == "High"
    assert determine_risk_band(85) == "Critical"
