"""Indicator normalization helpers for GERI."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Direction = Literal["positive", "negative"]


@dataclass(frozen=True)
class NormalizationParams:
    mean: float
    stddev: float
    direction: Direction


def compute_z_score(value: float, params: NormalizationParams) -> float:
    if params.stddev == 0:
        raise ValueError("Standard deviation must be non-zero")
    z = (value - params.mean) / params.stddev
    if params.direction == "negative":
        z = -z
    return z


def squash_to_scale(z_score: float) -> float:
    """Map z-score to 0-100 band per documentation."""
    raw = 50 + 20 * z_score
    return max(0.0, min(100.0, raw))


def normalize_indicator(value: float, params: NormalizationParams) -> tuple[float, float]:
    z = compute_z_score(value, params)
    score = squash_to_scale(z)
    return score, z
