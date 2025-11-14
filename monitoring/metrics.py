"""Prometheus metrics exporters shared by backend services.

Integrate this module with FastAPI router exposing `/metrics`. It collects both
infrastructure and ML-specific telemetry per documentation requirements."""

from prometheus_client import Counter, Gauge, Histogram

API_LATENCY = Histogram(
    "ris_api_latency_seconds",
    "Request latency distribution by endpoint",
    labelnames=("route", "method"),
)
API_REQUESTS = Counter(
    "ris_api_requests_total",
    "Total HTTP requests",
    labelnames=("route", "method", "status"),
)
CACHE_HIT_RATIO = Gauge(
    "ris_cache_hit_ratio",
    "Cache hit ratio per layer",
    labelnames=("tier",),
)
DATA_FRESHNESS = Gauge(
    "ris_data_freshness_seconds",
    "Age of latest observation per series",
    labelnames=("series",),
)
ML_PREDICTION_DURATION = Histogram(
    "ris_ml_prediction_duration_seconds",
    "Time spent per ML prediction",
    labelnames=("model_type",),
)
ML_MODEL_ACCURACY = Gauge(
    "ris_ml_model_accuracy",
    "ML model quality metrics",
    labelnames=("model_type", "metric_type"),
)


def record_request(route: str, method: str, status: int, duration_seconds: float) -> None:
    """Register a request/latency measurement."""
    API_REQUESTS.labels(route, method, str(status)).inc()
    API_LATENCY.labels(route, method).observe(duration_seconds)


def update_cache_hit_ratio(tier: str, ratio: float) -> None:
    CACHE_HIT_RATIO.labels(tier).set(ratio)


def update_data_freshness(series: str, age_seconds: float) -> None:
    DATA_FRESHNESS.labels(series).set(age_seconds)


def track_ml_prediction(model_type: str, elapsed_seconds: float) -> None:
    ML_PREDICTION_DURATION.labels(model_type).observe(elapsed_seconds)


def record_ml_accuracy(model_type: str, metric_type: str, value: float) -> None:
    ML_MODEL_ACCURACY.labels(model_type, metric_type).set(value)
