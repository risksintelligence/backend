"""Monitoring metrics registry."""
from prometheus_client import Counter, Gauge

API_LATENCY = Gauge(
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
ML_MODEL_ACCURACY = Gauge(
    "ris_ml_model_accuracy",
    "ML model quality metrics",
    labelnames=("model_type", "metric_type"),
)
SCENARIO_RUNS_TOTAL = Counter(
    "ris_scenario_runs_total",
    "Scenario simulations executed",
    labelnames=("channel",),
)
PEER_REVIEWS_TOTAL = Counter(
    "ris_peer_reviews_total",
    "Peer reviews submitted",
)
ALERT_DELIVERIES_TOTAL = Counter(
    "ris_alert_deliveries_total",
    "Alert deliveries per channel",
    labelnames=("channel",),
)
CRON_SNAPSHOT_RUNS = Counter(
    "ris_snapshot_jobs_total",
    "Cron snapshot executions",
)
CRON_JOB_LAST_RUN_TIMESTAMP = Gauge(
    "ris_cron_last_run_timestamp_seconds",
    "Timestamp of the most recent cron invocation",
    labelnames=("job_name",),
)
CRON_JOB_LAST_SUCCESS_TIMESTAMP = Gauge(
    "ris_cron_last_success_timestamp_seconds",
    "Timestamp of the most recent successful cron completion",
    labelnames=("job_name",),
)
CRON_JOB_LAST_FAILURE_TIMESTAMP = Gauge(
    "ris_cron_last_failure_timestamp_seconds",
    "Timestamp of the most recent cron failure",
    labelnames=("job_name",),
)
CRON_JOB_LAST_DURATION_SECONDS = Gauge(
    "ris_cron_last_duration_seconds",
    "Duration of the last cron execution in seconds",
    labelnames=("job_name",),
)
CRON_JOB_STATUS = Gauge(
    "ris_cron_last_status",
    "Latest cron status (1=success, 0=failure)",
    labelnames=("job_name",),
)
CRON_JOB_FAILURES_TOTAL = Counter(
    "ris_cron_failures_total",
    "Total cron job failures",
    labelnames=("job_name",),
)
CRON_JOB_EXPECTED_INTERVAL_SECONDS = Gauge(
    "ris_cron_expected_interval_seconds",
    "Expected interval between cron executions",
    labelnames=("job_name",),
)
