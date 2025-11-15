from pathlib import Path

import pytest

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "database" / "schema.sql"


@pytest.mark.parametrize(
    "token",
    [
        "CREATE TABLE raw_observations",
        "CREATE TABLE computed_indices",
        "CREATE TABLE cache_metadata",
        "CREATE TABLE feature_store_snapshots",
        "CREATE TABLE model_registry",
        "CREATE TABLE admin_audit_log",
        "CREATE TABLE research_api_requests",
        "CREATE TABLE scenario_runs",
        "CREATE TABLE alert_subscriptions",
        "CREATE TABLE peer_reviews",
    ],
)
def test_schema_contains_required_tables(token: str):
    content = SCHEMA_PATH.read_text()
    assert token in content
