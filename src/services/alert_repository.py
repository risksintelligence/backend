"""Database-backed repository for alert subscriptions and deliveries."""
from __future__ import annotations

import json
import os
import sqlite3
import logging
from urllib.parse import urlparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List


@dataclass
class SubscriptionRecord:
    id: int
    channel: str
    address: str
    conditions: List[Dict[str, object]]
    created_at: str
    active: bool = True


@dataclass
class DeliveryRecord:
    subscription_id: int
    channel: str
    address: str
    payload: dict
    delivered_at: str


logger = logging.getLogger(__name__)


class AlertRepository:
    def __init__(self, db_url: str | None = None) -> None:
        # Default to Postgres DSN for production, fallback to SQLite only if necessary (tests)
        default_sqlite = "sqlite:///:memory:"
        self._db_url = db_url or os.getenv("ALERT_DB_URL") or os.getenv("RIS_POSTGRES_DSN") or default_sqlite

        if self._db_url == default_sqlite:
            logger.warning("AlertRepository falling back to in-memory SQLite. Set ALERT_DB_URL or RIS_POSTGRES_DSN for persistence.")

        parsed = urlparse(self._db_url)
        self._is_sqlite = parsed.scheme == "sqlite"
        self._sqlite_conn: sqlite3.Connection | None = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Initialize database connection and tables lazily."""
        if self._initialized:
            return
            
        try:
            if self._is_sqlite:
                path = urlparse(self._db_url).path or ":memory:"
                if path in ("/:memory:", ":memory:"):
                    path = ":memory:"
                elif not os.path.isabs(path):
                    path = os.path.join(os.getcwd(), path.lstrip("/"))
                self._sqlite_conn = sqlite3.connect(path, check_same_thread=False)
                self._sqlite_conn.row_factory = sqlite3.Row
            self._ensure_tables()
            self._initialized = True
        except Exception as exc:
            logger.warning(f"Failed to initialize AlertRepository database: {exc}")
            # Fall back to in-memory SQLite on any connection error
            if not self._is_sqlite:
                logger.warning("Falling back to in-memory SQLite due to connection failure")
                self._db_url = "sqlite:///:memory:"
                self._is_sqlite = True
                self._sqlite_conn = sqlite3.connect(":memory:", check_same_thread=False)
                self._sqlite_conn.row_factory = sqlite3.Row
                self._ensure_tables()
                self._initialized = True

    @contextmanager
    def _connection(self):
        self._ensure_initialized()
        if self._is_sqlite:
            assert self._sqlite_conn is not None
            yield self._sqlite_conn
        else:
            try:
                import psycopg2  # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("psycopg2 is required for Postgres alert persistence") from exc
            conn = psycopg2.connect(self._db_url)
            try:
                yield conn
            finally:
                conn.close()

    def _ensure_tables(self) -> None:
        # Use existing schema table names: alert_subscriptions and alert_delivery_log
        if self._is_sqlite:
            subscription_sql = """
            CREATE TABLE IF NOT EXISTS alert_subscriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel TEXT NOT NULL,
                address TEXT NOT NULL,
                conditions TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                active BOOLEAN DEFAULT 1
            )
            """
            events_sql = """
            CREATE TABLE IF NOT EXISTS alert_delivery_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subscription_id INTEGER,
                payload TEXT NOT NULL,
                channel TEXT NOT NULL,
                address TEXT NOT NULL,
                delivery_status TEXT NOT NULL,
                error_message TEXT,
                delivered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        else:
            # Tables should already exist from schema.sql, but create if missing for resilience
            subscription_sql = """
            CREATE TABLE IF NOT EXISTS alert_subscriptions (
                id BIGSERIAL PRIMARY KEY,
                channel TEXT NOT NULL,
                address TEXT NOT NULL,
                conditions JSONB NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                active BOOLEAN DEFAULT TRUE
            )
            """
            events_sql = """
            CREATE TABLE IF NOT EXISTS alert_delivery_log (
                id BIGSERIAL PRIMARY KEY,
                subscription_id BIGINT,
                channel TEXT NOT NULL,
                address TEXT NOT NULL,
                payload JSONB NOT NULL,
                delivery_status TEXT NOT NULL,
                error_message TEXT,
                delivered_at TIMESTAMPTZ NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
            """
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(subscription_sql)
            cursor.execute(events_sql)
            conn.commit()

    def create_subscription(self, channel: str, address: str, conditions: List[Dict[str, object]]) -> SubscriptionRecord:
        payload = json.dumps(conditions)
        with self._connection() as conn:
            cursor = conn.cursor()
            if self._is_sqlite:
                cursor.execute(
                    "INSERT INTO alert_subscriptions (channel, address, conditions) VALUES (?, ?, ?)",
                    (channel, address, payload),
                )
                conn.commit()
                subscription_id = cursor.lastrowid
                cursor.execute(
                    "SELECT created_at FROM alert_subscriptions WHERE id = ?",
                    (subscription_id,),
                )
                created_at = cursor.fetchone()[0]
            else:  # pragma: no cover - requires Postgres
                cursor.execute(
                    "INSERT INTO alert_subscriptions (channel, address, conditions) VALUES (%s, %s, %s) RETURNING id, created_at",
                    (channel, address, payload),
                )
                subscription_id, created_at = cursor.fetchone()
                conn.commit()
        return SubscriptionRecord(
            id=subscription_id,
            channel=channel,
            address=address,
            conditions=conditions,
            created_at=str(created_at),
        )

    def list_subscriptions(self) -> List[SubscriptionRecord]:
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, channel, address, conditions, created_at, active FROM alert_subscriptions WHERE active = 1 ORDER BY created_at DESC"
            )
            rows = cursor.fetchall()
        records: List[SubscriptionRecord] = []
        for row in rows:
            if self._is_sqlite:
                conditions = json.loads(row[3])
            else:  # pragma: no cover
                conditions = row[3]
            records.append(
                SubscriptionRecord(
                    id=row[0],
                    channel=row[1],
                    address=row[2],
                    conditions=conditions,
                    created_at=str(row[4]),
                    active=bool(row[5]),
                )
            )
        return records

    def save_delivery(self, subscription_id: int, channel: str, address: str, payload: dict) -> DeliveryRecord:
        payload_json = json.dumps(payload)
        delivered_at = datetime.utcnow()
        with self._connection() as conn:
            cursor = conn.cursor()
            if self._is_sqlite:
                cursor.execute(
                    "INSERT INTO alert_delivery_log (subscription_id, payload, channel, address, delivery_status, delivered_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (subscription_id, payload_json, channel, address, "success", delivered_at.isoformat()),
                )
                conn.commit()
            else:  # pragma: no cover
                cursor.execute(
                    "INSERT INTO alert_delivery_log (subscription_id, payload, channel, address, delivery_status, delivered_at) VALUES (%s, %s, %s, %s, %s, %s)",
                    (subscription_id, payload_json, channel, address, "success", delivered_at),
                )
                conn.commit()
        return DeliveryRecord(
            subscription_id=subscription_id,
            channel=channel,
            address=address,
            payload=payload,
            delivered_at=str(delivered_at),
        )

    def deliveries(self, limit: int = 50) -> List[DeliveryRecord]:
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT subscription_id, payload, channel, address, delivered_at FROM alert_delivery_log ORDER BY delivered_at DESC LIMIT ?"
                if self._is_sqlite
                else "SELECT subscription_id, payload, channel, address, delivered_at FROM alert_delivery_log ORDER BY delivered_at DESC LIMIT %s",
                (limit,),
            )
            rows = cursor.fetchall()
        deliveries: List[DeliveryRecord] = []
        for row in rows:
            payload = json.loads(row[1]) if isinstance(row[1], str) else row[1]
            deliveries.append(
                DeliveryRecord(
                    subscription_id=row[0],
                    payload=payload,
                    channel=row[2],
                    address=row[3],
                    delivered_at=str(row[4]),
                )
            )
        return deliveries
