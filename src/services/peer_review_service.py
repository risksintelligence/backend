"""Peer review storage service."""
from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import List

from backend.src.monitoring.metrics import PEER_REVIEWS_TOTAL


@dataclass
class PeerReview:
    id: int
    reviewer_name: str
    reviewer_email: str | None
    decision: str
    comments: str | None
    created_at: str


class PeerReviewService:
    def __init__(self, db_path: str | None = None) -> None:
        path = db_path or os.getenv("PEER_REVIEW_DB", ":memory:")
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS peer_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                reviewer_name TEXT NOT NULL,
                reviewer_email TEXT,
                decision TEXT NOT NULL,
                comments TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._conn.commit()

    def submit(self, name: str, email: str | None, decision: str, comments: str | None) -> PeerReview:
        cursor = self._conn.cursor()
        cursor.execute(
            "INSERT INTO peer_reviews (reviewer_name, reviewer_email, decision, comments) VALUES (?, ?, ?, ?)",
            (name, email, decision, comments),
        )
        self._conn.commit()
        review_id = cursor.lastrowid
        cursor.execute("SELECT * FROM peer_reviews WHERE id = ?", (review_id,))
        row = cursor.fetchone()
        PEER_REVIEWS_TOTAL.inc()
        return PeerReview(
            id=row["id"],
            reviewer_name=row["reviewer_name"],
            reviewer_email=row["reviewer_email"],
            decision=row["decision"],
            comments=row["comments"],
            created_at=row["created_at"],
        )

    def list_reviews(self, limit: int = 20) -> List[PeerReview]:
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM peer_reviews ORDER BY created_at DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        return [
            PeerReview(
                id=row["id"],
                reviewer_name=row["reviewer_name"],
                reviewer_email=row["reviewer_email"],
                decision=row["decision"],
                comments=row["comments"],
                created_at=row["created_at"],
            )
            for row in rows
        ]
