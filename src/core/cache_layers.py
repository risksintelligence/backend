"""Multi-tier cache layer abstractions for RIS."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Protocol
import json


@dataclass
class CacheEntry:
    """Single cached value with metadata for TTL calculations."""

    value: Any
    stored_at: datetime
    soft_ttl: timedelta
    hard_ttl: timedelta
    metadata: Dict[str, Any] = field(default_factory=dict)

    def soft_expiry(self) -> datetime:
        return self.stored_at + self.soft_ttl

    def hard_expiry(self) -> datetime:
        return self.stored_at + self.hard_ttl

    def is_soft_expired(self, now: Optional[datetime] = None) -> bool:
        now = now or datetime.now(timezone.utc)
        return now >= self.soft_expiry()

    def is_hard_expired(self, now: Optional[datetime] = None) -> bool:
        now = now or datetime.now(timezone.utc)
        return now >= self.hard_expiry()


class CacheLayer(Protocol):
    async def get(self, key: str) -> Optional[CacheEntry]:
        ...

    async def set(self, key: str, entry: CacheEntry) -> None:
        ...


class MemoryCacheLayer:
    """In-process memory cache for sub-millisecond lookups."""

    def __init__(self) -> None:
        self._store: Dict[str, CacheEntry] = {}

    async def get(self, key: str) -> Optional[CacheEntry]:
        return self._store.get(key)

    async def set(self, key: str, entry: CacheEntry) -> None:
        self._store[key] = entry


class DictCacheLayer(MemoryCacheLayer):
    """Alias reserved for tests to simulate Redis/Postgres with dict backing."""

    pass


class FileCacheLayer:
    """Reads/writes JSON snapshots on disk for catastrophic fallback."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        safe_key = key.replace("/", "_")
        return self.directory / f"{safe_key}.json"

    async def get(self, key: str) -> Optional[CacheEntry]:
        path = self._path(key)
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        return CacheEntry(
            value=data["value"],
            stored_at=datetime.fromisoformat(data["stored_at"]),
            soft_ttl=timedelta(seconds=data["soft_ttl_seconds"]),
            hard_ttl=timedelta(seconds=data["hard_ttl_seconds"]),
            metadata=data.get("metadata", {}),
        )

    async def set(self, key: str, entry: CacheEntry) -> None:
        payload = {
            "value": entry.value,
            "stored_at": entry.stored_at.isoformat(),
            "soft_ttl_seconds": entry.soft_ttl.total_seconds(),
            "hard_ttl_seconds": entry.hard_ttl.total_seconds(),
            "metadata": entry.metadata,
        }
        self._path(key).write_text(json.dumps(payload, indent=2))
