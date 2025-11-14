"""Daily snapshot job for Global Economic Resilience Index (GERI).

This utility fetches the latest `/api/v1/analytics/geri` payload, validates the
schema, writes immutable JSON snapshots, and optionally publishes summaries for
press assets. It is safe to run in Render cron jobs or locally via `python`."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

SNAPSHOT_DIR = Path("snapshots")
LATEST_PATH = SNAPSHOT_DIR / "geri_latest.json"
DAILY_PATH_TEMPLATE = "geri_%Y%m%d.json"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def fetch_geri_snapshot(api_base: str, timeout: int = 10) -> Dict[str, Any]:
    url = f"{api_base.rstrip('/')}/api/v1/analytics/geri"
    request = Request(url, headers={"User-Agent": "RIS Snapshot Job/1.0"})
    try:
        with urlopen(request, timeout=timeout) as response:  # nosec: B310 (controlled URL)
            payload = response.read().decode("utf-8")
    except HTTPError as exc:  # pragma: no cover - network interaction
        raise RuntimeError(f"API responded with {exc.code}: {exc.reason}") from exc
    except URLError as exc:  # pragma: no cover - network interaction
        raise RuntimeError(f"Failed to reach GERI API: {exc.reason}") from exc

    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RuntimeError("API returned invalid JSON") from exc
    validate_snapshot(data)
    return data


def validate_snapshot(payload: Dict[str, Any]) -> None:
    required_keys = {"value", "band", "components", "data_freshness", "provenance"}
    missing = required_keys - payload.keys()
    if missing:
        raise RuntimeError(f"Snapshot missing keys: {', '.join(sorted(missing))}")

    if not isinstance(payload["components"], dict):
        raise RuntimeError("components must be a dict")
    if not isinstance(payload.get("value"), (int, float)):
        raise RuntimeError("value must be numeric")


@dataclass
class SnapshotWriter:
    output_dir: Path = SNAPSHOT_DIR

    def write(self, payload: Dict[str, Any]) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = _utc_now()
        daily_path = self.output_dir / timestamp.strftime(DAILY_PATH_TEMPLATE)
        self._write_file(daily_path, payload)
        self._write_file(LATEST_PATH, {"generated_at": timestamp.isoformat(), **payload})
        return daily_path

    @staticmethod
    def _write_file(path: Path, payload: Dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    api_base = os.environ.get("RIS_API_BASE_URL", "https://api.risksx.io")
    try:
        snapshot = fetch_geri_snapshot(api_base)
    except RuntimeError as exc:
        print(f"[snapshot] error: {exc}", file=sys.stderr)
        return 1

    writer = SnapshotWriter()
    output_path = writer.write(snapshot)
    print(f"[snapshot] wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
