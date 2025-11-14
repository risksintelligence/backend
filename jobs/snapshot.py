"""Cron job to capture GERI snapshots and store to file."""
import asyncio
from pathlib import Path
from datetime import datetime

from backend.src.services.geri_service import GERISnapshotService, seed_demo_data
from backend.src.monitoring.metrics import CRON_SNAPSHOT_RUNS

async def main() -> None:
    service = GERISnapshotService()
    seed_demo_data(service)
    payload, _, _ = service.build_payload()
    snapshots_dir = Path("snapshots")
    snapshots_dir.mkdir(exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = snapshots_dir / f"geri_snapshot_{timestamp}.json"
    path.write_text(str(payload), encoding="utf-8")
    CRON_SNAPSHOT_RUNS.inc()

if __name__ == "__main__":
    asyncio.run(main())
