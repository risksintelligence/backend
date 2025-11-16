import asyncio

from app.services.ingestion import ingest_local_series

async def run_worker(interval_seconds: int = 3600) -> None:
    while True:
        ingest_local_series()
        await asyncio.sleep(interval_seconds)
