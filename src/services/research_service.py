"""Research API support services."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

@dataclass
class HistoryRecord:
  ts_utc: str
  value: float
  band: str
  components: Dict[str, float]
  stale: bool


class ResearchService:
  def __init__(self) -> None:
    self._history: List[HistoryRecord] = []

  def record_snapshot(self, payload: Dict[str, object]) -> None:
    record = HistoryRecord(
      ts_utc=payload.get("snapshot_ts_utc") or datetime.utcnow().isoformat(),
      value=float(payload["value"]),
      band=str(payload["band"]),
      components={k: float(v) for k, v in payload.get("components", {}).items()},
      stale=bool(payload.get("stale")),
    )
    self._history.append(record)
    self._history = self._history[-1000:]

  def query_history(self, start: Optional[datetime], end: Optional[datetime], limit: int) -> List[HistoryRecord]:
    records = self._history
    def within(record: HistoryRecord) -> bool:
      ts = datetime.fromisoformat(record.ts_utc)
      if start and ts < start:
        return False
      if end and ts > end:
        return False
      return True
    filtered = [r for r in records if within(r)]
    return filtered[-limit:]

  def methodology_documents(self) -> List[Dict[str, str]]:
    docs_dir = Path("docs/methodology")
    documents = []
    for path in docs_dir.glob("*.md"):
      stat = path.stat()
      documents.append({
        "title": path.stem.replace("_", " ").upper(),
        "path": str(path),
        "updated_at": datetime.fromtimestamp(stat.st_mtime).date().isoformat(),
      })
    documents.sort(key=lambda doc: doc["path"])
    return documents


_RESEARCH_SERVICE: Optional[ResearchService] = None


def get_research_service() -> ResearchService:
  global _RESEARCH_SERVICE
  if _RESEARCH_SERVICE is None:
    _RESEARCH_SERVICE = ResearchService()
  return _RESEARCH_SERVICE
