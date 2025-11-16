from datetime import datetime, timedelta
from typing import List, Dict

from app.services.submissions import list_submissions


def get_component_history() -> List[Dict[str, object]]:
    base = datetime.utcnow()
    return [
        {
            "component": "Credit Spreads",
            "z_score": round(1.5 + i * 0.1, 2),
            "timestamp": (base - timedelta(days=5 - i)).isoformat() + 'Z',
        }
        for i in range(5)
    ]


def get_partner_highlights() -> List[Dict[str, str]]:
    submissions = list_submissions()
    return [
        {
            "title": entry.get("title", "Mission Insight"),
            "author": entry.get("author", "RRIO Fellow"),
            "mission": entry.get("mission", "Sector Mission"),
            "link": entry.get("link", "#"),
        }
        for entry in submissions[:3]
    ]


def get_newsletter_status() -> Dict[str, str]:
    return {
        "status": "Draft",
        "last_sent": (datetime.utcnow() - timedelta(days=7)).isoformat() + 'Z',
        "next_schedule": (datetime.utcnow() + timedelta(days=1)).isoformat() + 'Z',
    }


def get_scenario_prompts() -> List[Dict[str, str]]:
    return [
        {
            "prompt": "Anomaly score 0.78 detected in credit spreads - Evaluate liquidity stress scenario.",
            "timestamp": datetime.utcnow().isoformat() + 'Z'
        }
    ]
