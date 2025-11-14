import importlib
import logging
import os
import sys
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)

# Ensure local execution resolves `backend.*` imports
BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)


def load_router(module_path: str, attr: str = "router") -> Optional[object]:
    try:
        module = importlib.import_module(module_path)
        return getattr(module, attr)
    except ModuleNotFoundError:
        logger.warning("Router module missing: %s", module_path)
    except AttributeError:
        logger.warning("Router attribute '%s' missing in %s", attr, module_path)
    return None


router_modules = [
    "backend.src.api.v1.geri",
    "backend.src.api.v1.ai",
    "backend.src.api.v1.transparency",
    "backend.src.api.v1.research",
    "backend.src.api.v1.scenario",
    "backend.src.api.v1.scenario_alerts",
    "backend.src.api.v1.subscription",
    "backend.src.api.v1.admin",
    "backend.src.api.v1.alerts",
    "backend.src.api.v1.monitoring",
    "backend.src.api.v1.auth",
    "backend.src.api.v1.backup",
    "backend.src.api.v1.ml_admin",
    "backend.src.api.v1.scenario_sharing",
    "backend.src.api.v1.cron_admin",
    "backend.src.api.v1.advanced_exports",
]

app = FastAPI(title="RiskSX Intelligence System API")

raw_origins = os.getenv("RIS_ALLOWED_ORIGINS", "*")
allowed_origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]
wildcard = len(allowed_origins) == 1 and allowed_origins[0] == "*"
allow_credentials = not wildcard
origins_param = ["*"] if wildcard else allowed_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins_param,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

for module in router_modules:
    router = load_router(module)
    if router:
        app.include_router(router)


@app.get("/healthz")
async def health_check():
    return {"status": "ok"}
