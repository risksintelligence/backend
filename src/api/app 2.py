import importlib
import logging
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)

def load_router(module_path: str, attr: str = "router") -> Optional[object]:
    try:
        module = importlib.import_module(module_path)
        return getattr(module, attr)
    except ModuleNotFoundError as exc:
        logger.warning("Skipping router %s due to missing dependency: %s", module_path, exc)
    except AttributeError:
        logger.warning("Router attribute '%s' missing in %s", attr, module_path)
    except Exception as exc:
        logger.error("Failed to load router %s: %s", module_path, exc, exc_info=True)
    return None


router_modules = [
    "src.api.v1.geri",
    "src.api.v1.ai",
    "src.api.v1.transparency",
    "src.api.v1.research",
    "src.api.v1.scenario",
    "src.api.v1.scenario_alerts",
    "src.api.v1.subscription",
    "src.api.v1.admin",
    "src.api.v1.alerts",
    "src.api.v1.monitoring",
    "src.api.v1.auth",
    "src.api.v1.backup",
    "src.api.v1.ml_admin",
    "src.api.v1.scenario_sharing",
    "src.api.v1.cron_admin",
    "src.api.v1.advanced_exports",
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
