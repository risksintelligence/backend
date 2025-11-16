from fastapi import Depends, HTTPException
from fastapi.security import APIKeyHeader

from app.core.config import get_settings

api_key_header = APIKeyHeader(name='X-RRIO-REVIEWER', auto_error=False)

settings = get_settings()


def require_reviewer(api_key: str = Depends(api_key_header)) -> None:
    if not api_key or api_key != settings.reviewer_api_key:
        raise HTTPException(status_code=401, detail='Reviewer authentication failed')
