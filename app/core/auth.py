import os
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

SECRET = os.getenv('RRIO_JWT_SECRET', 'development-secret')
ALGORITHM = 'HS256'

bearer_scheme = HTTPBearer(auto_error=False)

def require_judge(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    if credentials is None:
        raise HTTPException(status_code=401, detail='Missing token')
    try:
        payload = jwt.decode(credentials.credentials, SECRET, algorithms=[ALGORITHM])
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail='Invalid token')
    if 'roles' not in payload or 'judge' not in payload['roles']:
        raise HTTPException(status_code=403, detail='Insufficient permissions')
    return payload
