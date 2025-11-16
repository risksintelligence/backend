import os
from typing import Optional, List
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

from app.core.config import get_settings

settings = get_settings()
bearer_scheme = HTTPBearer(auto_error=False)

def get_jwt_settings():
    """Get JWT configuration from settings."""
    return {
        'secret': os.getenv('RIS_JWT_SECRET'),
        'algorithm': 'HS256',
        'issuer': settings.auth_issuer,
        'audience': settings.auth_audience
    }

def verify_token(token: str) -> Optional[dict]:
    """Verify JWT token and return payload."""
    try:
        jwt_config = get_jwt_settings()
        
        # Check if JWT secret is configured
        if not jwt_config['secret']:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="JWT secret not configured"
            )
        
        # Decode with optional issuer/audience validation
        decode_kwargs = {
            'token': token,
            'key': jwt_config['secret'],
            'algorithms': [jwt_config['algorithm']]
        }
        
        # Only add issuer/audience if they are configured
        if jwt_config['issuer']:
            decode_kwargs['issuer'] = jwt_config['issuer']
        if jwt_config['audience']:
            decode_kwargs['audience'] = jwt_config['audience']
            
        payload = jwt.decode(**decode_kwargs)
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

def require_scopes(required_scopes: List[str]):
    """Require specific OAuth scopes."""
    def dependency(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
        if not credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing authorization header"
            )
        
        payload = verify_token(credentials.credentials)
        token_scopes = payload.get('scope', '').split()
        
        for required_scope in required_scopes:
            if required_scope not in token_scopes:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Insufficient permissions. Required scope: {required_scope}"
                )
        
        return payload
    
    return dependency

# Specific scope requirements per API documentation
require_observatory_read = require_scopes(['observatory:read'])
require_ai_read = require_scopes(['ai:read']) 
require_contributor_submit = require_scopes(['contributor:submit'])

# Legacy judge requirement for backward compatibility
def require_judge(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    """Legacy judge requirement - maps to contributor:submit scope."""
    if not credentials:
        raise HTTPException(status_code=401, detail='Missing token')
    
    payload = verify_token(credentials.credentials)
    
    # Check for judge role or contributor:submit scope
    has_judge_role = 'roles' in payload and 'judge' in payload['roles']
    has_submit_scope = 'contributor:submit' in payload.get('scope', '').split()
    
    if not (has_judge_role or has_submit_scope):
        raise HTTPException(
            status_code=403, 
            detail='Insufficient permissions - requires judge role or contributor:submit scope'
        )
    
    return payload

def optional_auth(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    """Optional authentication - returns None if no token provided."""
    if not credentials:
        return None
    
    try:
        return verify_token(credentials.credentials)
    except HTTPException:
        return None
