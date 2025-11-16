#!/usr/bin/env python3
"""
JWT Token Generator for RRIO Development

Generates JWT tokens for testing authentication in development.
In production, these tokens should come from a proper OAuth provider.

Usage:
    python scripts/generate_jwt.py --scope observatory:read
    python scripts/generate_jwt.py --scope ai:read --scope contributor:submit
    python scripts/generate_jwt.py --role judge
"""

import jwt
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import get_settings

def generate_token(scopes: list = None, roles: list = None, expires_in_hours: int = 24, user_id: str = "dev-user"):
    """Generate a JWT token for development/testing."""
    settings = get_settings()
    
    # Token payload
    now = datetime.utcnow()
    exp_time = now + timedelta(hours=expires_in_hours)
    payload = {
        'sub': user_id,
        'iss': settings.auth_issuer or 'https://auth.rrio.dev',
        'aud': settings.auth_audience or 'rrio-api',
        'iat': int(now.timestamp()),
        'exp': int(exp_time.timestamp()),
        'email': f'{user_id}@example.com',
        'name': f'Development User {user_id}'
    }
    
    # Add scopes if provided
    if scopes:
        payload['scope'] = ' '.join(scopes)
    
    # Add roles if provided
    if roles:
        payload['roles'] = roles
    
    # Get JWT secret
    jwt_secret = os.getenv('RIS_JWT_SECRET', 'development-secret')
    
    # Generate token
    token = jwt.encode(payload, jwt_secret, algorithm='HS256')
    
    return token, payload

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate JWT tokens for RRIO development')
    parser.add_argument('--scope', action='append', help='Add OAuth scope (can be used multiple times)')
    parser.add_argument('--role', action='append', help='Add user role (can be used multiple times)')
    parser.add_argument('--expires', type=int, default=24, help='Token expiration in hours (default: 24)')
    parser.add_argument('--user', default='dev-user', help='User ID (default: dev-user)')
    parser.add_argument('--verify', help='Verify an existing token instead of generating')
    
    args = parser.parse_args()
    
    if args.verify:
        # Verify existing token
        from app.core.auth import verify_token
        try:
            payload = verify_token(args.verify)
            print("Token is valid!")
            print("Payload:")
            for key, value in payload.items():
                print(f"  {key}: {value}")
        except Exception as e:
            print(f"Token verification failed: {e}")
        return
    
    # Generate new token
    token, payload = generate_token(
        scopes=args.scope,
        roles=args.role,
        expires_in_hours=args.expires,
        user_id=args.user
    )
    
    print("Generated JWT Token:")
    print("-" * 50)
    print(token)
    print()
    print("Token Payload:")
    for key, value in payload.items():
        if key == 'exp':
            # Convert timestamp to readable date
            exp_date = datetime.fromtimestamp(value)
            print(f"  {key}: {value} ({exp_date})")
        else:
            print(f"  {key}: {value}")
    print()
    print("Usage:")
    print(f"  curl -H 'Authorization: Bearer {token}' http://localhost:8000/api/v1/analytics/geri")
    print()
    print("Test in browser console:")
    print(f"  localStorage.setItem('rrio_token', '{token}');")

if __name__ == "__main__":
    main()