"""
Security utilities for API authentication and rate limiting.
"""
from fastapi import Security, HTTPException, status, Request, Depends
from fastapi.security import APIKeyHeader
from typing import Optional
import time
from collections import defaultdict
from app.config import settings

# API Key authentication
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Rate limiting storage (in-memory, resets on restart)
_rate_limit_storage = defaultdict(list)
_rate_limit_enabled = settings.RATE_LIMIT_ENABLED
_rate_limit_window = 60  # 1 minute window
_rate_limit_max_requests = settings.RATE_LIMIT_PER_MINUTE


def get_client_ip(request: Request) -> str:
    """Extract client IP address from request."""
    # Check for forwarded IP (from proxy/load balancer)
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    # Fallback to direct client
    if request.client:
        return request.client.host
    
    return "unknown"


def check_rate_limit(request: Request) -> bool:
    """
    Check if request is within rate limit.
    
    Returns:
        True if within limit, False if rate limited
    """
    if not _rate_limit_enabled:
        return True
    
    client_ip = get_client_ip(request)
    now = time.time()
    
    # Clean old entries (older than window)
    _rate_limit_storage[client_ip] = [
        timestamp for timestamp in _rate_limit_storage[client_ip]
        if now - timestamp < _rate_limit_window
    ]
    
    # Check if over limit
    if len(_rate_limit_storage[client_ip]) >= _rate_limit_max_requests:
        return False
    
    # Record this request
    _rate_limit_storage[client_ip].append(now)
    return True


def verify_api_key(
    api_key: Optional[str] = Security(api_key_header),
    request: Request = None
) -> str:
    """
    Verify API key and rate limit.
    
    Args:
        api_key: API key from X-API-Key header
        request: FastAPI request object (injected via dependency)
    
    Returns:
        API key if valid
    
    Raises:
        HTTPException: If API key invalid or rate limited
    """
    # Note: request will be injected via require_api_key dependency
    # This function signature works with Security() for header extraction
    
    # If no API keys configured, allow all (backward compatibility for development)
    valid_keys = settings.api_keys_list
    if not valid_keys:
        return api_key or "development"
    
    # Require API key if keys are configured
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required. Provide X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    if api_key not in valid_keys:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )
    
    return api_key


def require_api_key(
    request: Request,
    api_key: str = Depends(verify_api_key)
) -> str:
    """
    Dependency to require API key authentication with rate limiting.
    Use this for protected endpoints.
    
    This wrapper adds rate limiting before API key verification.
    """
    # Check rate limit first (applies to all requests)
    if not check_rate_limit(request):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Maximum {_rate_limit_max_requests} requests per minute."
        )
    
    # API key verification happens via verify_api_key dependency
    return api_key


def optional_api_key(api_key: Optional[str] = Security(api_key_header)) -> Optional[str]:
    """
    Dependency for optional API key (doesn't fail if missing).
    Use this for endpoints that have enhanced features with API key.
    """
    if not api_key:
        return None
    
    valid_keys = settings.api_keys_list
    if valid_keys and api_key in valid_keys:
        return api_key
    
    return None

