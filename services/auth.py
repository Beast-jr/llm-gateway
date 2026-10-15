"""API keys: create them, check them, and protect routes with them.

We never store the plain key, only its SHA-256 hash (a one-way scramble). The key is long and
random, so a plain fast hash is fine here (unlike passwords people choose themselves).
"""
import asyncio
import hashlib
import hmac
import logging
import secrets
import time
from dataclasses import dataclass

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select

import config
from models.user import User
from services.db import get_session, retry_once_if_dropped

logger = logging.getLogger(__name__)

KEY_PREFIX = "gw_"
# auto_error=False: we raise our own 401 with a clear message. This also adds the Authorize button to /docs.
_bearer = HTTPBearer(auto_error=False, description="Paste your API key (it starts with gw_). Do not type 'Bearer'.")


@dataclass(frozen=True)
class AuthUser:
    """What the rest of the app needs to know about the caller."""
    id: int
    name: str
    daily_token_budget: int
    is_active: bool


_cache: dict[str, tuple[float, AuthUser]] = {}  # key hash -> (time saved, user); lives in memory only
_CACHE_MAX = 1000


def generate_key() -> str:
    return KEY_PREFIX + secrets.token_urlsafe(32)


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def create_user(name: str, budget: int) -> tuple[AuthUser, str]:
    """Make a user and return (user, plain key). The plain key exists only in this return value."""
    key = generate_key()
    with get_session() as session:
        row = User(name=name, api_key_hash=hash_key(key), key_prefix=key[:8], daily_token_budget=budget)
        session.add(row)
        session.commit()
        return AuthUser(row.id, row.name, row.daily_token_budget, row.is_active), key


def _find_user(key_hash: str) -> AuthUser | None:
    """Sync DB lookup (run it with asyncio.to_thread)."""
    with get_session() as session:
        row = session.execute(select(User).where(User.api_key_hash == key_hash)).scalar_one_or_none()
        # Explicit constant-time comparison, so response time can't leak how much of a hash matched.
        if row is None or not hmac.compare_digest(row.api_key_hash, key_hash):
            return None
        return AuthUser(row.id, row.name, row.daily_token_budget, row.is_active)


async def authenticate(key: str) -> AuthUser | None:
    """Return the user for this key (None if the key is unknown). Remembers answers for AUTH_CACHE_SECONDS."""
    if not key.startswith(KEY_PREFIX):
        return None  # obviously not ours: don't bother the database
    key_hash = hash_key(key)
    hit = _cache.get(key_hash)
    if hit and time.monotonic() - hit[0] < config.AUTH_CACHE_SECONDS:
        return hit[1]
    user = await asyncio.to_thread(retry_once_if_dropped, lambda: _find_user(key_hash))
    if user is not None:  # unknown keys are not remembered, so random guesses can't fill the memory
        if len(_cache) >= _CACHE_MAX:
            _cache.clear()
        _cache[key_hash] = (time.monotonic(), user)
    return user


def _unauthorized(message: str) -> HTTPException:
    return HTTPException(status_code=401, detail=message, headers={"WWW-Authenticate": "Bearer"})


async def current_user(credentials: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> AuthUser:
    """FastAPI dependency: put it on a route to require a valid key."""
    if credentials is None:
        raise _unauthorized("Missing API key. Send the header 'Authorization: Bearer <your key>'.")
    try:
        user = await authenticate(credentials.credentials)
    except Exception:
        logger.exception("Could not check the API key")
        # We can't tell who this is, so we must not let them in. 503 = "try again", not "your key is wrong".
        raise HTTPException(status_code=503, detail="Could not verify your API key right now. Please try again.")
    if user is None:
        raise _unauthorized("Invalid API key.")
    if not user.is_active:
        raise _unauthorized("This API key is disabled.")
    return user
