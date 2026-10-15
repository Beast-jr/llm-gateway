import asyncio
import hashlib
import json
import logging
import os

from dotenv import load_dotenv
from redis.asyncio import Redis
from redis.asyncio.retry import Retry
from redis.backoff import NoBackoff

import config

load_dotenv()
logger = logging.getLogger(__name__)

_redis: Redis | None = None


def _get_redis() -> Redis | None:
    # Connect only when first needed; no REDIS_URL means "no cache" (not an error).
    global _redis
    if _redis is None and os.getenv("REDIS_URL"):
        _redis = Redis.from_url(
            os.environ["REDIS_URL"],
            decode_responses=True,
            socket_keepalive=True,  # TCP keepalive: idle connections are less likely to be dropped
            health_check_interval=30,  # after 30s idle, check the connection before using it
            retry=Retry(NoBackoff(), 1),  # a dead connection: reconnect and retry ONCE, immediately
        )
    return _redis


async def _within_timeout(coro, timeout: float | None = None):
    """Like asyncio.wait_for, but a timeout does NOT cancel the work. A reconnect that is still
    in progress keeps going in the background, so the NEXT request finds a working connection."""
    task = asyncio.ensure_future(coro)
    task.add_done_callback(lambda t: t.cancelled() or t.exception())  # mark errors as "seen"
    return await asyncio.wait_for(asyncio.shield(task), timeout or config.REDIS_TIMEOUT_SECONDS)


async def warm_up() -> None:
    """Open the Redis connection now. A brand-new connection takes ~3s, far over our 0.5s limit,
    and a timeout cancels the connect, so without this the first requests could never get a cache."""
    client = _get_redis()
    if client is not None:
        await asyncio.wait_for(client.ping(), 15)


def normalize(question: str) -> str:
    """'  What is   RAG? ' -> 'what is rag?' so tiny differences don't miss the cache."""
    return " ".join(question.lower().split())


def make_key(question: str, model: str, scope: str | None = None) -> str:
    # The model is part of the key: a different model should give a fresh answer.
    # So is the scope (e.g. "user:5"): the same question from two users gets two separate entries.
    digest = hashlib.sha256(f"{model}|{scope or 'shared'}|{normalize(question)}".encode()).hexdigest()
    return f"chat:{digest}"


async def get_cached(question: str, model: str, scope: str | None = None) -> dict | None:
    """Return the saved result, or None on a miss OR if Redis is down/slow."""
    client = _get_redis()
    if client is None:
        return None
    try:
        raw = await _within_timeout(client.get(make_key(question, model, scope)))
    except Exception as e:  # down, slow (TimeoutError), bad URL... never break /chat
        logger.warning("Redis read failed, skipping cache: %r", e)
        return None
    return json.loads(raw) if raw else None


async def set_cached(question: str, model: str, result: dict, scope: str | None = None) -> None:
    """Save a result with an expiry. Failures are logged and ignored."""
    client = _get_redis()
    if client is None:
        return
    try:
        await _within_timeout(
            client.set(make_key(question, model, scope), json.dumps(result), ex=config.CACHE_TTL_SECONDS)
        )
    except Exception as e:
        logger.warning("Redis write failed, answer not cached: %r", e)
