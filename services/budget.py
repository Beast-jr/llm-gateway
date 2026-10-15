"""Daily token budgets, counted in Redis: one counter per user per UTC day.

Key: budget:{user_id}:{YYYY-MM-DD}. Every Groq answer adds its real tokens (prompt + completion).
The counter expires after 48h by itself, so old days clean themselves up.

If Redis is down or slow we "fail open": the request is allowed (and we log a warning), because
blocking every user whenever our counter is down is worse than a few unmetered requests, and Groq
has its own rate limits as a backstop.
"""
import logging
import math
from datetime import datetime, timedelta, timezone

import config
from services import cache

logger = logging.getLogger(__name__)

COUNTER_TTL_SECONDS = 48 * 60 * 60


def utcnow() -> datetime:
    """The current time in UTC. Tests replace this to move the clock without waiting."""
    return datetime.now(timezone.utc)


def usage_key(user_id: int, now: datetime | None = None) -> str:
    return f"budget:{user_id}:{(now or utcnow()):%Y-%m-%d}"


def seconds_until_reset(now: datetime | None = None) -> int:
    """Seconds from now until the next UTC midnight (when everyone's budget starts again)."""
    now = now or utcnow()
    midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return max(1, math.ceil((midnight - now).total_seconds()))


class BudgetExceeded(Exception):
    """The user has already used their whole daily budget. The route turns this into a 429."""

    def __init__(self, used: int, budget: int, retry_after: int):
        self.used, self.budget, self.retry_after = used, budget, retry_after
        hours, minutes = divmod(retry_after // 60, 60)
        super().__init__(
            f"Daily token budget used up ({used} of {budget} tokens). Cached answers still work. "
            f"The budget resets at 00:00 UTC, in about {hours}h {minutes}m."
        )


def check(used: int | None, budget: int) -> None:
    """Raise BudgetExceeded if the user is already at or over their budget. used=None (unknown) is allowed."""
    if used is not None and used >= budget:
        raise BudgetExceeded(used, budget, seconds_until_reset())


async def get_used(user_id: int) -> int | None:
    """Tokens this user has used today, or None if we can't tell (Redis down or slow)."""
    client = cache._get_redis()
    if client is None:
        return None
    try:
        raw = await cache._within_timeout(client.get(usage_key(user_id)), config.BUDGET_TIMEOUT_SECONDS)
    except Exception as e:
        logger.warning("Redis budget read failed, allowing the request (fail open): %r", e)
        return None
    return int(raw) if raw else 0


async def _increase(client, key: str, tokens: int) -> int:
    async with client.pipeline(transaction=True) as pipe:  # both commands in one round trip
        pipe.incrby(key, tokens)
        pipe.expire(key, COUNTER_TTL_SECONDS)
        total, _ = await pipe.execute()
    return int(total)


async def add_usage(user_id: int, tokens: int) -> int | None:
    """Add tokens to today's counter. Returns the new total, or None if Redis failed."""
    client = cache._get_redis()
    if client is None:
        return None
    try:
        return await cache._within_timeout(_increase(client, usage_key(user_id), tokens), config.BUDGET_TIMEOUT_SECONDS)
    except Exception as e:
        logger.warning("Redis budget write failed, these tokens were not counted: %r", e)
        return None
